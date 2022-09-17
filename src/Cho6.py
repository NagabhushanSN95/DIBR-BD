# Shree KRISHNAya Namaha
# Updated from Cho5.py, and optimized
# Based on the paper: "Hole Filling Method for Depth Image Based Rendering Based on Boundary Decision" - SP Letters 2017
# Author: Nagabhushan S N
# Last Modified: 07/03/2021

import time
from pathlib import Path

import numpy
from matplotlib import pyplot
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from tqdm import tqdm
from sklearn.cluster import KMeans

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# TODO: Not sure what the default beta value is
class Inpainter():
    def __init__(self, image, mask, infilled_depth, patch_size=9, search_area_size=255, depth_map_bit_depth=16, beta=1,
                 sigma1=0.03, sigma2=7, lamda=10, T1=4, T2=0.4, w_bg=9, w_fg=1, plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.depth = infilled_depth
        self.max_depth = numpy.max(self.depth)
        self.patch_size = patch_size
        self.neighborhood_size = patch_size * 3
        self.search_area_size = search_area_size
        self.alpha = 2 ** depth_map_bit_depth - 1
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.lamda = lamda
        self.T1 = T1
        self.T2 = T2
        self.w_bg = w_bg
        self.w_fg = w_fg
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.working_opposite_indices = None
        self.front = None
        self.confidence = None
        self.data = None
        self.level_regularity = None
        self.background = None
        self.priority = None

        # Convergence variables
        self.num_prev_hole_pixels = -1
        self.no_improv_count = 0
        self.no_improv_threshold = 5
        return

    def inpaint(self):
        """ Compute the new image and return it """

        if self.mask.sum() == 0:
            return self.image

        self._validate_inputs()
        self._initialize_attributes()

        keep_going = True
        progress_bar = tqdm(total=self.mask.sum())
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()
            target_pixel = self._find_highest_priority_pixel()
            source_patch = self._find_source_patch(target_pixel)
            self._update_image(target_pixel, source_patch)
            self.update_opposite_indices(target_pixel)

            finished, progress = self._finished()
            progress_bar.update(progress)
            keep_going = not finished
        progress_bar.close()
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        pyplot.clf()
        pyplot.imshow(image)
        pyplot.draw()
        pyplot.pause(0.001)

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]

        self.confidence = (1 - self.mask).astype(float)
        self.data = numpy.zeros([height, width])
        self.level_regularity = numpy.zeros([height, width])
        self.background = numpy.zeros([height, width])

        self.working_image = numpy.copy(self.image)
        self.working_mask = numpy.copy(self.mask)
        self.working_opposite_indices = self.compute_opposite_indices(1 - self.mask)
        self.num_prev_hole_pixels = self.working_mask.sum()
        return

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')

    def _update_priority(self):
        ps = self.patch_size
        ps1 = (ps - 1) // 2
        h, w = self.confidence.shape
        h1, w1 = h + ps - 1, w + ps - 1
        image_mask = numpy.zeros(shape=(h1, w1), dtype=numpy.bool)
        image_mask[ps1:-ps1, ps1:-ps1] = 1
        known_mask = 1 - self.working_mask
        padded_known_mask = numpy.zeros(shape=(h1, w1), dtype=known_mask.dtype)
        padded_known_mask[ps1:-ps1, ps1:-ps1] = known_mask
        padded_confidence = numpy.zeros(shape=(h1, w1), dtype=self.confidence.dtype)
        padded_confidence[ps1:-ps1, ps1:-ps1] = self.confidence
        padded_depth = numpy.zeros(shape=(h1, w1), dtype=self.depth.dtype)
        padded_depth[ps1:-ps1, ps1:-ps1] = self.depth

        front_positions = numpy.argwhere(self.front == 1)
        num_front_pixels = front_positions.shape[0]
        if num_front_pixels == 0:
            return
        y_start = front_positions[:, 0]
        x_start = front_positions[:, 1]

        normal = self._calc_normal_matrix()
        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None
        gradient = numpy.stack(numpy.gradient(grey_image), axis=2)
        gradient[numpy.isnan(gradient)] = 0
        gradient_mag = numpy.sqrt(gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2)
        max_gradient = numpy.zeros([h, w, 2])
        padded_gradient = numpy.zeros(shape=(h1, w1, 2), dtype=gradient.dtype)
        padded_gradient[ps1:-ps1, ps1:-ps1, :] = gradient
        padded_gradient_mag = numpy.zeros(shape=(h1, w1), dtype=gradient_mag.dtype)
        padded_gradient_mag[ps1:-ps1, ps1:-ps1] = gradient_mag

        right_is_opposite = self.working_mask[front_positions[:, 0], front_positions[:, 1] + 1] == 1
        left_right_indices = self.working_opposite_indices[y_start, x_start]
        opposite_points_x = numpy.where(right_is_opposite, left_right_indices[:, 1], left_right_indices[:, 0])
        opposite_points = numpy.stack([front_positions[:, 0], opposite_points_x], axis=1)
        y_start1 = opposite_points[:, 0]
        x_start1 = opposite_points[:, 1]

        image_mask_patches = numpy.zeros(shape=(num_front_pixels, ps, ps))
        known_mask_patches = numpy.zeros(shape=(num_front_pixels, ps, ps))
        new_confidence = numpy.copy(self.confidence)
        old_confidence_patches = numpy.zeros(shape=(num_front_pixels, ps, ps))
        grad_patches = numpy.zeros(shape=(num_front_pixels, ps, ps, 2))
        grad_mag_patches = numpy.zeros(shape=(num_front_pixels, ps, ps))
        depth_patches = numpy.zeros(shape=(num_front_pixels, ps, ps))
        opposite_depth_patches = numpy.zeros(shape=(num_front_pixels, ps, ps))

        for i in range(ps):
            for j in range(ps):
                image_mask_patches[:, i, j] = image_mask[y_start + i, x_start + j]
                known_mask_patches[:, i, j] = padded_known_mask[y_start + i, x_start + j]
                old_confidence_patches[:, i, j] = padded_confidence[y_start + i, x_start + j]
                grad_patches[:, i, j] = padded_gradient[y_start + i, x_start + j]
                grad_mag_patches[:, i, j] = padded_gradient_mag[y_start + i, x_start + j]
                depth_patches[:, i, j] = padded_depth[y_start + i, x_start + j]
                opposite_depth_patches[:, i, j] = padded_depth[y_start1 + i, x_start1 + j]

        patch_areas = numpy.sum(image_mask_patches, axis=(1, 2))
        new_confidence_points = numpy.sum(old_confidence_patches, axis=(1, 2)) / patch_areas
        new_confidence[y_start, x_start] = new_confidence_points
        self.confidence = new_confidence

        grad_patches_vec = numpy.reshape(grad_patches, newshape=(grad_patches.shape[0], -1, 2))
        grad_mag_patches_vec = numpy.reshape(grad_mag_patches, newshape=(grad_mag_patches.shape[0], -1))
        grad_max_pos = numpy.argmax(grad_mag_patches_vec, axis=1)
        max_gradient[y_start, x_start] = grad_patches_vec[
            numpy.arange(grad_max_pos.size)[:, None], grad_max_pos[:, None], numpy.arange(2)[None]]
        normal_gradient = normal * max_gradient
        # 0.001 added to be sure to have a greater than 0 data
        self.data = numpy.sqrt(normal_gradient[:, :, 0] ** 2 + normal_gradient[:, :, 1] ** 2) + 0.001

        mean_depths = numpy.sum(depth_patches, axis=(1, 2)) / patch_areas
        sse = numpy.sum(numpy.square(depth_patches - mean_depths[:, None, None]) * known_mask_patches, axis=(1, 2))
        self.level_regularity[y_start, x_start] = patch_areas / (patch_areas + sse)

        B_l1 = numpy.zeros(self.mask.shape)
        B_l1[y_start, x_start] = (numpy.max(self.depth) - mean_depths) / self.alpha
        opposite_mean_depths = numpy.sum(opposite_depth_patches, axis=(1, 2)) / patch_areas
        B_n1 = numpy.zeros(self.mask.shape)
        with numpy.errstate(invalid='ignore', over='ignore'):
            B_n1[y_start, x_start] = 1 / (1 + numpy.exp(-(mean_depths - opposite_mean_depths) / (self.sigma1 ** 2)))
        self.background = B_l1 * B_n1

        self.priority = self.confidence * self.data * self.level_regularity * self.background * self.front
        return

    def _calc_normal_matrix(self):
        x_kernel = numpy.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = numpy.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = numpy.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = numpy.sqrt(y_normal ** 2 + x_normal ** 2).reshape(height, width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal / norm
        return unit_normal

    def _get_opposite_point(self, point):
        y, x = point
        if x + 1 == self.working_mask.shape[1]:
            return point
        if self.working_mask[y, x + 1] == 1:
            # The current point is at the left edge of hole
            candidates = numpy.argwhere(self.working_mask[y, x + 1:] == 0)
            if candidates.size > 0:
                x1 = numpy.min(candidates) + x + 1
            else:
                x1 = self.working_mask.shape[1] - 1
        else:
            # The current point is at the right edge of hole
            candidates = numpy.argwhere(self.working_mask[y, :x + 1] == 0)
            if candidates.size > 0:
                x1 = numpy.max(candidates)
            else:
                x1 = 0
        opposite_point = [y, x1]
        return opposite_point

    @staticmethod
    def compute_opposite_indices(mask):
        r, c = numpy.nonzero(mask)

        left_ind = numpy.zeros(mask.shape, dtype=int)
        left_ind[r, c] = c
        numpy.maximum.accumulate(left_ind, axis=1, out=left_ind)

        right_ind = numpy.zeros(mask.shape, dtype=int)
        right_ind[r, c] = mask.shape[1] - c
        right_ind = mask.shape[1] - numpy.maximum.accumulate(right_ind[:, ::-1], axis=1)[:, ::-1]

        opposite_indices = numpy.stack([left_ind, right_ind], axis=2)
        return opposite_indices

    def update_opposite_indices(self, target_pixel):
        target_patch_coords = self._get_patch(target_pixel)
        (y_start, y_end), (x_start, x_end) = target_patch_coords
        p_h = y_end - y_start + 1
        p_w = x_end - x_start + 1
        new_known_indices = numpy.arange(p_w)[None, :, None] + x_start
        new_known_indices = numpy.repeat(new_known_indices, repeats=p_h, axis=0)
        self.working_opposite_indices[y_start:y_end + 1, x_start:x_end + 1] = new_known_indices
        self.working_opposite_indices[y_start:y_end + 1, x_end + 1:, 0][
            self.working_opposite_indices[y_start:y_end + 1, x_end + 1:, 0] <= x_end
            ] = x_end
        self.working_opposite_indices[y_start:y_end+1, :x_start, 1][
            self.working_opposite_indices[y_start:y_end+1, :x_start, 1] >= x_start
        ] = x_start
        return

    def _find_highest_priority_pixel(self):
        point = numpy.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel, search_area_size=None):
        target_patch = self._get_patch(target_pixel)
        if search_area_size is None:
            search_area_size = self.search_area_size
        working_image_ngbrhd = self._get_search_area(target_pixel, search_area_size)
        height, width = self._patch_shape(working_image_ngbrhd)
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image)
        boundary_decision = self._compute_boundary_decision(target_pixel, target_patch)

        y1, x1 = working_image_ngbrhd[0][0], working_image_ngbrhd[1][0]
        for y in range(y1, y1 + height - patch_height + 1):
            for x in range(x1, x1 + width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height - 1],
                    [x, x + patch_width - 1]
                ]
                if self._patch_data(self.working_mask, source_patch).sum() != 0:
                    continue

                difference = self._calc_patch_difference(lab_image, target_patch, source_patch, boundary_decision)

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        if best_match is None:
            new_search_area_size = search_area_size * 2 + 1
            best_match = self._find_source_patch(target_pixel, new_search_area_size)
        return best_match

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = numpy.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)

        self._copy_to_patch(self.working_image, target_patch, new_data)
        self._copy_to_patch(self.working_mask, target_patch, 0)

    def _get_patch(self, point):
        half_patch_size = (self.patch_size - 1) // 2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height - 1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width - 1)
            ]
        ]
        return patch

    def _get_search_area(self, point, search_area_size):
        half_patch_size = (search_area_size - 1) // 2
        height, width = self.working_image.shape[:2]
        search_area = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height - 1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width - 1)
            ]
        ]
        return search_area

    def _get_neighborhood(self, point):
        half_patch_size = (self.neighborhood_size - 1) // 2
        height, width = self.working_image.shape[:2]
        neighborhood = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height - 1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width - 1)
            ]
        ]
        return neighborhood

    def _calc_patch_difference(self, lab_image, target_patch, source_patch, h):
        distance_tilde = self._compute_distance_tilde(target_patch, source_patch, lab_image, h)

        # Depth Variance
        source_depth = self._patch_data(self.depth, source_patch)
        source_depth_var = numpy.var(source_depth)

        total_distance = distance_tilde + self.lamda * source_depth_var
        return total_distance

    def _compute_boundary_decision(self, target_point: tuple, target_patch):
        mu_p = self._compute_mu_p(target_point, target_patch)
        target_depth = self._patch_data(self.depth, target_patch)
        Zp_var = numpy.var(target_depth)
        h = 1 if ((Zp_var > self.T1) and (mu_p < self.T2)) else 0
        return h

    def _compute_mu_p(self, target_point: tuple, target_patch: numpy.ndarray):
        Zp = self._patch_data(self.depth, target_patch)
        neighborhood = self._get_neighborhood(target_point)
        y1, x1 = neighborhood[0][0], neighborhood[1][0]
        y2, x2 = neighborhood[0][1], neighborhood[1][1]
        patch_height, patch_width = self._patch_shape(target_patch)

        Zq_patches = numpy.zeros(
            shape=(y2 - y1 - patch_height + 1, x2 - x1 - patch_width + 1, patch_height, patch_width))
        for i in range(patch_height):
            for j in range(patch_width):
                Zq_patches[:, :, i, j] = self.depth[y1 + i:y2 - patch_height + i + 1, x1 + j:x2 - patch_width + j + 1]
        mse = numpy.mean(numpy.square(Zp[None, None].astype('float') - Zq_patches.astype('float')), axis=(2, 3))
        mu_pq = numpy.exp(-mse / (self.sigma2 ** 2))
        mu_p = numpy.mean(mu_pq)
        return mu_p

    def _compute_distance_tilde(self, target_patch, source_patch, lab_image, h):
        target_data = self._patch_data(lab_image, target_patch)
        source_data = self._patch_data(lab_image, source_patch)
        target_depth = self._patch_data(self.depth, target_patch)
        source_depth = self._patch_data(self.depth, source_patch)
        known_mask = 1 - self._patch_data(self.working_mask, target_patch)

        if h == 1:
            # Apply K-means to segregate into background and foreground (k=2)
            fg_mask = self._compute_fg_mask(target_depth)
            bg_mask = ~fg_mask
            fg_mask = fg_mask & known_mask
            bg_mask = bg_mask & known_mask
            de_fg = self._compute_de(target_data, source_data, target_depth, source_depth, fg_mask)
            de_bg = self._compute_de(target_data, source_data, target_depth, source_depth, bg_mask)
            distance_tilde = self.w_fg * de_fg + self.w_bg * de_bg
        else:
            distance_tilde = self._compute_de(target_data, source_data, target_depth, source_depth, known_mask)
        return distance_tilde

    def _compute_de(self, target_data: numpy.ndarray, source_data: numpy.ndarray, target_depth: numpy.ndarray,
                    source_depth: numpy.ndarray, mask: numpy.ndarray):
        rgb_mask = self._to_rgb(mask)
        image_distance = numpy.sum(numpy.square(target_data - source_data) * rgb_mask)
        depth_distance = numpy.sum(numpy.square(target_depth - source_depth) * mask)
        de = image_distance + self.beta * depth_distance
        return de

    @staticmethod
    def _compute_fg_mask(depth_patch: numpy.ndarray):
        k_means = KMeans(n_clusters=2, n_init=1)
        k_means.fit(depth_patch.reshape(-1, 1))
        fg_mask = k_means.labels_.reshape(depth_patch.shape) == 1
        bg_mask = ~fg_mask
        fg_avg_depth = numpy.sum(depth_patch * fg_mask) / numpy.sum(fg_mask)
        bg_avg_depth = numpy.sum(depth_patch * bg_mask) / numpy.sum(bg_mask)
        if fg_avg_depth > bg_avg_depth:
            fg_mask = ~fg_mask
        return fg_mask

    @staticmethod
    def _mse(patch1: numpy.ndarray, patch2: numpy.ndarray):
        mse = numpy.mean(numpy.square(patch1.astype('float') - patch2.astype('float')))
        return mse

    def _finished(self):
        remaining = self.working_mask.sum()
        progress_update = self.num_prev_hole_pixels - remaining

        finished = remaining == 0
        if not finished:
            if remaining == self.num_prev_hole_pixels:
                self.no_improv_count += 1
            else:
                self.no_improv_count = 0
            if self.no_improv_count == self.no_improv_threshold:
                finished = True
            self.num_prev_hole_pixels = remaining
        return finished, progress_update

    @staticmethod
    def _patch_area(patch):
        return (1 + patch[0][1] - patch[0][0]) * (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1 + patch[0][1] - patch[0][0]), (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[patch[0][0]:patch[0][1] + 1, patch[1][0]:patch[1][1] + 1]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[dest_patch[0][0]:dest_patch[0][1] + 1, dest_patch[1][0]:dest_patch[1][1] + 1] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
