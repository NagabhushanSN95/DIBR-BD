# Shree KRISHNAya Namaha
# Updated from Criminisi2.py
# Based on the paper "Depth-aided image inpainting for Novel View Synthesis" - 2010
# Author: Nagabhushan S N
# Last Modified: 04/03/2021

from pathlib import Path

import numpy
from matplotlib import pyplot
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from tqdm import tqdm
import cv2

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# TODO: Not sure what the default beta value is
class Inpainter():
    def __init__(self, image, mask, warped_depth, patch_size=9, search_area_size=255, beta=1, plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.depth = cv2.inpaint(warped_depth, self.mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        self.patch_size = patch_size
        self.search_area_size = search_area_size
        self.beta = beta
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.level_regularity = None
        self.priority = None

        self.num_prev_hole_pixels = -1
        self.no_improv_count = 0
        self.no_improv_threshold = 5
        return

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        # start_time = time.time()
        keep_going = True
        progress_bar = tqdm(total=self.mask.sum())
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()

            target_pixel = self._find_highest_priority_pixel()
            # find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            # print('Time to find best: %f seconds' % (time.time()-find_start_time))

            self._update_image(target_pixel, source_patch)

            finished, progress = self._finished()
            progress_bar.update(progress)
            keep_going = not finished
        progress_bar.close()

        # print('Took %f seconds to complete' % (time.time() - start_time))
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

        self.working_image = numpy.copy(self.image)
        self.working_mask = numpy.copy(self.mask)
        self.num_prev_hole_pixels = self.working_mask.sum()

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self._update_level_regularity()
        self.priority = self.confidence * self.data * self.level_regularity * self.front

    def _update_confidence(self):
        new_confidence = numpy.copy(self.confidence)
        front_positions = numpy.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal*gradient
        self.data = numpy.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data
        return

    def _update_level_regularity(self):
        front_positions = numpy.argwhere(self.front == 1)
        for point in front_positions:
            patch_coordinates = self._get_patch(point)
            depth_patch = self._patch_data(self.depth, patch_coordinates)
            patch_area = self._patch_area(patch_coordinates)
            mean_depth = numpy.mean(depth_patch)

            known_mask = 1 - self._patch_data(self.working_mask, patch_coordinates)
            # rgb_mask = self._to_rgb(known_mask)
            sse = numpy.sum(numpy.square(depth_patch - mean_depth) * known_mask)
            point_level_regularity = patch_area / (patch_area + sse)
            self.level_regularity[point[0], point[1]] = point_level_regularity
        return

    def _calc_normal_matrix(self):
        x_kernel = numpy.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = numpy.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = numpy.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = numpy.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = numpy.nan_to_num(numpy.array(numpy.gradient(grey_image)))
        gradient_val = numpy.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = numpy.zeros([height, width, 2])

        front_positions = numpy.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = numpy.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        point = numpy.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        working_image_ngbrhd = self._get_search_area(target_pixel)
        # height, width = self.working_image.shape[:2]
        height, width = self._patch_shape(working_image_ngbrhd)
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image)

        y1, x1 = working_image_ngbrhd[0][0], working_image_ngbrhd[1][0]
        for y in range(y1, y1 + height - patch_height + 1):
            for x in range(x1, x1 + width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self._patch_data(self.working_mask, source_patch) \
                   .sum() != 0:
                    continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
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

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    def _get_search_area(self, point):
        half_patch_size = (self.search_area_size - 1) // 2
        height, width = self.working_image.shape[:2]
        search_area = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return search_area

    def _calc_patch_difference(self, image, target_patch, source_patch):
        known_mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(known_mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        # euclidean_distance = numpy.sqrt(
        #     (target_patch[0][0] - source_patch[0][0])**2 +
        #     (target_patch[1][0] - source_patch[1][0])**2
        # )  # tie-breaker factor

        # Depth distance
        target_depth = self._patch_data(self.depth, target_patch)
        source_depth = self._patch_data(self.depth, source_patch)
        depth_distance = numpy.sum(numpy.square(target_depth - source_depth) * known_mask)

        total_distance = squared_distance + self.beta * depth_distance
        return total_distance

    def _finished(self):
        # height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        # total = height * width
        progress_update = self.num_prev_hole_pixels - remaining
        # print('%d of %d completed' % (total-remaining, total))

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
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
