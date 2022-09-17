# Shree KRISHNAYa Namaha
# Copied from Ours01.py and updated for SceneNet
# Author: Nagabhushan S N
# Last Modified: 05/03/2021

from pathlib import Path
from typing import Optional

import numpy
import skimage.io
import skimage.transform


class SceneNetDataLoader:
    """
    Loads patches of resolution 256x256. Patches are selected such that they contain atleast 1 unknown pixel
    """

    def __init__(self, data_dirpath, split_name, patch_size):
        super(OurDataLoader, self).__init__()
        self.dataroot = Path(data_dirpath) / split_name
        self.patch_size = patch_size
        return

    def load_data(self, scene_num: int, frame1_num: int, patch_start_pt):
        frame2_num = frame1_num + 25
        warped_frame2_path = self.dataroot / f'{scene_num:05}/PoseWarping05/1step/warped_frames/{frame2_num:04}.png'
        mask2_path = self.dataroot / f'{scene_num:05}PoseWarping05/1step/disocclusion_masks/{frame2_num:04}.png'
        infilled_depth2_path = self.dataroot / f'{scene_num:05}InfilledDepth01/1step/infilled_depths/' \
                                               f'{frame2_num:04}.npz'

        warped_frame2 = self.get_image(warped_frame2_path, patch_start_pt)
        mask2 = self.get_mask(mask2_path, patch_start_pt)
        depth2 = self.get_infilled_depth(infilled_depth2_path, patch_start_pt)

        data_dict = {
            'warped_frame2': warped_frame2,
            'mask2': mask2,
            'infilled_depth2': depth2,
        }
        return data_dict

    def get_image(self, path: Path, patch_start_point: Optional[tuple]):
        image = skimage.io.imread(path.as_posix())
        if patch_start_point is not None:
            h, w = patch_start_point
            image = image[h:h + self.patch_size, w:w + self.patch_size]
        return image

    def get_mask(self, path: Path, patch_start_point: Optional[tuple]):
        mask = skimage.io.imread(path.as_posix())
        if patch_start_point is not None:
            h, w = patch_start_point
            mask = mask[h:h + self.patch_size, w:w + self.patch_size]
        bool_mask = mask == 255
        return bool_mask

    def get_infilled_depth(self, path: Path, patch_start_point: Optional[tuple]):
        with numpy.load(path.as_posix()) as depth_data:
            depth = depth_data['arr_0']
        if patch_start_point is not None:
            h, w = patch_start_point
            depth = depth[h:h + self.patch_size, w:w + self.patch_size]
        return depth

    def load_test_data(self, scene_num: int, frame3_num: int, patch_start_pt):
        data_dict = self.load_data(scene_num, frame3_num, patch_start_pt)
        return data_dict
