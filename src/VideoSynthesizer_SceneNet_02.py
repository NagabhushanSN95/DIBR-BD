# Shree KRISHNAya Namaha
# Copied from VSR005/VideoSynthesizer_SceneNet_02.py and updated for VSL010
# Author: Nagabhushan S N
# Date: 09/03/2021

import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
from matplotlib import pyplot
from tqdm import tqdm

from data_loaders import DataLoaderFactory

this_filename = Path(__file__).stem


class DibrModel:
    def __init__(self, configs: dict, database_dirpath, group, verbose_log: bool = False):
        self.configs = configs
        self.database_dirpath = database_dirpath
        self.group = group
        self.data_loader = self.get_data_loader()
        self.verbose_log = verbose_log
        return

    def get_data_loader(self):
        data_loader_name = self.configs['data_loader']
        data_loader = DataLoaderFactory.get_data_loader(data_loader_name, self.database_dirpath, split_name=self.group,
                                                        patch_size=None)
        return data_loader

    def load_data(self, scene_num, frame3_num):
        input_data = self.data_loader.load_test_data(scene_num, frame3_num, patch_start_pt=None)
        return input_data

    # noinspection PyUnusedLocal
    def infill_image(self, input_data):
        warped_frame2 = input_data['warped_frame2']
        mask2 = input_data['mask2']
        infilled_depth2 = input_data['infilled_depth2']

        infiller = self.get_infiller(warped_frame2, mask2, infilled_depth2)
        infilled_frame2 = infiller.inpaint()
        return infilled_frame2

    def get_infiller(self, warped_frame2, mask2, infilled_depth2):
        model_name = self.configs['model_name']
        patch_size = self.configs['patch_size']
        search_area_size = self.configs['search_area_size']
        plot_progress = self.verbose_log

        if model_name == 'Criminisi1':
            from Criminisi1 import Inpainter
            model = Inpainter(warped_frame2, mask2, patch_size, plot_progress)
        elif model_name == 'Criminisi2':
            from Criminisi2 import Inpainter
            model = Inpainter(warped_frame2, mask2, patch_size, search_area_size, plot_progress)
        elif model_name == 'Daribo1':
            from Daribo1 import Inpainter
            model = Inpainter(warped_frame2, mask2, infilled_depth2, patch_size, search_area_size, plot_progress)
        elif model_name == 'Cho6':
            from Cho6 import Inpainter
            model = Inpainter(warped_frame2, mask2, infilled_depth2, patch_size=patch_size,
                              search_area_size=search_area_size, plot_progress=plot_progress)
        else:
            raise RuntimeError(f'Unknown model name: {model_name}')
        return model


def get_image(path: Path):
    frame1 = skimage.io.imread(path.as_posix())[:, :, :3]
    return frame1


def save_configs(output_dirpath: Path, configs):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming testing')
    else:
        with open(configs_path.as_posix(), 'w') as configs_file:
            simplejson.dump(configs, configs_file, indent=4)
    return


def start_testing(configs: dict):
    test_videos_num = configs['test_videos_num']
    test_num = configs['test_num']

    database_dirpath = Path('../Data/Databases/SceneNet/Data/')
    output_dirpath = Path(f'../Runs/Testing/Test{test_num:04}')
    output_dirpath.mkdir(parents=True, exist_ok=True)

    save_configs(output_dirpath, configs)

    frames_data_path = database_dirpath / f'TestSets/Test{test_videos_num:02}/FramesData.csv'
    frames_data = pandas.read_csv(frames_data_path)
    test_frames_configs_path = database_dirpath / f'TestSets/Test{test_videos_num:02}/configs.json'
    with open(test_frames_configs_path.as_posix(), 'r') as configs_file:
        test_frames_configs = json.load(configs_file)
    group = test_frames_configs['group']

    dibr_model = DibrModel(configs, database_dirpath, group)

    num_frames = frames_data.shape[0]
    for row_num in range(num_frames):
        print(f'{row_num + 1}/{num_frames}')
        frame_data = frames_data.iloc[row_num]
        group, scene_num, frame3_num = frame_data
        scene_num, frame3_num = int(scene_num), int(frame3_num)

        scene_dirpath = database_dirpath / f'{group}/{scene_num:05}'
        frame3_path = scene_dirpath / f'photo/{frame3_num:04}.jpg'
        frame5_path = scene_dirpath / f'photo/{frame3_num + 2 * 25:04}.jpg'

        video_output_dirpath = output_dirpath / f'{scene_num:05}'
        pred_frames_dirpath = video_output_dirpath / 'PredictedFrames'
        pred_frames_dirpath.mkdir(parents=True, exist_ok=True)
        output_frame3_path = pred_frames_dirpath / f'{frame3_num:04}.png'
        infilled_frame4_path = pred_frames_dirpath / f'{frame3_num + 25:04}.png'
        output_frame5_path = pred_frames_dirpath / f'{frame3_num + 2 * 25:04}.png'

        if not output_frame3_path.exists():
            frame3 = get_image(frame3_path)
            skimage.io.imsave(output_frame3_path, frame3, check_contrast=False)

        if not infilled_frame4_path.exists():
            input_data = dibr_model.load_data(scene_num, frame3_num)
            infilled_frame4 = dibr_model.infill_image(input_data)
            skimage.io.imsave(infilled_frame4_path, infilled_frame4, check_contrast=False)

        if not output_frame5_path.exists():
            frame5 = get_image(frame5_path)
            skimage.io.imsave(output_frame5_path, frame5, check_contrast=False)
    return


def demo1():
    configs = {
        'Tester': this_filename,
        'test_num': 2,
        'model_name': 'Cho6',
        'data_loader': 'SceneNet01',
        'patch_size': 9,
        'search_area_size': 31,
        'test_videos_num': 3,
    }
    start_testing(configs)


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
