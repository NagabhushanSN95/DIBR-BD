# Shree KRISHNAya Namaha
# A Factory method that returns a Data Loader
# Author: Nagabhushan S N
# Date: 09/03/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot


def get_data_loader(name: str, data_dirpath, split_name, patch_size):
    if name == 'Ours01':
        from data_loaders.Ours01 import OurDataLoader
        data_loader = OurDataLoader(data_dirpath, split_name, patch_size)
    elif name == 'SceneNet01':
        from data_loaders.SceneNet01 import SceneNetDataLoader
        data_loader = SceneNetDataLoader(data_dirpath, split_name, patch_size)
    else:
        raise RuntimeError(f'Unknown data loader: {name}')
    return data_loader
