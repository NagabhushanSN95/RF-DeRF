# Shree KRISHNAya Namaha
# Common utility functions
# Author: Nagabhushan S N
# Last Modified: 23/06/2024

import importlib
import time
import datetime
import traceback
from typing import Dict, Any

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

from omegaconf import OmegaConf

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_test_videos_datapath(database_dirpath: Path, pred_train_dirpath: Path):
    train_configs_path = next(pred_train_dirpath.glob('Configs.*'))
    set_num = read_set_num_from_configs(train_configs_path)
    test_videos_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    return test_videos_datapath


def read_set_num_from_configs(configs_path: Path):
    if configs_path.suffix == '.py':
        spec = importlib.util.spec_from_file_location(configs_path.stem, configs_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        configs: Dict[str, Any] = cfg.config
        set_num = configs['set_num']
    elif configs_path.suffix == '.yaml':
        configs = OmegaConf.load(configs_path)
        set_num = configs.data.train_test_set_num
    else:
        raise ValueError(f'Unsupported file extension: {configs_path.as_posix()}')
    return set_num
