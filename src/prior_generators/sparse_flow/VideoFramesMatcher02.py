# Shree KRISHNAya Namaha
# Runs FrameMatchesComputer on all possible combinations of frames as specified by the configs. Creates PNG frames of
# all video frames using ffmpeg and copies them to colmap directory using shutil. Compatible with ImagesMatcher04.py
# Author: Nagabhushan S N
# Last Modified: 15/11/2023

import json
import os
import shutil
import time
import datetime
import traceback
from itertools import combinations

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path

from deepdiff import DeepDiff
from tqdm import tqdm
from matplotlib import pyplot

from colmap_utils.read_write_model import read_images_binary, read_points3d_binary

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class Matcher:
    def __init__(self, configs: dict, database_dirpath: Path, tmp_dirpath: Path, images_matcher):
        self.configs = configs
        self.database_dirpath = database_dirpath
        self.tmp_dirpath = tmp_dirpath
        self.tmp_dirpath_frames = tmp_dirpath / 'rgb_frames'
        self.tmp_dirpath_colmap = tmp_dirpath / 'colmap'
        self.frames_matcher = images_matcher(self.tmp_dirpath_colmap)

        set_num = self.configs['gen_set_num']
        video_datapath = self.database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
        self.video_data = pandas.read_csv(video_datapath)
        self.camera_suffix = self.configs['camera_suffix']
        self.res_suffix = self.configs['resolution_suffix']

        self.scene_name = None
        self.video_nums = None
        self.intrinsics = None
        self.extrinsics = None
        return

    def setup(self, scene_name: str):
        self.scene_name = scene_name
        self.video_nums = self.video_data.loc[self.video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()
        for video_num in self.video_nums:
            self.extract_frames(scene_name, video_num)
        intrinsics_path = self.database_dirpath / f'all/database_data/{scene_name}/CameraIntrinsics{self.camera_suffix}{self.res_suffix}.csv'
        extrinsics_path = self.database_dirpath / f'all/database_data/{scene_name}/CameraExtrinsics{self.camera_suffix}.csv'
        self.intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))[self.video_nums]
        self.extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))[self.video_nums]
        return

    def extract_frames(self, scene_name, video_num):
        video_path = self.database_dirpath / f'all/database_data/{scene_name}/rgb{self.camera_suffix}{self.res_suffix}/{video_num:04}.mp4'
        output_dirpath = self.tmp_dirpath_frames / f'{scene_name}/{video_num:04}'
        output_dirpath.mkdir(parents=True, exist_ok=True)
        cmd = f'ffmpeg -y -i {video_path.as_posix()} -start_number 0 {output_dirpath}/%04d.png'
        print(cmd)
        os.system(cmd)
        return

    def get_matches(self, video1_num: int, frame1_num: int, video2_num: int, frame2_num: int):
        video1_index = numpy.where(self.video_nums == video1_num)[0].item()
        video2_index = numpy.where(self.video_nums == video2_num)[0].item()
        frame1_path = self.tmp_dirpath_frames / f'{self.scene_name}/{video1_num:04}/{frame1_num:04}.png'
        frame2_path = self.tmp_dirpath_frames / f'{self.scene_name}/{video2_num:04}/{frame2_num:04}.png'
        extrinsic1 = self.extrinsics[video1_index]
        extrinsic2 = self.extrinsics[video2_index]
        intrinsic1 = self.intrinsics[video1_index]
        intrinsic2 = self.intrinsics[video2_index]
        matches_data = self.frames_matcher.estimate_sparse_flow([frame1_path, frame2_path], [extrinsic1, extrinsic2], [intrinsic1, intrinsic2])
        return matches_data

    @staticmethod
    def read_video(path: Path):
        video = skvideo.io.vread(path.as_posix())
        return video


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming generation: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return
