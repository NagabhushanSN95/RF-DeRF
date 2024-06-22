# Shree KRISHNAya Namaha
# Loads the entire video and saves the frames. Runs RAFT on the specified pair of frames and returns the flow.
# Author: Nagabhushan S N
# Last Modified: 25/12/2023

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

import torch.nn
from configargparse import Namespace
from deepdiff import DeepDiff
from tqdm import tqdm
from matplotlib import pyplot

from raft import RAFT
from utils.utils import InputPadder

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class FlowEstimator:
    def __init__(self, configs: dict, model_path: Path, database_dirpath: Path, tmp_dirpath: Path, device: int = 0):
        self.configs = configs
        self.model_path = model_path
        self.database_dirpath = database_dirpath
        self.device = self.get_device(device)

        self.tmp_dirpath = tmp_dirpath
        self.tmp_dirpath_frames = tmp_dirpath / 'rgb_frames'

        set_num = self.configs['gen_set_num']
        video_datapath = self.database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
        self.video_data = pandas.read_csv(video_datapath)
        self.camera_suffix = self.configs['camera_suffix']
        self.res_suffix = self.configs['resolution_suffix']

        self.scene_name = None
        self.video_nums = None

        raft_args = Namespace()
        raft_args.small = False
        raft_args.mixed_precision = False
        raft_args.alternate_corr = False
        self.model = torch.nn.DataParallel(RAFT(raft_args))
        self.model.load_state_dict(torch.load(model_path.as_posix(), map_location=self.device))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()
        self.padder = None
        return

    def setup(self, scene_name: str):
        self.scene_name = scene_name
        self.video_nums = self.video_data.loc[self.video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()
        for video_num in self.video_nums:
            self.extract_frames(scene_name, video_num)
        return

    def extract_frames(self, scene_name, video_num):
        video_path = self.database_dirpath / f'all/database_data/{scene_name}/rgb{self.camera_suffix}{self.res_suffix}/{video_num:04}.mp4'
        output_dirpath = self.tmp_dirpath_frames / f'{scene_name}/{video_num:04}'
        output_dirpath.mkdir(parents=True, exist_ok=True)
        cmd = f'ffmpeg -y -i {video_path.as_posix()} -start_number 0 {output_dirpath}/%04d.png'
        print(cmd)
        os.system(cmd)
        return

    def get_flow(self, video1_num: int, frame1_num: int, video2_num: int, frame2_num: int):
        frame1_path = self.tmp_dirpath_frames / f'{self.scene_name}/{video1_num:04}/{frame1_num:04}.png'
        frame2_path = self.tmp_dirpath_frames / f'{self.scene_name}/{video2_num:04}/{frame2_num:04}.png'
        frame1 = self.read_image(frame1_path)
        frame2 = self.read_image(frame2_path)
        estimated_flow = self.estimate_flow(frame1, frame2)
        return estimated_flow

    def estimate_flow(self, frame1: numpy.ndarray, frame2: numpy.ndarray):
        frame1_tr = self.preprocess_frame(frame1)
        frame2_tr = self.preprocess_frame(frame2)
        if self.padder is None:
            self.padder = InputPadder(frame1_tr.shape)
        frame1_padded, frame2_padded = self.padder.pad(frame1_tr, frame2_tr)
        with torch.no_grad():
            flow_low, flow_up = self.model(frame1_padded, frame2_padded, iters=20, test_mode=True)
        flow_np = self.postprocess_flow(flow_up)
        return flow_np

    def preprocess_frame(self, frame: numpy.ndarray) -> torch.Tensor:
        frame_tr = torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(self.device)
        return frame_tr

    def postprocess_flow(self, flow: torch.Tensor) -> numpy.ndarray:
        flow_unpad = self.padder.unpad(flow)
        flow_np = flow_unpad[0].permute(1, 2, 0).cpu().numpy()
        return flow_np

    @staticmethod
    def read_video(path: Path):
        video = skvideo.io.vread(path.as_posix())
        return video

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())#.astype('float32') / 255.0
        return image

    @staticmethod
    def save_flow(path: Path, flow: numpy.ndarray):
        match path.suffix:
            case '.npy':
                numpy.save(path, flow)
            case '.npz':
                numpy.savez_compressed(path, flow)
            case _:
                raise RuntimeError
        return

    @staticmethod
    def get_device(device: str):
        """
            Returns torch device object
            :param device: None//0/[0,],[0,1]. If multiple gpus are specified, first one is chosen
            :return:
            """
        if (device is None) or (device == '') or (not torch.cuda.is_available()):
            device = torch.device('cpu')
        else:
            device0 = device[0] if isinstance(device, list) else device
            device = torch.device(f'cuda:{device0}')
        return device


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
