# Shree KRISHNAya Namaha
# Downsamples the videos using ffmpeg command line and also saves the corresponding intrinsics
# Modified from N3DV/data_processors/VideoDownsampler.py
# Author: Nagabhushan S N
# Last Modified: 13/01/2024

import os
import shutil
import sys
import time
import datetime
import traceback
from collections import OrderedDict

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def downsample_data(database_dirpath: Path, camera_suffix: str, downsampling_factor: int, num_frames: int):
    resolution_suffix = f'_down{downsampling_factor}'
    for scene_dirpath in sorted(database_dirpath.iterdir()):
        # Downsample the RGB videos
        rgb_dirpath = scene_dirpath / f'rgb{camera_suffix}'
        rgb_down_dirpath = scene_dirpath / f'rgb{camera_suffix}{resolution_suffix}'
        rgb_down_dirpath.mkdir(parents=True, exist_ok=True)
        for video_path in sorted(rgb_dirpath.glob('*.mp4')):
            output_video_path = rgb_down_dirpath / video_path.name
            # Take only the first 300 frames and downsample the video
            # https://superuser.com/a/1297868/851274
            cmd = f'ffmpeg -y -i {video_path.as_posix()} -vframes {num_frames} -vf scale="iw/{downsampling_factor}:ih/{downsampling_factor}" {output_video_path.as_posix()}'
            os.system(cmd)

        # Save the intrinsics for downsampled videos
        intrinsics_path = scene_dirpath / f'CameraIntrinsics{camera_suffix}.csv'
        output_intrinsics_path = scene_dirpath / f'CameraIntrinsics{camera_suffix}{resolution_suffix}.csv'
        intrinsics = numpy.loadtxt(intrinsics_path, delimiter=',')
        intrinsics[:, 0] /= downsampling_factor
        intrinsics[:, 2] /= downsampling_factor
        intrinsics[:, 4] /= downsampling_factor
        intrinsics[:, 5] /= downsampling_factor
        numpy.savetxt(output_intrinsics_path.as_posix(), intrinsics, delimiter=',')
    return


def demo1():
    database_dirpath = Path('../../data/all/database_data')
    downsample_data(database_dirpath, camera_suffix='_original', downsampling_factor=2, num_frames=300)
    downsample_data(database_dirpath, camera_suffix='_undistorted', downsampling_factor=2, num_frames=300)
    downsample_data(database_dirpath, camera_suffix='_original', downsampling_factor=4, num_frames=300)
    downsample_data(database_dirpath, camera_suffix='_undistorted', downsampling_factor=4, num_frames=300)
    return


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
