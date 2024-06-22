# Shree KRISHNAya Namaha
# Extracts rgb frames, camera intrinsics and extrisics
# Author: Nagabhushan S N
# Last Modified: 07/03/2023

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


def change_coordinate_system(extrinsics, permuter):
    permuter = permuter[None]
    changed_extrinsics = numpy.linalg.inv(permuter) @ extrinsics @ permuter
    # for extrinsic in extrinsics:
    #     r = extrinsic[:3, :3]
    #     t = extrinsic[:3, 3:]
    #     rc = p.T @ r @ p
    #     tc = p @ t
    #     changed_pose = numpy.concatenate([numpy.concatenate([rc, tc], axis=1), extrinsic[3:]], axis=0)
    #     changed_extrinsics.append(changed_pose)
    # changed_extrinsics = numpy.stack(changed_extrinsics)
    return changed_extrinsics


def extract_data(unzipped_dirpath: Path, extracted_dirpath: Path):
    for src_scene_dirpath in sorted(unzipped_dirpath.iterdir()):
        if not src_scene_dirpath.is_dir():
            continue
        scene_name = src_scene_dirpath.name
        tgt_scene_dirpath = extracted_dirpath / scene_name
        tgt_scene_dirpath.mkdir(parents=True, exist_ok=False)

        # Extract RGB
        tgt_videos_dirpath = tgt_scene_dirpath / 'rgb'
        tgt_videos_dirpath.mkdir(parents=True, exist_ok=False)
        for view_num, src_video_filepath in enumerate(tqdm(sorted(src_scene_dirpath.glob('cam*.mp4')), desc=scene_name)):
            tgt_video_filepath = tgt_videos_dirpath / f'{view_num:04}.mp4'
            shutil.copy(src_video_filepath, tgt_video_filepath)

        # Extract Camera parameters
        src_poses_filepath = src_scene_dirpath / 'poses_bounds.npy'
        tgt_poses_filepath = tgt_scene_dirpath / 'poses_bounds.npy'
        shutil.copy(src_poses_filepath, tgt_poses_filepath)
        poses_bounds = numpy.load(src_poses_filepath.as_posix())
        num_views = poses_bounds.shape[0]
        cam_params = poses_bounds[:, :15].reshape((-1, 3, 5))

        intrinsics = numpy.repeat(numpy.eye(3)[None], repeats=num_views, axis=0)
        hwf = cam_params[:, :3, -1]
        intrinsics[:, 0, 0] = hwf[:, 2]
        intrinsics[:, 1, 1] = hwf[:, 2]
        intrinsics[:, 0, 2] = hwf[:, 1] / 2
        intrinsics[:, 1, 2] = hwf[:, 0] / 2

        extrinsics = numpy.repeat(numpy.eye(4)[None], repeats=num_views, axis=0)
        extrinsics[:, :3, :4] = cam_params[:, :3, :4]
        permuter = numpy.eye(4)
        permuter[:2] = permuter[:2][::-1]
        permuter[2,2] *= -1
        extrinsics = change_coordinate_system(extrinsics, permuter)
        extrinsics = numpy.linalg.inv(extrinsics)
        # extrinsics = extrinsics @ numpy.linalg.inv(extrinsics[:1])

        bounds = poses_bounds[:, 15:]
        extrinsics_filepath = tgt_scene_dirpath / 'CameraExtrinsics.csv'
        intrinsics_filepath = tgt_scene_dirpath / 'CameraIntrinsics.csv'
        bounds_filepath = tgt_scene_dirpath / 'DepthBounds.csv'
        numpy.savetxt(extrinsics_filepath.as_posix(), extrinsics.reshape((num_views, 16)), delimiter=',')
        numpy.savetxt(intrinsics_filepath.as_posix(), intrinsics.reshape((num_views, 9)), delimiter=',')
        numpy.savetxt(bounds_filepath.as_posix(), bounds, delimiter=',')
    return


def demo1():
    unzipped_dirpath = Path('../../data/raw/unzipped_data')
    extracted_dirpath = Path('../../data/all/database_data')
    extract_data(unzipped_dirpath, extracted_dirpath)
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
