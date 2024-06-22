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

from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


NUM_VIEWS = 16
# Depth bounds: https://github.com/facebookresearch/hyperreel/blob/main/datasets/technicolor.py#L126--L153
DEPTH_BOUNDS = {
    # 'Automaton': [0.65, 10.0],
    'Birthday': [1.75, 10.0],
    # 'Fabien': [0.35, 2.0],
    # 'Fatma': [0.65, 10.0],
    # 'Hands': [0.65, 10.0],
    'Painter': [1.75, 10.0],
    # 'Quentin': [0.65, 10.0],
    'Remy': [0.65, 10.0],
    # 'Rugby': [0.65, 10.0],
    'Theater': [0.65, 10.0],
    'Train': [0.65, 10.0],
    # 'Tristan': [0.65, 10.0],
}


def extract_data(unzipped_dirpath: Path, extracted_dirpath: Path):
    for scene_name in sorted(DEPTH_BOUNDS.keys()):
        tgt_scene_dirpath = extracted_dirpath / scene_name
        tgt_scene_dirpath.mkdir(parents=True, exist_ok=False)

        # Extract original data
        if (unzipped_dirpath / 'original' / scene_name).exists():
            camera_type = 'original'
            camera_suffix = '_original'

            # Save original RGB frames as mp4
            tgt_videos_dirpath = tgt_scene_dirpath / f'rgb{camera_suffix}'
            tgt_videos_dirpath.mkdir(parents=True, exist_ok=False)
            for view_num in tqdm(range(NUM_VIEWS), desc=scene_name):
                tgt_video_filepath = tgt_videos_dirpath / f'{view_num:04}.mp4'
                cmd = f'ffmpeg -framerate 30 -pattern_type sequence -start_number 1 -i "{unzipped_dirpath.as_posix()}/{camera_type}/{scene_name}/{scene_name}_%05d_{view_num:02}.png" -c:v libx264 -pix_fmt yuv420p {tgt_video_filepath.as_posix()}'
                os.system(cmd)

            # Copy original camera parameters
            src_camera_params_filepath = unzipped_dirpath / f'{camera_type}/{scene_name}/cameras_parameters.txt'
            tgt_camera_params_filepath = tgt_scene_dirpath / f'cameras_parameters{camera_suffix}.txt'
            shutil.copy(src_camera_params_filepath.as_posix(), tgt_camera_params_filepath.as_posix())

            # Extract original Camera Intrinsics
            camera_params_filepath = unzipped_dirpath / f'{camera_type}/{scene_name}/cameras_parameters.txt'
            camera_params_data = pandas.read_csv(camera_params_filepath, sep='\s+', header=None, skiprows=1)
            camera_params_data.columns = ['f', 'cu', 'cv', 'ar', 'sk', 'd1', 'd2', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
            zero_array = numpy.zeros_like(camera_params_data['f'].to_numpy())
            ones_array = numpy.ones_like(camera_params_data['f'].to_numpy())
            intrinsics = numpy.stack([camera_params_data['f'].to_numpy(), camera_params_data['sk'].to_numpy(), camera_params_data['cu'].to_numpy(),
                                      zero_array, camera_params_data['ar'].to_numpy()*camera_params_data['f'].to_numpy(), camera_params_data['cv'].to_numpy(),
                                      zero_array, zero_array, ones_array]).T  # (num_views, 9)
            intrinsics_filepath = tgt_scene_dirpath / f'CameraIntrinsics{camera_suffix}.csv'
            numpy.savetxt(intrinsics_filepath.as_posix(), intrinsics, delimiter=',')

            # Extract Camera Extrinsics
            quaternions = numpy.stack([camera_params_data['qx'].to_numpy(), camera_params_data['qy'].to_numpy(), camera_params_data['qz'].to_numpy(), camera_params_data['qw'].to_numpy()]).T
            rotations = Rotation.from_quat(quaternions).as_matrix()  # (num_views, 3, 3)
            translations = numpy.stack([camera_params_data['tx'].to_numpy(), camera_params_data['ty'].to_numpy(), camera_params_data['tz'].to_numpy()]).T[:, :, None]  # (num_views, 3, 1)
            extrinsics_3x4 = numpy.concatenate([rotations, translations], axis=2)  # (num_views, 3, 4)
            bottom_row = numpy.stack([zero_array, zero_array, zero_array, ones_array]).T[:, None, :]  # (num_views, 1, 4)
            extrinsics = numpy.concatenate([extrinsics_3x4, bottom_row], axis=1)  # (num_views, 4, 4)
            extrinsics_flat = extrinsics.reshape((-1, 16))  # (num_views, 16)
            extrinsics_filepath = tgt_scene_dirpath / f'CameraExtrinsics{camera_suffix}.csv'
            numpy.savetxt(extrinsics_filepath.as_posix(), extrinsics_flat, delimiter=',')

        # Extract undisorted data
        if (unzipped_dirpath / 'undistorted' / scene_name).exists():
            camera_type = 'undistorted'
            camera_suffix = '_undistorted'

            # Save undistorted RGB frames as mp4
            tgt_videos_dirpath = tgt_scene_dirpath / f'rgb{camera_suffix}'
            tgt_videos_dirpath.mkdir(parents=True, exist_ok=False)
            for view_num in tqdm(range(NUM_VIEWS), desc=scene_name):
                tgt_video_filepath = tgt_videos_dirpath / f'{view_num:04}.mp4'
                cmd = f'ffmpeg -framerate 30 -pattern_type sequence -start_number 1 -i "{unzipped_dirpath.as_posix()}/{camera_type}/{scene_name}/{scene_name}_undist_%05d_{view_num:02}.png" -c:v libx264 -pix_fmt yuv420p {tgt_video_filepath.as_posix()}'
                os.system(cmd)

            # Copy undistorted camera parameters
            src_camera_params_filepath = unzipped_dirpath / f'{camera_type}/{scene_name}/cameras_parameters.txt'
            tgt_camera_params_filepath = tgt_scene_dirpath / f'cameras_parameters{camera_suffix}.txt'
            shutil.copy(src_camera_params_filepath.as_posix(), tgt_camera_params_filepath.as_posix())

            # Extract undistorted Camera Intrinsics
            camera_params_filepath = unzipped_dirpath / f'{camera_type}/{scene_name}/cameras_parameters.txt'
            camera_params_data = pandas.read_csv(camera_params_filepath, sep='\s+', header=None, skiprows=1)
            camera_params_data.columns = ['f', 'cu', 'cv', 'ar', 'sk', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
            zero_array = numpy.zeros_like(camera_params_data['f'].to_numpy())
            ones_array = numpy.ones_like(camera_params_data['f'].to_numpy())
            intrinsics = numpy.stack([camera_params_data['f'].to_numpy(), camera_params_data['sk'].to_numpy(), camera_params_data['cu'].to_numpy(),
                                      zero_array, camera_params_data['ar'].to_numpy()*camera_params_data['f'].to_numpy(), camera_params_data['cv'].to_numpy(),
                                      zero_array, zero_array, ones_array]).T  # (num_views, 9)
            intrinsics_filepath = tgt_scene_dirpath / f'CameraIntrinsics{camera_suffix}.csv'
            numpy.savetxt(intrinsics_filepath.as_posix(), intrinsics, delimiter=',')

            # Extract Camera Extrinsics
            quaternions = numpy.stack([camera_params_data['qx'].to_numpy(), camera_params_data['qy'].to_numpy(), camera_params_data['qz'].to_numpy(), camera_params_data['qw'].to_numpy()]).T
            rotations = Rotation.from_quat(quaternions).as_matrix()  # (num_views, 3, 3)
            translations = numpy.stack([camera_params_data['tx'].to_numpy(), camera_params_data['ty'].to_numpy(), camera_params_data['tz'].to_numpy()]).T[:, :, None]  # (num_views, 3, 1)
            extrinsics_3x4 = numpy.concatenate([rotations, translations], axis=2)  # (num_views, 3, 4)
            bottom_row = numpy.stack([zero_array, zero_array, zero_array, ones_array]).T[:, None, :]  # (num_views, 1, 4)
            extrinsics = numpy.concatenate([extrinsics_3x4, bottom_row], axis=1)  # (num_views, 4, 4)
            extrinsics_flat = extrinsics.reshape((-1, 16))  # (num_views, 16)
            extrinsics_filepath = tgt_scene_dirpath / f'CameraExtrinsics{camera_suffix}.csv'
            numpy.savetxt(extrinsics_filepath.as_posix(), extrinsics_flat, delimiter=',')

        # Extract Depth Bounds
        if (unzipped_dirpath / 'original' / scene_name).exists() or (unzipped_dirpath / 'undistorted' / scene_name).exists():
            bounds = DEPTH_BOUNDS[scene_name]
            bounds = numpy.array([bounds] * NUM_VIEWS)
            bounds_filepath = tgt_scene_dirpath / 'DepthBounds.csv'
            numpy.savetxt(bounds_filepath.as_posix(), bounds, delimiter=',')
    return


def demo1():
    unzipped_dirpath = Path('../../data/raw/database_data_unzipped')
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
