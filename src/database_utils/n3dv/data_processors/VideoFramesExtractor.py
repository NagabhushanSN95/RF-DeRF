# Shree KRISHNAya Namaha
# Extracts video frames into png format. Also supports downsampling.
# Author: Nagabhushan S N
# Last Modified: 13/11/2023

import os
import shutil
import time
import datetime
import traceback
from typing import Optional

import numpy
import simplejson
import skimage.io
import skimage.transform
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def extract_frames(database_dirpath: Path, *, num_frames: int = Optional[None], downsampling_factor: int = 1):
    for scene_dirpath in sorted(database_dirpath.iterdir()):
        # Downsample the RGB videos
        rgb_dirpath = scene_dirpath / 'rgb'
        resolution_suffix = f'_down{downsampling_factor}' if downsampling_factor > 1 else ''
        output_rgb_dirpath = scene_dirpath / f'rgb{resolution_suffix}'
        output_rgb_dirpath.mkdir(parents=True, exist_ok=True)
        for video_path in sorted(rgb_dirpath.glob('*.mp4')):
            output_video_dirpath = output_rgb_dirpath / video_path.stem
            output_video_dirpath.mkdir(parents=True, exist_ok=True)
            cmd = f'ffmpeg -y -i {video_path.as_posix()} {output_video_dirpath}/%04d.png'
            os.system(cmd)

            if (downsampling_factor > 1) or (num_frames > 0):
                for frame_num, frame_path in enumerate(tqdm(sorted(output_video_dirpath.glob('*.png')), desc=f'{scene_dirpath.stem}/{video_path.stem}')):
                    if (num_frames > 0) and (frame_num >= num_frames):
                        frame_path.unlink()
                    if downsampling_factor > 1:
                        frame = read_image(frame_path)
                        downsampled_frame = downsample_image(frame, downsampling_factor)
                        frame_path.unlink()  # Delete the existing file
                        new_frame_path = frame_path.parent / f'{frame_num:04}.png'
                        save_image(new_frame_path, downsampled_frame)

        # Save the intrinsics for downsampled videos
        intrinsics_path = scene_dirpath / 'CameraIntrinsics.csv'
        output_intrinsics_path = scene_dirpath / f'CameraIntrinsics_down{downsampling_factor}.csv'
        intrinsics = numpy.loadtxt(intrinsics_path, delimiter=',')
        intrinsics[:, 2] /= downsampling_factor
        intrinsics[:, 5] /= downsampling_factor
        numpy.savetxt(output_intrinsics_path.as_posix(), intrinsics, delimiter=',')
    return


def downsample_image(image: numpy.ndarray, downsampling_factor: int):
    downsampled_image = skimage.transform.rescale(image, scale=1 / downsampling_factor, preserve_range=True,
                                                  multichannel=True, anti_aliasing=True)
    downsampled_image = numpy.round(downsampled_image).astype('uint8')
    return downsampled_image


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_image(path: Path, image: numpy.ndarray):
    skimage.io.imsave(path.as_posix(), image)
    return


def demo1():
    database_dirpath = Path('../../data/all/database_data')
    num_frames = 300
    downsampling_factor = 2
    extract_frames(database_dirpath, num_frames=num_frames, downsampling_factor=downsampling_factor)
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
