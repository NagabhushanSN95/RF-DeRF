# Shree KRISHNAya Namaha
# Creates videos using ffmpeg command line using the frames generated during validation/rendering.
# This is required since ffmpeg fails on some server to generate mp4 videos.
# Author: Nagabhushan S N
# Last Modified: 11/11/2023

import argparse
import os
import time
import datetime
import traceback

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def generate_videos(train_dirpaths):
    for train_dirpath in train_dirpaths:
        for scene_dirpath in sorted(train_dirpath.iterdir()):
            if not scene_dirpath.is_dir():
                continue
            for pred_videos_dirpath in sorted(scene_dirpath.glob('predicted_videos_iter*')):
                for domain_type in ['rgb', 'depth']:
                    domain_dirpath  = pred_videos_dirpath / domain_type
                    for pred_frames_dirpath in sorted(domain_dirpath.iterdir()):
                        if not pred_frames_dirpath.is_dir():
                            continue
                        pred_video_path = domain_dirpath / f'{pred_frames_dirpath.stem}.mp4'
                        cmd = f"ffmpeg -y -framerate 30 -pattern_type glob -i '{pred_frames_dirpath.as_posix()}/*.png' -c:v libx264 -pix_fmt yuv420p {pred_video_path.as_posix()}"
                        os.system(cmd)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dirpaths', nargs='+', type=str)

    args = parser.parse_args()
    train_dirpaths = [Path(train_dirpath) for train_dirpath in args.train_dirpaths]
    generate_videos(train_dirpaths)
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
