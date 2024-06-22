# Shree KRISHNAya Namaha
# Estimates sparse flow on N3DV scenes
# Modified from R19_SSLN/DEL001/DepthEstimator02_NeRF_LLFF.py
# Author: Nagabhushan S N
# Last Modified: 15/11/2023

import json
import time
import datetime
import traceback

import numpy
import simplejson
import skimage.io
import pandas

from pathlib import Path

import skvideo.io
from tqdm import tqdm
from matplotlib import pyplot
from deepdiff import DeepDiff

import ImagesMatcher04 as ImagesMatcher
import VideoFramesMatcher02 as VideoFramesMatcher

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_gen_num = int(this_filename[-2:])


NUM_FRAMES = 300


class Matcher(VideoFramesMatcher.Matcher):
    @staticmethod
    def read_video(path: Path):
        video = skvideo.io.vread(path.as_posix())[:NUM_FRAMES]
        return video


def start_generation(gen_configs: dict):
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases' / gen_configs['database_dirpath']
    tmp_dirpath = root_dirpath / 'tmp'

    output_dirpath = database_dirpath / f"all/estimated_flows/FEL001_FE{gen_configs['gen_num']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    VideoFramesMatcher.save_configs(output_dirpath, gen_configs)

    set_num = gen_configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = numpy.unique(video_data['scene_name'].to_numpy())

    matcher = Matcher(gen_configs, database_dirpath, tmp_dirpath, ImagesMatcher.ColmapTester)
    shifts = gen_configs['shifts']

    failed_pairs_path = output_dirpath / 'FailedPairs.csv'
    if failed_pairs_path.exists():
        failed_pairs_data = pandas.read_csv(failed_pairs_path)
    else:
        failed_pairs_data = pandas.DataFrame(columns=['scene_name', 'video1_num', 'frame1_num', 'video2_num', 'frame2_num'])

    for scene_name in scene_names:
        # if scene_name not in ['coffee_martini']:
        #     continue

        matcher.setup(scene_name)
        video_nums = video_data.loc[video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()
        num_frames = NUM_FRAMES

        for frame1_num in tqdm(range(num_frames), desc=scene_name):
            for shift in shifts:
                frame2_num = frame1_num + shift
                if frame2_num >= num_frames:
                    continue
                for video1_num in video_nums:
                    for video2_num in video_nums:
                        output_path = output_dirpath / f'{scene_name}/matched_pixels/{video1_num:04}_{frame1_num:04}__{video2_num:04}_{frame2_num:04}.csv'
                        is_failed_pair = failed_pairs_data.loc[
                                             (failed_pairs_data['scene_name'] == scene_name) &
                                             (failed_pairs_data['video1_num'] == video1_num) & (failed_pairs_data['frame1_num'] == frame1_num) &
                                             (failed_pairs_data['video2_num'] == video2_num) & (failed_pairs_data['frame2_num'] == frame2_num)
                                         ].shape[0] > 0
                        if output_path.exists() or is_failed_pair:
                            continue
                        matches_data = matcher.get_matches(video1_num, frame1_num, video2_num, frame2_num)
                        if matches_data is not None:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            matches_data.to_csv(output_path, index=False)
                        else:
                            pair_data = pandas.DataFrame.from_dict({
                                'scene_name': [scene_name],
                                'video1_num': [video1_num],
                                'frame1_num': [frame1_num],
                                'video2_num': [video2_num],
                                'frame2_num': [frame2_num],
                            })
                            failed_pairs_data = pandas.concat([failed_pairs_data, pair_data], ignore_index=True)
                            failed_pairs_data.to_csv(failed_pairs_path, index=False)
    return


def demo1():
    """
    For a gen set
    :return:
    """
    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesMatcher.this_filename}/{ImagesMatcher.this_filename}',
        'gen_num': 3,
        'gen_set_num': 3,
        'database_name': 'N3DV',
        'database_dirpath': 'N3DV/data',
        'camera_suffix': '',
        'resolution_suffix': '_down2',
        'shifts': [10],
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesMatcher.this_filename}/{ImagesMatcher.this_filename}',
        'gen_num': 4,
        'gen_set_num': 4,
        'database_name': 'N3DV',
        'database_dirpath': 'N3DV/data',
        'camera_suffix': '',
        'resolution_suffix': '_down2',
        'shifts': [10],
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesMatcher.this_filename}/{ImagesMatcher.this_filename}',
        'gen_num': 5,
        'gen_set_num': 5,
        'database_name': 'N3DV',
        'database_dirpath': 'N3DV/data',
        'camera_suffix': '',
        'resolution_suffix': '_down2',
        'shifts': [10],
    }
    start_generation(gen_configs)
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
