# Shree KRISHNAya Namaha
# Estimates dense flow on N3DV scenes. Computes cross-view flow also.
# Modified from FlowEstimator_N3DV01.py
# Author: Nagabhushan S N
# Last Modified: 25/12/2023

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

import FlowEstimator01 as FlowEstimatorModule

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_gen_num = int(this_filename[-2:])


NUM_FRAMES = 300


class FlowEstimator(FlowEstimatorModule.FlowEstimator):
    @staticmethod
    def read_video(path: Path):
        video = skvideo.io.vread(path.as_posix())[:NUM_FRAMES]
        return video

    @staticmethod
    def save_flow(path: Path, flow: numpy.ndarray):
        flow16 = flow.round().astype('int16')
        super(FlowEstimator, FlowEstimator).save_flow(path, flow16)
        return


def start_generation(gen_configs: dict):
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases' / gen_configs['database_dirpath']
    tmp_dirpath = root_dirpath / 'tmp'

    output_dirpath = database_dirpath / f"all/estimated_flows/FEL003_FE{gen_configs['gen_num']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    FlowEstimatorModule.save_configs(output_dirpath, gen_configs)

    set_num = gen_configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = numpy.unique(video_data['scene_name'].to_numpy())

    model_path = root_dirpath / f'PretrainedModels/{gen_configs["model_path"]}'
    flow_estimator = FlowEstimator(gen_configs, model_path, database_dirpath, tmp_dirpath)
    shifts = gen_configs['shifts']
    shifts_pn = []
    for shift in shifts:
        shifts_pn.extend([-shift, shift])

    for scene_name in scene_names:
        # if scene_name not in ['coffee_martini']:
        #     continue

        flow_estimator.setup(scene_name)
        video_nums = video_data.loc[video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()
        num_frames = NUM_FRAMES

        for video1_num in video_nums:
            for video2_num in video_nums:
                for frame1_num in tqdm(range(num_frames), desc=f'{scene_name}; video_nums: {video1_num}-{video2_num}'):
                    for shift in shifts_pn:
                        frame2_num = frame1_num + shift
                        if not (0 <= frame2_num < num_frames):
                            continue
                        output_path = output_dirpath / f'{scene_name}/estimated_flows/{video1_num:04}_{frame1_num:04}__{video2_num:04}_{frame2_num:04}.npz'
                        if output_path.exists():
                            continue
                        estimated_flow = flow_estimator.get_flow(video1_num, frame1_num, video2_num, frame2_num)
                        if estimated_flow is not None:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            flow_estimator.save_flow(output_path, estimated_flow)
    return


def demo1():
    """
    For a gen set
    :return:
    """
    gen_configs = {
        'generator': f'{this_filename}/{FlowEstimatorModule.this_filename}',
        'gen_num': 14,
        'gen_set_num': 4,
        'model_path': 'PaperModels/raft-sintel.pth',
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

    from snb_utils import Telegrammer

    time.sleep(5)
    message_content = f'R21/FEL003/{this_filename} has finished.\n' + run_result
    Telegrammer.send_message(message_content, chat_names=['Nagabhushan'])
