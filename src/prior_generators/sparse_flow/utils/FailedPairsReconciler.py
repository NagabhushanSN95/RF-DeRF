# Shree KRISHNAya Namaha
# Reconciles and saves failed pairs data for a folder that has already been generated
# Author: Nagabhushan S N
# Last Modified: 30/12/2023

import time
import datetime
import traceback
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


def reconcile_failed_pairs_data(database_dirpath: Path, flow_dirpath: Path, num_frames: int):
    configs_path = flow_dirpath / 'Configs.json'
    with open(configs_path, 'r') as configs_file:
        configs = simplejson.load(configs_file)
    set_num = configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = numpy.unique(video_data['scene_name'].to_numpy())
    shifts = configs['shifts']

    failed_pairs_path = flow_dirpath / 'FailedPairs.csv'
    if failed_pairs_path.exists():
        failed_pairs_data = pandas.read_csv(failed_pairs_path)
    else:
        failed_pairs_data = pandas.DataFrame(columns=['scene_name', 'video1_num', 'frame1_num', 'video2_num', 'frame2_num'])

    new_failed_pairs_list = []
    for scene_name in scene_names:
        video_nums = video_data[video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()

        for frame1_num in tqdm(range(num_frames), desc=scene_name):
            for shift in shifts:
                frame2_num = frame1_num + shift
                if frame2_num < 0 or frame2_num >= num_frames:
                    continue
                for video1_num in video_nums:
                    for video2_num in video_nums:
                        output_path = flow_dirpath / f'{scene_name}/matched_pixels/{video1_num:04}_{frame1_num:04}__{video2_num:04}_{frame2_num:04}.csv'
                        is_failed_pair = failed_pairs_data.loc[
                                             (failed_pairs_data['scene_name'] == scene_name) &
                                             (failed_pairs_data['video1_num'] == video1_num) & (failed_pairs_data['frame1_num'] == frame1_num) &
                                             (failed_pairs_data['video2_num'] == video2_num) & (failed_pairs_data['frame2_num'] == frame2_num)
                                         ].shape[0] > 0
                        if not (output_path.exists() or is_failed_pair):
                            new_failed_pairs_list.append([scene_name, video1_num, frame1_num, video2_num, frame2_num])

    new_failed_pairs_data = pandas.DataFrame(new_failed_pairs_list, columns=['scene_name', 'video1_num', 'frame1_num', 'video2_num', 'frame2_num'])
    failed_pairs_data = pandas.concat([failed_pairs_data, new_failed_pairs_data], ignore_index=True)
    failed_pairs_data.to_csv(failed_pairs_path, index=False)
    return


def demo1():
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases/N3DV/data'
    flow_dirpath = database_dirpath / 'all/estimated_flows/FEL001_FE33'

    reconcile_failed_pairs_data(database_dirpath, flow_dirpath, num_frames=300)
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
