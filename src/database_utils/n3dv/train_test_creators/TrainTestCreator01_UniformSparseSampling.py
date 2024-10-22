# Shree KRISHNAya Namaha
# Uses every 8th frame as the test frame as per the usual convention. Among the remaining frames, divides them into
# n+1 groups and picks n frames at the intersections
# Author: Nagabhushan S N
# Last Modified: 23/06/2024

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


def sample_sparse_train_videos(available_video_nums: list, num_videos: int):
    if num_videos is not None:
        selected_indices = numpy.round(numpy.linspace(-1, len(available_video_nums), num_videos + 2)).astype('int')[1:-1]
        selected_frame_nums = numpy.array(available_video_nums)[selected_indices]
    else:
        selected_frame_nums = available_video_nums
    return selected_frame_nums


def create_scene_frames_data(scene_name, frame_nums):
    frames_data = [[scene_name, frame_num] for frame_num in frame_nums]
    return frames_data


def create_data_frame(frames_data: list):
    frames_array = numpy.array(frames_data)
    frames_data = pandas.DataFrame(frames_array, columns=['scene_name', 'pred_video_num'])
    return frames_data


def create_train_test_set(configs: dict):
    root_dirpath = Path('../../')

    test_video_data = {
        'coffee_martini': [0, 3, 7, 13, 15],
        'cook_spinach': [0, 3, 8, 13, 18],
        'cut_roasted_beef': [0, 3, 7, 12, 17],
        'flame_salmon_1': [0, 3, 7, 12, 16],
        'flame_steak': [0, 3, 8, 13, 18],
        'sear_steak': [0, 3, 8, 13, 18],
    }
    excluded_videos = {
        'coffee_martini': [12],
        'cook_spinach': [],
        'cut_roasted_beef': [],
        'flame_salmon_1': [],
        'flame_steak': [],
        'sear_steak': [],
    }

    set_num = configs['set_num']
    num_train_videos = configs['num_train_videos']
    num_val_videos = configs['num_validation_videos']
    num_test_videos = configs['num_test_videos']

    set_dirpath = root_dirpath / f'data/train_test_sets/set{set_num:02}'
    set_dirpath.mkdir(parents=True, exist_ok=False)

    scenes_dirpath = root_dirpath / 'data/all/database_data/'
    scene_names = sorted(map(lambda path: path.stem, scenes_dirpath.iterdir()))
    all_data, train_data, val_data, test_data = [], [], [], []

    for scene_name in scene_names:
        videos_dirpath = scenes_dirpath / f'{scene_name}/rgb'
        all_video_nums = sorted(map(lambda path: int(path.stem), videos_dirpath.iterdir()))
        all_video_nums = list(set(all_video_nums) - set(excluded_videos[scene_name]))
        test_video_nums = test_video_data[scene_name][:num_test_videos]
        val_video_nums = test_video_nums[:num_val_videos]
        if num_train_videos == 'all':
            train_video_nums = list(set(all_video_nums) - set(test_video_nums))
        else:
            available_video_nums = list(set(all_video_nums) - set(test_video_data[scene_name]))
            train_video_nums = sample_sparse_train_videos(available_video_nums, num_train_videos)
        all_data.extend(create_scene_frames_data(scene_name, all_video_nums))
        train_data.extend(create_scene_frames_data(scene_name, train_video_nums))
        test_data.extend(create_scene_frames_data(scene_name, test_video_nums))
        val_data.extend(create_scene_frames_data(scene_name, val_video_nums))

    all_data = create_data_frame(all_data)
    all_data_path = set_dirpath / 'AllVideosData.csv'
    all_data.to_csv(all_data_path, index=False)

    train_data = create_data_frame(train_data)
    train_data_path = set_dirpath / 'TrainVideosData.csv'
    train_data.to_csv(train_data_path, index=False)

    test_data = create_data_frame(test_data)
    test_data_path = set_dirpath / 'TestVideosData.csv'
    test_data.to_csv(test_data_path, index=False)

    val_data = create_data_frame(val_data)
    val_data_path = set_dirpath / 'ValidationVideosData.csv'
    val_data.to_csv(val_data_path, index=False)

    configs_path = set_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)

    return


def demo1():
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 1,
        'num_test_videos': 1,
        'num_validation_videos': 1,
        'num_train_videos': 'all',
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 2,
        'num_test_videos': 5,
        'num_validation_videos': 1,
        'num_train_videos': 'all',
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 3,
        'num_test_videos': 1,
        'num_validation_videos': 1,
        'num_train_videos': 2,
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 4,
        'num_test_videos': 1,
        'num_validation_videos': 1,
        'num_train_videos': 3,
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 5,
        'num_test_videos': 1,
        'num_validation_videos': 1,
        'num_train_videos': 4,
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 6,
        'num_test_videos': 5,
        'num_validation_videos': 1,
        'num_train_videos': 2,
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 7,
        'num_test_videos': 5,
        'num_validation_videos': 1,
        'num_train_videos': 3,
    }
    create_train_test_set(configs)
    
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 8,
        'num_test_videos': 5,
        'num_validation_videos': 1,
        'num_train_videos': 4,
    }
    create_train_test_set(configs)
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
