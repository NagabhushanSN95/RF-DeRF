# Shree KRISHNAya Namaha
# Combines all the sparse flow (matches) files into a single file
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


def aggregate_matches_data(matches_data: pandas.DataFrame, new_matches_data: pandas.DataFrame):
    if matches_data is None:
        matches_data = new_matches_data
    else:
        matches_data = pandas.concat([matches_data, new_matches_data], axis=0)
    return matches_data


def aggregate_flow_priors(database_dirpath_suffix: str, flow_dirname: str):
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases' / database_dirpath_suffix
    flow_dirpath = database_dirpath / f'estimated_flows/{flow_dirname}'
    for scene_dirpath in sorted(flow_dirpath.iterdir()):
        if not scene_dirpath.is_dir():
            continue
        all_matches_data = []
        for matches_path in tqdm(sorted((scene_dirpath / 'matched_pixels').glob('*.csv')), desc=scene_dirpath.stem):
            matches_data = pandas.read_csv(matches_path)
            column_names = list(matches_data.columns)[:-1]
            video1_num, frame1_num = matches_path.stem.split('__')[0].split('_')
            video2_num, frame2_num = matches_path.stem.split('__')[1].split('_')
            matches_data['video1_num'] = int(video1_num)
            matches_data['video2_num'] = int(video2_num)
            matches_data['frame1_num'] = int(frame1_num)
            matches_data['frame2_num'] = int(frame2_num)
            matches_data = matches_data[['video1_num', 'frame1_num', 'video2_num', 'frame2_num'] + column_names]
            all_matches_data.append(matches_data)
        all_matches_data = pandas.concat(all_matches_data, axis=0)
        output_path = scene_dirpath / 'MatchedPixels.csv'
        all_matches_data.to_csv(output_path, index=False)
    return


def demo1():
    database_dirpath_suffix = 'N3DV/data/all'
    flow_dirname = 'FEL001_FE04'
    aggregate_flow_priors(database_dirpath_suffix, flow_dirname)
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
