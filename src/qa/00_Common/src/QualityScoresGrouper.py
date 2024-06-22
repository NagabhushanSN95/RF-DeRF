# Shree KRISHNAya Namaha
# Groups QA scores scene-wise
# Author: Nagabhushan S N
# Last Modified: 06/11/2022

import time
import datetime
import traceback
import pandas

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_grouped_qa_scores(qa_data: pandas.DataFrame):
    column_names = list(qa_data.columns)
    video_wise_scores = qa_data.groupby(by=column_names[:2]).mean().reset_index()[column_names[:2] + column_names[-1:]]
    scene_wise_scores = qa_data.groupby(by=column_names[:1]).mean().reset_index()[column_names[:1] + column_names[-1:]]

    video_wise_scores = video_wise_scores.round({column_names[-1]: 4, })
    scene_wise_scores = scene_wise_scores.round({column_names[-1]: 4, })
    return video_wise_scores, scene_wise_scores


def group_qa_scores(training_dirpath: Path, train_nums: list):
    for train_num in train_nums:
        qa_dirpath = training_dirpath / f'train{train_num:04}/quality_scores'
        for iter_dirpath in sorted(qa_dirpath.iterdir()):
            for frame_wise_filepath in sorted(iter_dirpath.glob('*_FrameWise.csv')):
                video_wise_filepath = frame_wise_filepath.parent / f'{frame_wise_filepath.stem[:-10]}_VideoWise.csv'
                scene_wise_filepath = frame_wise_filepath.parent / f'{frame_wise_filepath.stem[:-10]}_SceneWise.csv'
                # if video_wise_filepath.exists() and scene_wise_filepath.exists():
                #     continue

                frame_wise_scores = pandas.read_csv(frame_wise_filepath)
                video_wise_scores, scene_wise_scores = get_grouped_qa_scores(frame_wise_scores)
                video_wise_scores.to_csv(video_wise_filepath, index=False)
                scene_wise_scores.to_csv(scene_wise_filepath, index=False)
    return


def demo1():
    training_dirpath = Path('../../../view_synthesis/research/010_SameViewDenseFlow/runs/training')
    train_nums = [10]
    group_qa_scores(training_dirpath, train_nums)
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
