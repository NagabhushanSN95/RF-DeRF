# Shree KRISHNAya Namaha
# Groups QA scores video-wise, scene-wise and the overall average score
# Author: Nagabhushan S N
# Last Modified: 11/11/2023

import argparse
import json
from typing import Optional, List

import numpy
import pandas

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def update_qa_data(qa_data: Optional[pandas.DataFrame], new_qa_data: pandas.DataFrame):
    if qa_data is None:
        qa_data = new_qa_data
    else:
        qa_data = pandas.concat([qa_data, new_qa_data])
    return qa_data


def get_grouped_qa_scores(qa_data: pandas.DataFrame):
    column_names = list(qa_data.columns)
    video_wise_scores = qa_data.groupby(by=column_names[:2]).mean().reset_index()[column_names[:2] + column_names[-1:]]
    scene_wise_scores = qa_data.groupby(by=column_names[:1]).mean().reset_index()[column_names[:1] + column_names[-1:]]
    avg_score = qa_data[column_names[-1]].mean()

    frame_wise_scores = qa_data.round({column_names[-1]: 4, })
    video_wise_scores = video_wise_scores.round({column_names[-1]: 4, })
    scene_wise_scores = scene_wise_scores.round({column_names[-1]: 4, })
    avg_score = numpy.round(avg_score, 4)
    return avg_score, scene_wise_scores, video_wise_scores, frame_wise_scores


def group_qa_scores(train_dirpaths: List[Path]):
    for train_dirpath in train_dirpaths:
        scene_dirpaths = sorted(train_dirpath.iterdir())
        iters = []
        qa_names = []
        for scene_dirpath in scene_dirpaths:
            if not scene_dirpath.is_dir():
                continue
            # Determine all the iterations and QA methods for which QA scores are available
            scene_qa_dirpaths = sorted(scene_dirpath.glob('quality_scores_iter*'))
            scene_iters = sorted(map(lambda dirpath: int(dirpath.stem[len('quality_scores_iter'):]), scene_qa_dirpaths))
            iters.extend(scene_iters)
            for qa_dirpath in scene_qa_dirpaths:
                scene_qa_filepaths = sorted(qa_dirpath.glob('*.csv'))
                scene_qa_names = sorted(map(lambda filepath: filepath.stem, scene_qa_filepaths))
                qa_names.extend(scene_qa_names)
        iters = sorted(numpy.unique(iters))
        qa_names = sorted(numpy.unique(qa_names))

        qa_scores = {}
        for iter_num in iters:
            qa_scores[str(iter_num)] = {}
            for qa_name in qa_names:
                qa_data = None
                for scene_dirpath in scene_dirpaths:
                    if not scene_dirpath.is_dir():
                        continue
                    scene_qa_filepath = scene_dirpath / f'quality_scores_iter{iter_num:06}/{qa_name}.csv'
                    if scene_qa_filepath.exists():
                        scene_qa_data = pandas.read_csv(scene_qa_filepath)
                        qa_data = update_qa_data(qa_data, scene_qa_data)
                avg_score, scene_wise_scores, video_wise_scores, frame_wise_scores = get_grouped_qa_scores(qa_data)
                frame_wise_filepath = train_dirpath / f'quality_scores/iter{iter_num:06}/{qa_name}_FrameWise.csv'
                video_wise_filepath = train_dirpath / f'quality_scores/iter{iter_num:06}/{qa_name}_VideoWise.csv'
                scene_wise_filepath = train_dirpath / f'quality_scores/iter{iter_num:06}/{qa_name}_SceneWise.csv'
                frame_wise_filepath.parent.mkdir(parents=True, exist_ok=True)
                video_wise_filepath.parent.mkdir(parents=True, exist_ok=True)
                scene_wise_filepath.parent.mkdir(parents=True, exist_ok=True)
                frame_wise_scores.to_csv(frame_wise_filepath, index=False)
                video_wise_scores.to_csv(video_wise_filepath, index=False)
                scene_wise_scores.to_csv(scene_wise_filepath, index=False)
                qa_scores[str(iter_num)][qa_name] = avg_score
            # qa_scores[iter_num] = dict(sorted(qa_scores[iter_num].items()))
        qa_scores_path = train_dirpath / 'QualityScores.json'
        with open(qa_scores_path.as_posix(), 'w') as qa_file:
            json.dump(qa_scores, qa_file, indent=4)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dirpaths', nargs='+', type=str)

    args = parser.parse_args()
    train_dirpaths = [Path(train_dirpath) for train_dirpath in args.train_dirpaths]
    group_qa_scores(train_dirpaths)
    return


if __name__ == '__main__':
    main()
