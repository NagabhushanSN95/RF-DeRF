# Shree KRISHNAya Namaha
# Runs all metrics serially
# Author: Nagabhushan S N
# Last Modified: 28/12/2023

import argparse
import datetime
import importlib.util
import time
import traceback
from pathlib import Path
from typing import List

import QualityScoresGrouper, QualityScoresSorter

this_filepath = Path(__file__)
this_filename = Path(__file__).stem


def run_all_specified_qa(metric_filepaths: List[Path], pred_train_dirpath: Path, database_dirpath: Path,
                         dense_model_dirpath: Path, resolution_suffix: str, downsampling_factor: int):
    args_values = locals()
    qa_scores = {}
    for metric_file_path in metric_filepaths:
        spec = importlib.util.spec_from_file_location('module.name', metric_file_path.absolute().resolve().as_posix())
        qa_module = importlib.util.module_from_spec(spec)
        # noinspection PyUnresolvedReferences
        spec.loader.exec_module(qa_module)
        function_arguments = {}
        for arg_name in run_all_specified_qa.__code__.co_varnames[:run_all_specified_qa.__code__.co_argcount]:
            # noinspection PyUnresolvedReferences
            if arg_name in qa_module.start_qa.__code__.co_varnames[:qa_module.start_qa.__code__.co_argcount]:
                function_arguments[arg_name] = args_values[arg_name]
        # noinspection PyUnresolvedReferences
        qa_score = qa_module.start_qa(**function_arguments)
        # noinspection PyUnresolvedReferences
        qa_name = qa_module.this_metric_name
        qa_scores[qa_name] = qa_score
    return qa_scores


def run_all_qa(pred_train_dirpath: Path, database_dirpath: Path, dense_model_dirpath: Path, resolution_suffix: str,
               downsampling_factor: int):
    frame_metric_filepaths = [
        this_filepath.parent / '../../11_DepthRMSE/src/DepthRMSE01_N3DV.py',
        this_filepath.parent / '../../12_DepthMAE/src/DepthMAE01_N3DV.py',
        this_filepath.parent / '../../13_DepthSROCC/src/DepthSROCC01_N3DV.py',
    ]

    qa_scores = run_all_specified_qa(frame_metric_filepaths, pred_train_dirpath, database_dirpath, dense_model_dirpath,
                                     resolution_suffix, downsampling_factor)
    train_num = int(pred_train_dirpath.stem[-4:])
    QualityScoresGrouper.group_qa_scores(pred_train_dirpath.parent, [train_num])
    QualityScoresSorter.sort_qa_scores(pred_train_dirpath.parent, [train_num])
    return qa_scores


def demo1():
    root_dirpath = Path('../../')
    pred_train_dirpath = root_dirpath / 'runs/training/train0006'
    database_dirpath = root_dirpath / 'data/databases/N3DV/data'
    gt_depth_dirpath = root_dirpath / 'data/dense_input_radiance_fields/Kplanes/runs/training/train0001'
    resolution_suffix = '_down2'
    downsampling_factor = 1
    qa_scores = run_all_qa(pred_train_dirpath, database_dirpath, gt_depth_dirpath, resolution_suffix, downsampling_factor)
    return qa_scores


def demo2(args: dict):
    pred_train_dirpath = args['pred_train_dirpath']
    if pred_train_dirpath is None:
        raise RuntimeError(f'Please provide pred_train_dirpath')
    pred_train_dirpath = Path(pred_train_dirpath)

    database_dirpath = args['database_dirpath']
    if database_dirpath is None:
        raise RuntimeError(f'Please provide database_dirpath')
    database_dirpath = Path(database_dirpath)

    dense_model_dirpath = args['dense_model_dirpath']
    if dense_model_dirpath is None:
        raise RuntimeError(f'Please provide dense_model_dirpath')
    dense_model_dirpath = Path(dense_model_dirpath)

    resolution_suffix = args['resolution_suffix']
    downsampling_factor = args['downsampling_factor']

    qa_scores = run_all_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, resolution_suffix,  downsampling_factor)
    return qa_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_train_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--dense_model_dirpath')
    parser.add_argument('--resolution_suffix', default='_down4')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--chat_names', nargs='+')
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_train_dirpath': args.pred_train_dirpath,
        'database_dirpath': args.database_dirpath,
        'dense_model_dirpath': args.dense_model_dirpath,
        'resolution_suffix': args.resolution_suffix,
        'downsampling_factor': args.downsampling_factor,
        'chat_names': args.chat_names,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        qa_scores = demo1()
    elif args['demo_function_name'] == 'demo2':
        qa_scores = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return qa_scores


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    parsed_args = parse_args()
    try:
        qa_scores_dict = main(parsed_args)
        qa_scores_str = '\n'.join([f'{key}: {value}' for key, value in qa_scores_dict.items()])
        run_result = f'Program completed successfully!\n\n{parsed_args["pred_train_dirpath"]}\n{qa_scores_str}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
