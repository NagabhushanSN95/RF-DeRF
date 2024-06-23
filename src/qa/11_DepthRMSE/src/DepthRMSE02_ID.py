# Shree KRISHNAya Namaha
# RMSE measure between predicted depths and depths from dense input NeRF
# Author: Nagabhushan S N
# Last Modified: 23/06/2024

import argparse
import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import skvideo.io
from tqdm import tqdm

import CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_metric_name = this_filename[:-3]


class DepthRMSE:
    def __init__(self, videos_data: pandas.DataFrame, verbose_log: bool = True) -> None:
        super().__init__()
        self.videos_data = videos_data
        self.verbose_log = verbose_log
        return

    @staticmethod
    def compute_depth_rmse(gt_depth: numpy.ndarray, eval_depth: numpy.ndarray):
        error = gt_depth.astype('float') - eval_depth.astype('float')
        rmse = numpy.sqrt(numpy.mean(numpy.square(error)))
        return rmse

    def compute_avg_rmse(self, old_data: pandas.DataFrame, dense_model_dirpath: Path, pred_train_dirpath: Path,
                         iter_num: int, resolution_suffix: str, downsampling_factor: int):
        """

        :param old_data:
        :param dense_model_dirpath:
        :param pred_train_dirpath:
        :param iter_num:
        :param resolution_suffix:
        :param downsampling_factor:
        :return:
        """
        qa_scores = []
        for i, video_data in tqdm(self.videos_data.iterrows(), total=self.videos_data.shape[0], leave=self.verbose_log):
            scene_name, pred_video_num = video_data
            gt_depth_path = dense_model_dirpath / f'{scene_name}/predicted_videos_iter{90000:06}/depth/{pred_video_num:04}.mp4'
            gt_depth_scales_path = dense_model_dirpath / f'{scene_name}/predicted_videos_iter{90000:06}/depth/depth_scales_VideoWise.csv'
            pred_depth_path = pred_train_dirpath / f'{scene_name}/predicted_videos_iter{iter_num:06}/depth/{pred_video_num:04}.mp4'
            pred_depth_scales_path = pred_train_dirpath / f'{scene_name}/predicted_videos_iter{iter_num:06}/depth/depth_scales_VideoWise.csv'
            if not (gt_depth_path.exists() and gt_depth_scales_path.exists() and pred_depth_path.exists() and pred_depth_scales_path.exists()):
                continue
            gt_depth = self.read_depth(gt_depth_path, gt_depth_scales_path, pred_video_num)
            pred_depth = self.read_depth(pred_depth_path, pred_depth_scales_path, pred_video_num)
            if (downsampling_factor > 1) and (gt_depth.shape != pred_depth.shape):
                gt_depth = self.downsample_depth(gt_depth, downsampling_factor)

            for frame_num in range(gt_depth.shape[0]):
                if old_data is not None and old_data.loc[
                    (old_data['scene_name'] == scene_name) & (old_data['pred_video_num'] == pred_video_num) & (old_data['pred_frame_num'] == frame_num)
                ].size > 0:
                    continue
                qa_score = self.compute_depth_rmse(gt_depth[frame_num], pred_depth[frame_num])
                qa_scores.append([scene_name, pred_video_num, frame_num, qa_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=['scene_name', 'pred_video_num', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        avg_rmse = numpy.mean(merged_data[this_metric_name])
        merged_data = merged_data.round({this_metric_name: 4, })
        avg_rmse = numpy.round(avg_rmse, 4)
        if isinstance(avg_rmse, numpy.floating):
            avg_rmse = avg_rmse.item()
        return avg_rmse, merged_data

    @staticmethod
    def update_qa_frame_data(old_data: pandas.DataFrame, new_data: pandas.DataFrame):
        if (old_data is not None) and (new_data.size > 0):
            old_data = old_data.copy()
            new_data = new_data.copy()
            old_data.set_index(['scene_name', 'pred_video_num', 'pred_frame_num'], inplace=True)
            new_data.set_index(['scene_name', 'pred_video_num', 'pred_frame_num'], inplace=True)
            merged_data = old_data.combine_first(new_data).reset_index()
        elif old_data is not None:
            merged_data = old_data
        else:
            merged_data = new_data
        return merged_data

    @classmethod
    def read_depth(cls, depth_path: Path, depth_scales_path: Path, video_num: int):
        depth = skvideo.io.vread(depth_path.as_posix())[:, :, :, 0]
        depth_scale = cls.get_depth_scale(depth_scales_path, video_num)
        depth = depth.astype('float32') * depth_scale
        return depth

    @staticmethod
    def get_depth_scale(depth_scales_path: Path, video_num: int):
        depth_scales_data = pandas.read_csv(depth_scales_path)
        depth_scale = depth_scales_data.loc[depth_scales_data['video_num'] == video_num]['depth_scale'].values[0]
        if isinstance(depth_scale, numpy.floating):
            depth_scale = depth_scale.item()
        return depth_scale

    @staticmethod
    def downsample_depth(depth: numpy.ndarray, downsampling_factor: int):
        downsampled_depth = skvideo.transform.rescale(depth, scale=1 / downsampling_factor, preserve_range=True,
                                                      multichannel=False, anti_aliasing=True)
        return downsampled_depth


def get_iter_nums(pred_train_dirpath: Path):
    iter_nums = []
    for pred_videos_dirpath in sorted(pred_train_dirpath.glob('**/predicted_videos_iter*')):
        iter_num = int(pred_videos_dirpath.stem[-6:])
        iter_nums.append(iter_num)
    iter_nums = numpy.unique(iter_nums).tolist()
    return iter_nums


# noinspection PyUnusedLocal
def start_qa(pred_train_dirpath: Path, database_dirpath: Path, dense_model_dirpath: Path, resolution_suffix,
             downsampling_factor: int):
    if not pred_train_dirpath.exists():
        print(f'Skipping QA of folder: {pred_train_dirpath.stem}. Reason: pred_train_dirpath does not exist')
        return

    if not dense_model_dirpath.exists():
        print(f'Skipping QA of folder: {pred_train_dirpath.stem}. Reason: dense_model_dirpath does not exist')
        return

    test_videos_datapath = CommonUtils.get_test_videos_datapath(database_dirpath, pred_train_dirpath)
    videos_data = pandas.read_csv(test_videos_datapath)[['scene_name', 'pred_video_num']]
    rmse_computer = DepthRMSE(videos_data)

    qa_scores_filepath = pred_train_dirpath / 'QualityScores.json'
    iter_nums = get_iter_nums(pred_train_dirpath)
    avg_scores = {}
    for iter_num in iter_nums:
        if qa_scores_filepath.exists():
            with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
                qa_scores = json.load(qa_scores_file)
        else:
            qa_scores = {}

        if str(iter_num) in qa_scores:
            if this_metric_name in qa_scores[str(iter_num)]:
                avg_rmse = qa_scores[str(iter_num)][this_metric_name]
                print(f'Average {this_metric_name}: {pred_train_dirpath.as_posix()} - {iter_num:06}: {avg_rmse}')
                print('Running QA again.')
        else:
            qa_scores[str(iter_num)] = {}

        rmse_data_path = pred_train_dirpath / f'quality_scores/iter{iter_num:06}/{this_metric_name}_FrameWise.csv'
        if rmse_data_path.exists():
            rmse_data = pandas.read_csv(rmse_data_path)
        else:
            rmse_data = None

        avg_rmse, rmse_data = rmse_computer.compute_avg_rmse(rmse_data, dense_model_dirpath, pred_train_dirpath, iter_num,
                                                             resolution_suffix, downsampling_factor)
        if numpy.isfinite(avg_rmse):
            avg_scores[iter_num] = avg_rmse
            qa_scores[str(iter_num)][this_metric_name] = avg_rmse
            print(f'Average {this_metric_name}: {pred_train_dirpath.as_posix()} - {iter_num:06}: {avg_rmse}')
            with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
                simplejson.dump(qa_scores, qa_scores_file, indent=4)
            rmse_data_path.parent.mkdir(parents=True, exist_ok=True)
            rmse_data.to_csv(rmse_data_path, index=False)
    return avg_scores


def demo1():
    pred_train_dirpath = Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1001')
    database_dirpath = Path('../../../../databases/InterDigital/data')
    dense_model_dirpath = Path('../../../view_synthesis/literature/001_Kplanes/runs/training/train1001')
    resolution_suffix = '_down2'
    downsampling_factor = 1
    avg_score = start_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, resolution_suffix, downsampling_factor)
    return avg_score


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

    avg_score = start_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, resolution_suffix, downsampling_factor)
    return avg_score


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
        avg_score = demo1()
    elif args['demo_function_name'] == 'demo2':
        avg_score = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return avg_score


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    parsed_args = parse_args()
    try:
        output_score = main(parsed_args)
        run_result = f'Program completed successfully!\nAverage {this_metric_name}: {output_score}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
