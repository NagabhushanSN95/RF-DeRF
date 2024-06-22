import logging as log
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, MutableMapping, Union, Any, List

import numpy
import pandas
import pandas as pd
import skimage.io
import torch
import torch.utils.data

from datasets.video_datasets import Video360Dataset
from utils.ema import EMA
from utils.my_tqdm import tqdm
from ops.image import metrics
from ops.image.io import write_video_to_file
from models.lowrank_model import LowrankModel
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss, SparseFlowLoss, DenseFlowLoss, SparseDepthLoss
)


class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 save_true_depth: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        self.save_true_depth = save_true_depth
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = False
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            video_index = kwargs['video_index']
            near_far = data["near_fars"][video_index:video_index+1].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k:
                        preds[k].append(v.cpu())
                    # if ('pi' in k) or (k == 'weights'):
                    #     preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")

        pred_dirpath = Path(self.log_dir) / f'predicted_videos_iter{self.global_step:06}/'
        num_videos = dataset.median_imgs.shape[0]
        num_frames_per_video = dataset.num_samples // num_videos
        depth_extension = 'npy' if self.save_true_depth else 'png'
        pred_depth_scales_path = pred_dirpath / f'depth/depth_scales_FrameWise.csv'

        for img_idx, data in enumerate(dataset):
            video_index = img_idx // num_frames_per_video
            video_num = dataset.video_ids[video_index]
            frame_num = img_idx % num_frames_per_video

            pred_frame_path = pred_dirpath / f'rgb/{video_num:04}/{frame_num:04}.png'
            pred_depth_path = pred_dirpath / f'depth/{video_num:04}/{frame_num:04}.{depth_extension}'
            if pred_frame_path.exists() and pred_depth_path.exists():
                print('Warning @ video_trainer.py/validate(): Loading saved images and depth instead of re-rendering them')
                preds = {}
                frames_collage = torch.from_numpy(self.read_image(pred_frame_path).astype(numpy.float32) / 255)
                depths_collage = torch.from_numpy(self.read_depth(pred_depth_path)[:, :, None])
                h = frames_collage.shape[0] // 3
                preds['rgb'] = frames_collage[:h, :, :].reshape(-1, 3)
                preds['depth'] = depths_collage[:h, :, :].reshape(-1, 1)
                preds['prop_depth_0'] = depths_collage[h:2*h, :, :].reshape(-1, 1)
                preds['prop_depth_1'] = depths_collage[2*h:3*h, :, :].reshape(-1, 1)
            else:
                preds = self.eval_step(data, video_index=video_index)

            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            if self.compute_video_metrics:
                pred_frames.append(out_img)
                if out_depth is not None:
                    out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)

            if pred_frame_path.exists() and pred_depth_path.exists():
                pass
            else:
                self.save_image(pred_frame_path, out_img)
                depth_scale = self.save_depth(pred_depth_path, out_depth, as_png=True)
                self.save_frame_depth_scale(pred_depth_scales_path, video_num, frame_num, depth_scale)
            if self.save_video and (frame_num == num_frames_per_video - 1):
                self.generate_videos(pred_dirpath, video_num)

            pb.set_postfix_str(f"PSNR={out_metrics['PSNR']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        # if self.save_video:
        #     write_video_to_file(
        #         os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
        #         pred_frames
        #     )
        #     if len(out_depths) > 0:
        #         write_video_to_file(
        #             os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
        #             out_depths
        #         )
        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

        # Save individual frame metrics
        for qa_name in per_scene_metrics:
            qa_scores_path = Path(self.log_dir) / f'quality_scores_iter{self.global_step:06}/{qa_name}.csv'
            qa_scores = numpy.array(per_scene_metrics[qa_name])
            scene_names = numpy.array([dataset.name] * dataset.num_samples)
            video_nums = numpy.array(dataset.video_ids)[None].repeat(num_frames_per_video, axis=0).ravel(order='F')
            frame_nums = numpy.arange(num_frames_per_video)[None].repeat(num_videos, axis=0).ravel(order='C')
            column_names = ['scene_name', 'video_num', 'frame_num', qa_name]
            column_values = [scene_names, video_nums, frame_nums, qa_scores]
            qa_scores_dict = dict(zip(column_names, column_values))
            qa_scores_data = pandas.DataFrame.from_dict(qa_scores_dict)
            if qa_scores_path.exists():
                old_qa_scores = pandas.read_csv(qa_scores_path)
                qa_scores_data = pandas.concat([old_qa_scores, qa_scores_data])
                qa_scores_data = qa_scores_data.drop_duplicates(column_names[:-1])
            qa_scores_path.parent.mkdir(parents=True, exist_ok=True)
            qa_scores_data.to_csv(qa_scores_path, index=False)
        return

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
            SparseFlowLoss(kwargs.get('sparse_flow_loss_weight', 0.0),
                           threshold=kwargs.get('sparse_flow_loss_threshold', 0),
                           sfap=kwargs.get('sparse_flow_loss_average_point', False),
                           sfwe=kwargs.get('sparse_flow_loss_weighted_error', False),
                           sf_stop_gradient_weights=kwargs.get('sparse_flow_loss_stop_gradient_weights', False)),
            DenseFlowLoss(kwargs.get('dense_flow_loss_weight', 0.0),
                          threshold=kwargs.get('dense_flow_loss_threshold', 0),
                          dfap=kwargs.get('dense_flow_loss_average_point', False),
                          dfwe=kwargs.get('dense_flow_loss_weighted_error', False),
                          df_stop_gradient_weights=kwargs.get('dense_flow_loss_stop_gradient_weights', False)),
            SparseDepthLoss(kwargs.get('sparse_depth_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(path.as_posix(), image)
        return

    @staticmethod
    def read_depth(path: Path) -> numpy.ndarray:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def save_depth(path: Path, depth: numpy.ndarray, as_png: bool = False):
        if (depth.ndim == 3) and (depth.shape[2] == 1):
            depth = depth[:, :, 0]
        path.parent.mkdir(parents=True, exist_ok=True)
        depth_scale = depth.max() / 255
        depth_image = numpy.round(depth / depth_scale).astype('uint8')
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), depth_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), depth)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), depth_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth_scale

    @staticmethod
    def save_frame_depth_scale(path: Path, video_num: int, frame_num: int, depth_scale: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        scales_data = pandas.DataFrame.from_dict({'video_num': [video_num], 'frame_num': [frame_num], 'depth_scale': [depth_scale]})
        if path.exists():
            existing_data = pandas.read_csv(path)
            scales_data = pandas.concat([existing_data, scales_data] , ignore_index=True)
        scales_data.to_csv(path, index=False)
        return

    @staticmethod
    def save_video_depth_scale(path: Path, video_num: int, depth_scale: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        scales_data = pandas.DataFrame.from_dict({'video_num': [video_num], 'depth_scale': [depth_scale]})
        if path.exists():
            existing_data = pandas.read_csv(path)
            scales_data = pandas.concat([existing_data, scales_data] , ignore_index=True)
        scales_data.to_csv(path, index=False)
        return

    @classmethod
    def generate_videos(cls, videos_dirpath: Path, video_num: int, *, video_name_suffix: str = ''):
        rgb_dirpath = videos_dirpath / f'rgb/{video_num:04}{video_name_suffix}'
        rgb_video_path = videos_dirpath / f'rgb/{video_num:04}{video_name_suffix}.mp4'
        if rgb_dirpath.exists():
            cmd = f"ffmpeg -y -framerate 30 -pattern_type glob -i '{rgb_dirpath.as_posix()}/*.png' -c:v libx264 -pix_fmt yuv420p {rgb_video_path.as_posix()}"
            os.system(cmd)
        depth_dirpath = videos_dirpath / f'depth/{video_num:04}{video_name_suffix}'
        depth_video_path = videos_dirpath / f'depth/{video_num:04}{video_name_suffix}.mp4'
        if depth_dirpath.exists():
            frame_depth_scales_path = videos_dirpath / f'depth/depth_scales{video_name_suffix}_FrameWise.csv'
            if frame_depth_scales_path.exists():
                try:
                    frame_depth_scales_data = pandas.read_csv(frame_depth_scales_path)
                    video_depth_scale = max(frame_depth_scales_data[frame_depth_scales_data['video_num'] == video_num]['depth_scale'].to_numpy())
                    for depth_image_path in tqdm(sorted(depth_dirpath.glob('*.png')), desc='Rescaling depth images'):
                        frame_num = int(depth_image_path.stem)
                        depth_image = skimage.io.imread(depth_image_path.as_posix())
                        depth_scale = frame_depth_scales_data[
                            (frame_depth_scales_data['video_num'] == video_num) &
                            (frame_depth_scales_data['frame_num'] == frame_num)
                        ]['depth_scale'].to_numpy()[0]
                        depth_image_norm = numpy.round(depth_image * depth_scale / video_depth_scale).astype('uint8')
                        skimage.io.imsave(depth_image_path.as_posix(), depth_image_norm, check_contrast=False)
                    video_depth_scales_path = videos_dirpath / f'depth/depth_scales{video_name_suffix}_VideoWise.csv'
                    cls.save_video_depth_scale(video_depth_scales_path, video_num, video_depth_scale)
                except (IndexError,Exception) as e:
                    print(e)
                    traceback.print_exc()
            cmd = f"ffmpeg -y -framerate 30 -pattern_type glob -i '{depth_dirpath.as_posix()}/*.png' -c:v libx264 -pix_fmt yuv420p {depth_video_path.as_posix()}"
            os.system(cmd)
        return


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir,
        camera_suffix=kwargs.get('camera_suffix', ''),
        split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
        set_num=kwargs['set_num'],
        num_frames=kwargs['num_frames'],
        num_render_frames=kwargs.get('num_render_frames', 300),
        scene_name=kwargs['expname'],
        flow_dirpath=kwargs['flow_dirpath'],
        sparse_flow_dirnames=kwargs.get('sparse_flow_dirnames', []),
        num_sparse_flow_pixels=kwargs.get('num_sparse_flow_pixels', 0),
        dense_flow_dirnames=kwargs.get('dense_flow_dirnames', []),
        dense_flow_mask_dirnames=kwargs.get('dense_flow_mask_dirnames', None),
        dense_flow_cache_size=kwargs.get('dense_flow_cache_size', 0),
        dense_flow_reload_iters=kwargs.get('dense_flow_reload_iters', 0),
        depth_dirpath=kwargs['depth_dirpath'],
        sparse_depth_dirnames=kwargs.get('sparse_depth_dirnames', []),
        num_sparse_depth_pixels=kwargs.get('num_sparse_depth_pixels', 0),
    )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, **kwargs):
    downsample = 2.0  # Both D-NeRF and DyNeRF use downsampling by 2
    ts_dset = Video360Dataset(
        data_dir,
        camera_suffix=kwargs.get('camera_suffix', ''),
        split=split, downsample=downsample,
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
        set_num=kwargs['set_num'],
        num_frames=kwargs['num_frames'],
        num_render_frames=kwargs.get('num_render_frames', 300),
        scene_name=kwargs['expname'],
        flow_dirpath=kwargs['flow_dirpath'],
        sparse_flow_dirnames=kwargs.get('sparse_flow_dirnames', []),
        num_sparse_flow_pixels=kwargs.get('num_sparse_flow_pixels', 0),
        dense_flow_dirnames=kwargs.get('dense_flow_dirnames', []),
        dense_flow_mask_dirnames=kwargs.get('dense_flow_mask_dirnames', None),
        dense_flow_cache_size=kwargs.get('dense_flow_cache_size', 0),
        dense_flow_reload_iters=kwargs.get('dense_flow_reload_iters', 0),
        depth_dirpath=kwargs['depth_dirpath'],
        sparse_depth_dirnames=kwargs.get('sparse_depth_dirnames', []),
        num_sparse_depth_pixels=kwargs.get('num_sparse_depth_pixels', 0),
    )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, validate_train_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not validate_train_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    # test_split = 'render' if render_only else 'test'
    if render_only:
        test_split = 'render'
    elif validate_train_only:
        test_split = 'train'
    else:
        test_split = 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od
