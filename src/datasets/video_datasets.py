import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

import numpy
import numpy as np
import pandas
import skimage.io
import skvideo.io
import torch
from tqdm import tqdm

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import _split_poses_bounds
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path, center_poses
)
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
)


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 camera_suffix: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6, *,
                 set_num: int,
                 num_frames: int,
                 num_render_frames: int,
                 scene_name: str,
                 flow_dirpath: str,
                 sparse_flow_dirnames: List[str],
                 num_sparse_flow_pixels: int,
                 dense_flow_dirnames: List[str],
                 dense_flow_mask_dirnames: List[str],
                 dense_flow_cache_size: int,
                 dense_flow_reload_iters: int,
                 depth_dirpath: str,
                 sparse_depth_dirnames: List[str],
                 num_sparse_depth_pixels: int,
                 ):
        self.datadir = datadir
        self.camera_suffix = camera_suffix
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.set_num = set_num

        self.scene_name = scene_name
        self.flow_dirpath = Path(flow_dirpath)
        self.flow_masks_dirpath = self.flow_dirpath.parent / 'estimated_flow_masks'
        self.sparse_flow_dirnames = sparse_flow_dirnames
        self.num_sparse_flow_pixels = num_sparse_flow_pixels
        self.sparse_flow_indices = None
        self.sparse_flow_pointer = 0
        self.dense_flow_dirnames = dense_flow_dirnames
        self.dense_flow_mask_dirnames = dense_flow_mask_dirnames
        self.dense_flow_filepaths = None
        self.dense_flow_mask_filepaths = None
        self.dense_flow_index = 0
        self.dense_flow_data = None
        self.dense_flow_cache_size = dense_flow_cache_size
        self.dense_flow_iter_num = 0
        self.dense_flow_reload_iters = dense_flow_reload_iters
        self.depth_dirpath = Path(depth_dirpath)
        self.sparse_depth_dirnames = sparse_depth_dirnames
        self.num_sparse_depth_pixels = num_sparse_depth_pixels
        self.sparse_depth_indices = None
        self.sparse_depth_pointer = 0

        self.median_imgs = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        if "lego" in datadir or "dnerf" in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        # Note: timestamps are stored normalized between -1, 1.
        if dset_type == "llff":
            if split == "render":
                assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                per_cam_poses, per_cam_near_fars, per_cam_intrinsics, _, self.video_ids, resolution, depth_scale = load_llffvideo_poses(
                    datadir, camera_suffix=self.camera_suffix, downsample=self.downsample, split='all',
                    near_scaling=self.near_scaling, set_num=self.set_num)
                render_poses = generate_spiral_path(
                    per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=num_render_frames,
                    n_rots=2, zrate=0.5, dt=self.near_scaling, percentile=60)
                self.poses = torch.from_numpy(render_poses).float()
                self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]])
                timestamps = torch.linspace(0, num_render_frames-1, len(self.poses))
                imgs = None
                # Set intrinsics as mean of per_cam_intrinsics
                intrinsics = torch.mean(per_cam_intrinsics, dim=0, keepdim=True).repeat(num_render_frames, 1, 1)
            else:
                per_cam_poses, per_cam_near_fars, per_cam_intrinsics, videopaths, self.video_ids, resolution, depth_scale = load_llffvideo_poses(
                    datadir, camera_suffix=self.camera_suffix, downsample=self.downsample, split=split,
                    near_scaling=self.near_scaling, set_num=self.set_num)
                if split == 'test':
                    keyframes = False
                poses, intrinsics, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                    videopaths=videopaths, cam_poses=per_cam_poses, cam_intrinsics=per_cam_intrinsics,
                    num_frames=num_frames, resolution=resolution, split=split, keyframes=keyframes,
                    keyframes_take_each=30)
                self.poses = poses.float()
                if contraction:
                    self.per_cam_near_fars = per_cam_near_fars.float()
                else:
                    self.per_cam_near_fars = torch.tensor(
                        [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)

                # Load sparse flow prior
                if (split == 'train') and (self.num_sparse_flow_pixels > 0):
                    matches_data_list = []
                    for sparse_flow_dirname in self.sparse_flow_dirnames:
                        sparse_flow_path = self.flow_dirpath / f'{sparse_flow_dirname}/{scene_name}/MatchedPixels.csv'
                        matches_data = pandas.read_csv(sparse_flow_path)
                        filtered_matches_data = matches_data[
                            matches_data['video1_num'].isin(self.video_ids) &
                            matches_data['video2_num'].isin(self.video_ids)
                        ].reset_index()[matches_data.columns]
                        matches_data_list.append(filtered_matches_data)
                    self.matches_data = pandas.concat(matches_data_list, axis=0)
                    num_matches = self.matches_data.shape[0]
                    self.sparse_flow_indices = numpy.arange(num_matches)
                    numpy.random.shuffle(self.sparse_flow_indices)
                    print(f'Loaded {num_matches} matched pixels for sparse flow')

                # Load dense flow filenames
                if split == 'train':
                    self.dense_flow_filepaths = []
                    for dense_flow_dirname in self.dense_flow_dirnames:
                        dense_flow_dirpath = self.flow_dirpath / f'{dense_flow_dirname}/{scene_name}/estimated_flows'
                        dense_flow_filepaths = list(dense_flow_dirpath.iterdir())
                        filtered_flow_filepaths = list(filter(
                            lambda path: (int(path.stem.split('__')[0].split('_')[0]) in self.video_ids) and (
                                        int(path.stem.split('__')[1].split('_')[0]) in self.video_ids),
                            dense_flow_filepaths)
                        )
                        self.dense_flow_filepaths.extend(filtered_flow_filepaths)
                    print(f'Located {len(self.dense_flow_filepaths)} dense flow files')
                    self.dense_flow_data = dict.fromkeys(sorted(self.dense_flow_filepaths))

                    self.dense_flow_mask_filepaths = []
                    if self.dense_flow_mask_dirnames is not None:
                        for dense_flow_mask_dirnames in self.dense_flow_mask_dirnames:
                            dense_flow_masks_dirpath = self.flow_masks_dirpath / f'{dense_flow_mask_dirnames}/{scene_name}/valid_masks'
                            dense_flow_mask_filepaths = list(dense_flow_masks_dirpath.iterdir())
                            filtered_mask_filepaths = list(filter(
                                lambda path: (int(path.stem.split('__')[0].split('_')[0]) in self.video_ids) and (
                                        int(path.stem.split('__')[1].split('_')[0]) in self.video_ids),
                                dense_flow_mask_filepaths)
                            )
                            self.dense_flow_mask_filepaths.extend(filtered_mask_filepaths)
                        self.dense_flow_mask_filepaths = sorted(self.dense_flow_mask_filepaths)
                        assert len(self.dense_flow_mask_filepaths) == len(self.dense_flow_filepaths)
                        print(f'Located {len(self.dense_flow_mask_filepaths)} dense flow mask files')
                    else:
                        print('Warning! Dense flow masks not loaded.')

                    self.dense_flow_filepaths = numpy.array(self.dense_flow_filepaths)
                    numpy.random.shuffle(self.dense_flow_filepaths)

                    self.reload_dense_flow_data()

                # Load sparse depth prior
                if (split == 'train') and (self.num_sparse_depth_pixels > 0):
                    sparse_depth_data_list = []
                    for sparse_depth_dirname in self.sparse_depth_dirnames:
                        sparse_depth_path = self.depth_dirpath / f'{sparse_depth_dirname}/{scene_name}/EstimatedDepths.csv'
                        sparse_depth_data = pandas.read_csv(sparse_depth_path)
                        filtered_sparse_depth_data = sparse_depth_data[
                            sparse_depth_data['video_num'].isin(self.video_ids)
                        ].reset_index()[sparse_depth_data.columns]
                        filtered_sparse_depth_data['depth'] /= depth_scale
                        sparse_depth_data_list.append(filtered_sparse_depth_data)
                    self.sparse_depth_data = pandas.concat(sparse_depth_data_list, axis=0)
                    num_depths = self.sparse_depth_data.shape[0]
                    self.sparse_depth_indices = numpy.arange(num_depths)
                    numpy.random.shuffle(self.sparse_depth_indices)
                    print(f'Loaded {num_depths} sparse depth pixels')

            # These values are tuned for the salmon video
            self.global_translation = torch.tensor([0, 0, 2.])
            self.global_scale = torch.tensor([0.5, 0.6, 1])
            # Normalize timestamps between -1, 1
            timestamps = (timestamps.float() / (num_frames-1)) * 2 - 1
        elif dset_type == "synthetic":
            assert not contraction, "Synthetic video dataset does not work with contraction."
            assert not ndc, "Synthetic video dataset does not work with NDC."
            if split == 'render':
                num_tsteps = 120
                dnerf_durations = {'hellwarrior': 100, 'mutant': 150, 'hook': 100, 'bouncingballs': 150, 'lego': 50, 'trex': 200, 'standup': 150, 'jumpingjacks': 200}
                for scene in dnerf_durations.keys():
                    if 'dnerf' in datadir and scene in datadir:
                        num_tsteps = dnerf_durations[scene]
                render_poses = torch.stack([
                    generate_spherical_poses(angle, -30.0, 4.0)
                    for angle in np.linspace(-180, 180, num_tsteps + 1)[:-1]
                ], 0)
                imgs = None
                self.poses = render_poses
                timestamps = torch.linspace(0.0, 1.0, render_poses.shape[0])
                _, transform = load_360video_frames(
                    datadir, 'train', max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                img_h, img_w = 800, 800
            else:
                frames, transform = load_360video_frames(
                    datadir, split, max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                imgs, self.poses = load_360_images(frames, datadir, split, self.downsample)
                timestamps = torch.tensor(
                    [fetch_360vid_info(f)[0] for f in frames], dtype=torch.float32)
                img_h, img_w = imgs[0].shape[:2]
            if ndc:
                self.per_cam_near_fars = torch.tensor([[0.0, self.ndc_far]])
            else:
                self.per_cam_near_fars = torch.tensor([[2.0, 6.0]])
            if "dnerf" in datadir:
                # dnerf time is between 0, 1. Normalize to -1, 1
                timestamps = timestamps * 2 - 1
            else:
                # lego (our vid) time is like dynerf: between 0, 30.
                timestamps = (timestamps.float() / torch.amax(timestamps)) * 2 - 1
            intrinsics = load_360_intrinsics(
                transform, img_h=img_h, img_w=img_w, downsample=self.downsample)
            resolution = (img_h, img_w)
        else:
            raise ValueError(datadir)

        self.timestamps = timestamps
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(1, resolution[0], resolution[1]).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if imgs is not None and imgs.dtype != torch.uint8:
            imgs = (imgs * 255).to(torch.uint8)
        if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
            self.median_imgs = (self.median_imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        elif imgs is not None:
            imgs = imgs.view(-1, resolution[0] * resolution[1], imgs.shape[-1])

        # ISG/IST weights are computed on 4x subsampled data.
        weights_subsampled = int(4 / downsample)
        if scene_bbox is not None:
            scene_bbox = torch.tensor(scene_bbox)
        else:
            scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            resolution=resolution,
            imgs=imgs,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )

        if split == 'train':
            h, w = self.resolution
            self.num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)

        self.isg_weights = None
        self.ist_weights = None
        if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
            isg_weights_path = self.get_is_weights_path(datadir, is_name='isg')
            if os.path.exists(isg_weights_path):
                self.isg_weights = torch.load(isg_weights_path)
                log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file {isg_weights_path}.")
            else:
                # Precompute ISG weights
                t_s = time.time()
                gamma = 1e-3 if self.keyframes else 2e-2
                self.isg_weights = dynerf_isg_weight(
                    imgs.view(-1, resolution[0], resolution[1], imgs.shape[-1]),
                    median_imgs=self.median_imgs, gamma=gamma)
                # Normalize into a probability distribution, to speed up sampling
                self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                torch.save(self.isg_weights, isg_weights_path)
                t_e = time.time()
                log.info(f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s and saved to file {isg_weights_path}.")

            ist_weights_path = self.get_is_weights_path(datadir, is_name='ist')
            if os.path.exists(ist_weights_path):
                self.ist_weights = torch.load(ist_weights_path)
                log.info(f"Reloaded {self.ist_weights.shape[0]} IST weights from file {ist_weights_path}.")
            else:
                # Precompute IST weights
                t_s = time.time()
                self.ist_weights = dynerf_ist_weight(
                    imgs.view(-1, self.resolution[0], self.resolution[1], imgs.shape[-1]),
                    num_cameras=self.median_imgs.shape[0])
                # Normalize into a probability distribution, to speed up sampling
                self.ist_weights = (self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                torch.save(self.ist_weights, ist_weights_path)
                t_e = time.time()
                log.info(f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s and saved to file {ist_weights_path}.")

        if self.isg:
            self.enable_isg()

        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of size {self.resolution[0]}x{self.resolution[1]}. "
                 f"Images loaded: {self.imgs is not None}. "
                 f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                 f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def get_is_weights_path(self, datadir: str, *, is_name) -> str:
        if self.downsample == 2:
            # if downsample=2, this is actual training. But importance sampling weights are generated with downsample=4
            #  or downsample=8. So, check if they exist. If they exist, use the weights generated with higher sampling.
            #  If not, use weights with downsample=2. This should not happen - so print a warning.
            if (Path(datadir) / f'{is_name}_weights_set{self.set_num:02}_down4.pt').exists():
                is_weights_path = Path(datadir) / f'{is_name}_weights_set{self.set_num:02}_down4.pt'
            elif (Path(datadir) / f'{is_name}_weights_set{self.set_num:02}_down8.pt').exists():
                is_weights_path = Path(datadir) / f'{is_name}_weights_set{self.set_num:02}_down8.pt'
            else:
                print('Warning! Using importance sampling weights with downsample=2')
                is_weights_path = Path(datadir) / f'{is_name}_weights_set{self.set_num:02}_down{int(self.downsample)}.pt'
        else:
            # if downsample=4 or 8, this is case where the importance sampling weights are being generated. So, use
            #  appropriate name.
            is_weights_path = Path(datadir) / f'{is_name}_weights_set{self.set_num:02}_down{int(self.downsample)}.pt'
        return is_weights_path.as_posix()

    def __getitem__(self, index):
        h, w = self.resolution
        dev = "cpu"
        dense_flow_mask = None
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                if isinstance(index, tuple):
                    # index is a tuple of (index, dense_flow_mask)
                    index, dense_flow_mask = index
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                # We must deal with the fact that ISG/IST weights are computed on a dataset with
                # different 'downsampling' factor. E.g. if the weights were computed on 4x
                # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
                # will be 4 / 2 = 2.
                # Split each subsampled index into its 16 components in 2D.
                hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(self.weights_subsampled):
                    for aw in range(self.weights_subsampled):
                        x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w
            x, y = x + 0.5, y + 0.5
            video_index = torch.div(image_id, self.num_frames_per_camera, rounding_mode='floor')
            if isinstance(self.video_ids, list):
                self.video_ids = numpy.array(self.video_ids)
            video_num = torch.from_numpy(self.video_ids[video_index]).to(dev)
            frame_num = torch.remainder(image_id, self.num_frames_per_camera)
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)

        out = {
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "imgs": None,
        }
        if self.split == 'train':
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])

        c2w = self.poses[image_id]                                    # [num_rays or 1, 3, 4]
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics[image_id], True)  # [num_rays, 3]
        out['rays_o'], out['rays_d'] = get_rays(camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0,
                                                intrinsics=self.intrinsics[image_id], resolution=self.resolution,
                                                normalize_rd=True)  # [num_rays, 3]

        imgs = out['imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs

        if dense_flow_mask is not None:
            out['dense_flow_mask'] = dense_flow_mask

        if (self.split == 'train') and (self.num_sparse_flow_pixels > 0):
            out['video_ids'] = torch.from_numpy(self.video_ids[None]).to(dev)
            out['intrinsics'] = self.intrinsics
            extrinsics_last_row = torch.Tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat([video_num.shape[0], 1, 1]).to(c2w)
            extrinsics_c2w = torch.cat([c2w, extrinsics_last_row], dim=1)
            out['extrinsics1_c2w'] = extrinsics_c2w
            out['extrinsics2_c2w'] = torch.zeros_like(extrinsics_c2w)

            out['sparse_flow_mask'] = torch.zeros_like(video_num, dtype=bool)
            out['video1_num'] = video_num
            out['frame1_num'] = frame_num
            out['video2_num'] = -1 * torch.ones_like(video_num)
            out['frame2_num'] = -1 * torch.ones_like(frame_num)
            out['timestamps2'] = -2 * torch.ones_like(out['timestamps'])
            out['y1'] = y
            out['x1'] = x
            out['y2'] = -1 * torch.ones_like(y)
            out['x2'] = -1 * torch.ones_like(x)
            out = self.add_sparse_flow_data(out, dev)

        if (self.split == 'train') and (self.num_sparse_depth_pixels > 0):
            out['sparse_depth_mask'] = torch.zeros_like(out['timestamps'], dtype=bool)
            out['sparse_depth'] = torch.zeros_like(out['timestamps'])
            out = self.add_sparse_depth_data(out, dev)

        return out

    def get_rand_ids(self, index):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if self.sampling_weights is not None:
            if len(self.dense_flow_filepaths) > 0:
                raise NotImplementedError
            else:
                batch_size = self.batch_size // (self.weights_subsampled ** 2)
                num_weights = len(self.sampling_weights)
                if num_weights > self.sampling_batch_size:
                    # Take a uniform random sample first, then according to the weights
                    subset = torch.randint(
                        0, num_weights, size=(self.sampling_batch_size,),
                        dtype=torch.int64, device=self.sampling_weights.device)
                    samples = torch.multinomial(
                        input=self.sampling_weights[subset], num_samples=batch_size)
                    return subset[samples]
                return torch.multinomial(
                    input=self.sampling_weights, num_samples=batch_size)
        else:
            batch_size = self.batch_size
            if self.use_permutation:
                raise NotImplementedError
            elif len(self.dense_flow_filepaths) > 0:
                rand_ids, dense_flow_mask = self.get_rand_ids_dense_flow()
                return rand_ids, dense_flow_mask
            else:
                return torch.randint(0, self.num_samples, size=(batch_size,))

    def add_sparse_flow_data(self, return_dict: dict, device):
        sparse_flow_ids = self.get_sparse_flow_rand_ids()

        matches_data = self.matches_data.iloc[sparse_flow_ids]
        columns_data = numpy.split(matches_data.to_numpy()[:, :8], 8, axis=1)
        columns_data = map(lambda x: x[:, 0], columns_data)
        video1_num_sf, frame1_num_sf, video2_num_sf, frame2_num_sf, x1_sf, y1_sf, x2_sf, y2_sf = columns_data
        x1_sf, y1_sf, x2_sf, y2_sf = map(lambda x: x.round(), [x1_sf, y1_sf, x2_sf, y2_sf])
        video1_index_sf = numpy.where(self.video_ids == video1_num_sf[:, None])[1]
        video2_index_sf = numpy.where(self.video_ids == video2_num_sf[:, None])[1]
        image_id1_sf = video1_index_sf * self.num_frames_per_camera + frame1_num_sf.astype('int64')
        image_id2_sf = video2_index_sf * self.num_frames_per_camera + frame2_num_sf.astype('int64')
        h, w = self.resolution
        index1_sf = image_id1_sf * h * w + y1_sf.astype('int64') * w + x1_sf.astype('int64')
        index2_sf = image_id2_sf * h * w + y2_sf.astype('int64') * w + x2_sf.astype('int64')

        timestamps1_sf = self.timestamps[index1_sf]  # (num_rays or 1, )
        timestamps2_sf = self.timestamps[index2_sf]  # (num_rays or 1, )
        self.concat_dict_element(return_dict, 'timestamps', timestamps1_sf)
        self.concat_dict_element(return_dict, 'timestamps', timestamps2_sf)
        self.concat_dict_element(return_dict, 'timestamps2', timestamps2_sf)
        self.concat_dict_element(return_dict, 'timestamps2', timestamps1_sf)

        camera_id1_sf = image_id1_sf // self.num_frames_per_camera  # (num_rays)
        camera_id2_sf = image_id2_sf // self.num_frames_per_camera  # (num_rays)
        near_fars1_sf = self.per_cam_near_fars[camera_id1_sf, :]
        near_fars2_sf = self.per_cam_near_fars[camera_id1_sf, :]
        self.concat_dict_element(return_dict, 'near_fars', near_fars1_sf)
        self.concat_dict_element(return_dict, 'near_fars', near_fars2_sf)

        imgs1_sf = (self.imgs[index1_sf] / 255.0).view(-1, self.imgs.shape[-1])
        imgs2_sf = (self.imgs[index2_sf] / 255.0).view(-1, self.imgs.shape[-1])
        # Alpha compositing
        if imgs1_sf is not None and imgs1_sf.shape[-1] == 4:
            imgs1_sf = imgs1_sf[:, :3] * imgs1_sf[:, 3:] + return_dict['bg_color'] * (1.0 - imgs1_sf[:, 3:])
        if imgs2_sf is not None and imgs2_sf.shape[-1] == 4:
            imgs2_sf = imgs2_sf[:, :3] * imgs2_sf[:, 3:] + return_dict['bg_color'] * (1.0 - imgs2_sf[:, 3:])
        self.concat_dict_element(return_dict, 'imgs', imgs1_sf)
        self.concat_dict_element(return_dict, 'imgs', imgs2_sf)

        c2w1_sf = self.poses[image_id1_sf]  # [num_rays or 1, 3, 4]
        c2w2_sf = self.poses[image_id2_sf]  # [num_rays or 1, 3, 4]
        video1_num_sf, frame1_num_sf, video2_num_sf, frame2_num_sf = map(lambda x: torch.from_numpy(x).float().to(device), [video1_num_sf, frame1_num_sf, video2_num_sf, frame2_num_sf])
        x1_sf, y1_sf, x2_sf, y2_sf = map(lambda x: torch.from_numpy(x).float().to(device), [x1_sf, y1_sf, x2_sf, y2_sf])
        camera_dirs1_sf = stack_camera_dirs(x1_sf, y1_sf, self.intrinsics[image_id1_sf], True)  # [num_rays, 3]
        camera_dirs2_sf = stack_camera_dirs(x2_sf, y2_sf, self.intrinsics[image_id2_sf], True)  # [num_rays, 3]
        rays_o1_sf, rays_d1_sf = get_rays(camera_dirs1_sf, c2w1_sf, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics[image_id1_sf], resolution=self.resolution, normalize_rd=True)  # [num_rays, 3]
        rays_o2_sf, rays_d2_sf = get_rays(camera_dirs2_sf, c2w2_sf, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics[image_id2_sf], resolution=self.resolution, normalize_rd=True)  # [num_rays, 3]
        self.concat_dict_element(return_dict, 'rays_o', rays_o1_sf)
        self.concat_dict_element(return_dict, 'rays_o', rays_o2_sf)
        self.concat_dict_element(return_dict, 'rays_d', rays_d1_sf)
        self.concat_dict_element(return_dict, 'rays_d', rays_d2_sf)

        extrinsics_last_row = torch.Tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat([video1_num_sf.shape[0], 1, 1]).to(c2w1_sf)
        extrinsics1_c2w_sf = torch.cat([c2w1_sf, extrinsics_last_row], dim=1)
        extrinsics2_c2w_sf = torch.cat([c2w2_sf, extrinsics_last_row], dim=1)
        self.concat_dict_element(return_dict, 'extrinsics1_c2w', extrinsics1_c2w_sf)
        self.concat_dict_element(return_dict, 'extrinsics1_c2w', extrinsics2_c2w_sf)
        self.concat_dict_element(return_dict, 'extrinsics2_c2w', extrinsics2_c2w_sf)
        self.concat_dict_element(return_dict, 'extrinsics2_c2w', extrinsics1_c2w_sf)

        sf_mask1 = torch.ones_like(video1_num_sf, dtype=bool)
        sf_mask2 = torch.ones_like(video2_num_sf, dtype=bool)
        self.concat_dict_element(return_dict, 'sparse_flow_mask', sf_mask1)
        self.concat_dict_element(return_dict, 'sparse_flow_mask', sf_mask2)
        self.concat_dict_element(return_dict, 'video1_num', video1_num_sf)
        self.concat_dict_element(return_dict, 'video1_num', video2_num_sf)
        self.concat_dict_element(return_dict, 'frame1_num', frame1_num_sf)
        self.concat_dict_element(return_dict, 'frame1_num', frame2_num_sf)
        self.concat_dict_element(return_dict, 'video2_num', video2_num_sf)
        self.concat_dict_element(return_dict, 'video2_num', video1_num_sf)
        self.concat_dict_element(return_dict, 'frame2_num', frame2_num_sf)
        self.concat_dict_element(return_dict, 'frame2_num', frame1_num_sf)
        self.concat_dict_element(return_dict, 'y1', y1_sf)
        self.concat_dict_element(return_dict, 'y1', y2_sf)
        self.concat_dict_element(return_dict, 'x1', x1_sf)
        self.concat_dict_element(return_dict, 'x1', x2_sf)
        self.concat_dict_element(return_dict, 'y2', y2_sf)
        self.concat_dict_element(return_dict, 'y2', y1_sf)
        self.concat_dict_element(return_dict, 'x2', x2_sf)
        self.concat_dict_element(return_dict, 'x2', x1_sf)

        # Dense flow items
        if 'dense_flow_mask' in return_dict:
            df_mask1 = torch.zeros_like(video1_num_sf, dtype=bool)
            df_mask2 = torch.zeros_like(video2_num_sf, dtype=bool)
            self.concat_dict_element(return_dict, 'dense_flow_mask', df_mask1)
            self.concat_dict_element(return_dict, 'dense_flow_mask', df_mask2)

        # Sparse depth items
        if 'sparse_depth_mask' in return_dict:
            sd_mask1 = torch.zeros_like(video1_num_sf, dtype=bool)
            sd_mask2 = torch.zeros_like(video2_num_sf, dtype=bool)
            self.concat_dict_element(return_dict, 'sparse_depth_mask', sd_mask1)
            self.concat_dict_element(return_dict, 'sparse_depth_mask', sd_mask2)
            self.concat_dict_element(return_dict, 'sparse_depth', torch.zeros_like(video1_num_sf))

        return return_dict

    def get_sparse_flow_rand_ids(self):
        sparse_flow_ids = self.sparse_flow_indices[self.sparse_flow_pointer: self.sparse_flow_pointer + (self.num_sparse_flow_pixels // 2)]
        self.sparse_flow_pointer += (self.num_sparse_flow_pixels // 2)
        if self.sparse_flow_pointer >= self.matches_data.shape[0]:
            self.sparse_flow_pointer = 0
            numpy.random.shuffle(self.sparse_flow_indices)
        return sparse_flow_ids

    def add_sparse_depth_data(self, return_dict: dict, device):
        sparse_depth_ids = self.get_sparse_depth_rand_ids()

        depth_data = self.sparse_depth_data.iloc[sparse_depth_ids]
        columns_data = numpy.split(depth_data.to_numpy()[:, :5], 5, axis=1)
        columns_data = map(lambda x: x[:, 0], columns_data)
        video_num_sd, frame_num_sd, x_sd, y_sd, depth_sd = columns_data
        x_sd, y_sd = map(lambda x: x.round(), [x_sd, y_sd])
        video_index_sd = numpy.where(self.video_ids == video_num_sd[:, None])[1]
        image_id_sd = video_index_sd * self.num_frames_per_camera + frame_num_sd.astype('int64')
        h, w = self.resolution
        index_sd = image_id_sd * h * w + y_sd.astype('int64') * w + x_sd.astype('int64')

        timestamps_sd = self.timestamps[index_sd]  # (num_rays or 1, )
        self.concat_dict_element(return_dict, 'timestamps', timestamps_sd)

        camera_id_sd = image_id_sd // self.num_frames_per_camera  # (num_rays)
        near_fars_sd = self.per_cam_near_fars[camera_id_sd, :]
        self.concat_dict_element(return_dict, 'near_fars', near_fars_sd)

        imgs_sd = (self.imgs[index_sd] / 255.0).view(-1, self.imgs.shape[-1])
        # Alpha compositing
        if imgs_sd is not None and imgs_sd.shape[-1] == 4:
            imgs_sd = imgs_sd[:, :3] * imgs_sd[:, 3:] + return_dict['bg_color'] * (1.0 - imgs_sd[:, 3:])
        self.concat_dict_element(return_dict, 'imgs', imgs_sd)

        c2w_sd = self.poses[image_id_sd]  # [num_rays or 1, 3, 4]
        video_num_sd, frame_num_sd = map(lambda x: torch.from_numpy(x).float().to(device), [video_num_sd, frame_num_sd])
        x_sd, y_sd = map(lambda x: torch.from_numpy(x).float().to(device), [x_sd, y_sd])
        camera_dirs_sd = stack_camera_dirs(x_sd, y_sd, self.intrinsics[image_id_sd], True)  # [num_rays, 3]
        rays_o_sd, rays_d_sd = get_rays(camera_dirs_sd, c2w_sd, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics[image_id_sd], resolution=self.resolution, normalize_rd=True)  # [num_rays, 3]
        self.concat_dict_element(return_dict, 'rays_o', rays_o_sd)
        self.concat_dict_element(return_dict, 'rays_d', rays_d_sd)

        if 'sparse_flow_mask' in return_dict:
            extrinsics_last_row = torch.Tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat([video_num_sd.shape[0], 1, 1]).to(c2w_sd)
            extrinsics_c2w_sd = torch.cat([c2w_sd, extrinsics_last_row], dim=1)
            self.concat_dict_element(return_dict, 'extrinsics1_c2w', extrinsics_c2w_sd)
            self.concat_dict_element(return_dict, 'extrinsics2_c2w', torch.zeros_like(extrinsics_c2w_sd))

            sf_mask = torch.zeros_like(video_num_sd, dtype=bool)
            self.concat_dict_element(return_dict, 'sparse_flow_mask', sf_mask)
            self.concat_dict_element(return_dict, 'video1_num', video_num_sd)
            self.concat_dict_element(return_dict, 'frame1_num', frame_num_sd)
            self.concat_dict_element(return_dict, 'video2_num', -1 * torch.ones_like(video_num_sd))
            self.concat_dict_element(return_dict, 'frame2_num', -1 * torch.ones_like(frame_num_sd))
            self.concat_dict_element(return_dict, 'y1', y_sd)
            self.concat_dict_element(return_dict, 'x1', x_sd)
            self.concat_dict_element(return_dict, 'y2', -1 * torch.ones_like(y_sd))
            self.concat_dict_element(return_dict, 'x2', -1 * torch.ones_like(x_sd))

        if 'dense_flow_mask' in return_dict:
            df_mask = torch.zeros_like(video_num_sd, dtype=bool)
            self.concat_dict_element(return_dict, 'dense_flow_mask', df_mask)

        sd_mask = torch.ones_like(video_num_sd, dtype=bool)
        self.concat_dict_element(return_dict, 'sparse_depth_mask', sd_mask)
        self.concat_dict_element(return_dict, 'sparse_depth', torch.from_numpy(depth_sd).float().to(device))
        return return_dict

    def get_sparse_depth_rand_ids(self):
        sparse_depth_ids = self.sparse_depth_indices[self.sparse_depth_pointer: self.sparse_depth_pointer + self.num_sparse_depth_pixels]
        self.sparse_depth_pointer += self.num_sparse_depth_pixels
        if self.sparse_depth_pointer >= self.sparse_depth_data.shape[0]:
            self.sparse_depth_pointer = 0
            numpy.random.shuffle(self.sparse_depth_indices)
        return sparse_depth_ids

    @staticmethod
    def concat_dict_element(dictionary, key, new_value):
        dictionary[key] = torch.cat([dictionary[key], new_value])
        return

    def get_rand_ids_dense_flow(self):
        valid_flow_paths = list(filter(lambda path: self.dense_flow_data[path] is not None, self.dense_flow_filepaths))
        rand_flow_paths = numpy.random.choice(valid_flow_paths, size=self.batch_size, replace=True)
        video1_num = numpy.array([int(path.stem.split('__')[0].split('_')[0]) for path in rand_flow_paths])
        frame1_num = numpy.array([int(path.stem.split('__')[0].split('_')[1]) for path in rand_flow_paths])
        video1_id = numpy.where(self.video_ids == video1_num[:, None])[1]
        frame1_id = video1_id * self.num_frames_per_camera + frame1_num.astype('int64')
        y1_rand = numpy.random.randint(0, self.resolution[0], size=self.batch_size)
        x1_rand = numpy.random.randint(0, self.resolution[1], size=self.batch_size)
        flow1 = numpy.array([self.dense_flow_data[flow_path][0][y1_rand[i], x1_rand[i]] for i, flow_path in enumerate(rand_flow_paths)])
        mask1 = numpy.array([self.dense_flow_data[flow_path][1][y1_rand[i], x1_rand[i]] for i, flow_path in enumerate(rand_flow_paths)])
        frame1_id_final = frame1_id[:self.batch_size//2]
        y1_final = y1_rand[:self.batch_size//2]
        x1_final = x1_rand[:self.batch_size//2]
        flow1_final = flow1[:self.batch_size//2]
        mask1_final = mask1[:self.batch_size//2]

        video2_num = numpy.array([int(path.stem.split('__')[1].split('_')[0]) for path in rand_flow_paths])
        frame2_num = numpy.array([int(path.stem.split('__')[1].split('_')[1]) for path in rand_flow_paths])
        video2_id = numpy.where(self.video_ids == video2_num[:, None])[1]
        frame2_id = video2_id * self.num_frames_per_camera + frame2_num.astype('int64')
        frame2_id_from_flow = frame2_id[:self.batch_size//2]
        y2_from_flow = y1_final + flow1_final[:, 1]
        x2_from_flow = x1_final + flow1_final[:, 0]
        frame2_id_rand = frame2_id[self.batch_size//2:]
        y2_rand = y1_rand[self.batch_size//2:]
        x2_rand = x1_rand[self.batch_size//2:]
        # flow2_final = numpy.zeros_like(flow1_final)
        mask2_final = mask1_final
        frame2_id_final = mask1_final * frame2_id_from_flow + (1 - mask1_final) * frame2_id_rand
        y2_final = mask1_final * y2_from_flow + (1 - mask1_final) * y2_rand
        x2_final = mask1_final * x2_from_flow + (1 - mask1_final) * x2_rand

        frame_id = numpy.concatenate([frame1_id_final, frame2_id_final])
        y = numpy.concatenate([y1_final, y2_final])
        x = numpy.concatenate([x1_final, x2_final])
        # flow = numpy.concatenate([flow1_final, flow2_final])
        mask = numpy.concatenate([mask1_final, mask2_final])
        index = frame_id * self.resolution[0] * self.resolution[1] + y * self.resolution[1] + x

        index = torch.from_numpy(index).long()
        # flow = torch.from_numpy(flow).float()
        mask = torch.from_numpy(mask).bool()

        self.dense_flow_iter_num += 1
        if self.dense_flow_iter_num >= self.dense_flow_reload_iters:
            self.reload_dense_flow_data()
            self.dense_flow_iter_num = 0
        return index, mask

    def reload_dense_flow_data(self):
        print('Reloading dense flow data (partially)')
        # Reset the previously loaded data
        self.dense_flow_data = dict.fromkeys(sorted(self.dense_flow_filepaths))

        # Pick the next flow files to be loaded
        next_flow_paths = self.dense_flow_filepaths[self.dense_flow_index: self.dense_flow_index + self.dense_flow_cache_size]
        if len(next_flow_paths) < self.dense_flow_cache_size:
            # End of list reached. Load from beginning to complete the current batch and re-shuffle the list
            extra_next_flow_paths = self.dense_flow_filepaths[:self.dense_flow_cache_size - len(next_flow_paths)]
            next_flow_paths = numpy.concatenate([next_flow_paths, extra_next_flow_paths])
            self.dense_flow_index = 0
            numpy.random.shuffle(self.dense_flow_filepaths)

        # Load the next flow files
        for flow_path in next_flow_paths:
            flow = numpy.load(flow_path)['arr_0']
            if self.dense_flow_mask_dirnames is not None:
                flow_mask_path = next(filter(lambda x: x.stem == flow_path.stem, self.dense_flow_mask_filepaths))
                flow_mask = skimage.io.imread(flow_mask_path) == 255
            else:
                flow_mask = numpy.ones_like(flow[:, :, 0]).astype(bool)
            self.dense_flow_data[flow_path] = (flow, flow_mask)

        print('Reloading dense flow data (partially) done')
        return


def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def fetch_360vid_info(frame: Dict[str, Any]):
    timestamp = None
    fp = frame['file_path']
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
        pose_id = int(fp.split('r')[-1])
    if timestamp is None:  # will be None for dnerf
        timestamp = frame['time']
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: Optional[int]) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    fpath2poseid = defaultdict(list)
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    if max_cameras is not None:
        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        pose_ids = set(pose_ids[::subsample_poses])
        log.info(f"Selected subset of {len(pose_ids)} camera poses: {pose_ids}.")

    if max_tsteps is not None:
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
        timestamps = set(timestamps[::subsample_time])
        log.info(f"Selected subset of timestamps: {sorted(timestamps)} of length {len(timestamps)}")

    sub_frames = []
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
    # We need frames to be sorted by pose_id
    sub_frames = sorted(sub_frames, key=lambda f: fpath2poseid[f['file_path']])
    return sub_frames, meta


def get_video_indices(*, data_dir, set_num, split):
    database_dirpath = Path(data_dir).parent.parent.parent
    scene_name = Path(data_dir).stem
    set_path = database_dirpath / f'train_test_sets/set{set_num:02}/{split.title()}VideosData.csv'
    set_video_data = pandas.read_csv(set_path)
    video_indices = set_video_data[set_video_data['scene_name'] == scene_name]['pred_video_num'].tolist()
    return video_indices


def load_llffvideo_poses(datadir: str,
                         camera_suffix: str,
                         downsample: float,
                         split: str,
                         near_scaling: float, *,
                         set_num: int) -> Tuple[
                            torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int], Tuple[int, int], float]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        camera_suffix (str): Suffix for rgb directory and camera intrinsics file. E.g. '', '_original' or '_undistorted'
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    if os.path.exists(os.path.join(datadir, 'poses_bounds.npy')):
        poses, near_fars, intrinsics, depth_scale = load_llff_poses_helper(datadir, camera_suffix, downsample, near_scaling)
        resolution = (intrinsics.height, intrinsics.width)
        intrinsics = numpy.repeat(intrinsics.to_matrix()[None], poses.shape[0], axis=0)
    else:
        poses, near_fars, intrinsics, depth_scale = load_opencv_poses(datadir, camera_suffix, downsample, near_scaling)
        resolution = None

    reso_suffix = f'_down{int(downsample)}'
    if os.path.exists(os.path.join(datadir, f'rgb{camera_suffix}{reso_suffix}')):
        videopaths = np.array(glob.glob(os.path.join(datadir, f'rgb{camera_suffix}{reso_suffix}', '*.mp4')))  # [n_cameras]
    else:
        videopaths = np.array(glob.glob(os.path.join(datadir, f'rgb{camera_suffix}', '*.mp4')))  # [n_cameras]
    if resolution is None:
        resolution = read_resolution(videopaths[0])
    assert poses.shape[0] == len(videopaths), 'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    # if split == 'train':
    #     split_ids = np.arange(1, poses.shape[0])
    # elif split == 'test':
    #     split_ids = np.array([0])
    # else:
    #     split_ids = np.arange(poses.shape[0])
    split_ids = get_video_indices(data_dir=datadir, set_num=set_num, split=split)
    if 'coffee_martini' in datadir:
        # https://github.com/fengres/mixvoxels/blob/0013e4ad63c80e5f14eb70383e2b073052d07fba/dataLoader/llff_video.py#L323
        log.info(f"Deleting unsynchronized camera from coffee-martini video.")
        split_ids = np.setdiff1d(split_ids, 12)
    intrinsics = torch.from_numpy(intrinsics[split_ids].astype(numpy.float32))
    poses = torch.from_numpy(poses[split_ids].astype(numpy.float32))
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    return poses, near_fars, intrinsics, videopaths, split_ids, resolution, depth_scale


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        cam_intrinsics: torch.Tensor,
                        num_frames: int,
                        resolution: Tuple[int, int],
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    if ('_down2' in videopaths[0]) or ('_down4' in videopaths[0]) or ('_down8' in videopaths[0]):
        loaded = load_downsampled_videos(videopaths, cam_poses, cam_intrinsics, num_frames, keyframes, keyframes_take_each)
        imgs, poses, intrinsics, median_imgs, timestamps = loaded
    else:
        loaded = parallel_load_images(
            dset_type="video",
            tqdm_title=f"Loading {split} data",
            num_images=len(videopaths),
            paths=videopaths,
            poses=cam_poses,
            intrinsics=cam_intrinsics,
            num_frames=num_frames,
            out_h=resolution[0],
            out_w=resolution[1],
            load_every=keyframes_take_each if keyframes else 1,
        )
        imgs, poses, intrinsics, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    intrinsics = torch.cat(intrinsics, 0)  # [N, 3, 3]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, intrinsics, imgs, timestamps, median_imgs


def load_llff_poses_helper(datadir: str, camera_suffix: str, downsample: float, near_scaling: float) -> Tuple[np.ndarray, np.ndarray, Intrinsics, float]:
    poses_bounds = np.load(os.path.join(datadir, f'poses_bounds{camera_suffix}.npy'))  # (N_images, 17)
    poses, near_fars, intrinsics = _split_poses_bounds(poses_bounds)

    # Step 1: rescale focal length according to training resolution
    intrinsics.scale(1 / downsample)

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses)

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = np.min(near_fars)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    return poses, near_fars, intrinsics, scale_factor


def load_opencv_poses(datadir: str, camera_suffix: str, downsample: float, near_scaling: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    print('Loading poses in opencv format')
    resolution_suffix = f'_down{int(downsample)}'
    intrinsics_path = os.path.join(datadir, f'CameraIntrinsics{camera_suffix}{resolution_suffix}.csv')
    intrinsics = numpy.loadtxt(intrinsics_path, delimiter=',').reshape((-1, 3, 3))

    extrinsics_path = os.path.join(datadir, f'CameraExtrinsics{camera_suffix}.csv')
    extrinsics_w2c = numpy.loadtxt(extrinsics_path, delimiter=',').reshape((-1, 4, 4))
    extrinsics_c2w = numpy.linalg.inv(extrinsics_w2c)  # (x, -y, -z)
    extrinsics_llff = numpy.concatenate([extrinsics_c2w[:, :, 1:2], extrinsics_c2w[:, :, 0:1], -extrinsics_c2w[:, :, 2:3], extrinsics_c2w[:, :, 3:4]], axis=2)  # (-y, x, z)
    extrinsics_nerf = numpy.concatenate([extrinsics_llff[:, :, 1:2], -extrinsics_llff[:, :, :1], extrinsics_llff[:, :, 2:4]], axis=2)  # (x, y, z)
    poses, poses_avg = center_poses(extrinsics_nerf[:, :3, :4])

    bounds_path = os.path.join(datadir, f'DepthBounds.csv')
    bounds = numpy.loadtxt(bounds_path, delimiter=',')
    near_original = numpy.min(bounds)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    bounds /= scale_factor
    poses[:, :, 3] /= scale_factor
    return poses, bounds, intrinsics, scale_factor


def read_resolution(video_path: str) -> Tuple[int, int]:
    """Reads the resolution of a video using ffprobe."""
    import subprocess
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
               '-of', 'csv=s=x:p=0', video_path]
    w, h = subprocess.check_output(command).decode('utf-8').strip().split('x')
    resolution = (int(h), int(w))
    return resolution


def load_downsampled_videos(videopaths: List[str], cam_poses: torch.Tensor, cam_intrinsics: torch.Tensor,
                            num_frames: int, keyframes: bool, keyframes_take_each: Optional[int] = None):
    videos, poses, intrinsics, median_images, timestamps = [], [], [], [], []
    print('Loading downsampled videos')
    for video_idx in tqdm(range(len(videopaths))):
        video = skvideo.io.vread(videopaths[video_idx])[:num_frames]
        if keyframes:
            video = video[::keyframes_take_each]
        video_tr = torch.from_numpy(video)
        videos.append(video_tr)

        median_image, _ = torch.median(video_tr, dim=0)
        median_images.append(median_image)

        video_poses = cam_poses[video_idx].expand(video.shape[0], -1, -1)
        poses.append(video_poses)

        video_intrinsics = cam_intrinsics[video_idx].expand(video.shape[0], -1, -1)
        intrinsics.append(video_intrinsics)

        video_timestamps = torch.arange(video.shape[0], dtype=torch.int32)
        timestamps.append(video_timestamps)
    return videos, poses, intrinsics, median_images, timestamps


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff
