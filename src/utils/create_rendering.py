"""Entry point for simple renderings, given a trainer and some poses."""
import os
import logging as log
from pathlib import Path
from typing import Union

import torch

from models.lowrank_model import LowrankModel
from utils.my_tqdm import tqdm
from ops.image.io import write_video_to_file
from runners.static_trainer import StaticTrainer
from runners.video_trainer import VideoTrainer


@torch.no_grad()
def render_to_path(trainer: Union[VideoTrainer, StaticTrainer], extra_name: str = "") -> None:
    """Render all poses in the `test_dataset`, saving them to file
    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    dataset = trainer.test_dataset
    video_num = 0  # Only one spiral video will be generated irrespective of number of test poses
    pred_dirpath = Path(trainer.log_dir) / f'predicted_videos_iter{trainer.global_step:06}/'
    depth_extension = 'npy' if trainer.save_true_depth else 'png'
    pred_depth_scales_path = pred_dirpath / f'depth/depth_scales_spiral01_FrameWise.csv'

    pb = tqdm(total=dataset.timestamps.numel(), desc=f"Rendering scene")
    frames = []
    for img_idx, data in enumerate(dataset):
        frame_num = img_idx

        pred_frame_path = pred_dirpath / f'rgb/{video_num:04}_spiral01/{frame_num:04}.png'
        pred_depth_path = pred_dirpath / f'depth/{video_num:04}_spiral01/{frame_num:04}.{depth_extension}'
        if not (pred_frame_path.exists() and pred_depth_path.exists()):
            ts_render = trainer.eval_step(data, video_index=0)  # Only one spiral video will be generated irrespective of number of test poses

            if isinstance(dataset.img_h, int):
                img_h, img_w = dataset.img_h, dataset.img_w
            else:
                img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
            preds_rgb = (
                ts_render["rgb"]
                .reshape(img_h, img_w, 3)
                .cpu()
                .clamp(0, 1)
                .mul(255.0)
                .byte()
                .numpy()
            )
            preds_depth = (
                ts_render["depth"]
                .reshape(img_h, img_w, 1)
                .cpu()
                .numpy()
            )
            # frames.append(preds_rgb)
            trainer.save_image(pred_frame_path, preds_rgb)
            depth_scale = trainer.save_depth(pred_depth_path, preds_depth, as_png=True)
            trainer.save_frame_depth_scale(pred_depth_scales_path, video_num, frame_num, depth_scale)
        pb.update(1)
    pb.close()
    trainer.generate_videos(pred_dirpath, video_num, video_name_suffix='_spiral01')

    # out_fname = os.path.join(trainer.log_dir, f"rendering_path_{extra_name}.mp4")
    # write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {pred_frame_path.parent.as_posix()}")
    return


def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


@torch.no_grad()
def decompose_space_time(trainer: StaticTrainer, extra_name: str = "") -> None:
    """Render space-time decomposition videos for poses in the `test_dataset`.

    The space-only part of the decomposition is obtained by setting the time-planes to 1.
    The time-only part is obtained by simple subtraction of the space-only part from the full
    rendering.

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    chosen_cam_idx = 15
    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append([grid.data for grid in multires_grids])
    pn_parameters = []
    for pn in model.proposal_networks:
        pn_parameters.append([grid_plane.data for grid_plane in pn.grids])

    camdata = None
    for img_idx, data in enumerate(dataset):
        if img_idx == chosen_cam_idx:
            camdata = data
    if camdata is None:
        raise ValueError(f"Cam idx {chosen_cam_idx} invalid.")

    num_frames = img_idx + 1
    frames = []
    for img_idx in tqdm(range(num_frames), desc="Rendering scene with separate space and time components"):
        # Linearly interpolated timestamp, normalized between -1, 1
        camdata["timestamps"] = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model: turn on time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:
                model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = pn_parameters[i][plane_idx]
        preds = trainer.eval_step(camdata)
        full_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Space-only model: turn off time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:  # time-grids off
                model.field.grids[i][plane_idx].data = torch.ones_like(parameters[i][plane_idx])
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = torch.ones_like(pn_parameters[i][plane_idx])
        preds = trainer.eval_step(camdata)
        spatial_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Temporal model: full - space
        temporal_out = normalize_for_disp(full_out - spatial_out)

        frames.append(
            torch.cat([full_out, spatial_out, temporal_out], dim=1)
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
