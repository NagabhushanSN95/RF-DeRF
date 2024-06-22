import abc
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch import nn

from models.lowrank_model import LowrankModel
from ops.losses.histogram_loss import interlevel_loss
from raymarching.ray_samplers import RaySamples


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


class Regularizer():
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class PlaneTV(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'planeTV-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def step(self, global_step):
        pass

    def _regularize(self, model: LowrankModel, **kwargs):
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field_3d.grids + model.field_bf.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
        for grids in multi_res_grids:
            if len(grids) == 3:
                spatial_grids = [0, 1, 2]
            else:
                spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal
            for grid_id in spatial_grids:
                total += compute_plane_tv(grids[grid_id])
            for grid in grids:
                # grid: [1, c, h, w]
                total += compute_plane_tv(grid)
        return total


class TimeSmoothness(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'time-smooth-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field_3d.grids + model.field_bf.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return torch.as_tensor(total)


class HistogramLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('histogram-loss', initial_value)

        self.visualize = False
        self.count = 0

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        if self.visualize:
            if self.count % 500 == 0:
                prop_idx = 0
                fine_idx = 1
                # proposal info
                weights_proposal = model_out["weights_list"][prop_idx].detach().cpu().numpy()
                spacing_starts_proposal = model_out["ray_samples_list"][prop_idx].spacing_starts
                spacing_ends_proposal = model_out["ray_samples_list"][prop_idx].spacing_ends
                sdist_proposal = torch.cat([
                    spacing_starts_proposal[..., 0],
                    spacing_ends_proposal[..., -1:, 0]
                ], dim=-1).detach().cpu().numpy()

                # fine info
                weights_fine = model_out["weights_list"][fine_idx].detach().cpu().numpy()
                spacing_starts_fine = model_out["ray_samples_list"][fine_idx].spacing_starts
                spacing_ends_fine = model_out["ray_samples_list"][fine_idx].spacing_ends
                sdist_fine = torch.cat([
                    spacing_starts_fine[..., 0],
                    spacing_ends_fine[..., -1:, 0]
                ], dim=-1).detach().cpu().numpy()

                for i in range(10):  # plot 10 rays
                    fix, ax1 = plt.subplots()

                    delta = np.diff(sdist_proposal[i], axis=-1)
                    ax1.bar(sdist_proposal[i, :-1], weights_proposal[i].squeeze() / delta, width=delta, align="edge", label='proposal', alpha=0.7, color="b")
                    ax1.legend()
                    ax2 = ax1.twinx()

                    delta = np.diff(sdist_fine[i], axis=-1)
                    ax2.bar(sdist_fine[i, :-1], weights_fine[i].squeeze() / delta, width=delta, align="edge", label='fine', alpha=0.3, color='r')
                    ax2.legend()
                    os.makedirs(f'histogram_loss/{self.count}', exist_ok=True)
                    plt.savefig(f'./histogram_loss/{self.count}/batch_{i}.png')
                    plt.close()
                    plt.cla()
                    plt.clf()
            self.count += 1
        return interlevel_loss(model_out['weights_list'], model_out['ray_samples_list'])


class L1ProposalNetwork(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-proposal-network', initial_value)

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        grids = [p.grids for p in model.proposal_networks]
        total = 0.0
        for pn_grids in grids:
            for grid in pn_grids:
                total += torch.abs(grid).mean()
        return torch.as_tensor(total)


class DepthTV(Regularizer):
    def __init__(self, initial_value):
        super().__init__('tv-depth', initial_value)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        depth = model_out['depth']
        tv = compute_plane_tv(
            depth.reshape(64, 64)[None, None, :, :]
        )
        return tv


class L1TimePlanes(Regularizer):
    def __init__(self, initial_value, what='field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        super().__init__(f'l1-time-{what[:2]}', initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field_3d.grids + model.field_bf.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return torch.as_tensor(total)


class DistortionLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('distortion-loss', initial_value)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        """
        Efficient O(N) realization of distortion loss.
        from https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
        There are B rays each with N sampled points.
        """
        w = model_out['weights_list'][-1]
        rs: RaySamples = model_out['ray_samples_list'][-1]
        m = (rs.starts + rs.ends) / 2
        interval = rs.deltas

        loss_uni = (1/3) * (interval * w.pow(2)).sum(dim=-1).mean()
        wm = (w * m)
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)
        loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
        loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
        loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
        return loss_bi + loss_uni


class SparseFlowLoss(Regularizer):
    def __init__(self, initial_value, threshold, sfap, sfwe, sf_stop_gradient_weights):
        super().__init__('sparse-flow-loss', initial_value)
        self.threshold = threshold
        self.sfap = sfap
        self.sfwe = sfwe
        self.sf_stop_gradient_weights = sf_stop_gradient_weights
        if sfap and sfwe:
            raise RuntimeError('sparse_flow_loss_average_point and sparse_flow_loss_weighted_error both cannot be true')
        return

    def _regularize(self, model: LowrankModel, model_out, data, **kwargs) -> torch.Tensor:
        sf_mask = data['sparse_flow_mask']
        pi_prime = model_out['pi_prime'][sf_mask]  # (nr, ns, 3)
        nr, ns = pi_prime.shape[:2]
        weights = model_out['weights'][sf_mask]  # (nr, ns, 1)
        if self.sf_stop_gradient_weights:
            weights = weights.detach()
        if self.sfap:
            pi_prime = torch.sum(weights * pi_prime, dim=1, keepdim=True)  # (nr, 1, 3)

        pi_prime1 = pi_prime[:nr//2]
        pi_prime2 = pi_prime[nr//2:]
        error = pi_prime1 - pi_prime2  # (nr//2, ns/1, 2)
        abs_error = torch.relu(error.abs() - self.threshold)
        squared_error = torch.mean(torch.pow(abs_error, 2), dim=2)  # (nr//2, ns/1)
        if self.sfwe:
            weights1 = weights[:nr//2]  # (nr//2, ns, 1)
            weights2 = weights[nr//2:]
            max_weights = torch.maximum(weights1, weights2)
            ray_loss = torch.sum(max_weights[:, :, 0] * squared_error, dim=1)  # (nr//2, )
        else:
            ray_loss = torch.mean(squared_error, dim=1)  # (nr//2, )
        loss = torch.mean(ray_loss)  # (1, )
        return loss


class DenseFlowLoss(Regularizer):
    def __init__(self, initial_value, threshold, dfap, dfwe, df_stop_gradient_weights):
        super().__init__('dense-flow-loss', initial_value)
        self.threshold = threshold
        self.dfap = dfap
        self.dfwe = dfwe
        self.df_stop_gradient_weights = df_stop_gradient_weights
        if dfap and dfwe:
            raise RuntimeError('dense_flow_loss_average_point and dense_flow_loss_weighted_error both cannot be true')
        return

    def _regularize(self, model: LowrankModel, model_out, data, **kwargs) -> torch.Tensor:
        df_mask = data['dense_flow_mask']
        pi_prime = model_out['pi_prime']
        weights = model_out['weights']
        if 'sparse_flow_mask' in data:
            sf_mask = data['sparse_flow_mask']
            df_mask = df_mask[~sf_mask]
            pi_prime = pi_prime[~sf_mask]
            weights = weights[~sf_mask]
        pi_prime = pi_prime[df_mask]  # (nr, ns, 3)
        nr, ns = pi_prime.shape[:2]
        weights = weights[df_mask]  # (nr, ns, 1)
        if self.df_stop_gradient_weights:
            weights = weights.detach()
        if self.dfap:
            pi_prime = torch.sum(weights * pi_prime, dim=1, keepdim=True)  # (nr, 1, 3)

        pi_prime1 = pi_prime[:nr//2]
        pi_prime2 = pi_prime[nr//2:]
        error = pi_prime1 - pi_prime2  # (nr//2, ns/1, 2)
        abs_error = torch.relu(error.abs() - self.threshold)
        squared_error = torch.mean(torch.pow(abs_error, 2), dim=2)  # (nr//2, ns/1)
        if self.dfwe:
            weights1 = weights[:nr//2]  # (nr//2, ns, 1)
            weights2 = weights[nr//2:]
            max_weights = torch.maximum(weights1, weights2)
            ray_loss = torch.sum(max_weights[:, :, 0] * squared_error, dim=1)  # (nr//2, )
        else:
            ray_loss = torch.mean(squared_error, dim=1)  # (nr//2, )
        loss = torch.mean(ray_loss)  # (1, )
        return loss


class SparseDepthLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('sparse-depth-loss', initial_value)
        return

    def _regularize(self, model: LowrankModel, model_out, data, **kwargs) -> torch.Tensor:
        sd_mask = data['sparse_depth_mask']
        pred_depth = model_out['depth'][sd_mask]  # (nr, 1)
        gt_depth = data['sparse_depth'][sd_mask][:, None]  # (nr, 1)
        loss = torch.mean(torch.square(pred_depth - gt_depth))  # (1, )
        return loss
