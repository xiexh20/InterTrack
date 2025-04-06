"""
optimize the raw point clouds, with some better loss parameters
"""

import pickle as pkl
import sys, os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
import time
from typing import Iterable, Optional, Any

import cv2
import trimesh
from tqdm import tqdm

sys.path.append(os.getcwd())
import hydra
import torch
import wandb
import numpy as np
from pytorch3d.renderer.cameras import PerspectiveCameras

from configs.structured import ProjectConfig
import os.path as osp
from utils.losses import chamfer_distance
import torch.nn.functional as F
from training_utils import TrainState

from main_objopt_latent import TrainerObjOpt


class TrainerObjOptPoints(TrainerObjOpt):
    def compute_chamf_loss(self, obj_pts_no_trans, obj_pts_recon, obj_pts_live=None,
                           obj_trans_orig=None, obj_scale_orig=None):
        "compute chamfer on the interaction space, so the translation is regularized "
        obj_pts_recon_hoi = obj_pts_recon * obj_scale_orig[:, None] * 2 + obj_trans_orig[:, None]
        return chamfer_distance(obj_pts_live, obj_pts_recon_hoi).mean()

    def get_recon_pts_for_vis(self, batch):
        "recon points in hoi space"
        obj_pts_recon = torch.stack(batch['pred_obj'], 0).cuda()
        obj_scale_orig = torch.stack(batch['radius_obj_pred'], 0).to(self.device)
        obj_trans_orig = torch.stack(batch['cent_obj_pred'], 0).to(self.device)
        obj_pts_recon_hoi = obj_pts_recon * obj_scale_orig[:, None] * 2 + obj_trans_orig[:, None]
        return obj_pts_recon_hoi

    def get_opt_pts_for_vis(self, bs, obj_pts_no_trans, obj_pts_live=None):
        "use optimized points in hoi space "
        pts, feats = obj_pts_live, torch.tensor([[[0.0, 1., 1.0]]]).repeat(bs, obj_pts_live.shape[1], 1)
        return feats, pts

    def get_front_camera(self, batch, bs, cam_t):
        "use Kroi for H+O crop instead of object crop only"
        front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                        [0, -1., 0],
                                                        [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                       T=cam_t,
                                       K=torch.stack(batch['K'], 0).to(self.device),
                                       in_ndc=True,
                                       device=self.device)
        return front_cam


    def init_obj_scale(self, device, opt_obj_scale):
        "per-frame scale instead of one scale for all"
        if opt_obj_scale:
            obj_scale = torch.ones(2500, 1).to(device).requires_grad_(True)  # it should not be a single value, but also per-frame!
        else:
            obj_scale = torch.ones(2500, 1).to(device)
        return obj_scale

    def compute_loss_temp_t(self, cam_t, obj_trans_bi, obj_trans_orig):
        "different translation"
        trans_comb = obj_trans_bi + obj_trans_orig
        velo1 = trans_comb[1:-1] - trans_comb[:-2]
        velo2 = trans_comb[2:] - trans_comb[1:-1]
        # acceleration should be zero
        loss_temp_t = F.mse_loss(velo1, velo2, reduction='none').sum(-1).mean()
        return loss_temp_t


    def load_checkpoint(self, cfg, model, model_ema, optimizer, scheduler):
        "load optimizer, model state, scheduler etc. "
        return TrainState() # do not load any ckpt as we are optimizing raw points instead of the latents

    def canonical2live(self, obj_poses, obj_pts_can, obj_rot_bi, obj_trans_bi, obj_trans_orig,
                       obj_scale_bi, obj_scale_orig,
                       ret_no_trans=False):
        """
        the obj_scale is (T, 1),
        the live space now is the normalized H+O, not object only space
        such that using K of object, it can be projected to object only crop
        the obj_pts_no_trans will be the points with correct rotation, but not scale
        Parameters
        ----------
        obj_poses : network predicted pose
        obj_pts_can :
        obj_rot_bi :
        obj_trans_bi :
        obj_scale_bi : optimized per-frame scale
        obj_scale_orig: original predicted object scale
        ret_no_trans :

        Returns
        -------

        """
        # First apply predicted translation, then optimized translation
        obj_pts_live = torch.matmul(obj_pts_can, obj_poses[:, :3, :3].transpose(1, 2))
        obj_pts_live_no_trans = torch.matmul(obj_pts_live, obj_rot_bi.transpose(1, 2)) # (B, N, 3)
        # Apply scale and translation to HOI space
        obj_pts_live = obj_pts_live_no_trans * obj_scale_orig[:, None] * obj_scale_bi[:, None] * 2
        obj_pts_live = obj_pts_live + obj_trans_bi[:, None] + obj_trans_orig[:, None] # transform to H+O space now
        if ret_no_trans:
            return obj_pts_live, obj_pts_live_no_trans
        return obj_pts_live

    def get_obj_RT(self, batch, cfg):
        "the camera t is for normalizing H+O, not object only"
        if cfg.run.sample_mode in ['gt', 'recon-t']:
            obj_poses = torch.stack(batch['abs_pose'], 0).cuda()  # (B, 4, 4)
        else:
            # object pose doest not change
            obj_poses = torch.stack(batch['rot_pred'], 0).cuda()
        if cfg.run.sample_mode in ['recon-t', 'recon-rt']:
            # use predicted translation
            # translation of the object in camera space (normalized H+O)
            cam_t = torch.stack(batch['T'], 0).to(self.device)
        else:
            # print("Using GT translation")
            cam_t = torch.stack(batch['T'], 0).to(self.device)
        return cam_t, obj_poses



@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    # to prevent random shuffle
    cfg.run.job = 'sample' if cfg.run.job == 'train' else cfg.run.job
    cfg.run.freeze_feature_model = False
    cfg.dataloader.num_workers = 2  # reduce initialization waiting time
    cfg.dataset.load_obj_pose = False # do not load GT pose!
    wandb_cache = cfg.logging.wandb
    trainer = TrainerObjOptPoints(cfg)
    cfg.logging.wandb = wandb_cache
    print("Using wandb?", cfg.logging.wandb)
    trainer.run_sample(cfg)


if __name__ == '__main__':
    main()