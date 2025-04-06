"""
main entry point for object optimization
final version
"""
import glob
import pickle as pkl
import sys, os
import time
from typing import Iterable, Optional, Any

import cv2
import pytorch3d.io
import trimesh
from accelerate import Accelerator
from tqdm import tqdm

sys.path.append(os.getcwd())
import hydra
import torch
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf

from transformers import get_scheduler

from configs.structured import ProjectConfig
import training_utils
from main import TrainerCrossAttnHO
from model import CrossAttenHODiffusionModel
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os.path as osp
import joblib
from sklearn.neighbors import NearestNeighbors
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from utils.losses import chamfer_distance

from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras, PointsRenderer, AlphaCompositor
from render.pyt3d_wrapper import MeshRendererWrapper, get_kinect_camera, PcloudRenderer
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Pointclouds
from scipy.ndimage.morphology import distance_transform_edt


def filter_points_knn(pts, k=50, scale=3.0):
    """
    filter points based on knn
    k: number of nearest points to compute distance
    return: filtered points and mask
    """
    x_nn = NearestNeighbors(n_neighbors=k+1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(pts)
    min_y_to_x = x_nn.kneighbors(pts)[0]  # (N, K+1), one value is always zero
    min_y_to_x = np.sort(min_y_to_x, axis=1)
    min_y_to_x = min_y_to_x[:, 1:]
    dist_avg = np.mean(min_y_to_x, 1)
    avg, std = np.mean(dist_avg), np.std(dist_avg)

    # see: https://github.com/PointCloudLibrary/pcl/blob/2661588573771c7c32568fa3657c9f5d4f8a3c11/filters/src/statistical_outlier_removal.cpp#L70
    # filter based on mean dist of knn and std: https://github.com/PointCloudLibrary/pcl/blob/2661588573771c7c32568fa3657c9f5d4f8a3c11/filters/src/statistical_outlier_removal.cpp#L234
    mask = dist_avg > avg + std * scale
    pts_new = pts[~mask]

    print(f"{np.sum(mask)}/{len(pts)} points filtered out")
    return pts_new, mask


class TrainerObjOpt(TrainerCrossAttnHO):
    def sample(self, cfg: ProjectConfig,
                model: CrossAttenHODiffusionModel,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        """sanity check: given initial random points + GT pose, can we get the shape?"""
        from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras
        from render.pyt3d_wrapper import MeshRendererWrapper, get_kinect_camera, PcloudRenderer
        from pytorch3d.renderer import look_at_view_transform
        from pytorch3d.structures import Pointclouds
        import imageio

        # Visulization
        rend_size, device = cfg.model.image_size, 'cuda'
        renderer = self.init_renderer(device, rend_size)
        # renderer = PcloudRenderer(image_size=rend_size, radius=0.0075)

        # load pretrained object ae
        from model.pvcnn.pvcnn_enc import PVCNNAutoEncoder
        model_ae = model
        model_ae.eval()
        output_dir: Path = Path(output_dir)
        self.no_ae, DEBUG = cfg.model.obj_opt_noae, False
        import socket; DEBUG = 'volta' in socket.gethostname()

        seq_name = self.extract_seq_name(cfg)

        if 'sit' in seq_name:
            cfg.model.obj_opt_occ_thres = min(0.5, cfg.model.obj_opt_occ_thres)
            if seq_name == 'Date03_Sub03_chairblack_sit':
                cfg.model.obj_opt_occ_thres = 0.3
            print(f"Changed occ threshold to {cfg.model.obj_opt_occ_thres} for sitting sequence {seq_name}")
        # try to load from ckpt
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        opt_obj_trans = cfg.model.obj_opt_t
        opt_obj_rot = cfg.model.obj_opt_r
        opt_obj_scale = cfg.model.obj_opt_s
        opt_obj_shape = cfg.model.obj_opt_shape
        print(f"Optimize rotation? {cfg.model.obj_opt_r}, optimize translation? {cfg.model.obj_opt_t}, scale? {opt_obj_scale}, shape? {opt_obj_shape}, "
              f"lr={cfg.model.obj_opt_lr}, opt occlusion threshold: {cfg.model.obj_opt_occ_thres}, No AE? {cfg.model.obj_opt_noae}")
        if osp.isfile(ckpt_file):
            print(f"Loading from {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            latent = ckpt['latents'].requires_grad_(True)

            obj_rot_axis, obj_scale, obj_trans, optimizer, scheduler, train_state = self.prep_obj_optimization(cfg,
                                                                                                               ckpt,
                                                                                                               device,
                                                                                                               latent)
            init_frame_index = ckpt['init_frame_index']
        else:
            print(ckpt_file, 'does not exist!')
            latent, vc, optimizer, scheduler, obj_trans, obj_rot_axis, obj_scale = None, None, None, None, None, None, None
            train_state = training_utils.TrainState()
        # lw_chamf, lw_mask = 10.0, 0.001
        # lw_chamf, lw_mask, lw_dt = 10.0, 0.001, 0.0001
        lw_chamf = cfg.model.obj_lw_chamf
        lw_mask = cfg.model.obj_lw_mask
        lw_dt = cfg.model.obj_lw_dt
        lw_temp_t = cfg.model.obj_lw_temp_t
        lw_temp_r = cfg.model.obj_lw_temp_r
        lw_temp_s = cfg.model.obj_lw_temp_s
        lw_temp_v = cfg.model.obj_lw_temp_v # temporal smoothness applied to the transformed points
        print(f"Loss weights: chamf={lw_chamf}, mask={lw_mask}, dt={lw_dt}, trans={lw_temp_t}, rot={lw_temp_r}, scale={lw_temp_s}"
              f", opt mode: {cfg.run.sample_mode}")


        if cfg.run.job == 'vis':
            # simply render video and exit
            self.render_video(dataloader, latent, model_ae, output_dir, seq_name, train_state, obj_trans, obj_rot_axis, obj_scale,
                              show_gt=False,
                              show_recon=False, save_pc=True)
            # self.render_video(dataloader, latent, model_ae, output_dir, seq_name, train_state,
            #                   show_gt=cfg.run.sample_mode == 'gt',
            #                   show_recon='recon' in cfg.run.sample_mode, save_pc=True)
            print('visualization done, all done')
            return
        assert cfg.run.sample_mode in ['gt',  'recon-t', 'recon-r', 'recon-rt'], f'invalid sample mode {cfg.run.sample_mode}' # for rotation, translation

        # first load all batches and combine them
        batches_all = {}
        batches = []
        for batch in tqdm(dataloader, desc='Loading batches'):
            batches.append(batch)
            for key, value in batch.items():
                if key not in batches_all:
                    batches_all[key] = []
                if isinstance(value, list):
                    batches_all[key].extend(value)

            # if len(batches) == 2 and DEBUG:
            #     break
        seq_len = len(batches_all['images'])
        num_batches = len(dataloader)
        batch_size = cfg.dataloader.batch_size

        # Load from metadata directly
        hdm_out = cfg.dataset.ho_segm_pred_path
        meta_files = sorted(glob.glob(osp.join(hdm_out.replace('/pred', '/metadata'), seq_name, '*.pth')))
        occ_ratios = {}
        for file in meta_files:
            meta = torch.load(file, weights_only=False)
            frame = osp.splitext(osp.basename(file))[0]
            occ_ratios[frame] = meta['obj_visibility']

        self.occ_ratios = occ_ratios

        # additional loss using pretrained shape AE
        # if cfg.model.obj_lw_ae > 0:
        model_ae = PVCNNAutoEncoder(2048,
                                    6,
                                    [1024,512],
                                    2048,
                                    -1).to(device)
        model_ae.eval()

        # ckpt = torch.load('/BS/xxie-2/work/pc2-diff/experiments/outputs/aligned-all-2k-noise0.1/single/checkpoint-latest.pth')
        # state_dict = ckpt['model']
        # if any(k.startswith('module.') for k in state_dict.keys()):
        #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # model_ae.load_state_dict(state_dict, strict=True)
        self.model = model_ae
        assert torch.allclose(torch.tensor(batches_all['frame_index']), torch.arange(seq_len)), f'the frame order is incorrect: {seq_len}, {batches_all["frame_index"]}'

        # logging
        if cfg.logging.wandb:
            wandb.init(project='opt-obj', name=cfg.run.save_name + f'_{seq_name}', job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
            print("wandb initialized")

        while True:
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_chamf', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_tR', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_tT', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('loss_sil', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_edt', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('loss_mask', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('time_one_iter', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            # if len(batches) == 0:
            #     progress_bar: Iterable[Any] = metric_logger.log_every(dataloader, 10, header=log_header)
            # else:
            #     progress_bar: Iterable[Any] = metric_logger.log_every(batches, 10, header=log_header)
            progress_bar: Iterable[Any] = metric_logger.log_every(range(num_batches*10), 10, header=log_header)
            for i in progress_bar:
                torch.cuda.empty_cache()
                # randomly pick one batch from the sequence
                rid = np.random.randint(0, seq_len - batch_size)
                start_ind, end_ind = rid, min(seq_len, rid + batch_size)
                batch = {}
                for k, v in batches_all.items():
                    if len(v) == seq_len:
                        batch[k] = batches_all[k][start_ind:end_ind]

            # for i, batch in enumerate(progress_bar):
            #     if len(batches) < len(dataloader):
            #         batches.append(batch)
            #         print(f"Adding batch {i} to cache...")
                # obj_pts_recon = torch.stack(batch['pred_obj'],0).cuda()
                if cfg.run.sample_mode == 'gt':
                    obj_pts_recon = torch.stack(batch['pclouds_obj'], 0).cuda() # loaded from behave-attn dataset
                elif cfg.run.sample_mode in ['recon', 'recon-t', 'recon-r', 'recon-rt']:
                    obj_pts_recon = torch.stack(batch['pred_obj'], 0).cuda()
                else:
                    raise NotImplementedError

                # filter out heavy occlusions
                frame_times = [osp.basename(osp.dirname(x)) for x in batch['image_path']]
                occ_ratios_bi = [occ_ratios[k] for k in frame_times]
                vis_mask = torch.tensor(occ_ratios_bi).to('cuda') > cfg.model.obj_opt_occ_thres
                if torch.sum(vis_mask) == 0:
                    print("All frames occluded in this batch, skipped")
                    continue

                # obj_pts_recon = torch.stack(batch['pred_obj'],0).cuda()
                cam_t, obj_poses = self.get_obj_RT(batch, cfg) # based on sample mode, get object rotation and translation
                frame_indices = torch.cat(batch['frame_index']).to('cuda')
                if latent is None:
                    # file = str(batch['image_path'][0])
                    # if 't0003.' not in file:
                    #     continue # not using this to init
                    # Initialize latent code
                    # the dataset is behave-attn, so points are normalized to unit sphere
                    # TODO: select a canonical frame with good visibility
                    init_frame_index = frame_indices[vis_mask][0]
                    obj_pts_recon_init_frame = obj_pts_recon[vis_mask][0]
                    obj_pose_init_frame = obj_poses[vis_mask][0]
                    ind_i = init_frame_index - frame_indices[0]
                    print(f"Using shape from {batch['image_path'][ind_i]} as the canonical object shape.")
                    latent = self.get_cannonical_shape(cfg, model_ae, obj_pose_init_frame,
                                                       obj_pts_recon_init_frame)

                    # latent = latent.requires_grad_(True)
                    # if opt_obj_trans:
                    #     obj_trans = torch.zeros(750, 3).to(device).requires_grad_(True)
                    #     optimizer = optim.Adam([latent, obj_trans], lr=0.0006)
                    # else:
                    #     obj_trans = torch.zeros(750, 3).to(device)  # no grad, no optimization
                    #     optimizer = optim.Adam([latent], lr=0.0006)
                    obj_rot_axis, obj_trans, obj_scale, optimizer = self.init_optimizer(device, latent, opt_obj_rot, opt_obj_trans, opt_obj_scale, opt_obj_shape, cfg)
                    scheduler = get_scheduler(optimizer=optimizer, name='cosine',
                                              num_warmup_steps=cfg.run.max_steps//10,
                                              num_training_steps=int(cfg.run.max_steps * 1.5)) # do not go to zero in the end

                # frame_indices = frame_indices

                # get object poses for this frame
                # Forward: get points, transform to other frames, and then compute chamfer distance loss
                bs = torch.sum(vis_mask)
                obj_pts_recon = obj_pts_recon[vis_mask] # take only partial points, these points have been normalized in scale, but the normalization can be incorrect.
                cam_t = cam_t[vis_mask]
                # obj_poses = obj_poses[vis_mask]
                obj_pts_can = self.decode_can_pts(bs, latent, model_ae)

                obj_trans_bi = obj_trans[frame_indices]
                # print(f'Optimizes translation? {opt_obj_trans}, requires grad?{obj_trans_bi.requires_grad}')
                obj_rot_bi = axis_angle_to_matrix(obj_rot_axis[frame_indices])
                obj_scale_orig = torch.stack(batch['radius_obj_pred'], 0).to(self.device)
                obj_trans_orig = torch.stack(batch['cent_obj_pred'], 0).to(self.device)
                obj_scale_bi = obj_scale[frame_indices] if len(obj_scale) > 1 else obj_scale # per-frame scale

                obj_pts_live, obj_pts_no_trans = self.canonical2live(obj_poses[vis_mask], obj_pts_can, obj_rot_bi[vis_mask],
                                                                     obj_trans_bi[vis_mask],
                                                                     obj_trans_orig[vis_mask],
                                                                     obj_scale_bi[vis_mask],
                                                                     obj_scale_orig[vis_mask], ret_no_trans=True)
                # if i == 0:
                #     # if use raw points, the first example should be removed from computing loss
                #     chamf = chamfer_distance(obj_pts_live[1:], obj_pts_recon[1:]).mean()*lw_chamf
                # else:
                if lw_chamf > 0:
                    chamf = self.compute_chamf_loss(obj_pts_no_trans, obj_pts_recon, obj_pts_live,
                                                    obj_trans_orig[vis_mask], obj_scale_orig[vis_mask]) * lw_chamf # CD should not be applied after adding translation
                else:
                    chamf = torch.tensor(0)

                # for debug
                # if DEBUG:
                #     obj_pts_recon_hoi = obj_pts_recon * obj_scale_orig[:, None] * 2 + obj_trans_orig[:, None]
                if lw_dt >0 or lw_mask >0:
                    kernel_size = 7
                    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
                    masks_obj = torch.stack(batch['masks_obj'], 0).to('cuda')[vis_mask] # B, 2, H, W, human and obj mask
                    fore_mask = masks_obj[:, 1] > 0.5
                    ps_mask = masks_obj[:, 0] > 0.5
                    # mask_merge = ps_mask | fore_mask
                    mask_inv = - ps_mask.float() # other instances, ignore them
                    mask_inv[fore_mask] = 1 # obj + other instances are 1 bkg of ps_mask is zero
                    keep_mask = (mask_inv >= 0).float() # obj + bkg are kept
                    mask_ref = fore_mask.float()
                    # Render the points as mask

                    front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                                    [0, -1., 0],
                                                                    [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                                   T=cam_t, # camera that projects normalized object to object region
                                                   K=torch.stack(batch['K_obj'], 0).to(self.device)[vis_mask],
                                                   in_ndc=True,
                                                   device=self.device)
                    pc = Pointclouds(obj_pts_live, features=torch.tensor([[[1., 1., 1.]]]).repeat(bs, len(obj_pts_live[0]), 1).to(device))
                    images = renderer(pc, cameras=front_cam)
                    silhouette = images[..., :3].mean(-1) # the boolean operation will lead to no gradient
                    # print(silhouette.shape, silhouette.requires_grad, images.requires_grad)
                    # fragments = rasterizer(pc, cameras=front_cam)
                    # silhouette = (torch.mean(fragments.zbuf, -1) > 0).float() # B, H, W, no gradient here!
                    temp = keep_mask * silhouette
                    loss_mask = torch.sum((temp - mask_ref)**2, dim=(1, 2)).mean() * lw_mask

                    # edge loss:
                    edges_ref = pool(fore_mask.float()) - fore_mask.float()
                    edges_dt = [distance_transform_edt(1 - (mask_edge > 0)) ** (0.5) for mask_edge in edges_ref.cpu().numpy()]
                    edges_dt = torch.from_numpy(np.stack(edges_dt)).float().to(device)
                    edges_rend = pool(silhouette) - silhouette
                    # print(edges_rend.shape, edges_dt.shape)
                    loss_edt = torch.sum(edges_rend*edges_dt, dim=(1, 2)).mean() * lw_dt
                else:
                    loss_edt, loss_mask = 0., 0.

                # Temporal smoothness loss
                loss_temp_t, loss_temp_r, loss_temp_s, loss_temp_v = 0., 0., 0., 0.
                if opt_obj_trans and lw_temp_t >0:
                    loss_temp_t = self.compute_loss_temp_t(cam_t, obj_trans_bi, obj_trans_orig) * lw_temp_t
                    # this velocity term is also not helpful
                    # loss_temp_t = loss_temp_t + F.mse_loss(trans_comb[1:], trans_comb[:-1], reduction='none').sum(-1).mean()*lw_temp_t*0.5 # prevent strange jittering.
                    # print(loss_temp_t.requires_grad) # True
                if opt_obj_rot and lw_temp_r > 0:
                    rot_comb = torch.matmul(obj_rot_bi, obj_poses[:, :3, :3])
                    velo1, velo2 = self.compute_velocity(rot_comb)
                    # loss_temp_r = F.mse_loss(velo1, velo2, reduction='none').sum(dim=(-1, -2)).mean() * lw_temp_r

                    # compute error on points
                    # verts = torch.matmul(obj_pts_live, obj_rot_bi.transpose(1, 2))
                    loss_temp_r = F.mse_loss(velo1, velo2, reduction='mean') * lw_temp_r
                if opt_obj_scale and lw_temp_s > 0 and len(obj_scale) > 1:
                    # scale loss
                    scale_comb = obj_scale_orig * obj_scale_bi # B, 1
                    velo1, velo2 = self.compute_velocity(scale_comb)
                    loss_temp_s = F.mse_loss(velo1, velo2, reduction='mean') * lw_temp_s

                if lw_temp_v > 0:
                    with torch.no_grad():
                        # full mini batch
                        obj_pts_can_bi = self.decode_can_pts(len(obj_poses), latent, model_ae)
                    obj_pts_live_bi = self.canonical2live(obj_poses, obj_pts_can_bi,
                                                                     obj_rot_bi,
                                                                     obj_trans_bi,
                                                                     obj_trans_orig,
                                                                     obj_scale_bi,
                                                                     obj_scale_orig, ret_no_trans=False)
                    # velocity loss + acceleration
                    velo1, velo2 = self.compute_velocity(obj_pts_live_bi)
                    loss_temp_v1 = (velo1 **2).sum(-1).mean() * lw_temp_v
                    loss_temp_v2 = F.mse_loss(velo1, velo2, reduction='none').sum(-1).mean() * lw_temp_v
                    loss_temp_v = loss_temp_v2 + loss_temp_v1 # roughly equal value
                    # print(f"Velocity: {loss_temp_v1:.4f}, acc: {loss_temp_v2:.4f}, {loss_temp_v.requires_grad_()}")

                # AE shape regularization
                loss_ae = 0.
                if cfg.model.obj_lw_ae > 0 and train_state.step > 6000:
                    with torch.no_grad(): # this is not helpful!
                        obj_pts_aeout = model_ae(obj_pts_can[0:1]*2.)/2.0 # the AE input shape is [-1, 1]
                    loss_ae = self.compute_chamf_loss(obj_pts_can[0:1], obj_pts_aeout) * cfg.model.obj_lw_ae

                # print(silhouette.requires_grad, images.requires_grad)
                if DEBUG:
                    # visualize the mask
                    vis_ind = 0
                    sil = (silhouette[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)  # rendered image
                    img_ref = (mask_ref[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)
                    tmp = (temp[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)
                    h, w = sil.shape[:2]
                    vis1, vis2 = np.zeros((h, w, 3)), np.zeros((h, w, 3))
                    vis3, vis4 = np.zeros((h, w, 3)), np.zeros((h, w, 3))
                    # print(batch['images_obj'][vis_ind].min(), batch['images_obj'][vis_ind].max())
                    rgb = (batch['images_obj'][vis_ind].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                    vis3[:, :, 2] = tmp # rendered results
                    vis4[:, :, 2] = img_ref # reference mask
                    # yellow == red + green
                    # vis1[:, :, 0] = img_ref  # blue: the ref mask
                    # vis1[:, :, 2] = sil # red: the rendered points
                    # vis1[:, :, 1] = tmp # green: the one used to compute the loss, it should match blue
                    vis1[:, :, 0] = (edges_dt[vis_ind].cpu().numpy()*255).astype(np.uint8)  # black: the ref edge
                    vis1[:, :, 2] = (edges_rend[vis_ind].detach().cpu().numpy()*255).astype(np.uint8)
                    vis2[:, :, 0] = (keep_mask[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)  # blue: keep mask
                    vis2[:, :, 2] = (ps_mask[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8) # red: ps mask
                    vis2[:, :, 1] = (fore_mask[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8) # green: obj mask
                    # from left to right:
                    comb = np.concatenate([vis1, vis2, vis3, vis4, rgb], 1)
                    cv2.imshow(str(output_dir), comb)
                    cv2.waitKey(10)
                    cv2.moveWindow(str(output_dir), 600, 50)

                loss = chamf + loss_mask + loss_edt + loss_temp_r + loss_temp_t + loss_temp_s + loss_temp_v + loss_ae

                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'loss_chamf': chamf.item(),
                    'loss_mask': loss_mask,
                    "loss_edt": loss_edt,
                    "loss_tT": loss_temp_t,
                    'loss_tR': loss_temp_r,
                    'loss_ts': loss_temp_s,
                    'loss_tv': loss_temp_v,
                    'loss_ae': loss_ae
                }
                metric_logger.update(**log_dict)
                if cfg.logging.wandb:
                    wandb.log(log_dict, step=train_state.step)

                if torch.isnan(loss).any():
                    print("Found NAN in loss, stop training.")
                    return
                loss.backward()
                # print('nan in gradient?', torch.any(torch.isnan(latent.grad)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # save checkpoint
                if train_state.step % 250 == 0:
                    # image_path = str(batch['image_path'][0])
                    # ss = image_path.split(os.sep)
                    self.save_checkpoint_obj(cfg, latent, optimizer, output_dir, seq_name, train_state, scheduler,
                                             obj_trans, obj_rot_axis, obj_scale, batches_all, init_frame_index)

                # visualize
                if train_state.step % 2500 == 0: # and train_state.step > 0:
                    # train_state.step += 1
                    # continue
                    self.render_video(batches, latent, model_ae, output_dir, seq_name, train_state, obj_trans, obj_rot_axis, obj_scale,
                                      show_gt=cfg.run.sample_mode=='gt',
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True)


                if train_state.step >= cfg.run.max_steps or (optimizer.param_groups[0]['lr'] < 1e-8 and train_state.step > 2000):
                    import datetime
                    print(f'Ending training at: {datetime.datetime.now()}, lr={optimizer.param_groups[0]["lr"]}')
                    print(f'Final train state: {train_state}')
                    # wandb.finish()
                    # time.sleep(5)
                    self.save_checkpoint_obj(cfg, latent, optimizer, output_dir, seq_name, train_state, scheduler,
                                             obj_trans, obj_rot_axis, obj_scale, batches_all, init_frame_index)

                    # visualize before finish
                    batches = dataloader if len(batches) <=1 else batches # make sure we visualize full sequence
                    self.render_video(batches, latent, model_ae, output_dir, seq_name, train_state, obj_trans, obj_rot_axis, obj_scale,
                                      show_gt=False, show_recon=False, save_pc=False)
                    self.render_video(batches, latent, model_ae, output_dir, seq_name, train_state, obj_trans, obj_rot_axis, obj_scale,
                                      show_gt=cfg.run.sample_mode=='gt',
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True)
                    if cfg.logging.wandb:
                        wandb.finish()
                        time.sleep(10)
                    return
                train_state.step += 1
            train_state.epoch += 1

    def extract_seq_name(self, cfg):
        "extract the sequence name for this optimization"
        split_file = str(cfg.dataset.split_file)
        seq_name = pkl.load(open(split_file, 'rb'))['test'][0].split('/')[-3]  # format: root/seq_name/frame/k1.color.jpg
        return seq_name

    def get_cmap(self, vertices):
        min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
        cmap = (vertices - min_coord) / (max_coord - min_coord)
        return cmap

    def compute_chamf_loss(self, obj_pts_no_trans, obj_pts_recon, obj_pts_live=None,
                           obj_trans_orig=None, obj_scale_orig=None):
        return chamfer_distance(obj_pts_no_trans, obj_pts_recon).mean()

    def get_cannonical_shape(self, cfg, model_ae, obj_pose_init_frame, obj_pts_recon_init_frame):
        """
        get the object points/latent code in canonical space
        obj_pose_init_frame: predicted rotation for this frame, from canonical to camera space
        """
        pts_can = torch.matmul(obj_pts_recon_init_frame, obj_pose_init_frame[:3, :3])  # compute canonical object
        if self.no_ae:
            # optimize from raw pc, the latent is simply the raw pc
            latent = pts_can[None].clone()
            # add some small perturbation
            latent = latent + cfg.model.obj_opt_noise * torch.randn_like(latent)  # 0.2 already makes it almost Gaussian
            print("Noise added to the points.")
        else:
            # latent = model_ae.encode(pts_can[None]).detach()
            latent = model_ae.encode(pts_can[None] * 2).detach()  # need to scale up, as AE input is [-1, 1]
            latent = latent + 0.02 * torch.randn_like(latent)
        return latent

    def prep_2dlosses(self, fore_mask, ps_mask):
        """
        prepare images used for occlusion aware mask losses
        Parameters
        ----------
        fore_mask : the fore-ground object mask
        ps_mask : the interacting object (that can potentially occlude fore-ground object)

        Returns
        -------

        """
        mask_inv = - ps_mask.float()  # other instances, ignore them
        mask_inv[fore_mask] = 1  # obj + other instances are 1 bkg of ps_mask is zero
        keep_mask = (mask_inv >= 0).float()  # obj + bkg are kept
        mask_ref = fore_mask.float()
        kernel_size = 7
        pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
        edges_ref = pool(fore_mask.float()) - fore_mask.float()
        edges_dt = [distance_transform_edt(1 - (mask_edge > 0)) ** (0.5) for mask_edge in edges_ref.cpu().numpy()]
        edges_dt = torch.from_numpy(np.stack(edges_dt)).float()
        return edges_dt, keep_mask, mask_ref

    def load_occ_ratios(self, seq_name):
        pack_file = f'/scratch/inf0/user/xxie/behave-packed/{seq_name}_GT-packed.pkl'
        if osp.isfile(pack_file):
            pack_data = joblib.load(pack_file)
            occ_ratios = {k: v for k, v in zip(pack_data['frames'], pack_data['occ_ratios'][:, 1])}
        else:
            pack_data = pkl.load(open(f'/BS/xxie-2/work/pc2-diff/experiments/results/images/{seq_name}_proj_vis_reso512.pkl', 'rb'))
            occ_ratios = {k: v for k, v in zip(pack_data['frames'], pack_data['vis_hdm'])}

        return occ_ratios

    def prep_obj_optimization(self, cfg, ckpt, device, latent):
        opt_obj_trans = cfg.model.obj_opt_t
        opt_obj_rot = cfg.model.obj_opt_r
        opt_obj_scale = cfg.model.obj_opt_s
        opt_obj_shape = cfg.model.obj_opt_shape
        obj_rot_axis, obj_trans, obj_scale, optimizer = self.init_optimizer(device, latent, opt_obj_rot, opt_obj_trans,
                                                                            opt_obj_scale, opt_obj_shape, cfg)
        if 'obj_trans' in ckpt:
            obj_trans.data = ckpt['obj_trans'].data
        if 'obj_rot_axis' in ckpt:
            obj_rot_axis.data = ckpt['obj_rot_axis'].data
        if 'obj_scale' in ckpt:
            obj_scale.data = ckpt['obj_scale'].data
        if cfg.model.obj_opt_shape:
            # loading optimizer state
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                # import pdb;pdb.set_trace();
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"optimizer state loading failed due to {e}, skipped")
        else:
            print('Warning: Not loading optimizer state dict')
        train_state = ckpt['train_state'] if 'train_state' in ckpt else training_utils.TrainState()
        scheduler = get_scheduler(optimizer=optimizer, name='cosine',
                                  num_warmup_steps=cfg.run.max_steps//10,
                                  num_training_steps=int(cfg.run.max_steps * 1.25))
        scheduler.load_state_dict(ckpt['scheduler'])
        return obj_rot_axis, obj_scale, obj_trans, optimizer, scheduler, train_state

    def init_renderer(self, device, rend_size):
        self.device = device
        raster_settings = PointsRasterizationSettings(
            image_size=rend_size,
            # radius=0.003,
            radius=0.015,
            points_per_pixel=5,
            # bin_size=bin_size,
            max_points_per_bin=16000
        )
        rasterizer = PointsRasterizer(raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color
            # Set bkg to black
            compositor=AlphaCompositor(background_color=(0, 0, 0))
        )
        return renderer

    def compute_velocity(self, rot_comb):
        """
        compute velocity for batch tensors
        Parameters
        ----------
        rot_comb : (T, ...)

        Returns two velocity vectors, one forward, and one backward
        -------

        """
        velo1 = rot_comb[1:-1] - rot_comb[:-2]
        velo2 = rot_comb[2:] - rot_comb[1:-1]
        return velo1, velo2

    def compute_loss_temp_t(self, cam_t, obj_trans_bi, obj_trans_orig):
        trans_comb = obj_trans_bi * torch.tensor([[-1, -1., 1]]).to(cam_t.device) + cam_t  # the final translation applied to the object
        velo1, velo2 = self.compute_velocity(trans_comb)
        # the original translation jitters a lot, it makes no sense to ensure the optimized t is all zero
        loss_temp_t = F.mse_loss(velo1, velo2, reduction='none').sum(-1).mean()   # + (obj_trans_bi**2).sum(-1).mean()*lw_temp_t # prevent deviation from original translation
        return loss_temp_t

    def init_optimizer(self, device, latent, opt_obj_rot, opt_obj_trans, opt_obj_scale, opt_obj_shape, cfg):
        if opt_obj_shape:
            latent = latent.requires_grad_(True)
            params = [latent]
        else:
            params = []

        num_frames = 2500
        if opt_obj_trans:
            obj_trans = torch.zeros(num_frames, 3).to(device).requires_grad_(True)
            params.append(obj_trans)
        else:
            obj_trans = torch.zeros(num_frames, 3).to(device)

        if opt_obj_rot:
            obj_rot_axis = torch.zeros(num_frames, 3).to(device).requires_grad_(True)
            params.append(obj_rot_axis)
        else:
            obj_rot_axis = torch.zeros(num_frames, 3).to(device)
        obj_scale = self.init_obj_scale(device, opt_obj_scale)
        if opt_obj_scale:
            params.append(obj_scale)
        assert len(params) >=1, 'must select one parameter to optimize!'
        optimizer = optim.Adam(params, lr=cfg.model.obj_opt_lr)
        # if opt_obj_trans and opt_obj_rot:
        #     obj_trans = torch.zeros(750, 3).to(device).requires_grad_(True)
        #     obj_rot_axis = torch.zeros(750, 3).to(device).requires_grad_(True)
        #     params = [latent, obj_trans, obj_rot_axis]
        #     optimizer = optim.Adam(params, lr=0.0006)
        # elif opt_obj_trans and not opt_obj_rot:
        #     obj_trans = torch.zeros(750, 3).to(device).requires_grad_(True)
        #     obj_rot_axis = torch.zeros(750, 3).to(device)
        #     optimizer = optim.Adam([latent, obj_trans], lr=0.0006)
        # elif opt_obj_rot and not opt_obj_trans:
        #     # only optimize rotation
        #     obj_rot_axis = torch.zeros(750, 3).to(device).requires_grad_(True)
        #     optimizer = optim.Adam([latent, obj_rot_axis], lr=0.0006)
        #     obj_trans = torch.zeros(750, 3).to(device)
        # else:
        #     obj_trans = torch.zeros(750, 3).to(device)  # no grad, no optimization
        #     obj_rot_axis = torch.zeros(750, 3).to(device)
        #     optimizer = optim.Adam([latent], lr=0.0006)
        return obj_rot_axis, obj_trans, obj_scale, optimizer

    def init_obj_scale(self, device, opt_obj_scale):
        if opt_obj_scale:
            obj_scale = torch.ones(1, ).to(device).requires_grad_(True)  # it should not be a single value, but also per-frame!
        else:
            obj_scale = torch.ones(1, ).to(device)
        return obj_scale

    def canonical2live(self, obj_poses, obj_pts_can, obj_rot_bi, obj_trans_bi, obj_trans_orig,
                       obj_scale, obj_scale_orig, ret_no_trans=False):
        obj_pts_live = torch.matmul(obj_pts_can*obj_scale,
                                    obj_poses[:, :3, :3].transpose(1, 2))  # canonical points in current frame
        obj_pts_live_no_trans = torch.matmul(obj_pts_live, obj_rot_bi.transpose(1, 2))
        obj_pts_live = obj_pts_live_no_trans + obj_trans_bi[:, None]  # a relative global translation, initialized as zero
        if ret_no_trans:
            return obj_pts_live, obj_pts_live_no_trans
        return obj_pts_live

    def decode_can_pts(self, bs, latent, model_ae):
        """
        decode object points in canonical space
        Parameters
        ----------
        bs :
        latent :
        model_ae :

        Returns (bs, N, 3)
        -------

        """
        if self.no_ae:
            obj_pts_can = latent.repeat(bs, 1, 1)
        else:
            obj_pts_can = model_ae.decode(latent).repeat(bs, 1, 1)/2.  # B, N, 3, need to normalize to [-0.5, 0.5]
        # force center
        obj_pts_can = obj_pts_can - torch.mean(obj_pts_can, 1, keepdim=True)
        return obj_pts_can

    def get_obj_RT(self, batch, cfg):
        """
        return the camera translation to render, and predicted object rotation
        Parameters
        ----------
        batch :
        cfg :

        Returns
        -------

        """
        if cfg.run.sample_mode in ['gt', 'recon-t']:
            obj_poses = torch.stack(batch['abs_pose'], 0).cuda()  # (B, 4, 4)
            # print('Using GT rotation')
        else:
            # print('Using predicted rotation')
            obj_poses = torch.stack(batch['rot_pred'], 0).cuda()
        if cfg.run.sample_mode in ['recon-t', 'recon-rt']:
            # use predicted translation
            # print("Using predicted translation")
            # translation of the object in camera space (normalized object only)
            cam_t = torch.stack(batch['T_obj_scaled'], 0).to(self.device)
        else:
            # print("Using GT translation")
            cam_t = torch.stack(batch['T_obj'], 0).to(self.device)
        return cam_t, obj_poses

    @torch.no_grad()
    def render_video(self, batches, latent, model_ae, output_dir, seq_name, train_state, obj_trans, obj_rot_axis, obj_scale,
                     show_gt=True, show_recon=False, save_pc=False):
        import imageio
        # seq_name = str(batch['image_path'][0]).split(os.sep)[-3]
        video_file = output_dir / 'vis' / seq_name / f'step{train_state.step:06d}_gt-{show_gt}_recon-{show_recon}.mp4'
        os.makedirs(osp.dirname(video_file), exist_ok=True)
        rend_size = 224
        renderer = PcloudRenderer(image_size=rend_size, radius=0.0075)
        video_writer = imageio.get_writer(video_file, format='FFMPEG', mode='I', fps=15)
        with torch.no_grad():
            obj_pts_can = self.decode_can_pts(1, latent, model_ae) # force center happens inside

            # filter out points
            obj_pts_filter, _ = filter_points_knn(obj_pts_can[0].cpu().numpy())
            obj_pts_can = torch.from_numpy(obj_pts_filter[None]).float().to(self.device)

            # obj_pts_can = model_ae.decode(latent)
            vc_points = self.get_cmap(obj_pts_can.detach().cpu().numpy()[0])
        for i, batch in enumerate(tqdm(batches)):
            obj_pts_gt = torch.stack(batch['pclouds_obj'], 0).cuda()
            bs, NS = obj_pts_gt.shape[:2]
            # obj_poses = torch.stack(batch['abs_pose'], 0).cuda()  # (B, 4, 4)
            cam_t, obj_poses = self.get_obj_RT(batch, self.cfg)
            frame_indices = torch.cat(batch['frame_index']).to('cuda')
            obj_trans_bi = obj_trans[frame_indices] # optimized rot + translation
            obj_rot_bi = axis_angle_to_matrix(obj_rot_axis[frame_indices])

            vis_mask = frame_indices > -1 # all frames
            obj_scale_orig = torch.stack(batch['radius_obj_pred'], 0).to(self.device)[vis_mask]
            obj_trans_orig = torch.stack(batch['cent_obj_pred'], 0).to(self.device)[vis_mask]
            obj_scale_bi = obj_scale[frame_indices] if len(obj_scale) > 1 else obj_scale  # per-frame scale
            obj_pts_live, obj_pts_live_no_trans = self.canonical2live(obj_poses, obj_pts_can, obj_rot_bi, obj_trans_bi,
                                                                 obj_trans_orig,
                                                                 obj_scale_bi, obj_scale_orig, ret_no_trans=True)

            feats, pts = self.get_opt_pts_for_vis(bs, obj_pts_live_no_trans, obj_pts_live)
            if show_gt:
                pts = torch.cat([pts, obj_pts_gt], 1)
                feats = torch.cat([feats, torch.tensor([[[0.0, 1., 0]]]).repeat(bs, NS, 1)], 1)
            if show_recon:
                obj_pts_recon = self.get_recon_pts_for_vis(batch)
                pts = torch.cat([pts, obj_pts_recon], 1)
                feats = torch.cat([feats, torch.tensor([[[1.0, 0.0, 0]]]).repeat(bs, len(obj_pts_recon[0]), 1)], 1)
            pc = Pointclouds(pts, features=feats.to(self.device))

            front_cam = self.get_front_camera(batch, bs, cam_t)
            # side view camera: fixed at same place
            at = torch.zeros(bs, 3).to(self.device)
            R, T = look_at_view_transform(2.5, 0, 80, up=((0, -1, 0),),
                                          at=at, device=self.device)
            side_camera = PerspectiveCameras(image_size=((rend_size, rend_size),),
                                             device=self.device,
                                             R=R, T=T,
                                             focal_length=rend_size * 1.5,
                                             principal_point=torch.tensor(
                                                 ((rend_size / 2., rend_size / 2.))).repeat(bs, 1).to(
                                                 self.device),
                                             in_ndc=False)
            rends, masks = [torch.stack(batch['images_fullcrop']).cpu().permute(0, 2, 3, 1).numpy()], []
            for ii, cam in enumerate([front_cam, side_camera]):
                rend, mask = renderer.render(pc, cam, mode='mask')
                rends.append(rend)
                masks.append(mask)
            rend = np.concatenate(rends, 2)  # (B, H, W*3, 3)
            # add to video one by one
            for ii, img in enumerate(rend):
                comb = (img.copy() * 255).astype(np.uint8)
                # img_idx = int(batch['file_index'][ii])
                # show overlap
                img = comb
                h, w3 = img.shape[:2]
                rgb, front = img[:, :w3//3], img[:, w3//3:2*w3//3]
                mask = (front[:, :, 0] > 0) | (front[:, :, 1] > 0)
                overlap = rgb.copy()
                overlap[mask] = front[mask]
                img[:, w3 // 3:2 * w3 // 3] = overlap
                comb = img

                file = str(batch['image_path'][ii])
                ss = file.split(os.sep)
                # also add occlusion ratios
                cv2.putText(comb, f'{ss[-3]}/{ss[-2]} occ{self.occ_ratios[ss[-2]]:.2f}', (20, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.2,
                            (255, 0, 0), 2)  # color (0, 255, 255)=bright blue, same as human color
                video_writer.append_data(comb)

            # Epoch complete, log it and continue training

            # Save results
            if save_pc:
                for ii in range(bs):
                    pc_i = obj_pts_live[ii].cpu().numpy() # TODO: save without adding this translation.
                    # pc_i = obj_pts_live_no_trans[ii].cpu().numpy()
                    file = str(batch['image_path'][ii])
                    ss = file.split(os.sep)
                    outfile = output_dir / 'pred' / ss[-3] /f'{ss[-2]}.ply'
                    os.makedirs(osp.dirname(outfile), exist_ok=True)
                    trimesh.PointCloud(pc_i, np.array([[0, 1., 0.]]).repeat(len(pc_i), 0)).export(outfile)

                # add synlink
                # self.add_synlink(output_dir) # TODO: update this synlink link back to HDM results
        print("Visualization video saved to", video_file)
        if save_pc:
            print(f"pc saved to {output_dir.absolute()}, all done.")
        return seq_name

    def add_synlink(self, output_dir):
        for pat in ['images', 'metadata', 'gt']:
            syn_file = output_dir / pat
            if not osp.islink(str(syn_file)):
                cmd = f'ln -s /BS/xxie-2/work/pc2-diff/experiments/outputs/sround3_segm-1m/single/15fps-ddim/{pat} {str(syn_file)}'
                os.system(cmd)

    def get_front_camera(self, batch, bs, cam_t):
        front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                        [0, -1., 0],
                                                        [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                       T=cam_t,
                                       K=torch.stack(batch['K_obj'], 0).to(self.device),
                                       # visualize with full image parameters
                                       in_ndc=True,
                                       device=self.device)
        return front_cam

    def get_opt_pts_for_vis(self, bs, obj_pts_no_trans, obj_pts_live=None):
        pts, feats = obj_pts_no_trans, torch.tensor([[[0.0, 1., 1.0]]]).repeat(bs, obj_pts_live.shape[1], 1)
        return feats, pts

    def get_recon_pts_for_vis(self, batch):
        obj_pts_recon = torch.stack(batch['pred_obj'], 0).cuda()
        return obj_pts_recon

    def save_checkpoint_obj(self, cfg, latent, optimizer, output_dir, ss, train_state, scheduler, obj_trans,
                            obj_rot_axis, obj_scale, batch, init_frame_index=None):
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        obj_pts_can = self.decode_can_pts(1, latent, self.model)
        ckpt_dict = {
            "latents": latent,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": train_state.epoch,
            "step": train_state.step,
            "cfg": cfg,
            'train_state': train_state,
            'obj_trans': obj_trans,
            "obj_rot_axis": obj_rot_axis,
            'obj_scale': obj_scale,
            'init_frame_index': init_frame_index, # the frame used to get canonical object shape
            'rot_pred': torch.stack(batch['rot_pred'], 0), # network predicted object pose

            'frame_indices': torch.cat(batch['frame_index']),
            'cent_obj_pred': torch.stack(batch['cent_obj_pred']),
            'radius_obj_pred': torch.stack(batch['radius_obj_pred']),

            # canonical object point
            'obj_pts_canonical': obj_pts_can.detach().cpu(),
            'image_path': batch['image_path']
        }
        # checkpoint_path = 'latents-latest.pth'
        ckpt_file = output_dir / f'checkpoint-{ss}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        os.makedirs(osp.dirname(ckpt_file), exist_ok=True)
        torch.save(ckpt_dict, ckpt_file)
        # torch.save(ckpt_dict, osp.join(self.exp_path, f'latents-step{train_state.step:06d}.pth'))
        print('checkpoint saved to', ckpt_file)


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    # to prevent random shuffle
    cfg.run.job = 'sample' if cfg.run.job == 'train' else cfg.run.job
    cfg.run.freeze_feature_model = False
    cfg.dataloader.num_workers = 2  # reduce initialization waiting time
    cfg.dataset.load_obj_pose = False # do not load GT pose!
    wandb_cache = cfg.logging.wandb
    trainer = TrainerObjOpt(cfg)
    cfg.logging.wandb = wandb_cache
    print("Using wandb?", cfg.logging.wandb)
    trainer.run_sample(cfg)


if __name__ == '__main__':
    main()



