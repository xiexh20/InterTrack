"""
optimize human + object together, human is represented as the latent of AE
"""

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
from trainer import Trainer
import training_utils
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os.path as osp
from torchvision.transforms import functional as TVF
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from pytorch3d.ops import knn_points
from utils.losses import chamfer_distance

from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras
from render.pyt3d_wrapper import MeshRendererWrapper, get_kinect_camera, PcloudRenderer
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Pointclouds

from model.pvcnn.pvcnn_enc import PVCNNAutoEncoder
from main_objopt_pts import TrainerObjOptPoints
from main_objopt_latent import filter_points_knn


class TrainerHOIOpt(TrainerObjOptPoints):
    def sample(self, cfg: ProjectConfig,
               model: PVCNNAutoEncoder,
               dataloader: Iterable,
               accelerator: Accelerator,
               output_dir: str = 'sample', ):
        "given optimized object shape and all human in correspondence, optimize global configuration in the interaction space"
        # Visulization
        rend_size, device = cfg.model.image_size, 'cuda'
        renderer = self.init_renderer(device, rend_size)
        import socket;
        self.DEBUG, self.no_ae = 'volta' in socket.gethostname(), True
        output_dir: Path = Path(output_dir)

        # human AE
        human_ae: PVCNNAutoEncoder = model
        human_ae.eval()

        # try to load latents, parameters from ckpt
        split_file = str(cfg.dataset.split_file)
        seq_name = osp.splitext(osp.basename(split_file))[0].split('-')[1]
        # try to load from ckpt
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        opt_obj_trans = cfg.model.obj_opt_t
        opt_obj_rot = cfg.model.obj_opt_r
        opt_obj_scale = cfg.model.obj_opt_s
        opt_obj_shape = cfg.model.obj_opt_shape
        print(
            f"Object: Optimize rotation? {cfg.model.obj_opt_r}, optimize translation? {cfg.model.obj_opt_t}, scale? {opt_obj_scale}, shape? {opt_obj_shape}, "
            f"lr={cfg.model.obj_opt_lr}, opt occlusion threshold: {cfg.model.obj_opt_occ_thres}")

        if osp.isfile(ckpt_file):
            print(f"Loading from {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)

            # object optimizer
            latent_obj = ckpt['latents']
            obj_rot_axis, obj_scale, obj_trans, optimizer_obj, scheduler_obj, train_state = self.prep_obj_optimization(
                cfg,
                ckpt,
                device,
                latent_obj)
            # human optimizer
            latent_hum = ckpt['latents_hum'].detach()
            hum_scale = ckpt['hum_scale'].detach()
            hum_trans = ckpt['hum_trans'].detach()
            # latent_hum.data = ckpt['latents_hum']
            # optimizer.load_state_dict(ckpt['optimizer_hum'])
            # scheduler.load_state_dict(ckpt['scheduler_hum'])
        else:
            print(ckpt_file, 'does not exist!')
            ckpt_file_obj = osp.join(cfg.dataset.hoi_opt_obj_shape_path,
                                     f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth')
            ckpt_obj = torch.load(ckpt_file_obj, map_location=self.device)
            latent_obj = ckpt_obj['latents']

            # filter object points
            assert self.no_ae
            # filter out outlier points
            obj_pts_filter, _ = filter_points_knn(latent_obj[0].detach().cpu().numpy())
            latent_obj = torch.from_numpy(obj_pts_filter[None]).float().to(self.device)

            obj_rot_axis, obj_trans, obj_scale, optimizer_obj = self.init_optimizer(device, latent_obj, opt_obj_rot,
                                                                                    opt_obj_trans,
                                                                                    opt_obj_scale, opt_obj_shape, cfg)
            obj_trans.data = ckpt_obj['obj_trans'].data
            obj_rot_axis.data = ckpt_obj['obj_rot_axis'].data
            obj_scale.data = ckpt_obj['obj_scale'].data
            scheduler_obj = get_scheduler(optimizer=optimizer_obj, name='cosine',
                                          num_warmup_steps=100,
                                          num_training_steps=cfg.run.max_steps)

            train_state = training_utils.TrainState()

            # load human state from ckpt
            ckpt_file_hum = osp.join(cfg.dataset.hoi_opt_hum_shape_path,
                                     f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth')
            ckpt_hum = torch.load(ckpt_file_hum, map_location=self.device)
            latent_hum = ckpt_hum['latents_hum'].detach()  # disable all gradients
            hum_scale = ckpt_hum['hum_scale'].detach()
            hum_trans = ckpt_hum['hum_trans'].detach()

        lw_chamf_obj = cfg.model.obj_lw_chamf
        lw_mask = cfg.model.obj_lw_mask
        lw_dt = cfg.model.obj_lw_dt
        lw_temp_t = cfg.model.obj_lw_temp_t
        lw_temp_r = cfg.model.obj_lw_temp_r
        lw_temp_s = cfg.model.obj_lw_temp_s
        lw_hoi_cont = cfg.model.hoi_lw_cont
        lw_chamf_hum = cfg.model.hoi_lw_cd_h
        lw_temp_v = cfg.model.obj_lw_temp_v  # temporal smoothness applied to the transformed points
        print(
            f"Loss weights: chamf={lw_chamf_obj}, mask={lw_mask}, dt={lw_dt}, trans={lw_temp_t}, rot={lw_temp_r}, scale={lw_temp_s}"
            f", opt mode: {cfg.run.sample_mode}")

        # First load all batches and combine them
        batches, batches_all, latent_hum_batches = self.load_all_batches(cfg, dataloader, human_ae, latent_hum,
                                                                         latent_obj, obj_rot_axis)

        # Prepare human optimizer
        if latent_hum is None:
            latent_hum = torch.cat(latent_hum_batches, 0)  # N, D
        opt_params = []
        if cfg.model.hum_opt_lat:
            latent_hum = latent_hum.requires_grad_(True)
            opt_params.append(latent_hum)
        if cfg.model.hum_opt_t:
            hum_trans = hum_trans.requires_grad_(True)
            opt_params.append(hum_trans)
        if cfg.model.hum_opt_s:
            hum_scale = hum_scale.requires_grad_(True)
            opt_params.append(hum_scale)
        optimizer_hum = optim.Adam(opt_params, lr=0.001)
        scheduler_hum = get_scheduler(optimizer=optimizer_hum, name='cosine',
                                      num_warmup_steps=100,
                                      num_training_steps=cfg.run.max_steps)
        print(
            f"Human: optimize latent? {cfg.model.hum_opt_lat} translation? {cfg.model.hum_opt_t} scale? {cfg.model.hum_opt_s}")

        if osp.isfile(ckpt_file):
            scheduler_hum.load_state_dict(ckpt['scheduler_hum'])
            optimizer_hum.load_state_dict(ckpt['optimizer_hum'])
        hdm_out = cfg.dataset.ho_segm_pred_path
        occ_ratios = self.load_occ_ratios(hdm_out, seq_name)
        self.occ_ratios = occ_ratios

        # logging
        if cfg.logging.wandb:
            wandb.init(project='opt-hoi', name=str(output_dir) + f'_{seq_name}', job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
        seq_len = len(batches_all['images'])
        num_batches = len(batches)
        batch_size = cfg.dataloader.batch_size
        data_dict = {'hum_trans': hum_trans,
                     'hum_scale': hum_scale,
                     }
        while True:
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            # metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('loss_chamf_h', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_tR', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_tT', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('loss_sil', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_edt', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_mask', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('time_one_iter', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            progress_bar: Iterable[Any] = metric_logger.log_every(range(num_batches * 10), 10, header=log_header)
            for i in progress_bar:
                batch = self.get_random_batch(batch_size, batches_all, seq_len)
                weight_vis = self.get_vis_weights(batch, occ_ratios)

                # Object losses
                chamf_obj, loss_edt, loss_mask, loss_temp_r, loss_temp_s, loss_temp_t, obj_pts_live = self.compute_object_losses(
                    cfg,
                    batch, latent_obj, obj_rot_axis, obj_scale, obj_trans, renderer, weight_vis)

                frame_indices = torch.cat(batch['frame_index']).to('cuda')
                bs = len(frame_indices)

                # Human losses
                # Chamfer
                cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
                radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')
                latent_hum_bi = latent_hum[frame_indices]
                hum_pts_aeout = human_ae.decode(latent_hum_bi)
                hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()
                hum_pts_aeout_hoi = hum_pts_aeout * 2 * radius_hum[:, None] * hum_scale[frame_indices, None] + cent_hum[
                                                                                                               :,
                                                                                                               None] + \
                                    hum_trans[frame_indices, None]
                hum_pts_recon_hoi = hum_pts_recon * 2 * radius_hum[:, None] + cent_hum[:, None]
                cd_hum = chamfer_distance(hum_pts_aeout_hoi, hum_pts_recon_hoi).mean() * cfg.model.hum_lw_cd

                # TODO: decide if we want latent code loss or not.
                lat_orig = torch.stack(batch['latent'], 0).to(self.device)
                loss_lat = F.mse_loss(latent_hum_bi, lat_orig) * cfg.model.hum_lw_lat

                # Contact losses: all transform to interaction space
                loss_cont = self.compute_contact_loss(batch, cfg, hum_pts_aeout_hoi, obj_pts_live, weight_vis)

                # Human temporal loss
                velo1, velo2 = self.compute_velocity(hum_pts_aeout_hoi)
                lw_temp_h = cfg.model.hoi_lw_temp_h
                loss_temp_h = ((velo1 - velo2) ** 2).sum(-1).mean() * lw_temp_h + (velo1 ** 2).sum(
                    -1).mean() * 0.5 * lw_temp_h

                loss = chamf_obj + cd_hum + loss_mask + loss_edt + loss_temp_r + loss_temp_t + loss_temp_s + loss_cont + loss_temp_h

                log_dict = {
                    'lr_h': optimizer_hum.param_groups[0]["lr"],
                    'lr_o': optimizer_obj.param_groups[0]["lr"],
                    'step': train_state.step,
                    'loss_cd_h': cd_hum.item(),
                    'loss_cd_o': chamf_obj.item(),
                    'loss_mask': loss_mask.item(),
                    "loss_edt": loss_edt.item(),
                    "loss_tT": loss_temp_t,
                    'loss_tR': loss_temp_r,
                    'loss_ts': loss_temp_s,
                    'loss_th': loss_temp_h,
                    'loss_cont': loss_cont,
                    'loss_lat': loss_lat
                }
                metric_logger.update(**log_dict)
                if cfg.logging.wandb:
                    wandb.log(log_dict, step=train_state.step)

                if torch.isnan(loss).any():
                    print("Found NAN in loss, stop training.")
                    return

                loss.backward()
                optimizer_hum.step()
                optimizer_obj.step()
                scheduler_hum.step()
                scheduler_obj.step()
                optimizer_hum.zero_grad()
                optimizer_obj.zero_grad()

                if train_state.step % 250 == 0:
                    # save state
                    self.save_checkpoint_hoi(cfg, latent_hum, latent_obj, obj_rot_axis, obj_scale, obj_trans,
                                             optimizer_hum,
                                             optimizer_obj, output_dir, scheduler_hum, scheduler_obj, seq_name,
                                             train_state, batches_all,
                                             data_dict={'hum_trans': hum_trans,
                                                        'hum_scale': hum_scale,
                                                        }
                                             )

                if train_state.step % 1000 == 0:
                    self.render_video(batches, latent_obj, human_ae, output_dir, seq_name, train_state, obj_trans,
                                      obj_rot_axis, obj_scale,
                                      show_gt=cfg.run.sample_mode == 'gt',
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      latent_hum=latent_hum, hum_scale=hum_scale, hum_trans=hum_trans)

                if train_state.step >= cfg.run.max_steps or (
                        optimizer_obj.param_groups[0]['lr'] < 1e-8 and train_state.step > 2000):
                    import datetime
                    print(f'Ending training at: {datetime.datetime.now()}, lr={optimizer_obj.param_groups[0]["lr"]}')
                    print(f'Final train state: {train_state}')
                    # wandb.finish()
                    # time.sleep(5)
                    self.save_checkpoint_hoi(cfg, latent_hum, latent_obj, obj_rot_axis, obj_scale, obj_trans,
                                             optimizer_hum,
                                             optimizer_obj, output_dir, scheduler_hum, scheduler_obj, seq_name,
                                             train_state, batches_all,
                                             data_dict={'hum_trans': hum_trans,
                                                        'hum_scale': hum_scale, }
                                             )

                    # visualize before finish
                    batches = dataloader if len(batches) <= 1 else batches  # make sure we visualize full sequence
                    self.render_video(batches, latent_obj, human_ae, output_dir, seq_name, train_state, obj_trans,
                                      obj_rot_axis, obj_scale,
                                      show_gt=cfg.run.sample_mode == 'gt',
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      latent_hum=latent_hum, hum_scale=hum_scale, hum_trans=hum_trans)
                    if cfg.logging.wandb:
                        wandb.finish()
                        time.sleep(10)
                    return
                train_state.step += 1
            train_state.epoch += 1


    def load_checkpoint(self, cfg, model, model_ema, optimizer, scheduler):
        "load optimizer, model state, scheduler etc. "
        return training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler,
                                                     model_ema)

    def compute_contact_loss(self, batch, cfg, hum_pts_aeout_hoi, obj_pts_live, weight_vis):
        cont_dist = torch.stack(batch['contact_dist']).to(self.device)
        indices_hum = torch.stack(batch['indices_in_hum']).to(self.device)
        indices_obj = torch.stack(batch['indices_in_obj']).to(self.device)
        cont_mask = cont_dist < cfg.model.hoi_cont_thres
        bs = len(batch['frame_index'])
        loss_cont = 0.
        for ii in range(bs):
            if torch.sum(cont_mask[ii]) < 5:
                continue
            ind_hum, ind_obj = indices_hum[ii, cont_mask[ii, :, 0]], indices_obj[ii, cont_mask[ii, :, 0]]
            cont_hum = hum_pts_aeout_hoi[ii, ind_hum]
            cont_obj = obj_pts_live[ii, ind_obj]
            cont_dist = ((cont_obj - cont_hum) ** 2).sum(-1).mean() * weight_vis[ii]
            loss_cont += cont_dist
        loss_cont = loss_cont * cfg.model.hoi_lw_cont
        return loss_cont

    def compute_object_losses(self, cfg, batch, latent_obj, obj_rot_axis, obj_scale, obj_trans, renderer, weight_vis):
        lw_chamf_obj = cfg.model.obj_lw_chamf
        lw_mask = cfg.model.obj_lw_mask
        lw_dt = cfg.model.obj_lw_dt
        lw_temp_t = cfg.model.obj_lw_temp_t
        lw_temp_r = cfg.model.obj_lw_temp_r
        lw_temp_s = cfg.model.obj_lw_temp_s
        lw_hoi_cont = cfg.model.hoi_lw_cont
        opt_obj_trans = cfg.model.obj_opt_t
        opt_obj_rot = cfg.model.obj_opt_r
        opt_obj_scale = cfg.model.obj_opt_s
        opt_obj_shape = cfg.model.obj_opt_shape
        device = self.device

        obj_poses = torch.stack(batch['rot_pred'], 0).cuda()
        frame_indices = torch.cat(batch['frame_index']).to('cuda')
        # filter out heavy occlusions
        bs = len(frame_indices)
        obj_pts_can = self.decode_can_pts(bs, latent_obj, None)
        obj_trans_bi = obj_trans[frame_indices]
        obj_rot_bi = axis_angle_to_matrix(obj_rot_axis[frame_indices])
        obj_scale_orig = torch.stack(batch['radius_obj_pred'], 0).to(self.device)
        obj_trans_orig = torch.stack(batch['cent_obj_pred'], 0).to(self.device)
        obj_scale_bi = obj_scale[frame_indices]
        obj_pts_live, obj_pts_no_trans = self.canonical2live(obj_poses, obj_pts_can,
                                                             obj_rot_bi,
                                                             obj_trans_bi,
                                                             obj_trans_orig,
                                                             obj_scale_bi,
                                                             obj_scale_orig, ret_no_trans=True)
        obj_pts_recon = torch.stack(batch['pred_obj'], 0).cuda()
        chamf_obj = self.compute_chamf_obj(obj_pts_no_trans, obj_pts_recon, weight_vis,
                                           obj_pts_live, obj_trans_orig,
                                           obj_scale_orig) * lw_chamf_obj  # CD should not be applied after adding translation
        cam_t = torch.stack(batch['T'], 0).to(self.device)

        if lw_mask > 0 or lw_dt > 0:
            front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                            [0, -1., 0],
                                                            [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                           T=cam_t,  # camera that projects normalized object to object region
                                           K=torch.stack(batch['K_obj'], 0).to(self.device),
                                           in_ndc=True,
                                           device=self.device)
            pc = Pointclouds(obj_pts_live,
                             features=torch.tensor([[[1., 1., 1.]]]).repeat(bs, len(obj_pts_live[0]), 1).to(device))
            images = renderer(pc, cameras=front_cam)
            silhouette = images[..., :3].mean(-1)  # the boolean operation will lead to no gradient
            kernel_size = 7
            pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
            keep_mask, mask_ref, fore_mask = torch.stack(batch['keep_mask_obj']).to(self.device), torch.stack(
                batch['mask_ref_obj']).to(self.device), torch.stack(batch['fore_mask_obj']).to(self.device)
            temp = keep_mask * silhouette
            loss_mask = (torch.sum((temp - mask_ref) ** 2, dim=(1, 2)) * weight_vis).mean() * lw_mask
            # edge loss:
            edges_dt = torch.stack(batch['edges_dt_obj']).to(self.device)
            edges_rend = pool(silhouette) - silhouette
            loss_edt = (torch.sum(edges_rend * edges_dt, dim=(1, 2)) * weight_vis).mean() * lw_dt
        else:
            loss_mask, loss_edt = 0., 0.
        # Temporal smoothness loss
        loss_temp_t, loss_temp_r, loss_temp_s, loss_temp_v = 0., 0., 0., 0.
        if opt_obj_trans and lw_temp_t > 0:
            loss_temp_t = self.compute_loss_temp_t(cam_t, obj_trans_bi, obj_trans_orig) * lw_temp_t
        if opt_obj_rot and lw_temp_r > 0:
            rot_comb = torch.matmul(obj_rot_bi, obj_poses[:, :3, :3])
            velo1, velo2 = self.compute_velocity(rot_comb)
            loss_temp_r = F.mse_loss(velo1, velo2, reduction='mean') * lw_temp_r
        if opt_obj_scale and lw_temp_s > 0 and len(obj_scale) > 1:
            # scale loss
            scale_comb = obj_scale_orig * obj_scale_bi  # B, 1
            velo1, velo2 = self.compute_velocity(scale_comb)
            loss_temp_s = F.mse_loss(velo1, velo2, reduction='mean') * lw_temp_s
        return chamf_obj, loss_edt, loss_mask, loss_temp_r, loss_temp_s, loss_temp_t, obj_pts_live

    def compute_chamf_obj(self, obj_pts_no_trans, obj_pts_recon, weight_vis,
                          obj_pts_live, obj_trans_orig, obj_scale_orig):
        "compute chamfer distance in the interaction space, all others remain the same"
        obj_pts_recon_hoi = obj_pts_recon * obj_scale_orig[:, None] * 2 + obj_trans_orig[:, None]
        return chamfer_distance(obj_pts_live, obj_pts_recon_hoi).mean()

    def get_vis_weights(self, batch, occ_ratios):
        frame_times = [osp.basename(osp.dirname(x)) for x in batch['image_path']]
        occ_ratios_bi = [occ_ratios[k] for k in frame_times]
        weight_vis = torch.tensor(occ_ratios_bi).to('cuda')
        return weight_vis

    def get_random_batch(self, batch_size, batches_all, seq_len):
        torch.cuda.empty_cache()
        # randomly pick one chunk from the sequence
        rid = np.random.randint(0, seq_len - batch_size)
        start_ind, end_ind = rid, min(seq_len, rid + batch_size)
        batch = {}
        for k, v in batches_all.items():
            if len(v) == seq_len:
                batch[k] = batches_all[k][start_ind:end_ind]
        return batch

    def load_all_batches(self, cfg, dataloader, human_ae, latent_hum, latent_obj, obj_rot_axis):
        batches_all = {}
        batches, latent_hum_batches = [], []
        for batch in tqdm(dataloader, desc='Loading batches'):
            hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()  # already centered and normalized
            obj_pts_recon = torch.stack(batch['pred_obj'], 0).cuda()  # already centered and normalized
            frame_indices = torch.cat(batch['frame_index']).to('cuda')

            # human latent
            if latent_hum is None:
                # compute latents online
                latent_bi = []
                with torch.no_grad():
                    for ii in range(0, len(hum_pts_recon), 16):
                        latent = human_ae.encode(hum_pts_recon[ii:ii + 16])
                        latent_bi.append(latent)
                torch.cuda.empty_cache()
                latent = torch.cat(latent_bi)
                latent_hum_batches.append(latent)
            else:
                latent = latent_hum[frame_indices]
            batch['latent'] = [x for x in latent.detach().cpu().clone()]
            # decode human
            hum_ae_out = human_ae.decode(latent)
            bs = len(hum_ae_out)

            # Compute contact points: first transform recon points back to interaction space
            cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
            cent_obj = torch.stack(batch['cent_obj_pred'], 0).to('cuda')  # B, 3
            radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')  # B, 1
            radius_obj = torch.stack(batch['radius_obj_pred'], 0).to('cuda')
            hum_pts_aeout = hum_ae_out * radius_hum[:, None] * 2 + cent_hum[:, None]
            # hum_pts_aeout_hoi = hum_ae_out * 2 * radius_hum[:, None] * hum_scale[frame_indices, None] + cent_hum[:,None] + hum_trans[frame_indices, None]
            obj_pts_hoi = obj_pts_recon * radius_obj[:, None] * 2 + cent_obj[:, None]

            obj_pts_can = self.decode_can_pts(bs, latent_obj, None)
            cam_t, obj_poses = self.get_obj_RT(batch, cfg)
            with torch.no_grad():
                obj_rot_bi = axis_angle_to_matrix(obj_rot_axis[frame_indices])
                rot_comb = torch.matmul(obj_rot_bi, obj_poses[:, :3, :3])
                obj_pts_live_no_trans = torch.matmul(obj_pts_can, rot_comb.transpose(1, 2))  # normalized

            # closest_dist_in_obj = knn_points(hum_pts_aeout, obj_pts_hoi, K=1)
            # dist = closest_dist_in_obj.dists**0.5, hum_pts_aeout has wrong translation
            # print(hum_pts_aeout[:2, :3], hum_ae_out[hum_ae_out:2, :3], obj_pts_hoi[:2, :3])
            dist, indices_in_hum, indices_in_obj = self.compute_contacts(hum_pts_aeout, obj_pts_hoi,
                                                                         obj_pts_live_no_trans, obj_pts_recon)
            # print(indices_in_hum[:3], indices_in_obj[:3], indices_in_hum.shape, indices_in_obj.shape)
            # print(torch.sum(dist < 0.02)) # something is wrong here, it is significantly less points
            batch['contact_dist'] = [x for x in dist.cpu()]
            batch['indices_in_hum'] = [x for x in indices_in_hum.cpu()]
            batch['indices_in_obj'] = [x for x in indices_in_obj.cpu()]  # save for later optimization

            # also re-compute the masks for object loss
            masks_ho = torch.stack(batch['masks_obj'], 0)  # B, 2, H, W, human and obj mask
            fore_mask = masks_ho[:, 1] > 0.5
            ps_mask = masks_ho[:, 0] > 0.5
            # mask_merge = ps_mask | fore_mask
            edges_dt, keep_mask, mask_ref = self.prep_2dlosses(fore_mask, ps_mask)
            batch['keep_mask_obj'] = [x for x in keep_mask]
            batch['mask_ref_obj'] = [x for x in mask_ref]
            batch['fore_mask_obj'] = [x for x in fore_mask]
            batch['edges_dt_obj'] = [x for x in edges_dt]

            for key, value in batch.items():
                if key not in batches_all:
                    batches_all[key] = []
                if isinstance(value, list):
                    batches_all[key].extend(value)
            batches.append(batch)
            # if len(batches) == 2:
            #     break
        # exit(-1)
        return batches, batches_all, latent_hum_batches

    def compute_contacts(self, hum_pts_aeout, obj_pts_hoi, obj_pts_live_no_trans, obj_pts_recon):
        """first find contacts in the object, then map them to human surfaces to have 1-1 pairs"""
        # Find indices on optimized object shape, for 1 to 1 contact matching
        closest_dist_in_obj = knn_points(hum_pts_aeout, obj_pts_hoi, K=1)
        dist = closest_dist_in_obj.dists ** 0.5
        # human is simply the ordering
        indices_in_hum = torch.arange(hum_pts_aeout.shape[1])[None].repeat(len(hum_pts_aeout), 1)[:, :, None]
        indices_in_obj = closest_dist_in_obj.idx # (B, len(hum), K)
        # index these points in normalized space
        obj_pts_corr = []
        # for ind, pts in zip(obj_pts_recon, indices_in_obj):
        for i in range(len(obj_pts_recon)):
            # print(indices_in_obj.shape, obj_pts_recon.shape)
            obj_pts_corr.append(obj_pts_recon[i, indices_in_obj[i, :, 0]])
        obj_pts_corr = torch.stack(obj_pts_corr, 0)
        # transfer these points to the shape to be optimized
        closest_dist_in_obj_opt = knn_points(obj_pts_corr, obj_pts_live_no_trans, K=1)
        indices_in_obj = closest_dist_in_obj_opt.idx
        # import pdb;pdb.set_trace()
        assert indices_in_hum.shape == indices_in_obj.shape, 'error'
        return dist, indices_in_hum, indices_in_obj

    def save_checkpoint_hoi(self, cfg, latent_hum, latent_obj, obj_rot_axis, obj_scale, obj_trans, optimizer_hum,
                            optimizer_obj,
                            output_dir, scheduler_hum, scheduler_obj, seq_name, train_state, batch, data_dict=None):
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        obj_pts_can = self.decode_can_pts(1, latent_obj, self.model)
        hum_trans, hum_scale = data_dict['hum_trans'], data_dict['hum_scale']
        ckpt_dict = {
            # these use the same name convention from object optimizer
            "latents": latent_obj,
            "optimizer": optimizer_obj.state_dict(),
            "scheduler": scheduler_obj.state_dict(),
            "epoch": train_state.epoch,
            "step": train_state.step,
            "cfg": cfg,
            'train_state': train_state,
            'obj_trans': obj_trans,
            "obj_rot_axis": obj_rot_axis,
            'obj_scale': obj_scale,
            # human
            'latents_hum': latent_hum,
            "optimizer_hum": optimizer_hum.state_dict(),
            'scheduler_hum': scheduler_hum.state_dict(),
            'hum_trans': hum_trans,
            'hum_scale': hum_scale,

            # normalization parameters
            'frame_indices': torch.cat(batch['frame_index']),
            "center_hum": torch.stack(batch['cent_hum_pred'], 0),
            "radius_hum": torch.stack(batch['radius_hum_pred'], 0),
            'cent_obj_pred': torch.stack(batch['cent_obj_pred']),
            'radius_obj_pred': torch.stack(batch['radius_obj_pred']),

            # canonical object point
            'obj_pts_canonical': obj_pts_can.detach().cpu(),
            'image_path': batch['image_path']
        }
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        os.makedirs(osp.dirname(ckpt_file), exist_ok=True)
        torch.save(ckpt_dict, ckpt_file)
        print('checkpoint saved to', ckpt_file)

    @torch.no_grad()
    def render_video(self, batches, latent_obj, model_ae,
                     output_dir, seq_name, train_state,
                     obj_trans, obj_rot_axis, obj_scale,
                     show_gt=True, show_recon=False, save_pc=False,
                     latent_hum=None, hum_scale=None, hum_trans=None, data_dict=None):
        import imageio
        # seq_name = str(batch['image_path'][0]).split(os.sep)[-3]
        video_file = output_dir / 'vis' / seq_name / f'step{train_state.step:06d}_gt-{show_gt}_recon-{show_recon}.mp4'
        os.makedirs(osp.dirname(video_file), exist_ok=True)
        rend_size = 224
        renderer = PcloudRenderer(image_size=rend_size, radius=0.0075)
        video_writer = imageio.get_writer(video_file, format='FFMPEG', mode='I', fps=10)
        with torch.no_grad():
            obj_pts_can = self.decode_can_pts(1, latent_obj, None)  # force center happens inside
            # obj_pts_can = model_ae.decode(latent)
            # vc_points = self.get_cmap(obj_pts_can.detach().cpu().numpy()[0])

        for i, batch in enumerate(tqdm(batches)):
            obj_pts_gt = torch.stack(batch['pclouds_obj'], 0).cuda()
            bs, NS = obj_pts_gt.shape[:2]
            cam_t, obj_poses = self.get_obj_RT(batch, self.cfg)
            frame_indices = torch.cat(batch['frame_index']).to('cuda')
            obj_trans_bi = obj_trans[frame_indices]  # optimized rot + translation
            obj_rot_bi = axis_angle_to_matrix(obj_rot_axis[frame_indices])

            vis_mask = frame_indices > -1  # all frames
            obj_scale_orig = torch.stack(batch['radius_obj_pred'], 0).to(self.device)[vis_mask]
            obj_trans_orig = torch.stack(batch['cent_obj_pred'], 0).to(self.device)[vis_mask]
            obj_scale_bi = obj_scale[frame_indices] if len(obj_scale) > 1 else obj_scale  # per-frame scale
            obj_pts_live, obj_pts_live_no_trans = self.canonical2live(obj_poses, obj_pts_can, obj_rot_bi, obj_trans_bi,
                                                                      obj_trans_orig,
                                                                      obj_scale_bi, obj_scale_orig, ret_no_trans=True)

            hum_pts_aeout, hum_pts_aeout_hoi = self.get_hum_pts_hoi(batch,
                                                                    hum_scale,
                                                                    hum_trans,
                                                                    latent_hum,
                                                                    model_ae,
                                                                    data_dict)

            # visualize the contact points
            cont_dist = torch.stack(batch['contact_dist']).to(self.device)
            indices_hum = torch.stack(batch['indices_in_hum']).to(self.device)
            indices_obj = torch.stack(batch['indices_in_obj']).to(self.device)
            cont_mask = cont_dist < 0.02
            pts_colors = []
            for ii in range(bs):
                vc_h = np.array([[0.1, 1., 0.9]]).repeat(len(hum_pts_aeout_hoi[0]), 0)  # blue
                vc_o = np.array([[1., 0., 0.]]).repeat(len(obj_pts_live[0]), 0)  # red
                if torch.sum(cont_mask[ii]) < 5:
                    pts_colors.append(np.concatenate([vc_h, vc_o], 0))
                    continue
                ind_hum, ind_obj = indices_hum[ii, cont_mask[ii, :, 0]], indices_obj[ii, cont_mask[ii, :, 0]]
                cont_hum = hum_pts_aeout_hoi[ii, ind_hum]
                # compute contact vc
                vc_cont = self.get_cmap(cont_hum.cpu().numpy())
                vc_h[ind_hum.cpu().numpy()] = vc_cont
                vc_o[ind_obj.cpu().numpy()] = vc_cont
                pts_colors.append(np.concatenate([vc_h, vc_o], 0))

            pts = torch.cat([hum_pts_aeout_hoi, obj_pts_live], 1)
            feats = torch.from_numpy(np.stack(pts_colors, 0)).float().to(self.device)
            pc = Pointclouds(pts, features=feats.to(self.device))

            front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                            [0, -1., 0],
                                                            [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                           T=cam_t,
                                           K=torch.stack(batch['K'], 0).to(self.device),  # project to H+O crop
                                           # visualize with full image parameters
                                           in_ndc=True,
                                           device=self.device)
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
            rends, masks = [torch.stack(batch['images']).cpu().permute(0, 2, 3, 1).numpy()], []
            for ii, cam in enumerate([front_cam, side_camera]):
                rend, mask = renderer.render(pc, cam, mode='mask')
                rends.append(rend)
                masks.append(mask)
            rend = np.concatenate(rends, 2)  # (B, H, W*3, 3)
            # add to video one by one
            for ii, img in enumerate(rend):
                comb = (img.copy() * 255).astype(np.uint8)
                # img_idx = int(batch['file_index'][ii])
                file = str(batch['image_path'][ii])
                ss = file.split(os.sep)
                cv2.putText(comb, f'{ss[-3]}/{ss[-2]} occ {self.occ_ratios[ss[-2]]:.2f}', (20, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 0, 0), 2)  # color (0, 255, 255)=bright blue, same as human color
                video_writer.append_data(comb)

            # Save results
            if save_pc:
                rot_comb = torch.matmul(obj_rot_bi, obj_poses[:, :3, :3])
                for ii in range(bs):
                    # pc_i = obj_pts_live[ii].cpu().numpy() # TODO: save without adding this translation.
                    # pc_i = obj_pts_live_no_trans[ii].cpu().numpy()
                    file = str(batch['image_path'][ii])
                    ss = file.split(os.sep)
                    outfile = output_dir / 'pred' / ss[-3] / f'{ss[-2]}.ply'
                    os.makedirs(osp.dirname(outfile), exist_ok=True)
                    pc_i = pts[ii].cpu().numpy()
                    vc_i = np.concatenate([
                        np.array([[0, 1.0, 1.0]]).repeat(hum_pts_aeout[ii].shape[0], 0),
                        np.array([[0.0, 1.0, 0]]).repeat(obj_pts_live[ii].shape[0], 0)
                    ], 0)
                    trimesh.PointCloud(pc_i, vc_i).export(outfile)

                # add synlink
                # self.add_synlink(output_dir) # TODO: update synlink target dir
            print("Visualization video saved to", video_file)
        if save_pc:
            print(f"pc saved to {output_dir.absolute()}, all done.")
        return seq_name

    def get_hum_pts_hoi(self, batch, hum_scale, hum_trans, latent, model_ae, data_dict=None):
        frame_indices = torch.cat(batch['frame_index']).to('cuda')
        latent_hum_bi = latent[frame_indices]
        hum_pts_aeout = model_ae.decode(latent_hum_bi)
        cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
        radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')
        hum_pts_aeout_hoi = (hum_pts_aeout * 2 * radius_hum[:, None] * hum_scale[frame_indices, None]
                             + cent_hum[:, None] + hum_trans[frame_indices, None])
        return hum_pts_aeout, hum_pts_aeout_hoi


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    cfg.run.job = 'sample' if cfg.run.job == 'train' else cfg.run.job
    cfg.run.freeze_feature_model = False
    cfg.dataloader.num_workers = 2  # reduce initialization waiting time
    cfg.dataset.load_obj_pose = True
    wandb_cache = cfg.logging.wandb
    trainer = TrainerHOIOpt(cfg)
    cfg.logging.wandb = wandb_cache
    import traceback
    try:
        trainer.run_sample(cfg)
    except Exception as e:
        print(traceback.print_exc())


if __name__ == '__main__':
    main()
