"""
use AE to optimize human, loading recon from HDM folder, optimize the AE latent vector
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
import training_utils
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os.path as osp
from utils.losses import chamfer_distance, rigid_loss


from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras
from render.pyt3d_wrapper import MeshRendererWrapper, get_kinect_camera, PcloudRenderer
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Pointclouds

from model.pvcnn.pvcnn_enc import PVCNNAutoEncoder
from main_objopt_latent import TrainerObjOpt


class TrainerHumOpt(TrainerObjOpt):
    "optimize the latent of human autoencoder"
    def sample(self, cfg: ProjectConfig,
                model: PVCNNAutoEncoder,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        ""
        # Visulization
        rend_size, device = cfg.model.image_size, 'cuda'
        renderer = self.init_renderer(device, rend_size)
        import socket;self.DEBUG, self.no_ae = 'volta' in socket.gethostname(), True
        output_dir: Path = Path(output_dir)

        # human AE
        human_ae: PVCNNAutoEncoder = model
        human_ae.eval()

        # try to load latents, parameters from ckpt
        seq_name = self.extract_seq_name(cfg)
        # try to load from ckpt
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        if osp.isfile(ckpt_file):
            print(f"Loading from {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device)
            latent_hum = ckpt['latents_hum']
            train_state = ckpt['train_state']
        else:
            latent_hum = None
            print("No checkpoint for optimization found!")
            train_state = training_utils.TrainState()

        # preload
        # First load all batches and combine them
        batches_all = {}
        batches, latent_hum_batches = [], []
        for batch in tqdm(dataloader, desc='Loading batches'):
            hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()  # already centered and normalized
            # obj_pts_recon = torch.stack(batch['pred_obj'], 0).cuda()  # already centered and normalized
            frame_indices = torch.cat(batch['frame_index']).to('cuda')

            # if hum_pts_recon.shape[1] > 6890:
            #     # downsample
            #     hum_pts_down = []
            #     for pts in hum_pts_recon:
            #         choice = torch.from_numpy(np.random.choice(len(pts), 6890, replace=False))
            #         hum_pts_down.append(pts[choice.cuda()])
            #     hum_pts_recon = torch.stack(hum_pts_down)
            print("Input human points shape:", hum_pts_recon.shape)

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

            # pre-compute for mask losses
            if cfg.model.hum_lw_dt > 0:
                masks_ho = torch.stack(batch['masks'], 0)  # B, 2, H, W, human and obj mask, in the human crop
                fore_mask = masks_ho[:, 0] > 0.5 # now fore is human, the other is obj
                obj_mask = masks_ho[:, 1] > 0.5
                # mask_merge = ps_mask | fore_mask
                edges_dt, keep_mask, mask_ref = self.prep_2dlosses(fore_mask, obj_mask)
                batch['keep_mask_hum'] = [x for x in keep_mask]
                batch['mask_ref_hum'] = [x for x in mask_ref]
                batch['fore_mask_hum'] = [x for x in fore_mask]
                batch['edges_dt_hum'] = [x for x in edges_dt]

            # decode human
            for key, value in batch.items():
                if key not in batches_all:
                    batches_all[key] = []
                if isinstance(value, list):
                    batches_all[key].extend(value)
            batches.append(batch)
            # if len(batches) == 2:
            #     break
        # Prepare human optimizer
        if latent_hum is None:
            latent_hum = torch.cat(latent_hum_batches, 0)  # N, D

        # TODO: load from state dict if exist
        latent_hum = latent_hum.requires_grad_(True)
        opt_params= [latent_hum]
        hum_trans = torch.zeros(2500, 3).to(device)
        hum_scale = torch.ones(2500, 1).to(device)
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
        if osp.isfile(ckpt_file):
            print("Load state of optimizer and scheduler")
            scheduler_hum.load_state_dict(ckpt['scheduler_hum'])
            optimizer_hum.load_state_dict(ckpt['optimizer_hum'])
            hum_scale.data = ckpt['hum_scale'].data
            hum_trans.data = ckpt['hum_trans'].data
        if cfg.logging.wandb:
            wandb.init(project='opt-hum', name=cfg.run.save_name + f'_{seq_name}', job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
            print('Initialized wandb')
        seq_len = len(batches_all['images'])
        num_batches = len(batches)
        batch_size = cfg.dataloader.batch_size
        print(f"Optimize translation? {cfg.model.hum_opt_t}, scale? {cfg.model.hum_opt_s}")

        while True:
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('loss_cd', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('loss_temp', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('loss_lat', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('time_one_iter', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            progress_bar: Iterable[Any] = metric_logger.log_every(range(num_batches * 10), 10, header=log_header)
            for i in progress_bar:
                torch.cuda.empty_cache()
                # randomly pick one chunk from the sequence
                rid = np.random.randint(0, seq_len - batch_size)
                start_ind, end_ind = rid, min(seq_len, rid + batch_size)
                batch = {}
                for k, v in batches_all.items():
                    if len(v) == seq_len:
                        batch[k] = batches_all[k][start_ind:end_ind]

                frame_indices = torch.cat(batch['frame_index']).to('cuda')
                latent_hum_bi = latent_hum[frame_indices]
                hum_pts_aeout = human_ae.decode(latent_hum_bi)
                hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()
                cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
                radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')
                hum_pts_aeout_hoi = hum_pts_aeout * 2 * radius_hum[:, None] * hum_scale[frame_indices, None] + cent_hum[:,None] + hum_trans[frame_indices, None]
                # 1. Chamfer distance
                hum_pts_recon_hoi = hum_pts_recon* 2 * radius_hum[:, None]+ cent_hum[:, None]
                loss_cd = chamfer_distance(hum_pts_aeout_hoi, hum_pts_recon_hoi).mean() * cfg.model.hum_lw_cd

                # 2. Latent code regularization
                lat_orig = torch.stack(batch['latent'], 0).to(self.device)
                loss_lat = F.mse_loss(latent_hum_bi, lat_orig) * cfg.model.hum_lw_lat

                # 3. temporal smoothness, on the interaction space
                velo1, velo2 = self.compute_velocity(hum_pts_aeout_hoi)
                lw_temp_h = cfg.model.hoi_lw_temp_h
                loss_temp_h = ((velo1 - velo2) ** 2).sum(-1).mean() * lw_temp_h + (velo1 ** 2).sum(-1).mean() * 0.5 * lw_temp_h

                # temporal smoothness on the normalized space: not hepful.
                # loss_temp_hn = 0.
                # if cfg.model.hum_lw_temp_hn > 0:
                #     velo1, velo2 = self.compute_velocity(hum_pts_aeout)
                #     lw_temp_hn = cfg.model.hum_lw_temp_hn
                #     loss_temp_hn = ((velo1 - velo2) ** 2).sum(-1).mean() * lw_temp_hn + (velo1 ** 2).sum(-1).mean() * 0.5 * lw_temp_hn

                # local rigidity loss
                loss_rigid = 0
                if cfg.model.hum_lw_rigid > 0:
                    loss_rigid = rigid_loss(hum_pts_aeout[1:], hum_pts_aeout[0:1]) * cfg.model.hum_lw_rigid

                # scale regularization
                loss_scale = 0.
                if cfg.model.hum_opt_s:
                    loss_scale = ((hum_scale - 1.)**2).mean()

                # 2D projection loss: edge dt
                loss_edt, loss_mask = 0, 0
                if cfg.model.hum_lw_dt > 0:
                    # if only 2d edt loss: it will shrink to one dot.
                    bs = len(frame_indices)
                    front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                                    [0, -1., 0],
                                                                    [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                                   # camera that projects normalized H+O to human region
                                                   T=torch.stack(batch['T'], 0).to(self.device),
                                                   K=torch.stack(batch['K_hum'], 0).to(self.device),
                                                   in_ndc=True,
                                                   device=self.device)
                    pc = Pointclouds(hum_pts_aeout_hoi, features=torch.tensor([[[1., 1., 1.]]]).repeat(bs, len(hum_pts_aeout_hoi[0]), 1).to(device))
                    images = renderer(pc, cameras=front_cam)
                    silhouette = images[..., :3].mean(-1)
                    keep_mask = torch.stack(batch['keep_mask_hum']).to(self.device)
                    edges_dt = torch.stack(batch['edges_dt_hum']).to(self.device)
                    mask_ref = torch.stack(batch['mask_ref_hum']).to(self.device)
                    obj_mask = torch.stack(batch['masks']).to(self.device)[:, 1]
                    fore_mask = torch.stack(batch['masks']).to(self.device)[:, 0]
                    kernel_size = 7
                    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
                    edges_rend = pool(silhouette) - silhouette
                    loss_edt = (torch.sum(edges_rend * edges_dt, dim=(1, 2))).mean() * cfg.model.hum_lw_dt
                    temp = keep_mask * silhouette
                    loss_mask = torch.sum((temp - mask_ref) ** 2, dim=(1, 2)).mean() * cfg.model.hum_lw_mask

                loss = loss_cd + loss_lat + loss_temp_h + loss_scale + loss_rigid # + loss_edt
                log_dict = {
                    'lr': optimizer_hum.param_groups[0]["lr"],
                    'step': train_state.step,
                    'loss_cd_h': loss_cd.item(),
                    'loss_th': loss_temp_h.item(),
                    # 'loss_thn': loss_temp_hn,
                    'loss_lat': loss_lat.item(),
                    'loss_scale': loss_scale,
                    # 'loss_edt': loss_edt,
                    # 'loss_mask': loss_mask,
                    'loss_rigid': loss_rigid
                }
                metric_logger.update(**log_dict)
                if cfg.logging.wandb:
                    wandb.log(log_dict, step=train_state.step)

                # visualization
                if self.DEBUG and cfg.model.hum_lw_dt > 0:
                    vis_ind = 0
                    temp = keep_mask * silhouette
                    sil = (silhouette[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)  # rendered image
                    img_ref = (mask_ref[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)
                    tmp = (temp[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)
                    h, w = sil.shape[:2]

                    vis1, vis2 = np.zeros((h, w, 3)), np.zeros((h, w, 3))
                    vis3, vis4 = np.zeros((h, w, 3)), np.zeros((h, w, 3))
                    rgb = (batch['images'][vis_ind].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    vis3[:, :, 2] = tmp
                    vis4[:, :, 2] = img_ref

                    vis1[:, :, 0] = (edges_dt[vis_ind].cpu().numpy() * 255).astype(np.uint8)  # blue: the ref edge, pink: overlap
                    vis1[:, :, 2] = (edges_rend[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8) # red: the rendered edge
                    vis2[:, :, 0] = (keep_mask[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)  # blue: keep mask
                    vis2[:, :, 2] = (obj_mask[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)  # red: ps mask
                    vis2[:, :, 1] = (fore_mask[vis_ind].detach().cpu().numpy() * 255).astype(np.uint8)  # green: obj mask
                    comb = np.concatenate([vis1, vis2, vis3, vis4, rgb], 1)
                    cv2.imshow(str(output_dir), comb)
                    cv2.waitKey(10)
                    cv2.moveWindow(str(output_dir), 600, 50)

                if torch.isnan(loss).any():
                    print("Found NAN in loss, stop training.")
                    return

                loss.backward()
                optimizer_hum.step()
                scheduler_hum.step()
                optimizer_hum.zero_grad()

                if train_state.step % 500 == 0:
                    # save checkpoint
                    self.save_checkpoint_hum(cfg, latent_hum, optimizer_hum, output_dir, scheduler_hum, seq_name,
                                             train_state, hum_trans, hum_scale, batches_all)

                if train_state.step % 1000 == 0:
                    # render
                    self.render_video(batches, latent_hum, human_ae, output_dir, seq_name, train_state, hum_trans,
                                      None, hum_scale,
                                      show_gt=True,
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True)
                    # print('all done')
                    # exit(0)

                if train_state.step >= cfg.run.max_steps or (optimizer_hum.param_groups[0]['lr'] < 1e-8 and train_state.step > 2000):
                    import datetime
                    print(f'Ending training at: {datetime.datetime.now()}, lr={optimizer_hum.param_groups[0]["lr"]}')
                    print(f'Final train state: {train_state}')
                    self.save_checkpoint_hum(cfg, latent_hum, optimizer_hum, output_dir, scheduler_hum, seq_name,
                                             train_state, hum_trans, hum_scale, batches_all)
                    self.render_video(batches, latent_hum, human_ae, output_dir, seq_name, train_state, hum_trans,
                                      None, hum_scale,
                                      show_gt=False,
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True)
                    self.render_video(batches, latent_hum, human_ae, output_dir, seq_name, train_state, hum_trans,
                                      None, hum_scale,
                                      show_gt=True,
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=False)
                    if cfg.logging.wandb:
                        wandb.finish()
                        time.sleep(10)
                    return

                train_state.step += 1
            train_state.epoch += 1

    def save_checkpoint_hum(self, cfg, latent_hum, optimizer_hum, output_dir, scheduler_hum, seq_name, train_state,
                            hum_trans, hum_scale, batches_all):
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        # also the parameters of translation and scale
        ckpt_dict = {
            # these use the same name convention from object optimizer
            "epoch": train_state.epoch,
            "step": train_state.step,
            "cfg": cfg,
            'train_state': train_state,
            # human
            'latents_hum': latent_hum,
            'hum_trans': hum_trans,
            'hum_scale': hum_scale,
            "optimizer_hum": optimizer_hum.state_dict(),
            'scheduler_hum': scheduler_hum.state_dict(),
            # with this we have the full parameters to compute the final results
            'frame_indices': torch.cat(batches_all['frame_index']),
            "center_hum": torch.stack(batches_all['cent_hum_pred'], 0),
            "radius_hum": torch.stack(batches_all['radius_hum_pred'], 0),
            'image_path': batches_all['image_path']
        }
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        os.makedirs(osp.dirname(ckpt_file), exist_ok=True)
        torch.save(ckpt_dict, ckpt_file)
        print('checkpoint saved to', ckpt_file)

    @torch.no_grad()
    def render_video(self, batches, latent, model_ae, output_dir, seq_name, train_state, hum_trans, obj_rot_axis, hum_scale,
                     show_gt=True, show_recon=False, save_pc=False, data_dict=None):
        ""
        import imageio
        # seq_name = str(batch['image_path'][0]).split(os.sep)[-3]
        video_file = output_dir / 'vis' / seq_name / f'step{train_state.step:06d}_gt-{show_gt}_recon-{show_recon}.mp4'
        os.makedirs(osp.dirname(video_file), exist_ok=True)
        rend_size = 224
        renderer = PcloudRenderer(image_size=rend_size, radius=0.0075)
        video_writer = imageio.get_writer(video_file, format='FFMPEG', mode='I', fps=15)
        for i, batch in enumerate(tqdm(batches)):
            frame_indices = torch.cat(batch['frame_index']).to('cuda')
            cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
            radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')
            hum_pts_aeout, hum_pts_aeout_hoi = self.get_hum_pts_hoi(batch,
                                                                     hum_scale,
                                                                     hum_trans,
                                                                     latent,
                                                                     model_ae,
                                                                        data_dict)

            bs = len(frame_indices)
            pts, feats = hum_pts_aeout_hoi, torch.tensor([[[0.0, 1., 1.0]]]).repeat(bs, hum_pts_aeout.shape[1], 1)
            if show_gt:
                pts_gt = torch.stack(batch['pclouds'], 0).to(self.device) # this is normalized
                pts_gt = pts_gt * 2 * radius_hum[:, None] * hum_scale[frame_indices, None] + cent_hum[:, None] + hum_trans[frame_indices, None]
                pts = torch.cat([pts, pts_gt], 1)
                bs, NS = pts_gt.shape[:2]
                feats = torch.cat([feats, torch.tensor([[[1.0, 0., 0]]]).repeat(bs, NS, 1)], 1)
            if show_recon:
                hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()
                hum_pts_recon = hum_pts_recon * 2 * radius_hum[:, None] * hum_scale[frame_indices, None] + cent_hum[:, None] + hum_trans[frame_indices, None]
                pts = torch.cat([pts, hum_pts_recon], 1)
                feats = torch.cat([feats, torch.tensor([[[1.0, 1.0, 0]]]).repeat(bs, len(hum_pts_recon[0]), 1)], 1)
            pc = Pointclouds(pts, features=feats.to(self.device))
            cam_t = torch.stack(batch['T'], 0).to(self.device)
            front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                            [0, -1., 0],
                                                            [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                           T=cam_t,
                                           K=torch.stack(batch['K'], 0).to(self.device), # h+o space
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
                cv2.putText(comb, f'{ss[-3]}/{ss[-2]}', (50, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 0, 0), 2)  # color (0, 255, 255)=bright blue, same as human color
                video_writer.append_data(comb)

            # Save results
            if save_pc:
                # compute optimized scale, translation and rotation to save
                for ii in range(bs):
                    pc_i = hum_pts_aeout_hoi[ii].cpu().numpy()
                    file = str(batch['image_path'][ii])
                    ss = file.split(os.sep)
                    outfile = output_dir / 'pred' / ss[-3] / f'{ss[-2]}.ply'
                    os.makedirs(osp.dirname(outfile), exist_ok=True)
                    trimesh.PointCloud(pc_i, np.array([[0, 1., 1.]]).repeat(len(pc_i), 0)).export(outfile)

                # add synlink
                # self.add_synlink(output_dir)
        print("Visualization video saved to", video_file)
        if save_pc:
            print(f"pc saved to {output_dir.absolute()}, all done.")
        return seq_name

    def load_checkpoint(self, cfg, model, model_ema, optimizer, scheduler):
        "load optimizer, model state, scheduler etc. "
        return training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler,
                                                     model_ema)

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
    # assert cfg.model.model_name == 'diff-ho-attn'
    # cfg.logging.wandb = False
     # to prevent random shuffle
    cfg.run.job = 'sample' if cfg.run.job == 'train' else cfg.run.job
    # shuffle only if it is train
    cfg.run.freeze_feature_model = False
    cfg.dataloader.num_workers = 2  # reduce initialization waiting time
    cfg.dataset.load_obj_pose = False
    # cfg.run.max_steps = max(5000, cfg.run.max_steps)
    # print('pose path:', cfg.dataset.pred_obj_pose_path)
    wandb_cache = cfg.logging.wandb
    trainer = TrainerHumOpt(cfg)
    cfg.logging.wandb = wandb_cache
    print("Using wandb?", cfg.logging.wandb)
    trainer.run_sample(cfg)


if __name__ == '__main__':
    main()