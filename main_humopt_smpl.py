"""
optimize through a SMPL layer
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
from pathlib import Path
import os.path as osp
from utils.losses import chamfer_distance, rigid_loss

from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras
from lib_smpl import get_smpl
from lib_smpl.body_landmark import BodyLandmarks
from lib_smpl.th_smpl_prior import get_prior
from lib_smpl.th_hand_prior import HandPrior

from model.pvcnn.pvcnn_enc import PVCNNAutoEncoder
from main_humopt import TrainerHumOpt


class TrainerHumOptSMPL(TrainerHumOpt):
    "Optimize human via the SMPL layer"
    def sample(self, cfg: ProjectConfig,
                model: PVCNNAutoEncoder,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        "load SMPL pose, shape, and compute a mean shape for optimization"
        rend_size, device = cfg.model.image_size, 'cuda'
        self.device = device
        # renderer = self.init_renderer(device, rend_size)
        import socket; self.DEBUG, self.no_ae = 'volta' in socket.gethostname(), True
        output_dir: Path = Path(output_dir)

        # human AE
        human_ae: PVCNNAutoEncoder = model
        human_ae.eval()
        smpl_layer = get_smpl('male', True).to(device)
        self.smpl_layer = smpl_layer

        # try to load latents, parameters from ckpt
        seq_name = self.extract_seq_name(cfg)
        # try to load from ckpt
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        if osp.isfile(ckpt_file):
            print(f"Loading from {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            # load SMPL pose parameters instead of latent
            # latent_hum = ckpt['latents_hum']
            train_state = ckpt['train_state']
            poses_hum, betas_hum, transl_smplh = ckpt['poses_hum'], ckpt['betas_hum'], ckpt['transl_smpl']
            hum_trans, hum_scale  = ckpt['hum_trans'], ckpt['hum_scale']
        else:
            poses_hum, betas_hum, transl_smplh = None, None, None
            hum_trans = torch.zeros(2500, 3).to(device)
            hum_scale = torch.ones(2500, 1).to(device)
            print("No checkpoint for optimization found!")
            train_state = training_utils.TrainState()

        batches_all = {}
        batches, latent_hum_batches = [], []
        betas, poses, transl, verts_smplh = [], [], [], [] # SMPL parameters
        centers, scale = [], []
        for batch in tqdm(dataloader, desc='Loading batches'):
            # hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()  # already centered and normalized
            # frame_indices = torch.cat(batch['frame_index']).to('cuda')
            # load SMPLH and parameters
            if poses_hum is None:
                for file in batch['pred_file']:
                    smplh_mesh = trimesh.load(file.replace('/pred/', '/smplh/'), process=False)
                    smplh_params = pkl.load(open(file.replace('/pred/', '/smplh/').replace('.ply', '.pkl'), 'rb'))
                    betas.append(smplh_params['betas'])
                    poses.append(smplh_params['pose'])
                    transl.append(smplh_params['trans'])
                    verts_smplh.append(np.array(smplh_mesh.vertices))
                    centers.append(smplh_params['center'])
                    scale.append(float(smplh_params['scale'])) # make sure this is not an array

            # add to all batches
            for key, value in batch.items():
                if key not in batches_all:
                    batches_all[key] = []
                if isinstance(value, list):
                    batches_all[key].extend(value)
            batches.append(batch)

            # if len(batches) == 5:
            #     break

        # Initialize pose and shapes
        if betas_hum is None:
            betas_avg = np.mean(np.stack(betas, 0), 0)
            poses_hum = torch.from_numpy(np.stack(poses, 0)).float().to(device)
            transl_smplh = torch.from_numpy(np.stack(transl, 0)).float().to(device)
            betas_hum = torch.from_numpy(betas_avg[None]).float().to(device)
            verts_new, _, _, _ = smpl_layer(poses_hum,
                                        betas_hum.repeat(len(poses_hum), 1),
                                        transl_smplh,
                                        )
            verts_new = verts_new.cpu().numpy()

            scales_new, center_new = [], []
            for i in range(len(verts_smplh)):
                # transform to normalized space, and compute an alignment
                verts_orig_norm = (verts_smplh[i] - centers[i]) / scale[i]
                mat, transformed, cost1 = trimesh.registration.procrustes(verts_new[i], verts_orig_norm, scale=True)
                # Get the scale factor and translation, this is used later to compute Chamfer loss
                u, s, vt = np.linalg.svd(mat[:3, :3])
                scales_new.append(s[0]*scale[i])
                center_new.append(centers[i] + scale[i]*mat[:3, 3]) # this new params will transform verts_new to interaction space
            scales_new = torch.from_numpy(np.array(scales_new)[:, None]).float()
            center_new = torch.from_numpy(np.stack(center_new, 0)).float()
            # print(center_new.shape, scales_new.shape)
            batches_all['scales_smplh_aligned'] = scales_new
            batches_all['centers_smplh_aligned'] = center_new
        else:
            batches_all['scales_smplh_aligned'] = ckpt['scales_smplh_aligned']
            batches_all['centers_smplh_aligned'] = ckpt['centers_smplh_aligned']

        # feed back to each batch
        idx = 0
        for batch in batches:
            batch['scales_smplh_aligned'] = batches_all['scales_smplh_aligned'][idx:idx+len(batch['frame_index'])]
            batch['centers_smplh_aligned'] = batches_all['centers_smplh_aligned'][idx:idx + len(batch['frame_index'])]
            idx = idx + len(batch['frame_index']) # accumulate

        # setup optimization params
        poses_hum = poses_hum.requires_grad_(True)
        opt_params = [poses_hum]
        if cfg.model.hum_opt_t:
            hum_trans = hum_trans.requires_grad_(True)
            opt_params.append(hum_trans)
        if cfg.model.hum_opt_s:
            hum_scale = hum_scale.requires_grad_(True)
            opt_params.append(hum_scale)
        if cfg.model.hum_opt_betas:
            betas_hum = betas_hum.requires_grad_(True)
            opt_params.append(betas_hum)
        optimizer_hum = optim.Adam(opt_params, lr=0.001)
        scheduler_hum = get_scheduler(optimizer=optimizer_hum, name='cosine',
                                      num_warmup_steps=100,
                                      num_training_steps=int(cfg.run.max_steps*1.2))
        body_prior = get_prior()
        hand_prior = HandPrior()

        if osp.isfile(ckpt_file):
            print("Load state of optimizer and scheduler")
            scheduler_hum.load_state_dict(ckpt['scheduler_hum'])
            optimizer_hum.load_state_dict(ckpt['optimizer_hum'])

        if cfg.logging.wandb:
            wandb.init(project='opt-hum', name=cfg.run.save_name + f'_{seq_name}', job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
            print('Initialized wandb')
        seq_len = len(batches_all['images'])
        num_batches = len(batches)
        batch_size = min(cfg.dataloader.batch_size, seq_len-1)
        assert torch.allclose(torch.tensor(batches_all['frame_index']), torch.arange(seq_len)), 'the frame order is incorrect!'
        print(f"Optimize translation? {cfg.model.hum_opt_t}, scale? {cfg.model.hum_opt_s}")
        data_dict = {"poses_hum": poses_hum,
                     "transl_smplh": transl_smplh,
                     "betas_hum": betas_hum}
        # keypoint based loss
        landmark = None
        if cfg.model.hum_lw_kpts > 0.:
            landmark = BodyLandmarks(osp.join(cfg.data.demo_data_path, 'assets'), batch_size, self.device)


        while True:
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

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
                bs = len(frame_indices)
                hum_pts_aeout = smpl_layer(poses_hum[start_ind:end_ind],
                                        betas_hum.repeat(bs, 1),
                                        transl_smplh[start_ind:end_ind],
                                        )[0] # in normalized space
                hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()
                cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
                radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')
                # hum_pts_aeout_hoi = hum_pts_aeout * 2 * radius_hum[:, None] * hum_scale[frame_indices, None] + cent_hum[:,None] + hum_trans[frame_indices, None]
                # Transform to interaction space
                hum_pts_aeout_hoi = hum_pts_aeout * batch['scales_smplh_aligned'][:, None].to(device) * hum_scale[
                    frame_indices, None] + hum_trans[frame_indices, None] + batch['centers_smplh_aligned'][:, None].to(device)
                # 1. Chamfer distance
                hum_pts_recon_hoi = hum_pts_recon* 2 * radius_hum[:, None]+ cent_hum[:, None]
                # print(hum_pts_aeout_hoi.shape, hum_pts_recon_hoi.shape, hum_pts_aeout.shape)
                loss_cd = 0.
                if cfg.model.hum_lw_cd > 0:
                    # import pdb; pdb.set_trace()
                    loss_cd = chamfer_distance(hum_pts_aeout_hoi, hum_pts_recon_hoi).mean() * cfg.model.hum_lw_cd

                # 2. temporal smoothness, on the interaction space
                velo1, velo2 = self.compute_velocity(hum_pts_aeout_hoi)
                lw_temp_h = cfg.model.hoi_lw_temp_h
                loss_temp_h = ((velo1 - velo2) ** 2).sum(-1).mean() * lw_temp_h + (velo1 ** 2).sum(-1).mean() * 0.5 * lw_temp_h

                # 3. human pose prior
                loss_bpr = torch.mean(body_prior(poses_hum[start_ind:end_ind, :72])) * cfg.model.hum_lw_bprior
                loss_hpr = torch.mean(hand_prior(poses_hum[start_ind:end_ind])) * cfg.model.hum_lw_hprior

                # 4. scale regularization
                loss_scale = 0.
                if cfg.model.hum_opt_s:
                    loss_scale = ((hum_scale - 1.) ** 2).mean()

                # 5. 2D body keypoint projection loss if used
                loss_j2d = 0
                if cfg.model.hum_lw_kpts > 0:
                    j3d = landmark.get_body_kpts_th(hum_pts_aeout_hoi)
                    # projection
                    cam_t = torch.stack(batch['T'], 0).to(self.device)
                    front_cam = PerspectiveCameras(R=torch.tensor([[[-1, 0, 0.],
                                                                    [0, -1., 0],
                                                                    [0, 0, 1.]]]).repeat(bs, 1, 1).to(self.device),
                                                   T=cam_t,  # camera that projects normalized object to object region
                                                   K=torch.stack(batch['K_hum'], 0).to(self.device),
                                                   in_ndc=True,
                                                   device=self.device)
                    j2d_proj = front_cam.transform_points_ndc(j3d) # ndc space: x-left, y-up
                    j2d_proj_cv = (1 - j2d_proj)/2. # convert to opencv convention and normalized to 0-1
                    j2d_op = torch.stack(batch['joints2d'], 0).to(self.device)
                    loss_j2d = (torch.sum((j2d_op - j2d_proj_cv)**2, -1) * j2d_op[:, :, 2]).mean()* cfg.model.hum_lw_kpts

                    if self.DEBUG:
                        # visualize
                        vis_ind = 0
                        rgb = (batch['images'][vis_ind].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        rh = rgb.shape[0]
                        j2d_proj_np = (j2d_proj_cv[vis_ind].detach().cpu().numpy() * rh).astype(int)
                        j2d_op_np = (j2d_op[vis_ind].cpu().numpy() * rh).astype(int)
                        img_vis = rgb.copy()
                        for i, (x, y) in enumerate(zip(j2d_op_np[:, 0], j2d_op_np[:, 1])):
                            cv2.circle(img_vis, (int(x), int(y)), 2, (0, 255, 255), 1, cv2.LINE_8) # blue
                        for i, (x, y) in enumerate(zip(j2d_proj_np[:, 0], j2d_proj_np[:, 1])):
                            cv2.circle(img_vis, (int(x), int(y)), 2, (255, 0, 0), 1, cv2.LINE_8) # red
                        comb = np.concatenate([rgb, img_vis], 1)[:, :, ::-1]
                        cv2.imshow(str(output_dir), comb)
                        cv2.waitKey(10)
                        cv2.moveWindow(str(output_dir), 600, 50)

                loss = loss_cd + loss_bpr + loss_hpr + loss_temp_h + loss_scale + loss_j2d # + loss_rigid  # + loss_edt
                log_dict = {
                    'lr': optimizer_hum.param_groups[0]["lr"],
                    'step': train_state.step,
                    'loss_cd_h': loss_cd,
                    'loss_th': loss_temp_h.item(),
                    'loss_bpr': loss_bpr.item(),
                    'loss_hpr': loss_hpr.item(),
                    'loss_scale': loss_scale,
                    'loss_j2d': loss_j2d
                }
                metric_logger.update(**log_dict)
                if cfg.logging.wandb:
                    wandb.log(log_dict, step=train_state.step)

                if torch.isnan(loss).any():
                    print("Found NAN in loss, stop training.")
                    return

                loss.backward()
                optimizer_hum.step()
                scheduler_hum.step()
                optimizer_hum.zero_grad()

                if train_state.step % 500 == 0:
                    # save checkpoint
                    self.save_checkpoint_smplh(batches_all, betas_hum, cfg, hum_scale, hum_trans, optimizer_hum,
                                               output_dir, poses_hum, scheduler_hum, train_state, transl_smplh)

                if train_state.step % 1000 == 0:
                    self.render_video(batches, None, human_ae, output_dir, seq_name, train_state, hum_trans,
                                      None, hum_scale,
                                      show_gt=True,
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      data_dict=data_dict)

                if train_state.step >= cfg.run.max_steps or (optimizer_hum.param_groups[0]['lr'] < 1e-8 and train_state.step > 2000):
                    import datetime
                    print(f'Ending training at: {datetime.datetime.now()}, lr={optimizer_hum.param_groups[0]["lr"]}')
                    print(f'Final train state: {train_state}')
                    self.save_checkpoint_smplh(batches_all, betas_hum, cfg, hum_scale, hum_trans, optimizer_hum,
                                               output_dir, poses_hum, scheduler_hum, train_state, transl_smplh)
                    self.render_video(batches, None, human_ae, output_dir, seq_name, train_state, hum_trans,
                                      None, hum_scale,
                                      show_gt=False,
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      data_dict=data_dict)
                    self.render_video(batches, None, human_ae, output_dir, seq_name, train_state, hum_trans,
                                      None, hum_scale,
                                      show_gt=True,
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      data_dict=data_dict)
                    if cfg.logging.wandb:
                        wandb.finish()
                        time.sleep(10)
                    return
                train_state.step += 1
            train_state.epoch += 1

    def get_hum_pts_hoi(self, batch, hum_scale, hum_trans, latent, model_ae, data_dict=None):
        ""
        frame_indices = torch.cat(batch['frame_index']).to('cuda')
        poses_hum = data_dict['poses_hum']
        transl_smplh = data_dict['transl_smplh']
        betas_hum = data_dict['betas_hum']
        hum_pts_aeout = self.smpl_layer(poses_hum[frame_indices],
                                   betas_hum.repeat(len(frame_indices), 1),
                                   transl_smplh[frame_indices],
                                   )[0]
        cent_hum = batch['centers_smplh_aligned'].to('cuda')
        radius_hum = batch['scales_smplh_aligned'].to('cuda')
        hum_pts_aeout_hoi = hum_pts_aeout * radius_hum[:, None] * hum_scale[
            frame_indices, None] + cent_hum[:, None] + hum_trans[frame_indices, None]

        return hum_pts_aeout, hum_pts_aeout_hoi

    def save_checkpoint_smplh(self, batches_all, betas_hum, cfg, hum_scale, hum_trans, optimizer_hum, output_dir,
                              poses_hum, scheduler_hum, train_state, transl_smplh):
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        # also the parameters of translation and scale
        ckpt_dict = {
            # these use the same name convention from object optimizer
            "epoch": train_state.epoch,
            "step": train_state.step,
            "cfg": cfg,
            'train_state': train_state,
            # smpl params
            'poses_hum': poses_hum,
            'betas_hum': betas_hum,
            'transl_smpl': transl_smplh,
            'scales_smplh_aligned': batches_all['scales_smplh_aligned'],
            'centers_smplh_aligned': batches_all['centers_smplh_aligned'],

            # human
            'hum_trans': hum_trans,
            'hum_scale': hum_scale,
            "optimizer_hum": optimizer_hum.state_dict(),
            'scheduler_hum': scheduler_hum.state_dict(),
            # some normalization params
            'frame_indices': torch.cat(batches_all['frame_index']),
            "center_hum": torch.stack(batches_all['cent_hum_pred'], 0),
            "radius_hum": torch.stack(batches_all['radius_hum_pred'], 0)
        }
        seq_name = self.extract_seq_name(cfg)
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        os.makedirs(osp.dirname(ckpt_file), exist_ok=True)
        torch.save(ckpt_dict, ckpt_file)
        print('checkpoint saved to', ckpt_file)


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    # to prevent random shuffle
    cfg.run.job = 'sample' if cfg.run.job == 'train' else cfg.run.job
    # shuffle only if it is train
    cfg.run.freeze_feature_model = False
    cfg.dataloader.num_workers = 2  # reduce initialization waiting time
    cfg.dataset.load_obj_pose = False
    wandb_cache = cfg.logging.wandb
    trainer = TrainerHumOptSMPL(cfg)
    cfg.logging.wandb = wandb_cache
    trainer.run_sample(cfg)

if __name__ == '__main__':
    main()