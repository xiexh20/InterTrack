"""
optimize human + object together, human is represented SMPLH body model
"""
import sys, os
import time
from typing import Iterable, Optional, Any

from accelerate import Accelerator

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
from utils.losses import chamfer_distance


from pytorch3d.renderer import PerspectiveCameras
from lib_smpl import get_smpl
from lib_smpl.body_landmark import BodyLandmarks
from lib_smpl.th_smpl_prior import get_prior
from lib_smpl.th_hand_prior import HandPrior

from model.pvcnn.pvcnn_enc import PVCNNAutoEncoder
from main_objopt_latent import filter_points_knn
from main_hoiopt_latent import TrainerHOIOpt


class TrainerHOISMPLOpt(TrainerHOIOpt):
    def sample(self, cfg: ProjectConfig,
                model: PVCNNAutoEncoder,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        "use SMPL for human"
        rend_size, device = cfg.model.image_size, 'cuda'
        renderer = self.init_renderer(device, rend_size)
        import socket; self.DEBUG, self.no_ae = 'volta' in socket.gethostname(), True
        output_dir: Path = Path(output_dir)

        # human AE
        human_ae: PVCNNAutoEncoder = model
        human_ae.eval()

        # try to load latents, parameters from ckpt
        seq_name = self.extract_seq_name(cfg)
        # try to load from ckpt
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        opt_obj_trans = cfg.model.obj_opt_t
        opt_obj_rot = cfg.model.obj_opt_r
        opt_obj_scale = cfg.model.obj_opt_s
        opt_obj_shape = cfg.model.obj_opt_shape
        print(f"Object: Optimize rotation? {cfg.model.obj_opt_r}, optimize translation? {cfg.model.obj_opt_t}, scale? {opt_obj_scale}, shape? {opt_obj_shape}, "
            f"lr={cfg.model.obj_opt_lr}, opt occlusion threshold: {cfg.model.obj_opt_occ_thres}")

        # Load checkpoint
        if osp.isfile(ckpt_file):
            print(f"Loading from {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device)

            # object optimizer
            latent_obj = ckpt['latents']
            obj_rot_axis, obj_scale, obj_trans, optimizer_obj, scheduler_obj, train_state = self.prep_obj_optimization(
                                            cfg,
                                            ckpt,
                                            device,
                                            latent_obj)
            init_frame_index = ckpt['init_frame_index'] # canonical frame index

            # human optimizer
            poses_hum, betas_hum, transl_smplh = ckpt['poses_hum'], ckpt['betas_hum'], ckpt['transl_smpl']
            hum_trans, hum_scale = ckpt['hum_trans'], ckpt['hum_scale']
            scales_smplh, centers_smplh = ckpt['scales_smplh_aligned'], ckpt['centers_smplh_aligned']
        else:
            print(ckpt_file, 'does not exist!')
            ckpt_file_obj = osp.join(cfg.dataset.hoi_opt_obj_shape_path,
                                     f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth')
            ckpt_obj = torch.load(ckpt_file_obj, map_location=self.device)
            latent_obj = ckpt_obj['latents']

            # filter out outlier points
            assert self.no_ae, 'do not support optimize object via shape ae!'
            obj_pts_filter, _ = filter_points_knn(latent_obj[0].detach().cpu().numpy())
            latent_obj = torch.from_numpy(obj_pts_filter[None]).float().to(self.device)

            obj_rot_axis, obj_trans, obj_scale, optimizer_obj = self.init_optimizer(device, latent_obj, opt_obj_rot,
                                                                                    opt_obj_trans,
                                                                                    opt_obj_scale, opt_obj_shape, cfg)
            obj_trans.data = ckpt_obj['obj_trans'].data
            obj_rot_axis.data = ckpt_obj['obj_rot_axis'].data
            obj_scale.data = ckpt_obj['obj_scale'].data
            init_frame_index = ckpt_obj['init_frame_index']  # canonical frame index
            scheduler_obj = get_scheduler(optimizer=optimizer_obj, name='cosine',
                                          num_warmup_steps=cfg.run.max_steps // 10,
                                          num_training_steps=int(cfg.run.max_steps * 1.5))

            train_state = training_utils.TrainState()

            # load human state from ckpt
            ckpt_file_hum = osp.join(cfg.dataset.hoi_opt_hum_shape_path,
                                     f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth')
            ckpt_hum = torch.load(ckpt_file_hum, map_location=self.device)
            poses_hum, betas_hum, transl_smplh = ckpt_hum['poses_hum'], ckpt_hum['betas_hum'], ckpt_hum['transl_smpl']
            hum_trans, hum_scale = ckpt_hum['hum_trans'], ckpt_hum['hum_scale']
            scales_smplh, centers_smplh = ckpt_hum['scales_smplh_aligned'], ckpt_hum['centers_smplh_aligned']

        # First load all batches and combine them
        batches, batches_all, latent_hum_batches = self.load_all_batches(cfg, dataloader, human_ae, None,
                                                                         latent_obj, obj_rot_axis)
        # feed back to each batch
        batches_all['scales_smplh_aligned'] = scales_smplh
        batches_all['centers_smplh_aligned'] = centers_smplh
        idx = 0
        for batch in batches:
            batch['scales_smplh_aligned'] = batches_all['scales_smplh_aligned'][idx:idx + len(batch['frame_index'])]
            batch['centers_smplh_aligned'] = batches_all['centers_smplh_aligned'][idx:idx + len(batch['frame_index'])]
            idx = idx + len(batch['frame_index'])  # accumulate

        # prepare human optimizer
        opt_params = []
        if cfg.model.hum_opt_lat:
            poses_hum = poses_hum.requires_grad_(True)
            opt_params.append(poses_hum)
        if cfg.model.hum_opt_t:
            hum_trans = hum_trans.requires_grad_(True)
            opt_params.append(hum_trans)
        if cfg.model.hum_opt_s:
            hum_scale = hum_scale.requires_grad_(True)
            opt_params.append(hum_scale)
        optimize_hum = len(opt_params) > 0
        if optimize_hum:
            optimizer_hum = optim.Adam(opt_params, lr=cfg.model.hoi_lr_hum)
            scheduler_hum = get_scheduler(optimizer=optimizer_hum, name='cosine',
                                          num_warmup_steps=cfg.run.max_steps//10,
                                          num_training_steps=int(cfg.run.max_steps*1.5))
        smpl_layer = get_smpl('male', True).to(device)
        self.smpl_layer = smpl_layer
        body_prior = get_prior()
        hand_prior = HandPrior()
        print(f"Human: optimize latent? {cfg.model.hum_opt_lat} translation? {cfg.model.hum_opt_t} scale? {cfg.model.hum_opt_s}")

        if osp.isfile(ckpt_file) and optimize_hum:
            scheduler_hum.load_state_dict(ckpt['scheduler_hum'])
            optimizer_hum.load_state_dict(ckpt['optimizer_hum'])
        occ_ratios = self.load_occ_ratios(seq_name)
        self.occ_ratios = occ_ratios

        # logging
        if cfg.logging.wandb:
            wandb.init(project='opt-hoi', name=str(output_dir) + f'_{seq_name}', job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
        seq_len = len(batches_all['images'])
        num_batches = len(batches)
        batch_size = cfg.dataloader.batch_size
        assert torch.allclose(torch.tensor(batches_all['frame_index']), torch.arange(seq_len)), 'the frame order is incorrect!'
        data_dict = {"poses_hum": poses_hum,
                     "transl_smplh": transl_smplh,
                     "betas_hum": betas_hum,
                     'hum_trans': hum_trans,
                     'hum_scale': hum_scale,
                     "init_frame_index": init_frame_index
                     }

        # keypoint based loss, by default this is not used
        landmark = None
        if cfg.model.hum_lw_kpts > 0.:
            landmark = BodyLandmarks("/BS/xxie-2/work/chore-video/assets/", batch_size, self.device)

        while True:
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            # metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            progress_bar: Iterable[Any] = metric_logger.log_every(range(num_batches * 10), 10, header=log_header)
            for i in progress_bar:
                batch = self.get_random_batch(batch_size, batches_all, seq_len)
                weight_vis = self.get_vis_weights(batch, occ_ratios)

                # Object losses
                chamf_obj, loss_edt, loss_mask, loss_temp_r, loss_temp_s, loss_temp_t, obj_pts_live = self.compute_object_losses(
                    cfg, batch, latent_obj, obj_rot_axis, obj_scale, obj_trans, renderer, weight_vis)

                frame_indices = torch.cat(batch['frame_index']).to('cuda')
                bs = len(frame_indices)

                # Human losses
                hum_pts_aeout = smpl_layer(poses_hum[frame_indices],
                                           betas_hum.repeat(bs, 1),
                                           transl_smplh[frame_indices],
                                           )[0]  # in normalized space
                hum_pts_recon = torch.stack(batch['pred_hum'], 0).cuda()
                cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
                radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')
                # Transform to interaction space
                hum_pts_aeout_hoi = hum_pts_aeout * batch['scales_smplh_aligned'][:, None].to(device) * hum_scale[
                    frame_indices, None] + hum_trans[frame_indices, None] + batch['centers_smplh_aligned'][:, None].to(device)

                # 1. Chamfer distance
                hum_pts_recon_hoi = hum_pts_recon * 2 * radius_hum[:, None] + cent_hum[:, None]
                # print(hum_pts_aeout_hoi.shape, hum_pts_recon_hoi.shape, hum_pts_aeout.shape)
                loss_cd = chamfer_distance(hum_pts_aeout_hoi, hum_pts_recon_hoi).mean() * cfg.model.hum_lw_cd if optimize_hum else 0.

                # 2. temporal smoothness, on the interaction space
                velo1, velo2 = self.compute_velocity(hum_pts_aeout_hoi)
                lw_temp_h = cfg.model.hoi_lw_temp_h
                loss_temp_h = ((velo1 - velo2) ** 2).sum(-1).mean() * lw_temp_h + (velo1 ** 2).sum(-1).mean() * 0.5 * lw_temp_h if optimize_hum else 0.

                # 3. human pose prior
                loss_bpr = torch.mean(body_prior(poses_hum[frame_indices, :72])) * cfg.model.hum_lw_bprior if optimize_hum else 0.
                loss_hpr = torch.mean(hand_prior(poses_hum[frame_indices])) * cfg.model.hum_lw_hprior if optimize_hum else 0.

                # 4. 2d keypoint loss, not used by default
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
                    j2d_proj = front_cam.transform_points_ndc(j3d)  # ndc space: x-left, y-up
                    j2d_proj_cv = (1 - j2d_proj) / 2.  # convert to opencv convention and normalized to 0-1
                    j2d_op = torch.stack(batch['joints2d'], 0).to(self.device)
                    loss_j2d = (torch.sum((j2d_op - j2d_proj_cv) ** 2, -1) * j2d_op[:, :, 2]).mean() * cfg.model.hum_lw_kpts

                # Contact HOI loss
                loss_cont = self.compute_contact_loss(batch, cfg, hum_pts_aeout_hoi, obj_pts_live, weight_vis)

                loss = chamf_obj + loss_edt + loss_mask + loss_temp_r + loss_temp_s + loss_temp_t + loss_cd + loss_temp_h + loss_hpr + loss_bpr + loss_cont + loss_j2d

                log_dict = {
                    'lr_h': optimizer_hum.param_groups[0]["lr"],
                    'lr_o': optimizer_obj.param_groups[0]["lr"],
                    'step': train_state.step,
                    'loss_cd_h': loss_cd,
                    'loss_cd_o': chamf_obj.item(),
                    'loss_mask': loss_mask,
                    "loss_edt": loss_edt,
                    "loss_tT": loss_temp_t,
                    'loss_tR': loss_temp_r,
                    'loss_ts': loss_temp_s,
                    'loss_th': loss_temp_h,
                    'loss_cont': loss_cont,
                    'loss_bpr': loss_bpr,
                    'loss_hpr': loss_hpr,
                    'loss_j2d': loss_j2d
                }

                metric_logger.update(**log_dict)
                if cfg.logging.wandb:
                    wandb.log(log_dict, step=train_state.step)

                if torch.isnan(loss).any():
                    print("Found NAN in loss, stop training.")
                    return

                loss.backward()
                optimizer_obj.step()
                scheduler_obj.step()

                if optimize_hum:
                    optimizer_hum.step()
                    scheduler_hum.step()
                    optimizer_hum.zero_grad()
                optimizer_obj.zero_grad()

                if train_state.step% 250 == 0:
                    # save state
                    self.save_checkpoint_hoi(cfg, None, latent_obj, obj_rot_axis, obj_scale, obj_trans,
                                             optimizer_hum,
                                             optimizer_obj, output_dir, scheduler_hum, scheduler_obj, seq_name,
                                             train_state, batches_all, data_dict)

                if train_state.step % 1000 == 0:
                    self.render_video(batches, latent_obj, human_ae, output_dir, seq_name, train_state, obj_trans,
                                      obj_rot_axis, obj_scale,
                                      show_gt=cfg.run.sample_mode == 'gt',
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      latent_hum=None, hum_scale=hum_scale, hum_trans=hum_trans, data_dict=data_dict)

                if train_state.step >= cfg.run.max_steps or (optimizer_obj.param_groups[0]['lr'] < 1e-8 and train_state.step > 2000):
                    import datetime
                    print(f'Ending training at: {datetime.datetime.now()}, lr={optimizer_obj.param_groups[0]["lr"]}')
                    print(f'Final train state: {train_state}')
                    self.save_checkpoint_hoi(cfg, None, latent_obj, obj_rot_axis, obj_scale, obj_trans,
                                             optimizer_hum,
                                             optimizer_obj, output_dir, scheduler_hum, scheduler_obj, seq_name,
                                             train_state, batches_all, data_dict)

                    # visualize before finish
                    batches = dataloader if len(batches) <= 1 else batches  # make sure we visualize full sequence
                    self.render_video(batches, latent_obj, human_ae, output_dir, seq_name, train_state, obj_trans,
                                      obj_rot_axis, obj_scale,
                                      show_gt=cfg.run.sample_mode == 'gt',
                                      show_recon='recon' in cfg.run.sample_mode, save_pc=True,
                                      latent_hum=None, hum_scale=hum_scale, hum_trans=hum_trans,
                                      data_dict=data_dict)
                    if cfg.logging.wandb:
                        wandb.finish()
                        time.sleep(10)
                    return
                train_state.step += 1
            train_state.epoch += 1

    def save_checkpoint_hoi(self, cfg, latent_hum, latent_obj, obj_rot_axis, obj_scale, obj_trans, optimizer_hum, optimizer_obj,
                            output_dir, scheduler_hum, scheduler_obj, seq_name, train_state, batch, data_dict=None):
        ""
        # save state
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        obj_pts_can = self.decode_can_pts(1, latent_obj, self.model)
        poses_hum, betas_hum, transl_smplh = data_dict['poses_hum'], data_dict['betas_hum'], data_dict['transl_smplh']
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
            # smpl params
            'poses_hum': poses_hum,
            'betas_hum': betas_hum,
            'transl_smpl': transl_smplh,
            'hum_trans': hum_trans,
            'hum_scale': hum_scale,
            'scales_smplh_aligned': batch['scales_smplh_aligned'],
            'centers_smplh_aligned': batch['centers_smplh_aligned'],
            "optimizer_hum": optimizer_hum.state_dict(),
            'scheduler_hum': scheduler_hum.state_dict(),

            # normalization parameters
            'frame_indices': torch.cat(batch['frame_index']),
            "center_hum": torch.stack(batch['cent_hum_pred'], 0),
            "radius_hum": torch.stack(batch['radius_hum_pred'], 0),
            'cent_obj_pred': torch.stack(batch['cent_obj_pred']),
            'radius_obj_pred': torch.stack(batch['radius_obj_pred']),

            # canonical object point
            'obj_pts_canonical': obj_pts_can.detach().cpu(),
            "init_frame_index": data_dict['init_frame_index'],
            'rot_pred': torch.stack(batch['rot_pred'], 0),  # network predicted object pose
            'image_path': batch['image_path']
        }
        ckpt_file = output_dir / f'checkpoint-{seq_name}_k{cfg.dataset.cam_id}.pth'  # ckpt+seq_name
        os.makedirs(osp.dirname(ckpt_file), exist_ok=True)
        torch.save(ckpt_dict, ckpt_file)
        print('checkpoint saved to', ckpt_file)

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



@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    cfg.run.job = 'sample' if cfg.run.job == 'train' else cfg.run.job
    cfg.run.freeze_feature_model = False
    cfg.dataloader.num_workers = 2 # reduce initialization waiting time
    cfg.dataset.load_obj_pose = True
    wandb_cache = cfg.logging.wandb
    trainer = TrainerHOISMPLOpt(cfg)
    cfg.logging.wandb = wandb_cache
    import traceback
    try:
        trainer.run_sample(cfg)
    except Exception as e:
        print(traceback.print_exc())



if __name__ == '__main__':
    main()

