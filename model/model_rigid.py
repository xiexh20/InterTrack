"""
for rigid object diffusion

"""
import glob
import time
from typing import Optional, List

import cv2
import numpy as np
import torch
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
from torch.nn.modules.module import T
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
from .point_cloud_model import PointCloudModel

from .model_diff_data import ConditionalPCDiffusionBehave
from .geometry_utils import rotmat_to_6d, rot6d_to_rotmat, quat_to_rotmat, rotation_matrix_to_quaternion
from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix, quaternion_to_matrix
from pytorch3d.transforms import so3_log_map, so3_exp_map


def get_pose_in_dim(smpl_cond_type):
    if smpl_cond_type in ['theta', 'theta-local']:
        pose_in_dim = 144
    elif smpl_cond_type in ['joints']:
        pose_in_dim = 23 * 3  # SMPL body joints
    elif smpl_cond_type in ['joints25']:
        pose_in_dim = 25 * 3
    elif smpl_cond_type in ['joints25+grot']:
        pose_in_dim = 25 * 3 + 6  # global rotation + joint orientation
    elif smpl_cond_type in ['theta+joints25']:
        pose_in_dim = 25 * 3 + 144
    elif smpl_cond_type in ['joints25+velo']:
        pose_in_dim = 25 * 3 * 2
    elif smpl_cond_type in ['joints25+velo+grot']:
        pose_in_dim = 25 * 3 * 2 + 6
    elif smpl_cond_type in ['joints25+velo+theta']:
        pose_in_dim = 25 * 3 * 2 + 144
    elif smpl_cond_type in ['joints25+velo+theta+avelo']:
        pose_in_dim = 25 * 3 * 2 + 144 + 72  # angular velocity of the rotations
    elif smpl_cond_type in ['none']:
        pose_in_dim = 6  # simply some dummy data
    else:
        raise NotImplementedError
    print(f"SMPL conditioning type={smpl_cond_type}")
    return pose_in_dim

class RigidShapeDiffusion(ConditionalPCDiffusionBehave):
    def forward(self, batch, mode: str = 'train', **kwargs):
        ""
        if mode == 'train':
            # the batch data contain numpy array
            images = torch.from_numpy(np.stack(batch['images'], 0)).float().to('cuda')
            masks = torch.from_numpy(np.stack(batch['masks'], 0)).to('cuda')
            dt = torch.from_numpy(np.stack(batch['dist_transform'], 0)).to('cuda')
            cameras = [PerspectiveCameras(
                R=torch.from_numpy(batch['R'][i]).to('cuda').float(),
                T=torch.from_numpy(batch['T'][i]).to('cuda').float(),
                K=torch.from_numpy(batch['K'][i]).to('cuda').float(),
                device='cuda',
                in_ndc=True
            ) for i in range(len(batch['images']))]
            rela_poses = torch.from_numpy(np.stack(batch['rela_poses'], 0)).to('cuda').float()
            abs_poses = torch.from_numpy(np.stack(batch['abs_poses'], 0)).to('cuda').float()
            occ_ratios = torch.from_numpy(np.stack(batch['occ_ratios'], 0)).to('cuda').float()
            smpl_poses = torch.from_numpy(np.stack(batch['smpl_poses'], 0)).to('cuda').float()
            smpl_joints = torch.from_numpy(np.stack(batch['smpl_joints'], 0)).to('cuda').float()
            body_joints25 = torch.from_numpy(np.stack(batch['body_joints25'], 0)).to('cuda').float()
            frame_indices = torch.from_numpy(np.stack(batch['frame_indices'], 0)).to('cuda').float()
            return self.forward_train(pc=batch['pclouds'],
                                      camera=cameras, 
                                      image_rgb=images,
                                      mask=masks,
                                      rela_poses=rela_poses,
                                      dist_transform=dt,
                                      abs_poses=abs_poses,
                                      occ_ratios=occ_ratios,
                                      smpl_poses=smpl_poses,
                                      smpl_joints=smpl_joints,
                                      body_joints25=body_joints25,
                                      frame_indices=frame_indices,
                                      **kwargs)
        elif mode in ['sample', 'abspose', 'pred-abs', 'pred-rela', 'pred-rela-t']:
            images = torch.stack(batch['images'], 0).to('cuda')
            masks = torch.stack(batch['masks'], 0).to('cuda')
            dt = torch.stack(batch['dist_transform'], 0).to('cuda')
            cameras = [PerspectiveCameras(
                R=batch['R'][i],
                T=batch['T'][i],
                K=batch['K'][i],
                device='cuda',
                in_ndc=True
            ) for i in range(len(batch['images']))]
            num_points = kwargs.pop('num_points', batch['pclouds'][0].shape[0])
            rela_poses = torch.stack([x.to(self.device) for x in batch['rela_poses']])
            abs_poses = torch.stack([x.to(self.device) for x in batch['abs_poses']]) # B, T, 4, 4
            occ_ratios = torch.stack(batch['occ_ratios'], 0).to('cuda')
            smpl_poses = torch.stack(batch['smpl_poses'], 0).to('cuda')
            smpl_joints = torch.stack(batch['smpl_joints'], 0).to('cuda')
            body_joints25 = torch.stack(batch['body_joints25'], 0).to('cuda')
            frame_indices = torch.stack(batch['frame_indices'], 0).to('cuda')
            # ret_rot = kwargs.get('ret_rot', False)
            rela_trans = rela_poses[:, :, :3, 3]
            assert torch.allclose(rela_trans, torch.zeros_like(rela_trans, device=rela_trans.device), atol=1e-5), 'not all zeros!'
            if mode == 'abspose':
                abs_poses[:, :, :3, 3] = 0 # reset all translation
                return self.forward_sample(num_points=num_points,
                                           camera=cameras,
                                           image_rgb=images,
                                           mask=masks,
                                           rela_poses=abs_poses, # diffuse same pc at canonical space
                                           dist_transform=dt,
                                           gt_pc=batch['pclouds'],
                                           abs_poses=abs_poses,
                                           occ_ratios=occ_ratios,
                                           smpl_poses=smpl_poses,
                                           smpl_joints=smpl_joints,
                                           body_joints25=body_joints25,
                                           frame_indices=frame_indices,
                                           **kwargs)
            elif mode == 'pred-abs':
                # use predicted poses as relative pose as input
                pred_poses = torch.from_numpy(np.stack(batch['pred_poses'], 0)).to('cuda').float()
                return self.forward_sample(num_points=num_points,
                                           camera=cameras,
                                           image_rgb=images,
                                           mask=masks,
                                           rela_poses=pred_poses,  # diffuse same pc at canonical space
                                           dist_transform=dt,
                                           gt_pc=batch['pclouds'],
                                           abs_poses=abs_poses,
                                           occ_ratios=occ_ratios,
                                           smpl_poses=smpl_poses,
                                           smpl_joints=smpl_joints,
                                           body_joints25=body_joints25,
                                           frame_indices=frame_indices,
                                           **kwargs)
            elif mode == 'pred-rela':
                print("Using estimated rotation + GT translation!")
                # use predicted poses as relative pose as input
                pred_poses = torch.from_numpy(np.stack(batch['pred_poses'], 0)).to('cuda').float()
                # compute a relative pose
                pred_poses_rela = torch.matmul(pred_poses, pred_poses[:, 0:1].transpose(-2, -1))
                return self.forward_sample(num_points=num_points,
                                           camera=cameras,
                                           image_rgb=images,
                                           mask=masks,
                                           rela_poses=pred_poses_rela,  # diffuse same pc at canonical space
                                           dist_transform=dt,
                                           gt_pc=batch['pclouds'],
                                           abs_poses=abs_poses,
                                           occ_ratios=occ_ratios,
                                           smpl_poses=smpl_poses,
                                           smpl_joints=smpl_joints,
                                           body_joints25=body_joints25,
                                           frame_indices=frame_indices,
                                           **kwargs)
            elif mode == 'pred-rela-t':
                print("Using estimated rotation + translation!")
                # translation is also predicted
                pred_poses = torch.from_numpy(np.stack(batch['pred_poses'], 0)).to('cuda').float()
                # compute a relative pose
                pred_poses_rela = torch.matmul(pred_poses, pred_poses[:, 0:1].transpose(-2, -1))
                # Use predicted translation
                cameras_pred = [PerspectiveCameras(
                    R=batch['R'][i],
                    T=torch.from_numpy(batch['T_obj_scaled'][i]).float(),
                    K=batch['K'][i],
                    device='cuda',
                    in_ndc=True
                ) for i in range(len(batch['images']))]
                return self.forward_sample(num_points=num_points,
                                           camera=cameras_pred,
                                           image_rgb=images,
                                           mask=masks,
                                           rela_poses=pred_poses_rela,  # diffuse same pc at canonical space
                                           dist_transform=dt,
                                           gt_pc=batch['pclouds'],
                                           abs_poses=abs_poses,
                                           occ_ratios=occ_ratios,
                                           smpl_poses=smpl_poses,
                                           smpl_joints=smpl_joints,
                                           body_joints25=body_joints25,
                                           frame_indices=frame_indices,
                                           **kwargs)
            else:
                return self.forward_sample(num_points=num_points,
                                          camera=cameras,
                                          image_rgb=images,
                                          mask=masks,
                                          rela_poses=rela_poses,
                                           dist_transform=dt,
                                           gt_pc=batch['pclouds'],
                                           abs_poses=abs_poses,
                                           occ_ratios=occ_ratios,
                                           smpl_poses=smpl_poses,
                                           smpl_joints=smpl_joints,
                                           body_joints25=body_joints25,
                                           frame_indices=frame_indices,
                                           **kwargs)
        else:
            raise NotImplementedError

    def forward_train(
        self,
        pc: List[Tensor],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        """
        the pc is the template object points

        Parameters
        ----------
        pc : a list of points of shape (N, 3) 
        camera : a list of cameras
        image_rgb : (B, T, 3, H, W)
        mask : (B, T, 2, H, W)
        return_intermediate_steps :
        kwargs : include object rotation parameters

        Returns
        -------

        """
        assert self.consistent_center
        assert not return_intermediate_steps
        pc_feats, noises = [], []
        B, T = image_rgb.shape[:2]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        # timesteps = torch.zeros(B, device=self.device, dtype=torch.long)
        rela_poses = kwargs.get('rela_poses', None).clone()
        rela_poses[:, :, :3, 3] *= self.scale_factor # scale to 7
        assert rela_poses is not None
        # print(B, T)
        frame_indices = kwargs.get('frame_indices')
        occ_ratios = kwargs.get('occ_ratios')
        for i in range(len(pc)):
            # print(i, rela_poses[i]) # clip=1, all identity
            pc_i = torch.from_numpy(pc[i]).to(self.device) * self.scale_factor # (N, 3)
            # first diffuse, then transform to each local frame, then aggregate
            noise = torch.randn_like(pc_i)
            noise = noise - torch.mean(noise, dim=0, keepdim=True)
            # print(noise.shape, torch.mean(noise, 0)) # the noise is not centered properly!
            x_t = self.scheduler.add_noise(pc_i, noise, timesteps[i])[None].repeat(T, 1, 1)
            # transform to local
            xt_local = torch.matmul(x_t, rela_poses[i, :, :3, :3].transpose(1, 2)) + rela_poses[i, :, :3, 3].unsqueeze(-2)
            xt_input = self.get_point_feats(camera, i, image_rgb, kwargs, mask, xt_local)
            feat_agg = self.aggregate_features(x_t, xt_input, frame_indices[i], occ_ratios[i])
            pc_feats.append(feat_agg) # do not average the coordinates
            noises.append(noise)
            # print(timesteps[i], torch.mean(xt_input[:, :, :3], 1)) # Now it is centered properly.
        xt_feat = torch.stack(pc_feats, 0)
        noise_pred = self.point_cloud_model(xt_feat, timesteps)
        noise_pred = noise_pred - torch.mean(noise_pred, dim=1, keepdim=True)
        noises = torch.stack(noises, 0)

        loss = F.mse_loss(noise_pred, noises)
        return loss

    def aggregate_features(self, x_t, xt_input, frame_indices, occ_ratios=None):
        """

        Parameters
        ----------
        x_t : (T, N, 3), point coordinates
        xt_input : (T, N, D), per point features, the first 3 channels in the last dimension are the coordinates
        frame_indices: (T, ), the index for the images in a full sequence
        occ_ratios: (T, ), the object visibility information

        Returns (N, 3), aggregated features to one point cloud
        -------

        """
        if self.human_feat in ['add', 'masked']:
            # simply average
            feat_agg = torch.cat([x_t[0], torch.mean(xt_input[:, :, 3:], 0)], 1)
        elif self.human_feat in ['posenc+mlp', 'posenc+mlp-mask']:
            # use mlp to aggregate, and also add positional encoding
            ind_rela = frame_indices - frame_indices[0]
            T, N, D = xt_input.shape
            pos_feat = self.posi_encoder_frames(ind_rela)[:, None].repeat(1, N, 1) # T -> (T, 64) -> (T, N, 64)
            feat_sep = self.mlp_shared(torch.cat([xt_input, pos_feat], -1))
            feat_sum = feat_sep.sum(0) # -> (N, D)
            feat_agg = self.mlp_aggregate(feat_sum)
            feat_agg = torch.cat([x_t[0], feat_agg], 1) # N, D
        elif self.human_feat in ['posenc+mlp-avg', 'posenc+mlp-avg-mask']:
            # use mlp to aggregate, and also add positional encoding
            ind_rela = frame_indices - frame_indices[0]
            T, N, D = xt_input.shape
            pos_feat = self.posi_encoder_frames(ind_rela)[:, None].repeat(1, N, 1)  # T -> (T, 64) -> (T, N, 64)
            feat_sep = self.mlp_shared(torch.cat([xt_input, pos_feat], -1))
            feat_mean = feat_sep.mean(0)  # -> (N, D)
            feat_agg = self.mlp_aggregate(feat_mean)
            feat_agg = torch.cat([x_t[0], feat_agg], 1)  # N, D
        elif self.human_feat in ['posenc+mlp-avg-vis', 'posenc+mlp-avg-vis-mask']:
            ind_rela = frame_indices - frame_indices[0]
            T, N, D = xt_input.shape
            pos_feat = self.posi_encoder_frames(ind_rela)[:, None].repeat(1, N, 1)  # T -> (T, 64) -> (T, N, 64)
            occ_repeat = occ_ratios[:, None, None].repeat(1, N, 1)
            feat_sep = self.mlp_shared(torch.cat([xt_input, pos_feat, occ_repeat], -1))
            feat_mean = feat_sep.mean(0)  # -> (N, D)
            feat_agg = self.mlp_aggregate(feat_mean)
            feat_agg = torch.cat([x_t[0], feat_agg], 1)  # N, D

        else:
            raise NotImplementedError

        return feat_agg

    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "add additional MLP layers if needed"
        from .topnet import GaussianFourierProjection
        # init point diffusion model
        super().init_pcloud_model(kwargs, point_cloud_model, point_cloud_model_embed_dim)
        if self.human_feat in ['posenc+mlp', 'posenc+mlp-avg', 'posenc+mlp-avg-vis',
                               'posenc+mlp-mask', 'posenc+mlp-avg-mask', 'posenc+mlp-avg-vis-mask']:
            if self.human_feat in ['posenc+mlp-avg-vis', 'posenc+mlp-avg-vis-mask']:
                feat_dim, out_dim = self.in_channels + 64 + 1, self.in_channels - 3 # also add one channel for visibility
            else:
                feat_dim, out_dim = self.in_channels + 64, self.in_channels - 3

            # add MLP and positional encoding layer
            self.posi_encoder_frames = nn.Sequential(
                GaussianFourierProjection(embed_dim=64),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
            )
            # feat_dim = self.in_channels + 64 # with positional encoding
            self.mlp_shared = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),  # pose dimension is 6
                nn.LeakyReLU(),
                nn.Linear(feat_dim, out_dim), # need to concate point features in the end
                nn.LeakyReLU(),
            )
            # MLP layer after summing all features
            self.mlp_aggregate = nn.Sequential(
                nn.Linear(out_dim, out_dim),  # pose dimension is 6
                nn.LeakyReLU()
            )
            print(f"MLP and positional encoding initialized! feature mode: {self.human_feat}")

    def get_point_feats(self, camera, i, image_rgb, kwargs, mask, xt_local, use_cache=False):
        """
        get point feature of one chunk in the batch

        Parameters
        ----------
        camera :
        i :
        image_rgb :
        kwargs :
        mask :
        xt_local :

        Returns (B, N, D), per point features
        -------

        """
        if self.human_feat in ['add', 'posenc+mlp', 'posenc+mlp-avg', 'posenc+mlp-avg-vis']:
            xt_input = self.get_input_with_conditioning(xt_local, camera[i], image_rgb[i], mask[i],
                                                    None, kwargs.get('dist_transform')[i], use_cache)
        elif self.human_feat in ['masked', 'posenc+mlp-mask', 'posenc+mlp-avg-mask', 'posenc+mlp-avg-vis-mask']:
            # print("Masking out non-obj features")
            # mask the features where human mask is presented (=1)
            xt_input = self.get_input_with_conditioning(xt_local, camera[i], image_rgb[i], mask[i],
                                                        None, kwargs.get('dist_transform')[i])
            assert not self.use_global_features
            hum_cidx = 3 + 3 + self.feature_model.feature_dim
            hum_mask = xt_input[:, :, hum_cidx:hum_cidx+1] < 0.5 # (B, N, 1)
            # mask out points that are projected to the human mask
            xt_input[:, :, 6:6+self.feature_model.feature_dim] = xt_input[:, :, 6:6+self.feature_model.feature_dim] * hum_mask
            # print("Number of human masks:", torch.sum(hum_mask, 1))

            # debug: save the points and see
            # import trimesh
            # out = '/BS/xxie-2/work/pc2-diff/experiments/debug/meshes/rigid'
            # fcount = len(glob.glob(out + "/pts_*.ply"))
            # for x in range(len(hum_mask)):
            #     pts = xt_input[x, :, :3].cpu().numpy()
            #     mask_i = hum_mask[x, :, 0].cpu().numpy()
            #     vc = np.zeros_like(pts) + np.array([1., 0, 0.])
            #     vc[mask_i, :] = np.array([0, 1., 0.]) # green human, red obj
            #     trimesh.PointCloud(pts, colors=vc).export(out + f'/pts_{fcount}_{x}.ply')
            #     cv2.imwrite(out + f'/pts_{fcount}_{x}.png', (image_rgb[i, x].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)[:, :, ::-1])


        else:
            raise NotImplementedError
        return xt_input

    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            **kwargs
    ):
        """
        at each step, transform points to local, do projection and feature agg, then run diffusion

        """
        print(f"Reverse diffusion scheduler={scheduler}, eta={eta}.")
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        # B = 1 if image_rgb is None else image_rgb.shape[0]
        B, T= image_rgb.shape[:2]
        D = self.get_x_T_channel()
        device = self.device if image_rgb is None else image_rgb.device

        x_t = self.initialize_x_T(device, gt_pc, (B, N, D), -1, scheduler)

        # Set timesteps
        extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor
        frame_indices = kwargs.get('frame_indices') # abs frame indices
        occ_ratios = kwargs.get('occ_ratios')
        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))

            # Conditioning
            xt_feats = []
            for j in range(B):
                xt_local = torch.matmul(x_t[j:j+1].repeat(T, 1, 1), rela_poses[j, :, :3, :3].transpose(1, 2)) + rela_poses[j, :, :3,
                                                                                         3].unsqueeze(-2)
                xt_input = self.get_point_feats(camera, j, image_rgb, kwargs, mask, xt_local)
                # xt_feats.append(torch.cat([xt_local[0], torch.mean(xt_input[:, :, 3:], 0)], 1)) # only average per point features.
                # all the input shoud have been centered
                # print(torch.mean(xt_input[:, :, :3], 1))
                feat_agg = self.aggregate_features(x_t, xt_input, frame_indices[j], occ_ratios[j])
                xt_feats.append(feat_agg)
            xt_feats = torch.stack(xt_feats, 0)

            inference_binary = (i == len(progress_bar) - 1) | add_interm_output
            # One reverse step with conditioning
            x_t = self.reverse_step(extra_step_kwargs, scheduler, t, x_t, xt_feats,
                                    inference_binary=inference_binary)  # (B, N, D), D=3 or 4

            # Append to output list if desired
            if add_interm_output:
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(x_t, denormalize=True,
                                            unscale=True)  # this convert the points back to original scale
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs]

        return (output, all_outputs) if return_all_outputs else output


class CondPairConsistentDiffusion(ConditionalPCDiffusionBehave):
    def forward(self, batch, mode: str = 'train', **kwargs):
        images = torch.stack(batch['images'], 0).to('cuda')
        masks = torch.stack(batch['masks'], 0).to('cuda')

        if mode == 'train':
            pc = [Pointclouds([x[i].to('cuda') for x in batch['pclouds']]) for i in range(2)]
            camera = [
                PerspectiveCameras(
                R=torch.stack([x[i] for x in batch['R']]),
                T=torch.stack([x[i] for x in batch['T']]),
                K=torch.stack([x[i] for x in batch['K']]),
                device='cuda',
                in_ndc=True
            ) for i in range(2)
            ]
            grid_df = torch.stack(batch['grid_df'], 0).to('cuda') if 'grid_df' in batch else None
            return self.forward_train(
                pc=pc,
                camera=camera,
                image_rgb=images,
                mask=masks,
                grid_df=grid_df,
                **kwargs)
        elif mode == 'sample':
            pc = self.get_input_pc(batch)
            camera = PerspectiveCameras(
                R=torch.stack(batch['R']),
                T=torch.stack(batch['T']),
                K=torch.stack(batch['K']),
                device='cuda',
                in_ndc=True
            )
            num_points = kwargs.pop('num_points', len(batch['pclouds'][0]))
            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                **kwargs)
        else:
            raise NotImplementedError

    def forward_train(
        self,
        pc: List[Pointclouds],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        """
        different data format for train and sample
        Parameters
        ----------
        pc :
        camera :
        image_rgb : (B, 2, 3,H,W) # 2 images are nearby frames
        mask :
        return_intermediate_steps :
        kwargs :

        Returns
        -------

        """
        assert self.consistent_center
        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc[0], normalize=True, scale=True)  # this will not pack the point colors
        B, N, D = x_0.shape
        # print("Point cloud shape:", x_0.shape)

        # Sample random noise
        noise = torch.randn_like(x_0)
        noise = noise - torch.mean(noise, dim=1, keepdim=True)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,),
                                 device=self.device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x_0, noise, timestep)
        x_t_input = self.get_diffu_input(camera[0], image_rgb[:, 0], mask[:, 0], timestep, x_t, **kwargs)
        noise_pred0 = self.point_cloud_model(x_t_input, timestep)

        x_0_1 = self.point_cloud_to_tensor(pc[1], normalize=True, scale=True)
        x_t = self.scheduler.add_noise(x_0_1, noise, timestep)
        x_t_input = self.get_diffu_input(camera[1], image_rgb[:, 1], mask[:, 1], timestep, x_t, **kwargs)
        noise_pred1 = self.point_cloud_model(x_t_input, timestep)

        loss_1 = F.mse_loss(noise_pred0, noise)
        loss_2 = F.mse_loss(noise_pred1, noise)

        return loss_1 + loss_2, torch.tensor([loss_1, loss_2])



class ObjectCombinedDiffusion(RigidShapeDiffusion):
    def forward(self, batch, mode: str = 'train', **kwargs):
        "input points_all, so we have a uniform representation for human and object"
        images = torch.stack(batch['images'], 0).to('cuda')
        masks = torch.stack(batch['masks'], 0).to('cuda')
        dt = torch.stack(batch['dist_transform'], 0).to('cuda')
        cameras = [PerspectiveCameras(
            R=batch['R'][i],
            T=batch['T'][i],
            K=batch['K'][i],
            device='cuda',
            in_ndc=True
        ) for i in range(len(batch['images']))]
        num_points = kwargs.pop('num_points', batch['pclouds'][0].shape[0])
        rela_poses = torch.stack([x.to(self.device) for x in batch['rela_poses']])
        occ_ratios = torch.stack(batch['occ_ratios'], 0).to('cuda')
        if mode == 'train':
            return self.forward_train(pc=batch['points_all'], # a list of (T, N, 3) tensor
                                      camera=cameras,
                                      image_rgb=images,
                                      mask=masks,
                                      rela_poses=rela_poses,
                                      dist_transform=dt,
                                      occ_ratios=occ_ratios,
                                      **kwargs)
        elif mode == 'sample':
            return self.forward_sample(num_points=num_points,
                                       camera=cameras,
                                       image_rgb=images,
                                       mask=masks,
                                       rela_poses=rela_poses,
                                       dist_transform=dt,
                                       occ_ratios=occ_ratios,
                                       **kwargs)
        else:
            raise NotImplementedError

    def forward_train(
        self,
        pc: List[Tensor],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        """
        the pc is the template object points

        Parameters
        ----------
        pc : a list of points of shape (N, 3)
        camera : a list of cameras
        image_rgb : (B, T, 3, H, W)
        mask : (B, T, 2, H, W)
        return_intermediate_steps :
        kwargs : include object rotation parameters

        Returns
        -------

        """
        assert self.consistent_center
        assert not return_intermediate_steps
        pc_feats, noises = [], []
        B, T = image_rgb.shape[:2]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        # timesteps = torch.zeros(B, device=self.device, dtype=torch.long) + 3
        # rela_poses = kwargs.get('rela_poses', None).clone()
        # rela_poses[:, :, :3, 3] *= self.scale_factor # scale to 7
        # assert rela_poses is not None
        # print(B, T)
        # losses = 0
        xt_feats, noises = [], []
        # pc_all = kwargs.get('points_all')
        for i in range(len(pc)):
            # print(i, rela_poses[i]) # clip=1, all identity
            pc_i = pc[i].to(self.device) * self.scale_factor # (T, N, 3)
            T, N = pc_i.shape[:2]
            xt_local = pc_i # already T, N, 3, and in local frame coordinate
            noise = torch.randn_like(xt_local[0])[None].repeat(T, 1, 1) # all pts in this clip should have same noise added
            noise = noise - torch.mean(noise, dim=1, keepdim=True)

            x_t = self.scheduler.add_noise(xt_local, noise, timesteps[i]) # add noise to different
            # transform to local
            xt_input = self.get_point_feats(camera, i, image_rgb, kwargs, mask, x_t)
            xt_feats.append(xt_input.reshape(T*N, -1))
            noises.append(noise.reshape(T*N, -1))

        # loss = F.mse_loss(noise_pred, noises)
        noise_pred = self.point_cloud_model(torch.stack(xt_feats, 0), timesteps)
        loss = F.mse_loss(noise_pred, torch.stack(noises, 0))
        return loss

    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            **kwargs
    ):
        """
        at each step, transform points to local, do projection and feature agg, then run diffusion

        """
        print(f"Reverse diffusion scheduler={scheduler}, eta={eta}.")
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        # B = 1 if image_rgb is None else image_rgb.shape[0]
        B, T= image_rgb.shape[:2]
        D = self.get_x_T_channel()
        device = self.device if image_rgb is None else image_rgb.device

        x_t = self.initialize_x_T(device, gt_pc, (T, N, D), -1, scheduler) # fixed sample
        xt_all = x_t[None].repeat(B, 1, 1, 1)

        # Set timesteps
        extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor
        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))

            # Conditioning
            xt_feats = []
            for j in range(B):
                xt_local = xt_all[j].clone()
                xt_feat = self.get_point_feats(camera, j, image_rgb, kwargs, mask, xt_local)
                xt_feats.append(xt_feat.reshape(T*N, -1))

            xt_all = self.reverse_step(extra_step_kwargs, scheduler, t,
                                       xt_all, torch.stack(xt_feats, 0),
                                        inference_binary=False)  # (B, T*N, D)
            # Append to output list if desired
            if add_interm_output:
                all_outputs.append(xt_all)

        # Convert output back into a point cloud, undoing normalization and scaling
        # a list of point clouds
        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        if return_all_outputs:
            # each iterm inside all_outputs: (B, T, N, D)
            all_outputs = torch.stack(all_outputs, dim=2)  # (B, T, sample_steps, N, D)
            all_outputs = [[self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs[x]] for x in range(B)]

        return (output, all_outputs) if return_all_outputs else output

    def reverse_step(self, extra_step_kwargs, scheduler, t, x_t, x_t_input, **kwargs):
        """
        
        Parameters
        ----------
        extra_step_kwargs : 
        scheduler : 
        t : 
        x_t : (B, T, N, 3)
        x_t_input : (B, T*N, D)
        kwargs : 

        Returns (B, T, N, 3), updated 
        -------

        """
        B, T, N = x_t.shape[:3]
        # Forward
        noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))
        noise_pred = noise_pred.reshape(B, T, N, 3)
        if self.consistent_center:
            assert self.dm_pred_type != 'sample', 'incompatible dm predition type for CCD!'
            # suggested by the CCD-3DR paper
            noise_pred = noise_pred - torch.mean(noise_pred, dim=2, keepdim=True)

        # compute a weighted average of current frame prediction and nearby frames
        # if self.temporal_avg != 'none':
        #     noise_pred = self.agg_predictions(noise_pred)
        assert self.temporal_avg == 'none', 'invalid temporal_avg mode!'

        # Step
        if self.fixed_reverse_var:
            x_t = self.fixed_var_step(extra_step_kwargs, noise_pred, scheduler, t, x_t[..., :3])
        else:
            x_t = scheduler.step(noise_pred, t, x_t[..., :3], **extra_step_kwargs).prev_sample

        if self.consistent_center:
            x_t = x_t - torch.mean(x_t, dim=2, keepdim=True)
        return x_t
    
    def fixed_var_step(self, extra_step_kwargs, noise_pred, scheduler, t, x_t):
        "x_t has shape (B, T, N, 3)"
        assert isinstance(scheduler, DDPMScheduler) or isinstance(scheduler, DDIMScheduler), 'invalid scheduler!'
        B, T, N = x_t.shape[:3]
        if self.reverse_var_type == 'fixed':
            variance_noise = self.fixed_samples_t[f'sample{N}_step{t}'].to(x_t.device)[None].repeat(B, T, 1, 1)
        elif self.reverse_var_type == 'predict':
            variance_noise = noise_pred  # suggested by vladmir, just use predicted noise as the variance
        else:
            raise NotImplementedError
        if isinstance(scheduler, DDPMScheduler):
            out, variance = scheduler.step(noise_pred, t, x_t, return_var=True, **extra_step_kwargs)
            prev_sample_no_noise = out.prev_sample - variance  # remove the added variance
            # copy the same variance for all
            if t == 0:
                x_t = out.prev_sample
            else:
                assert scheduler.variance_type not in ["learned", "learned_range"], 'incompatible config!'
                predicted_variance = None
                # add a fixed variance for every step
                if scheduler.variance_type == "fixed_small_log":
                    variance = scheduler._get_variance(t, predicted_variance=predicted_variance) * variance_noise
                else:
                    variance = (scheduler._get_variance(t,
                                                        predicted_variance=predicted_variance) ** 0.5) * variance_noise

                x_t = prev_sample_no_noise + variance
        else:
            # print(f"Using same sample, {variance_noise.shape}, {x_t.shape}") # [16, 4, 6890, 3]), [16, 4, 6890, 3]
            # DDIM: provide noise ourselves
            x_t = scheduler.step(noise_pred, t, x_t[..., :3], variance_noise=variance_noise,
                                 **extra_step_kwargs).prev_sample
        return x_t

    def get_point_feats(self, camera, i, image_rgb, kwargs, mask, xt_local):
        xt_input = super().get_point_feats(camera, i, image_rgb, kwargs, mask, xt_local)
        if self.comb_tdiff_add_feats == 'none':
            return xt_input
        elif self.comb_tdiff_add_feats == 'posenc':
            # print("Adding positional encoding for the points")
            # positional encoding of the relative position of points in this clip
            T, N = xt_input.shape[:2]
            pos = self.posi_encoder(torch.arange(0, T, device=xt_input.device)[:, None]/T) # (T, 2*10+1)
            xt_input = torch.cat([xt_input, pos[:, None].repeat(1, N, 1)], -1) # (T, N, D+21)
            return xt_input
        else:
            raise NotImplementedError

    def add_extra_input_chennels(self, input_channels):
        ""
        if self.comb_tdiff_add_feats == 'posenc':
            return input_channels + 21
        else:
            return input_channels


class RigidRotationOnlyModel6D(RigidShapeDiffusion):
    "diffuse rotation only, use 6D representation"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "a transformer to predict rotation"
        from .topnet import RotationModel
        embed_dim = 512
        input_dim = 128 + 64 + 64 + self.feature_model.feature_dim + 1 # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = RotationModel(input_dim, embed_dim)

    # def train(self: T, mode: bool = True) -> T:
    #     ""

    def forward_train(
        self,
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        "use relative pose only"
        B, T = image_rgb.shape[:2]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        rela_poses = kwargs.get('rela_poses', None).clone()
        rot6d = rotmat_to_6d(rela_poses[:, :, :3, :3].reshape(B*T, 3, 3)).reshape(B, T, 6).to(self.device)
        noise = torch.randn(B, 1, 6, device=self.device)
        noise = (noise - torch.mean(noise, 0)).repeat(1, T, 1)
        x_t = self.scheduler.add_noise(rot6d, noise, timesteps)

        # get image global features
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        pred = self.point_cloud_model(x_t, img_feats, timesteps)
        
        if self.dm_pred_type == 'epsilon':
            loss = F.mse_loss(noise, pred)
        elif self.dm_pred_type == 'sample':
            rot_mat = rot6d_to_rotmat(pred.reshape(B*T, 6))
            rot6d_pre = rotmat_to_6d(rot_mat).reshape(B, T, 6)
            loss = F.mse_loss(rot6d_pre, rot6d)
        else:
            raise NotImplementedError
        return loss

    def extract_features(self, image_rgb, mask, occ_ratios=None):
        "return (bs, T, D)"
        feats = []
        for j, img in enumerate(image_rgb):
            feat = self.feature_model(img, return_type='cls_token', return_upscaled_features=False)  # (T, feat_dim)
            # feat = self.feature_model(img, return_type='feat_avg', return_upscaled_features=False)
            # also compute object mask: subtitute of visibility
            h, w = mask.shape[-2:]
            if occ_ratios is None:
                obj_ratio = torch.sum(mask[j, :, 1], dim=(-1, -2)) * 1.5 * 1.5 / (h * w)   # 1.5 accounts for bbox expansion
                print("Warning: using estimated visibility ratio")
            else:
                obj_ratio = occ_ratios[j].clone() # (B, T)->(T)
            # print(f'h={h}, w={w}, object mask area ratio:', obj_ratio)
            feats.append(torch.cat([feat, obj_ratio[:, None]], -1))
        img_feats = torch.stack(feats)
        return img_feats

    def forward_sample(
        self,
        num_points: int,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            **kwargs
    ):
        ""
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        B, T = image_rgb.shape[:2]
        # x_t = torch.randn(B, T, 6, device=self.device)
        x_t = torch.randn(B, 1, 6, device=self.device).repeat(1, T, 1)

        # Set timesteps
        extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(self.device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        # GT translation and rotation
        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor

        # do manual casting, ref: https://github.com/TimDettmers/bitsandbytes/issues/240#issuecomment-1761692886
        # need to set run.mixed_precision=no as well!!!
        for param in self.point_cloud_model.parameters():
            # Check if parameter dtype is  Float (float16)
            if param.dtype == torch.float16:
            # if param.dtype == torch.float32:
                param.data = param.data.to(torch.float32)
                # print('changing param to float16')

        # Conditioning, onle need to compute once
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))

            # with torch.autocast("cuda", dtype=torch.float16, enabled=True): # autocast datatypes, not working! need to disable mixed precision
            # noise_pred = self.point_cloud_model(x_t.to(torch.float16), img_feats.to(torch.float16), t.reshape(1).expand(B).to(torch.float16))
            noise_pred = self.point_cloud_model(x_t, img_feats, t.reshape(1).expand(B))
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
            # Append to output list if desired
            if add_interm_output:
                xt_all = self.transform_points(gt_pc, rela_poses, x_t)
                all_outputs.append(torch.stack(xt_all))

        # compute output from predicted x_t, same format as other model for better evaluation
        xt_all = self.transform_points(gt_pc, rela_poses, x_t)
        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        if return_all_outputs:
            # each iterm inside all_outputs: (B, T, N, D)
            all_outputs = torch.stack(all_outputs, dim=2)  # (B, T, sample_steps, N, D)
            all_outputs = [[self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs[x]] for x in range(B)]

        return (output, all_outputs) if return_all_outputs else output

    def transform_points(self, gt_pc, rela_poses, x_t):
        """
        use predicted x_t (pose) to transform GT points
        Parameters
        ----------
        gt_pc :
        rela_poses :
        x_t : (B, T, 6)

        Returns
        -------

        """
        B, T = x_t.shape[:2]
        rotmat = rot6d_to_rotmat(x_t.reshape(B * T, 6))
        poses = rela_poses.clone()
        poses[:, :, :3, :3] = rotmat.reshape(B, T, 3, 3)
        xt_all = []
        for j in range(B):
            pc_i = gt_pc[j].to(self.device) * self.scale_factor
            xt_local = torch.matmul(pc_i.repeat(T, 1, 1),
                                    poses[j, :, :3, :3].transpose(1, 2)) + poses[j, :, :3, 3].unsqueeze(-2)
            xt_all.append(xt_local)
        return xt_all
        

class RigidSO3Diffusion(RigidRotationOnlyModel6D):
    def estimate_velocity(self, FPS, j25):
        """
        estimate velocity of joints
        reference: https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py#L309
        Parameters
        ----------
        FPS : scalar, the fps of the data
        j25 : (B, T, ...)

        Returns (B, T, ...)
        -------

        """
        velo = (j25[:, 2:] - j25[:, :-2]) / (2 / FPS)  # follow humor
        velo = torch.cat([velo[:, 0:1], velo, velo[:, -1:]], 1)  # duplicate last and first velocity
        return velo

    def estimate_angular_velocity(self, FPS, rotmat):
        """
        References: https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py#L320
        Parameters
        ----------
        FPS :
        rotmat : (B, T, J, 3, 3)

        Returns
        -------

        """
        dRdt = self.estimate_velocity(FPS, rotmat)
        w_mat = torch.matmul(dRdt, rotmat.transpose(-1, -2))

        # pull out angular velocity vector
        # average symmetric entries
        w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
        w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
        w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
        w = torch.stack([w_x, w_y, w_z], axis=-1)

        return w

    def get_hum_feat(self, B, T, kwargs):
        "get human feature to be sent to the SMPL condition network"
        FPS = 15
        if self.smpl_cond_type in ['theta', 'theta-local']:
            smpl_poses = kwargs.get('smpl_poses', None)
        elif self.smpl_cond_type in ['joints']:
            smpl_poses = kwargs.get('smpl_joints')  # SMPL body joints (B, T, J, 3)
            # subtract the SMPL joint center
            smpl_poses = smpl_poses - smpl_poses[:, :, 0].unsqueeze(-2)
        elif self.smpl_cond_type in ['joints25']:
            smpl_poses = kwargs.get('body_joints25')
            smpl_poses = smpl_poses - smpl_poses[:, :, 8].unsqueeze(-2)
        elif self.smpl_cond_type in ['joints25+grot']:
            j25 = kwargs.get('body_joints25')
            j25 = (j25 - j25[:, :, 8].unsqueeze(-2)).reshape(B, T, -1)
            grot = kwargs.get('smpl_poses', None).reshape(B, T, -1)
            smpl_poses = torch.cat([j25, grot[:, :, :6]], -1)
        elif self.smpl_cond_type in ['theta+joints25']:
            j25 = kwargs.get('body_joints25')
            j25 = (j25 - j25[:, :, 8].unsqueeze(-2)).reshape(B, T, -1)
            theta = kwargs.get('smpl_poses', None).reshape(B, T, -1)
            smpl_poses = torch.cat([j25, theta], -1)
        elif self.smpl_cond_type in ['joints25+velo']:
            j25 = kwargs.get('body_joints25')
            j25 = j25 - j25[:, :, 8].unsqueeze(-2)  # B, T, J, 3
            velo = self.estimate_velocity(FPS, j25)
            smpl_poses = torch.cat([j25, velo], -1)
        elif self.smpl_cond_type in ['joints25+velo+grot']:
            j25 = kwargs.get('body_joints25')
            j25 = j25 - j25[:, :, 8].unsqueeze(-2)  # B, T, J, 3
            velo = self.estimate_velocity(FPS, j25)
            grot = kwargs.get('smpl_poses', None).reshape(B, T, -1)
            smpl_poses = torch.cat([j25, velo], -1)
            smpl_poses = torch.cat([smpl_poses.reshape(B, T, -1), grot[:, :, :6]], -1)
        elif self.smpl_cond_type in ['joints25+velo+theta']:
            j25 = kwargs.get('body_joints25')  # this is the condition that works best!
            j25 = j25 - j25[:, :, 8].unsqueeze(-2)  # B, T, J, 3, relative body joints
            velo = self.estimate_velocity(FPS, j25)
            grot = kwargs.get('smpl_poses', None).reshape(B, T, -1)
            smpl_poses = torch.cat([j25, velo], -1)
            smpl_poses = torch.cat([smpl_poses.reshape(B, T, -1), grot], -1)
        elif self.smpl_cond_type in ['joints25+velo+theta+avelo']:
            j25 = kwargs.get('body_joints25')
            j25 = j25 - j25[:, :, 8].unsqueeze(-2)  # B, T, J, 3
            velo = self.estimate_velocity(FPS, j25)
            theta = kwargs.get('smpl_poses', None)  # B, T, J, 6
            rotmat = rot6d_to_rotmat(theta.reshape(-1, 6)).reshape(B, T, -1, 3, 3)
            avelo = self.estimate_angular_velocity(FPS, rotmat)
            joints = torch.cat([j25, velo], -1)
            thetas = torch.cat([theta, avelo], -1)
            smpl_poses = torch.cat([joints.reshape(B, T, -1), thetas.reshape(B, T, -1)], -1)
        elif self.smpl_cond_type in ['none']:
            # no conditioning
            smpl_poses = torch.zeros(B, T, 6, device=self.device)
        else:
            raise NotImplementedError
        smpl_poses = smpl_poses.reshape(B, T, -1)
        return smpl_poses

    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "input rotation is 3x3 matrix"
        from .topnet import RotationModel
        embed_dim = 512
        pose_dim = kwargs.get('pose_feat_dim', 128)
        input_dim = 128 + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        if self.dm_pred_type == 'epsilon':
            self.point_cloud_model = RotationModel(input_dim, embed_dim, pose_dim=9, out_dim=3) # predict skew vector
        elif self.dm_pred_type == 'sample':
            # directly predict x0 as a 6d rotation vector
            self.point_cloud_model = RotationModel(input_dim, embed_dim, pose_dim=9, out_dim=6,
                                                   norm=kwargs.get('norm_layer', 'none'),
                                                   pose_feat_dim=pose_dim,
                                                   add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
                                                   )  # predict skew vector
        else:
            raise ValueError(f"Unknown prediction type {self.dm_pred_type}")
        # self.lw_rot_acc = kwargs.get('lw_rot_acc', 0.1)
        self.forward_count = 0

    def forward_train(
        self,
        pc: List[Tensor],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        "use exp and log map to do diffusion"

        B, T = image_rgb.shape[:2]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        rela_poses = kwargs.get('rela_poses', None).clone()
        abs_poses = kwargs.get('abs_poses', None).clone()

        if self.so3_rot_type == 'rela':
            rot_mat = rela_poses[:, :, :3, :3] # B, T, 3, 3
        elif self.so3_rot_type == 'abs':
            # print('using abs pose')
            rot_mat = abs_poses[:, :, :3, :3]  # B, T, 3, 3
        else:
            raise NotImplementedError
        start = time.time()
        descaled_noise, x_t = self.diffuse_rotation(rot_mat, timesteps)
        end = time.time()
        # print("Time to diffuse rotation:", end-start) # this is fast

        # get image global features
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        x_t = torch.flatten(x_t, start_dim=-2).reshape(B, T, -1)
        pred = self.model_forward(img_feats, timesteps, x_t, kwargs)

        # print(descaled_noise.shape, pred.shape, x_t.shape, noise.shape, eps.shape)
        if self.dm_pred_type == 'epsilon':
            loss = F.mse_loss(pred, descaled_noise.reshape(B, T, -1))
        elif self.dm_pred_type == 'sample':
            gt_rot = rotmat_to_6d(rot_mat.reshape(B*T, 3, 3)).reshape(B, T, 6)
            loss = self.compute_loss_x0(gt_rot, pred) # u
        else:
            raise NotImplementedError

        return loss

    def model_forward(self, img_feats, timesteps, x_t, kwargs):
        """

        Parameters
        ----------
        img_feats :
        timesteps :
        x_t : (B, T, 9)
        kwargs :

        Returns
        -------

        """
        # B, T = img_feats.shape[:2]
        pred = self.point_cloud_model(x_t, img_feats, timesteps)
        return pred

    def compute_loss_x0(self, gt_pose, pred):
        """
        compute loss for predicted x0
        Parameters
        ----------
        gt_pose :  (B, T, 6)
        pred : (B, T, 6)

        Returns
        -------

        """
        if self.so3_loss_type == 'rot-l2':
            loss = F.mse_loss(pred, gt_pose)
        elif self.so3_loss_type == 'rot-l1':
            loss = F.l1_loss(pred, gt_pose)
        elif self.so3_loss_type == 'rot+acc-l2':
            loss_rot = F.mse_loss(pred, gt_pose)
            acc_gt = gt_pose[:, :-2] - 2 * gt_pose[:, 1:-1] + gt_pose[:, 2:]
            acc_pr = pred[:, :-2] - 2 * pred[:, 1:-1] + pred[:, 2:]
            loss_acc = F.mse_loss(acc_pr, acc_gt) * self.lw_rot_acc
            loss = loss_acc + loss_rot
            # print(f'rot={loss_rot:.4f}, acc={loss_acc:.4f}, lw={self.lw_rot_acc:.4f}')
        elif self.so3_loss_type == 'rot+acc-l1':
            # acceleration loss
            loss_rot = F.l1_loss(pred, gt_pose)
            acc_gt = gt_pose[:, :-2] - 2 * gt_pose[:, 1:-1] + gt_pose[:, 2:]
            acc_pr = pred[:, :-2] - 2 * pred[:, 1:-1] + pred[:, 2:]
            loss_acc = F.l1_loss(acc_pr, acc_gt) * self.lw_rot_acc
            self.forward_count += 1
            if self.forward_count == 800:
                print(f'rot={loss_rot:.4f}, acc={loss_acc:.4f}, lw={self.lw_rot_acc:.4f}')
                self.forward_count = 0
            loss = loss_acc + loss_rot
            print(f'rot={loss_rot:.4f}, acc={loss_acc:.4f}, lw={self.lw_rot_acc:.4f}')
        else:
            raise NotImplementedError
        return loss

    def diffuse_rotation(self, rot_mat, timesteps):
        """
        diffuse rotation matrices
        Parameters
        ----------
        rot_mat : *B, T, 3, 3
        timesteps : (B, )

        Returns scaled noise to compute rotation loss and x_t
        x_t: (BT, 3, 3)
        -------

        """
        B, T = rot_mat.shape[:2]
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[timesteps]).flatten() ** 0.5  # coeff for noise
        sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[timesteps].flatten() ** 0.5  # coeff for sample
        noisedist = IsotropicGaussianSO3(sqrt_one_minus_alpha_prod*self.so3_eps_scale,
                                         torch.eye(3, device=self.device)[None].repeat(B, 1, 1)) # scale done the noise
        noise = noisedist.sample()
        # Add noise to the input rotation
        x_0 = rot_mat.reshape(B * T, 3, 3)
        noise = noise[:, None].repeat(1, T, 1, 1).reshape(B * T, 3, 3)
        # Interpolation between identity and x_0, with scale defined by sqrt_alpha_prod
        scale = sqrt_alpha_prod[:, None].repeat(1, T).flatten() # B, T -> BT,
        x_blend = so3_utils.so3_scale_pyt3d(x_0, scale)
        x_t = torch.matmul(x_blend, noise)
        # use skewvect representation, essentially predicting an axis angle
        eps = sqrt_one_minus_alpha_prod[:, None].repeat(1, T).flatten()  # (B, T) -> (BT
        descaled_noise = so3_log_map(noise) / eps[:, None]  # (BT, 3)
        return descaled_noise, x_t

    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            ret_rot: bool = False,
            **kwargs
    ):
        ""
        # Set timesteps
        # extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)
        # print('changing param to float16')
        # Conditioning: always the same, only need to compute once
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None)) # B, T, D
        # compute numeric difference: very small, almost no diff!
        # from tool.plt_utils import PltVisualizer
        # name = '15fps-dino-clstoken'
        # fcount = len(glob.glob(f'/BS/xxie-2/work/pc2-diff/experiments/debug/images/{name}_*.png'))
        # for ii, feat in enumerate(img_feats):
        #     feat_np = feat.cpu().numpy().T # (T, D)->(D, T)
        #     outfile = f'/BS/xxie-2/work/pc2-diff/experiments/debug/images/{name}_{fcount}_{ii:02d}.png'
        #     PltVisualizer.plot_and_save(np.arange(len(feat_np))[:200], feat_np[:200, :4],
        #                                 'feature index',
        #                                 'feature value',
        #                                 'features in one clip',
        #                                 outfile, (20, 9),
        #                                 legend=[f'frame {x}' for x in range(len(feat_np))], # ylim=[-0.2, 0.2],
        #                                 mute=True)
        #     # (T, D)
        #     # mean = torch.mean(feat, dim=0, keepdim=True)
        #     for i in range(len(feat)-1):
        #         diff = torch.abs(feat[i]-feat[i+1])
        #         print(f"Frame {i} and {i+1}: feat range: {feat[i].min():.4f}~{feat[i].max():.4f}, avg feat diff: {diff.mean():.4f}, max diff: {diff.max():.4f}")
            # std = torch.std(feat, dim=0)
            # print(f"Feat range: {feat[0].min():.4f}-{feat[0].max():.4f}, Mean std: {std.mean():.4f}, max: {std.max():.4f}")

        return_all_outputs = (return_sample_every_n_steps > 0)
        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor
        abs_poses = kwargs.get('abs_poses').clone()
        all_outputs, rela_poses, x_t = self.reverse_process(disable_tqdm, eta,
                                                                                                 gt_pc, img_feats,
                                                                                                 rela_poses, abs_poses,
                                                                                                 return_sample_every_n_steps,
                                                                                                 scheduler,
                                                                                                return_all_outputs, kwargs)

        B, T = img_feats.shape[:2]


        # compute output from predicted x_t, same format as other model for better evaluation
        xt_all = self.transform_points(gt_pc, rela_poses, x_t, abs_poses)
        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        # print(f"Predicted rotation:", x_t[0:2], x_t.shape)
        # print(f"GT rotation:", abs_poses[0, 0:2, :3, :3])

        if ret_rot:
            rotmat = x_t.reshape(B, T, 3, 3)
            return output, rotmat

        if return_all_outputs:
            # each iterm inside all_outputs: (B, T, N, D)
            all_outputs = torch.stack(all_outputs, dim=2)  # (B, T, sample_steps, N, D)
            all_outputs = [[self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs[x]] for x
                           in range(B)]

        return (output, all_outputs) if return_all_outputs else output

    def reverse_process(self, disable_tqdm, eta, gt_pc, img_feats, rela_poses, abs_poses, return_sample_every_n_steps, scheduler,
                        return_all_outputs, kwargs):
        """
        run one reverse process to sample rotations
        Parameters
        ----------
        disable_tqdm :
        eta :
        gt_pc :
        img_feats :
        kwargs :
        return_sample_every_n_steps :
        scheduler :

        Returns x_t: (BT, 3, 3)
        -------

        """
        print(f"Reverse diffusion scheduler={scheduler}, eta={eta}.")
        assert scheduler == 'ddpm', 'only support ddpm for now!'
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        B, T = img_feats.shape[:2]
        # x_t, _ = torch.linalg.qr(torch.randn((B, 3, 3))) # random sample from full SO3 space
        x_t = IsotropicGaussianSO3(torch.tensor(self.so3_eps_scale)).sample(
            (B,))  # sample from SO3 space with a scale to limit the deviation from identity
        x_t = x_t[:, None].repeat(1, T, 1, 1).to(self.device).reshape(B * T, 3, 3)
        # Loop over timesteps
        all_outputs = []
        progress_bar = tqdm(scheduler.timesteps.to(self.device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)
        # GT translation and rotation

        # do manual casting, ref: https://github.com/TimDettmers/bitsandbytes/issues/240#issuecomment-1761692886
        # need to set run.mixed_precision=no as well!!!
        for param in self.point_cloud_model.parameters():
            # Check if parameter dtype is  Float (float16)
            if param.dtype == torch.float16:
                # if param.dtype == torch.float32:
                param.data = param.data.to(torch.float32)
        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (
                    i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))

            # noise_pred = self.point_cloud_model(torch.flatten(x_t, start_dim=-2).reshape(B, T, -1),
            #                                     img_feats, t.reshape(1).expand(B))  # this is B, T, 3
            noise_pred = self.model_forward(img_feats, t.reshape(1).expand(B), torch.flatten(x_t, start_dim=-2).reshape(B, T, -1), kwargs)
            x_t, x0_onestep = self.reverse_step_rot(noise_pred, scheduler, t, x_t, ret_x0=True)  # This is BT, 3,3

            # print(f"Step {t} Predicted rotation:", x_t[0:2], x_t.shape)
            # print(f"GT rotation {t}:", abs_poses[0, 0:2, :3, :3])
            # exit(0)

            # Append to output list if desired
            if add_interm_output:
                xt_all = self.transform_points(gt_pc, rela_poses, x_t, abs_poses)  # TODO: update rotation with abs one
                all_outputs.append(torch.stack(xt_all))
            # if i == 10:
            #     break # one step prediction.
            # x_t = x0_onestep # one step prediction
            # break
        return all_outputs, rela_poses, x_t

    def rotation_forward(self, img_feats):
        "run reverse diffusion with the given image features, return: B, T, 6"
        B, T = img_feats.shape[:2]
        x_t = IsotropicGaussianSO3(torch.tensor(self.so3_eps_scale)).sample((B,))  # sample from SO3 space with a scale to limit the deviation from identity
        x_t = x_t[:, None].repeat(1, T, 1, 1).to(self.device).reshape(B * T, 3, 3)
        progress_bar = tqdm(self.scheduler.timesteps.to(self.device), desc=f'Sampling ({x_t.shape})', disable=False)
        scheduler = self.scheduler
        for i, t in enumerate(progress_bar):
            noise_pred = self.point_cloud_model(torch.flatten(x_t, start_dim=-2).reshape(B, T, -1),
                                                img_feats, t.reshape(1).expand(B)) # this is B, T, 3
            x_t = self.reverse_step_rot(noise_pred, scheduler, t, x_t) # This is BT, 3,3
        return rotmat_to_6d(x_t).reshape(B, T, 6)

    def reverse_step_rot(self, noise_pred, scheduler:DDPMScheduler, t, x_t, ret_x0=False):
        """
        one reverse step for rotation
        Parameters
        ----------
        noise_pred : (B, T, 3) skew vector if predict noise, else (B, T, 6), rotation 6d vector
        scheduler :
        t :
        x_t : (BT, 3, 3)

        Returns
        ------- (BT, 3, 3), x_t in next step

        """
        # rot_trace = x_t[:, 0, 0] + x_t[:, 1, 1] + x_t[:, 2, 2]
        # eps = 0.0001
        # if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        # print(f'step: {t:3d}', t, x_t[1]) # xt is already a degenerated matrix, 3rd column is all zero.
        # understand why!

        B, T = noise_pred.shape[:2]

        # Reverse step
        # Compute x_0 from prediction
        alpha_prod_t = scheduler.alphas_cumprod[t].to(self.device)
        alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
        beta_prod_t = 1 - alpha_prod_t.to(self.device)
        beta_prod_t_prev = 1 - alpha_prod_t_prev.to(self.device)
        # pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        if self.dm_pred_type == 'epsilon':
            xt_term = so3_utils.so3_scale_pyt3d(x_t, 1 / alpha_prod_t ** 0.5)
            noise_pred = noise_pred.reshape(B * T, 3)
            noise_term = beta_prod_t ** (0.5) * noise_pred / alpha_prod_t ** (0.5)
            noise_term = so3_exp_map(noise_term)
            x_0 = torch.matmul(xt_term, noise_term.transpose(-1, -2))  # subtraction is achieved by inverse
        elif self.dm_pred_type == 'sample':
            # print(noise_pred.shape, xt_term.shape)
            x_0 = rot6d_to_rotmat(noise_pred.reshape(B*T, 6)).reshape(B*T, 3, 3)
            # print("Using sample prediction")
        else:
            raise NotImplementedError
        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
        current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
        # 5. Compute predicted previous sample µ_t
        c1 = so3_utils.so3_scale_pyt3d(x_0, pred_original_sample_coeff.to(self.device))
        c2 = so3_utils.so3_scale_pyt3d(x_t, current_sample_coeff.to(self.device))
        mu_t = torch.matmul(c1, c2)
        if t == 0:
            x_t = mu_t
        else:
            # add noise
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * scheduler.betas[t]
            variance = torch.maximum(variance, torch.Tensor([1e-10]).to(self.device))
            stdev = torch.exp(0.5 * torch.log(variance))
            # scale down by 0.5 already smooths the sampling a lot
            sample = IsotropicGaussianSO3(stdev[0]*self.so3_eps_scale).sample([B]) # at step T, it is uniform sample over the SO3 space

            x_t = torch.matmul(mu_t, sample[:, None].repeat(1, T, 1, 1).reshape(B*T, 3, 3))
        if torch.isnan(x_t).any():
            # import pdb
            # pdb.set_trace()
            print(f"Found {torch.isnan(x_t).sum()} nan values at step {t}!") # if multiply with 0.1, we will have nan from step 0 to 14
            debug = 0
        if ret_x0:
            return x_t, x_0
        return x_t

    def transform_points(self, gt_pc, rela_poses, x_t, abs_poses=None):
        """
        use predicted x_t (pose) to transform GT points
        Parameters
        ----------
        gt_pc : (B, N, 3)
        rela_poses : (B, T, 4, 4), the GT transformation, already scaled by the scale_factor
        x_t : (B, T, 9)

        Returns
        -------

        """
        B, T = rela_poses.shape[:2]
        # rotmat = rot6d_to_rotmat(x_t.reshape(B * T, 6))
        if self.so3_rot_type == 'rela':
            rotmat = x_t.clone().reshape(B*T, 3, 3)
        elif self.so3_rot_type == 'abs':
            # compute a relative rotation, should not use this, because if first frame is wrong then all others are affected!
            # use gt abs pose to rotate GT points to canonical space, and then apply to others
            gt_pc_can = torch.stack(gt_pc, 0).clone().to(self.device)
            rotmat = x_t.reshape(B, T, 3, 3).clone()  # simply take the abs rotation
            for j in range(B):
                gt_pc_can[j] = torch.matmul(gt_pc_can[j], abs_poses[j, 0, :3, :3]) # inverse of the rotation
                # print('GT abs:', abs_poses[j, 0, :3, :3])
                # print('predicted rot:', rotmat[j, 0])
                # print("Matmul:", torch.matmul(abs_poses[j, 0, :3, :3], rotmat[j, 0].T))
            # x_t = x_t.reshape(B, T, 3, 3)
            # rot0 = x_t[:, 0:1].repeat(1, T, 1, 1)
            # rotmat = torch.matmul(x_t, rot0.transpose(2, 3))
            gt_pc = gt_pc_can # now use this to compute the resulting pc
            # print("Using noise")
            # print("Translation:", rela_poses[:, :, :3, 3])
            # yes they are all zeros!
            # print("All zeros?", torch.allclose(rela_poses[:, :, :3, 3], torch.zeros_like(rela_poses[:, :, :3, 3]), atol=1e-5))
        else:
            raise NotImplementedError
        poses = rela_poses.clone()
        poses[:, :, :3, :3] = rotmat.reshape(B, T, 3, 3)
        xt_all = []
        for j in range(B):
            pc_i = gt_pc[j].to(self.device) * self.scale_factor
            xt_local = torch.matmul(pc_i.repeat(T, 1, 1),
                                    poses[j, :, :3, :3].transpose(1, 2)) + poses[j, :, :3, 3].unsqueeze(-2)
            xt_all.append(xt_local)
        return xt_all



class RigidSMPLCondSO3Diffusion(RigidSO3Diffusion):
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "use SMPL conditional model"
        from .topnet import SMPLCondRotationModel, RotationModel, SMPLCondRotationModelv2
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        # Runname=so3smpl_5obj, loss not decreasing...
        # self.point_cloud_model = SMPLCondRotationModel(input_dim, embed_dim,
        #                                        pose_dim=kwargs.get('smpl_pose_dim', 144),
        #                                        norm=kwargs.get('norm_layer', 'none'),
        #                                        pose_feat_dim=pose_dim,
        #                                        add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
        #                                        )
        assert smpl_cond_type == 'joints25+velo+theta', 'invalid condition type!'
        self.point_cloud_model = SMPLCondRotationModelv2(input_dim, embed_dim,
                                                         pose_dim=pose_in_dim + 9, # also x_t rotation
                                                         norm=kwargs.get('norm_layer', 'none'),
                                                         pose_feat_dim=pose_feat_dim,
                                                         add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
                                                         )
        self.forward_count = 0

    def model_forward(self, img_feats, timesteps, x_t, kwargs):
        "add smpl conditioning"
        B, T = img_feats.shape[:2]
        smpl_poses = self.get_hum_feat(B, T, kwargs)
        xt_smpl = torch.cat([x_t, smpl_poses], -1)
        pred = self.point_cloud_model(xt_smpl, img_feats, timesteps)
        return pred



class RigidSO3Regression(RigidSO3Diffusion):
    "directly regress the abs rotation, without any diffusion process"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "a transformer to predict rotation"
        from .topnet import RotationModel
        embed_dim = 512
        pose_dim = kwargs.get('pose_feat_dim', 128)
        input_dim = pose_dim + 64 + 64 + self.feature_model.feature_dim + 1 # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = RotationModel(input_dim, embed_dim,
                                               # pose_dim=kwargs.get(''),
                                               norm=kwargs.get('norm_layer', 'none'),
                                               pose_feat_dim=pose_dim,
                                               add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
                                               )
        # self.lw_rot_acc = kwargs.get('lw_rot_acc', 0.1)


    def forward_train(
        self,
        pc: List[Tensor],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        "simply input zero pose and timestep, conditioned on image feature to do regression, regress 6d pose"
        abs_poses = kwargs.get('abs_poses', None).clone()
        assert self.so3_rot_type == 'abs'
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        B, T = image_rgb.shape[:2]
        pred = self.rotation_forward(img_feats, **kwargs)
        gt_pose = rotmat_to_6d(abs_poses[:, :, :3, :3].reshape(B*T, 3, 3)).reshape(B, T, 6).to(self.device)

        loss = self.compute_loss_x0(gt_pose, pred)

        return loss

    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            ret_rot: bool= False,
            **kwargs
    ):
        "direct regress"
        rotmat = self.regress_poses(image_rgb, mask, **kwargs)
        B, T = image_rgb.shape[:2]

        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor
        abs_poses = kwargs.get('abs_poses', None).clone()
        print(f"Predicted rotation:", rotmat[0, 0:2], rotmat.shape)
        print(f"GT rotation:", abs_poses[0, 0:2, :3, :3])
        # if ret_rot:
        #     return rotmat

        xt_all = self.transform_points(gt_pc, rela_poses, rotmat, abs_poses)

        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        return_all_outputs = (return_sample_every_n_steps > 0)

        if ret_rot:
            return output, rotmat

        return (output, [[x]*T for x in output]) if return_all_outputs else output

    def regress_poses(self, image_rgb, mask, **kwargs):
        B, T = image_rgb.shape[:2]
        occ_ratios = kwargs.get('occ_ratios', None)
        img_feats = self.extract_features(image_rgb, mask, occ_ratios)
        pred = self.rotation_forward(img_feats, **kwargs)
        rotmat = rot6d_to_rotmat(pred.reshape(B * T, 6)).reshape(B, T, 3, 3)
        return rotmat

    def rotation_forward(self, img_feats, **kwargs):
        """
        given image feature, use the model to run one forward prediction step
        Parameters
        ----------
        img_feats :

        Returns *B, T, 6
        -------

        """
        B, T = img_feats.shape[:2]
        x_t = torch.zeros(B, T, 6, device=self.device)
        timesteps = torch.zeros(B, device=self.device)
        pred = self.point_cloud_model(torch.flatten(x_t, start_dim=-2).reshape(B, T, -1), img_feats, timesteps)
        return pred


class RigidSO3RegressQuaternion(RigidSO3Regression):
    "predict quaternion instead of rot6d"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        ""
        from .topnet import RotationModel
        embed_dim = 512
        pose_dim = kwargs.get('pose_feat_dim', 128)
        input_dim = pose_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = RotationModel(input_dim, embed_dim,
                                               out_dim=4,
                                               # pose_dim=kwargs.get(''),
                                               norm=kwargs.get('norm_layer', 'none'),
                                               pose_feat_dim=pose_dim,
                                               add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
                                               )

    def regress_poses(self, image_rgb, mask, **kwargs):
        """
        add post processing to convert quaternion to rotmat
        Parameters
        ----------
        image_rgb :
        mask :
        kwargs :

        Returns
        -------

        """
        B, T = image_rgb.shape[:2]
        occ_ratios = kwargs.get('occ_ratios', None)
        img_feats = self.extract_features(image_rgb, mask, occ_ratios)
        pred = self.rotation_forward(img_feats, **kwargs)
        # convert quaternion to rotmat
        # rotmat = quat_to_rotmat(pred.reshape(B * T, 4)).reshape(B, T, 3, 3)
        rotmat = quaternion_to_matrix(pred)
        return rotmat

    def compute_loss_x0(self, gt_pose, pred):
        """

        Parameters
        ----------
        gt_pose : (B, T, 6), GT pose represented as rot6d vector
        pred : (B, T, 4), predicted quaternion

        Returns
        -------

        """
        B, T = gt_pose.shape[:2]
        gt_quat = matrix_to_quaternion(rotation_6d_to_matrix(gt_pose))
        norm_quat = pred / pred.norm(p=2, dim=-1, keepdim=True) # normalize
        if self.so3_loss_type == 'rot-l2':
            # L2 distance for the quaternion, ref: https://math.stackexchange.com/questions/90081/quaternion-distance
            qd = (gt_quat * norm_quat).sum(-1)**2
            loss = (1 - qd).mean()
        elif self.so3_loss_type == 'rot-l1':
            # L1 distance: abs of the dot product
            qd = (gt_quat * norm_quat).sum(-1)
            loss = (1 - torch.abs(qd)).mean()
        elif self.so3_loss_type == 'rot+acc-l2':
            # L2 distance + acceleration of quaternion
            acc_gt = gt_quat[:, :-2] - 2 * gt_quat[:, 1:-1] + gt_quat[:, 2:]
            acc_pr = norm_quat[:, :-2] - 2 * norm_quat[:, 1:-1] + norm_quat[:, 2:]
            loss_rot = (1 - (gt_quat * norm_quat).sum(-1) ** 2).mean()
            loss_acc = F.mse_loss(acc_pr, acc_gt) # do not multiply 0.1 to have roughly 1/10 of the rotation loss
            print(f"Loss rot={loss_rot:.4f}, loss acc={loss_acc}")
            loss = loss_acc + loss_rot
        elif self.so3_loss_type == 'rot+acc-l1':
            # acceleration loss
            acc_gt = gt_quat[:, :-2] - 2 * gt_quat[:, 1:-1] + gt_quat[:, 2:]
            acc_pr = norm_quat[:, :-2] - 2 * norm_quat[:, 1:-1] + norm_quat[:, 2:]
            loss_rot = (1 - (gt_quat * norm_quat).sum(-1) ** 2).mean()
            loss_acc = F.l1_loss(acc_pr, acc_gt) # do not multiply 0.1 to have roughly 1/10 of the rotation loss
            print(f"Loss rot={loss_rot:.4f}, loss acc={loss_acc}")
            loss = loss_acc + loss_rot
        else:
            raise NotImplementedError
        return loss


class RigidSMPLCondSO3Regression(RigidSO3Regression):
    "condition on human pose as well"
    def rotation_forward(self, img_feats, **kwargs):
        "get SMPL pose"

        B, T = img_feats.shape[:2]
        smpl_poses = self.get_hum_feat(B, T, kwargs)
        x_t = smpl_poses
        timesteps = torch.zeros(B, device=self.device)
        pred = self.point_cloud_model(x_t, img_feats, timesteps)
        return pred

    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        ""
        from .topnet import SMPLCondRotationModel, RotationModel, SMPLCondRotationModelv2
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        # Runname=so3smpl_5obj, loss not decreasing...
        # self.point_cloud_model = SMPLCondRotationModel(input_dim, embed_dim,
        #                                        pose_dim=kwargs.get('smpl_pose_dim', 144),
        #                                        norm=kwargs.get('norm_layer', 'none'),
        #                                        pose_feat_dim=pose_dim,
        #                                        add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
        #                                        )
        self.point_cloud_model = SMPLCondRotationModelv2(input_dim, embed_dim,
                                               pose_dim=pose_in_dim,
                                               norm=kwargs.get('norm_layer', 'none'),
                                               pose_feat_dim=pose_feat_dim,
                                               add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
                                               )




class RigidSMPLCondQposeSO3Regression(RigidSMPLCondSO3Regression):
    "Q=pose, K=V=image feature"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "use cross attention model"
        from .topnet import SMPLCondCrossAttnQpose
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = SMPLCondCrossAttnQpose(input_dim, embed_dim,
                                                        pose_dim=pose_in_dim,
                                                        norm=kwargs.get('norm_layer', 'none'),
                                                        pose_feat_dim=pose_feat_dim,
                                                        add_src_key_mask=kwargs.get('add_src_key_mask', -1.))

class RigidSMPLCondQimgSO3Regression(RigidSMPLCondSO3Regression):
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "Q=image, K=V=pose feature"
        from .topnet import SMPLCondCrossAttnQimg
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = SMPLCondCrossAttnQimg(input_dim, embed_dim,
                                                        pose_dim=pose_in_dim,
                                                        norm=kwargs.get('norm_layer', 'none'),
                                                        pose_feat_dim=pose_feat_dim,
                                                        add_src_key_mask=kwargs.get('add_src_key_mask', -1.))

class RigidSMPLCondQimgSASO3Regression(RigidSMPLCondSO3Regression):
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "Q=image, K=V=pose feature"
        from .topnet import SMPLCondCrossAttnQimgSelfAttn
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = SMPLCondCrossAttnQimgSelfAttn(input_dim, embed_dim,
                                                        pose_dim=pose_in_dim,
                                                        norm=kwargs.get('norm_layer', 'none'),
                                                        pose_feat_dim=pose_feat_dim,
                                                        add_src_key_mask=kwargs.get('add_src_key_mask', -1.))



class RigidSMPLCondQposeCombSO3Regression(RigidSMPLCondSO3Regression):
    "Q=pose, K=V=image feature"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "use cross attention model"
        from .topnet import SMPLCondCrossAttnQposeCombine
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = SMPLCondCrossAttnQposeCombine(input_dim, embed_dim,
                                                        pose_dim=pose_in_dim,
                                                        norm=kwargs.get('norm_layer', 'none'),
                                                        pose_feat_dim=pose_feat_dim,
                                                        add_src_key_mask=kwargs.get('add_src_key_mask', -1.))

class RigidSMPLCondTwoHead(RigidSMPLCondSO3Regression):
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        from .topnet import SMPLCondCrossAttnTwoHead
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = SMPLCondCrossAttnTwoHead(input_dim, embed_dim,
                                                       pose_dim=pose_in_dim,
                                                       norm=kwargs.get('norm_layer', 'none'),
                                                       pose_feat_dim=pose_feat_dim,
                                                       add_src_key_mask=kwargs.get('add_src_key_mask', -1.))



    def compute_loss_x0(self, gt_pose, pred):
        "compute loss for two heads predictions and then average"
        D = pred.shape[-1]
        assert D == 12, f'the given object pose prediction dimension is {D}!'
        loss1 = super().compute_loss_x0(gt_pose, pred[..., :D//2])
        loss2 = super().compute_loss_x0(gt_pose, pred[..., D//2:])
        return loss1 + loss2

    def rotation_forward(self, img_feats, **kwargs):
        "train: same as before, test: average predictions of two heads"
        pred = super().rotation_forward(img_feats, **kwargs)
        if self.training:
            return pred
        else:
            D = pred.shape[-1]
            assert D == 12, f'the given object pose prediction dimension is {D}!'
            p1, p2 = pred[..., :D//2], pred[..., D//2:]
            return (p1 + p2)/2 # average predictions


class RigidSMPLCondUncertainRegression(RigidSMPLCondSO3Regression):
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        ""
        from .topnet import SMPLRotationWithUncertaintyModel
        embed_dim = 512
        pose_feat_dim = kwargs.get('pose_feat_dim', 128)
        smpl_cond_type = kwargs.get('smpl_cond_type', 'theta')
        pose_in_dim = get_pose_in_dim(smpl_cond_type)
        self.smpl_cond_type = smpl_cond_type
        self.smpl_cond_dim = pose_in_dim
        input_dim = pose_feat_dim + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = SMPLRotationWithUncertaintyModel(input_dim, embed_dim,
                                                         pose_dim=pose_in_dim,
                                                         norm=kwargs.get('norm_layer', 'none'),
                                                         pose_feat_dim=pose_feat_dim,
                                                         add_src_key_mask=kwargs.get('add_src_key_mask', -1.)
                                                         )
        self.nnl_beta = kwargs.get('nnl_beta', 0.5) # beta factor for nnl loss
        print(f"Beta for NNL computation={self.nnl_beta}.")

    def compute_loss_x0(self, gt_pose, pred):
        ""
        import math
        D = pred.shape[-1]
        mean, var = pred[..., :D//2], pred[..., D//2:]
        # the err is too small, so adding up a log it becomes negative, adding a constant lift it up
        loss = 0.5 * (F.l1_loss(mean, gt_pose, reduction='none') / var + torch.log(var) + 6*math.log(2 * math.pi))
        if self.nnl_beta > 0:
            loss = loss * (var.detach()**self.nnl_beta) # L2 can also have negative values

        # loss for acc
        acc_gt = gt_pose[:, :-2] - 2 * gt_pose[:, 1:-1] + gt_pose[:, 2:]
        acc_pr = mean[:, :-2] - 2 * mean[:, 1:-1] + mean[:, 2:]
        # loss_acc = 0.5 * (F.l1_loss(acc_pr, acc_gt, reduction='none') / var[:, 1:-1] + var[:, 1:-1].log())
        # if self.nnl_beta > 0:
        #     loss_acc = loss_acc * (var[:, 1:-1].detach() ** self.nnl_beta)
        loss_acc = F.l1_loss(acc_pr, acc_gt) * 3 * (self.lw_rot_acc * 10)

        # loss_acc = loss_acc.mean() * 0.1 # acc loss is negative if use l1!
        loss_rot = loss.mean()
        # print(f"Acc loss: {loss_acc:.4f}, rot loss: {loss_rot:.4f}, var min={torch.min(var)}")
        return loss_acc + loss_rot

    def regress_poses(self, image_rgb, mask, **kwargs):
        "for simplicity, just return the mean"
        B, T = image_rgb.shape[:2]
        occ_ratios = kwargs.get('occ_ratios', None)
        img_feats = self.extract_features(image_rgb, mask, occ_ratios)
        pred = self.rotation_forward(img_feats, **kwargs)
        D = pred.shape[-1]
        rotmat = rot6d_to_rotmat(pred[..., :D//2].reshape(B * T, 6)).reshape(B, T, 3, 3)
        return rotmat

    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            ret_rot: bool= False,
            **kwargs
    ):
        ""

        B, T = image_rgb.shape[:2]
        occ_ratios = kwargs.get('occ_ratios', None)
        img_feats = self.extract_features(image_rgb, mask, occ_ratios)
        pred = self.rotation_forward(img_feats, **kwargs)
        ret_uncert = kwargs.get('ret_uncertainty', False)
        rotmat = rot6d_to_rotmat(pred[..., :6].reshape(B * T, 6)).reshape(B, T, 3, 3)

        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor
        abs_poses = kwargs.get('abs_poses', None).clone()
        print(f"Predicted rotation:", rotmat[0, 0:2], rotmat.shape)
        print(f"GT rotation:", abs_poses[0, 0:2, :3, :3])
        # if ret_rot:
        #     return rotmat

        xt_all = self.transform_points(gt_pc, rela_poses, rotmat, abs_poses)

        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        return_all_outputs = (return_sample_every_n_steps > 0)

        if ret_rot:
            if ret_uncert:
                return output, pred
            return output, rotmat

        return (output, [[x] * T for x in output]) if return_all_outputs else output


class RigidSO3ShapeDiffusion(RigidSO3Diffusion):
    "diffuse both rotation and shape"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        ""
        from .topnet import RotationModel
        embed_dim = 512
        input_dim = 128 + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.so3_model = RotationModel(input_dim, embed_dim, pose_dim=9, out_dim=3)  # predict skew vector
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,  # voxel resolution multiplier is 1.
            voxel_resolution_multiplier=kwargs.get('voxel_resolution_multiplier', 1)
        )

    def forward_train(
        self,
        pc: List[Tensor],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        "diffuse shape and rotation, then use diffused rotation to transform points to other frames"

        B, T = image_rgb.shape[:2]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        rela_poses = kwargs.get('rela_poses', None).clone()
        rela_poses[:, :, :3, 3] *= self.scale_factor  # scale to 7
        assert rela_poses is not None
        rot_mat = rela_poses[:, :, :3, :3]  # B, T, 3, 3
        descaled_noise, x_t = self.diffuse_rotation(rot_mat, timesteps)
        # get image global features
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        pred = self.so3_model(torch.flatten(x_t, start_dim=-2).reshape(B, T, -1), img_feats, timesteps)

        loss_rot = F.mse_loss(pred, descaled_noise.reshape(B, T, -1))
        rot_t = x_t.clone().reshape(B, T, 3, 3)

        # Now diffuse pc and transform
        noises, pc_feats = [], []
        for i in range(len(pc)):
            # print(i, rela_poses[i]) # clip=1, all identity
            pc_i = pc[i].to(self.device) * self.scale_factor # (N, 3)
            # first diffuse, then transform to each local frame, then aggregate
            noise = torch.randn_like(pc_i)
            noise = noise - torch.mean(noise, dim=0, keepdim=True)
            x_t = self.scheduler.add_noise(pc_i, noise, timesteps[i])[None].repeat(T, 1, 1)
            # transform to local use diffused rotation
            rela_pose_i = rela_poses[i].clone() # Assume GT translation
            rela_pose_i[:, :3, :3] = rot_t[i].clone()
            xt_local = torch.matmul(x_t, rela_pose_i[:, :3, :3].transpose(1, 2)) + rela_pose_i[:, :3, 3].unsqueeze(-2)
            xt_input = self.get_point_feats(camera, i, image_rgb, kwargs, mask, xt_local)
            pc_feats.append(torch.cat([x_t[0], torch.mean(xt_input[:, :, 3:], 0)], 1)) # do not average the coordinates
            noises.append(noise)
        xt_feat = torch.stack(pc_feats, 0)
        noise_pred = self.point_cloud_model(xt_feat, timesteps)
        noises = torch.stack(noises, 0)

        loss_pc = F.mse_loss(noise_pred, noises)

        loss = loss_rot + loss_pc

        return loss, torch.tensor([loss_pc.clone().detach(), loss_rot.clone().detach()])


    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            **kwargs
    ):
        ""
        from .so3.distributions import IsotropicGaussianSO3
        print(f"Reverse diffusion scheduler={scheduler}, eta={eta}.")
        assert scheduler == 'ddpm', 'only support ddpm for now!'
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        B, T = image_rgb.shape[:2]
        device, D = self.device, self.get_x_T_channel()
        # rot_t, _ = torch.linalg.qr(torch.randn((B, 3, 3)))
        rot_t = IsotropicGaussianSO3(torch.tensor(self.so3_eps_scale)).sample((B,))
        rot_t = rot_t[:, None].repeat(1, T, 1, 1).to(self.device).reshape(B * T, 3, 3)

        # Init object points
        N = num_points
        x_t = self.initialize_x_T(device, gt_pc, (B, N, D), -1, scheduler)

        # Set timesteps
        extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(self.device), desc=f'Sampling ({rot_t.shape})', disable=disable_tqdm)

        # GT translation
        rela_poses = kwargs.get('rela_poses', None)
        rela_poses[:, :, :3, 3] *= self.scale_factor

        # Conditioning for rotation: always the same, only need to compute once
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (
                        i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))

            noise_pred = self.so3_model(torch.flatten(rot_t, start_dim=-2).reshape(B, T, -1),
                                                img_feats, t.reshape(1).expand(B)) # this is B, T, 3
            rot_t = self.reverse_step_rot(noise_pred, scheduler, t, rot_t)

            # use updated rot_t to do transformation and get conditioning
            xt_feats = []
            rot_t_tmp = rot_t.clone().reshape(B, T, 3, 3)
            for j in range(B):
                pose_j = rela_poses[j].clone()
                pose_j[:, :3, :3] = rot_t_tmp[j]
                xt_local = torch.matmul(x_t[j:j + 1].repeat(T, 1, 1),
                                        pose_j[:, :3, :3].transpose(1, 2)) + pose_j[:, :3, 3].unsqueeze(-2)
                xt_input = self.get_point_feats(camera, j, image_rgb, kwargs, mask, xt_local)
                xt_feats.append(torch.cat([xt_local[0], torch.mean(xt_input[:, :, 3:], 0)],
                                          1))  # here the xyz coordinates are also averaged!
            xt_feats = torch.stack(xt_feats, 0)
            inference_binary = (i == len(progress_bar) - 1) | add_interm_output
            # One reverse step with conditioning
            x_t = self.reverse_step(extra_step_kwargs, scheduler, t, x_t, xt_feats,
                                    inference_binary=inference_binary)  # (B, N, D), D=3 or 4

            # Append to output list if desired
            if add_interm_output:
                # transform to other frames use predicted shape and rotation
                xt_all = self.transform_points(x_t/self.scale_factor, rela_poses, rot_t)
                all_outputs.append(torch.stack(xt_all))

        # Convert output back into a point cloud, undoing normalization and scaling
        xt_all = self.transform_points(x_t/self.scale_factor, rela_poses, rot_t)
        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        if return_all_outputs:
            # each iterm inside all_outputs: (B, T, N, D)
            all_outputs = torch.stack(all_outputs, dim=2)  # (B, T, sample_steps, N, D)
            all_outputs = [[self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs[x]] for x
                           in range(B)]

        return (output, all_outputs) if return_all_outputs else output


class RigidSE3Diffusion(RigidSO3Diffusion):
    "also diffuse a translation vector"
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        "input rotation is 3x3 matrix"
        from .topnet import RotationModel
        embed_dim = 512
        input_dim = 128 + 64 + 64 + self.feature_model.feature_dim + 1  # pose, timestep, positional encoding and image feature, and mask area ratio
        self.point_cloud_model = RotationModel(input_dim, embed_dim, pose_dim=12, out_dim=6) # predict skew vector + translation
        self.shift_scale = 0.3 # scale for the translation distribution, i.e. std. of the normal distribution
    def forward_train(
        self,
        pc: List[Tensor],
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        ""
        B, T = image_rgb.shape[:2]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        rela_poses = kwargs.get('rela_poses', None).clone() # this is not scaled!

        rot_mat = rela_poses[:, :, :3, :3]  # B, T, 3, 3
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]).to(self.device).flatten() ** 0.5  # coeff for noise
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[timesteps].to(self.device).flatten() ** 0.5  # coeff for sample
        affine = so3_utils.AffineT(torch.eye(3, device=self.device)[None].repeat(B*T, 1, 1),
                                   torch.zeros(B*T, 3, device=self.device))
        eps = sqrt_one_minus_alpha_prod[:, None].repeat(1, T).flatten()
        noisedist = IGSO3xR3(eps, affine, shift_scale=self.shift_scale) # constrain translation to [-1, 1]
        noise = noisedist.sample()
        # Add noise to the input rotation
        x_0 = so3_utils.AffineT(rot_mat.reshape(B*T, 3, 3), rela_poses[:, :, :3, 3].reshape(B*T, 3))
        # noise = noise[:, None].repeat(1, T, 1, 1).reshape(B * T, 3, 3)
        # Interpolation between identity and x_0, with scale defined by sqrt_alpha_prod
        scale = sqrt_alpha_prod[:, None].repeat(1, T).flatten()
        x_blend = so3_utils.se3_scale_pyt3d(x_0, scale) # change to se3
        # x_blend = so3_utils.so3_scale_pyt3d(x_0, scale)
        # compute x_t for rotation and translation separately
        xt_rot = x_blend.rot @ noise.rot
        xt_trans = x_blend.shift + noise.shift
        # use skewvect representation, essentially predicting an axis angle
        eps = sqrt_one_minus_alpha_prod[:, None].repeat(1, T).flatten()  # (B, T) -> (BT
        descaled_noise = so3_log_map(noise.rot) / eps[:, None]  # (BT, 3)
        descaled_shift = (noise.shift) * (1 / (eps[:, None] * self.shift_scale)) # (BT, 3)

        # get image global features
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        x_t = torch.cat([xt_rot, xt_trans.unsqueeze(1)], 1)
        pred = self.point_cloud_model(torch.flatten(x_t, start_dim=-2).reshape(B, T, -1), img_feats, timesteps)

        # print(descaled_noise.shape, pred.shape, x_t.shape, noise.shape, eps.shape)
        loss_rot = F.mse_loss(pred[:, :, :3], descaled_noise.reshape(B, T, -1))
        loss_t = F.mse_loss(pred[:, :, 3:], descaled_shift.reshape(B, T, -1))
        loss = loss_t + loss_rot

        return loss, torch.tensor([loss_t.clone().detach(), loss_rot.clone().detach()])

    def forward_sample(
        self,
        num_points: int,
        camera: List[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM if eta=0, pure deterministic, small steps work
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            **kwargs
    ):
        """"""
        assert scheduler == 'ddpm', 'only support ddpm for now!'
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        B, T = image_rgb.shape[:2]
        x_t, _ = torch.linalg.qr(torch.randn((B, 3, 3)))
        rot_t = x_t[:, None].repeat(1, T, 1, 1).to(self.device).reshape(B * T, 3, 3)
        trans_t = torch.randn((B, 1, 3)).repeat(1, T, 1).to(self.device).reshape(B * T, 3)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(self.device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        # GT translation and rotation
        rela_poses = kwargs.get('rela_poses', None)
        # rela_poses[:, :, :3, 3] *= self.scale_factor

        # Conditioning: always the same, only need to compute once
        img_feats = self.extract_features(image_rgb, mask, kwargs.get('occ_ratios', None))
        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (
                    i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))

            x_t = torch.cat([rot_t, trans_t.unsqueeze(1)], 1) # BT, 4, 3
            noise_pred = self.point_cloud_model(torch.flatten(x_t, start_dim=-2).reshape(B, T, -1),
                                                img_feats, t.reshape(1).expand(B))  # this is B, T, 6
            x_t = self.reverse_step_rot(noise_pred, scheduler, t, x_t)

            # Append to output list if desired
            if add_interm_output:
                # replace GT translation with predicted one
                rela_poses_pred = rela_poses.clone()
                rela_poses_pred[:, :, :3, 3] = x_t[:, 3, :3].reshape(B, T, -1) * self.scale_factor
                xt_all = self.transform_points(gt_pc, rela_poses_pred, x_t[:, :3, :3])  # take predicted rotation
                all_outputs.append(torch.stack(xt_all))

        # compute output from predicted x_t, same format as other model for better evaluation
        rela_poses_pred = rela_poses.clone()
        rela_poses_pred[:, :, :3, 3] = x_t[:, 3, :3].reshape(B, T, -1) * self.scale_factor
        xt_all = self.transform_points(gt_pc, rela_poses_pred, x_t[:, :3, :3])
        output = [self.tensor_to_point_cloud(xt_all[x], denormalize=True, unscale=True) for x in range(B)]

        if return_all_outputs:
            # each iterm inside all_outputs: (B, T, N, D)
            all_outputs = torch.stack(all_outputs, dim=2)  # (B, T, sample_steps, N, D)
            all_outputs = [[self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs[x]] for x
                           in range(B)]

        return (output, all_outputs) if return_all_outputs else output


    def reverse_step_rot(self, noise_pred, scheduler:DDPMScheduler, t, x_t):
        """
        one reverse step for rotation
        Parameters
        ----------
        noise_pred : (B, T, 6) skew vector + translation
        scheduler :
        t :
        x_t : (BT, 4, 3), rotation + translation

        Returns
        ------- (BT, 3, 3), x_t in next step

        """
        B, T = noise_pred.shape[:2]
        noise_pred = noise_pred.reshape(B * T, 6)
        vec_pred, trans_pred = noise_pred[:, :3], noise_pred[:, 3:]
        # Reverse step
        # Compute x_0 from prediction
        alpha_prod_t = scheduler.alphas_cumprod[t].to(self.device)
        alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
        beta_prod_t = 1 - alpha_prod_t.to(self.device)
        beta_prod_t_prev = 1 - alpha_prod_t_prev.to(self.device)
        # Compute x0 for the rotation
        xt_term_rot = so3_utils.so3_scale_pyt3d(x_t[:, :3, :3], 1 / alpha_prod_t ** 0.5)
        noise_term_rot = beta_prod_t ** (0.5) * vec_pred / alpha_prod_t ** (0.5)
        noise_term_rot = so3_exp_map(noise_term_rot)
        x0_rot = torch.matmul(xt_term_rot, noise_term_rot.transpose(-1, -2)) # subtraction is achieved by inverse
        # Compute x0 for the translation
        xt_term_trans = x_t[:, 3, :3] / alpha_prod_t ** 0.5
        noise_term_trans = beta_prod_t ** (0.5) * trans_pred / alpha_prod_t ** (0.5)
        x0_trans = xt_term_trans - noise_term_trans # simply subtraction as it is in SE3

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev.to(self.device) ** (0.5) * scheduler.betas[t]) / beta_prod_t
        current_sample_coeff = scheduler.alphas[t].to(self.device) ** (0.5) * beta_prod_t_prev / beta_prod_t
        # 5. Compute predicted previous sample µ_t for rotation
        c1 = so3_utils.so3_scale_pyt3d(x0_rot, pred_original_sample_coeff)
        c2 = so3_utils.so3_scale_pyt3d(x_t[:, :3, :3], current_sample_coeff)
        mu_t_rot = torch.matmul(c1, c2)
        # Compute predicted previous mu_t for translation
        mu_t_trans = x0_trans * pred_original_sample_coeff + x_t[:, 3, :3]*current_sample_coeff
        if t == 0:
            x_t = torch.cat([mu_t_rot, mu_t_trans.unsqueeze(1)], 1)
        else:
            # add noise to rotation and translation separately
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * scheduler.betas[t]
            variance = torch.maximum(variance, torch.Tensor([1e-10]).to(self.device))
            stdev = torch.exp(0.5 * torch.log(variance))
            # sample = IsotropicGaussianSO3(stdev[0]).sample([B * T])
            # xt_rot = torch.matmul(mu_t, sample)
            affine = so3_utils.AffineT(mu_t_rot, mu_t_trans)
            sample = IGSO3xR3(stdev[0], affine, shift_scale=self.shift_scale).sample()
            x_t = torch.cat([sample.rot, sample.shift.unsqueeze(1)], 1)

        return x_t

