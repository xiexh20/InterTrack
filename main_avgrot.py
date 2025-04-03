"""
one frame is run multiple times, each at different batch, and then compute an average of it
"""
import glob
import pickle as pkl
import sys, os
import time
from typing import Iterable, Optional

import trimesh
from accelerate import Accelerator
from tqdm import tqdm

sys.path.append(os.getcwd())
import hydra
import torch
import wandb
import numpy as np
import os.path as osp
from torchvision.transforms import functional as TVF
from pytorch3d.renderer.cameras import PerspectiveCameras


from model.geometry_utils import rotmat_to_6d, rot6d_to_rotmat
from configs.structured import ProjectConfig
from pathlib import Path

# from main_video import TrainerCombinedObj
from main import TrainerBehave


def clips2seq_fast(clips, step, window_size):
    """
    10 times faster version
    Parameters
    ----------
    clips : (B, T, D)
    step : step between nearby frames
    window_size : sliding window size

    Returns
    -------

    """
    assert step == 1, 'currently only support step size 1!'
    B, T = clips.shape[:2]
    L = (B-1)*step + window_size
    out_all = torch.zeros(L, window_size//step, *clips.shape[2:]).to(clips.device)

    masks = []
    for t in range(T):
        out_b_idx = torch.arange(0, L).to(clips.device)
        in_b_idx = torch.arange(-T+1+t, -T+L+t+1).to(clips.device)
        in_t_idx = T - 1 - t
        mask = (in_b_idx < B) & (in_b_idx >=0)
        out_all[out_b_idx[mask], t] = clips[in_b_idx[mask], in_t_idx]
        masks.append(mask)
    masks = torch.stack(masks, 1)
    seq = torch.sum(out_all, 1) / torch.sum(masks, 1).unsqueeze(-1)

    # ind = T
    # for i in range(T):
    #     print(f'Prediction {i}:', out_all[ind, i])
    # print("Avg:", torch.sum(out_all[ind], 0)/torch.sum(masks[ind], 0))
    return seq


class SamplerAvgRot(TrainerBehave):
    @torch.no_grad()
    def sample(self, cfg: ProjectConfig,
                model: torch.nn.Module,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        "the dataloader has only one sequence"
        model.eval() # super important! otherwise random dropout in transformer!!!

        batches = [] # non-repeat batches
        rot6d_preds, abs_poses, uncertainties = [], [], [] # assumes the window is 1
        rot6d_preds_real = []
        file_paths = []
        pc_can = None # canonical points
        output_dir: Path = Path(output_dir)
        cam_trans = []
        cam_K = []

        assert cfg.dataset.window == 1, 'only support window size=1!'

        os.makedirs(output_dir, exist_ok=True)

        for bid, batch in enumerate(tqdm(dataloader)):
            for file_list in batch['image_path']:
                for file in file_list:
                    if file not in file_paths:
                        file_paths.append(file)
            # check done
            ss = str(file_paths[0]).split(os.sep)
            sequence_category = ss[-3]
            pred_files = sorted(glob.glob(osp.join(output_dir, f'pred/{sequence_category}/*.ply')))
            gt_files = sorted(glob.glob(osp.join(output_dir, f'gt/{sequence_category}/*.ply')))
            if len(pred_files) >= len(gt_files) and len(gt_files) >0 and not cfg.run.redo:
                print("all done, skipped")
                return

            images = torch.stack(batch['images'], 0).to('cuda')
            masks = torch.stack(batch['masks'], 0).to('cuda')
            occ_ratios = torch.stack(batch['occ_ratios'], 0).to('cuda')
            # additional SMPL condition
            smpl_poses = torch.stack(batch['smpl_poses'], 0).to('cuda')
            smpl_joints = torch.stack(batch['smpl_joints'], 0).to('cuda')
            body_joints25 = torch.stack(batch['body_joints25'], 0).to('cuda')

            time_start = time.time()
            feats = model.extract_features(images, masks, occ_ratios)
            B, T = feats.shape[:2] # (B, T, D)
            batch['img_feats'] = feats.reshape(B*T, -1)
            # img_feats.append(feats.reshape(B*T, -1))

            if cfg.model.model_name in ['diff-so3', 'diff-so3-smpl']:
                # diffusion
                kwargs = {
                    "occ_ratios": occ_ratios,
                    "smpl_poses": smpl_poses,
                    "smpl_joints": smpl_joints,
                    "body_joints25": body_joints25
                }
                _, _, x_t = model.reverse_process(False, 1.0,
                                                None,  # not transforming GT pc
                                                 feats,
                                                None, None, # not using rela and abs poses
                                                1,
                                                cfg.run.diffusion_scheduler,
                                                False, kwargs) # x_t is shape (BT, 3, 3)
                # pred = matrix_to_rotation_6d(x_t.reshape(B*T, 3, 3)).reshape(B, T, 6) # pytorch3d is different from our own rot6d!!!
                pred = rotmat_to_6d(x_t).reshape(B, T, 6)
            else:
                # feedforward regressuib
                pred = model.rotation_forward(feats,
                                          occ_ratios=occ_ratios,
                                          smpl_poses=smpl_poses,
                                          smpl_joints=smpl_joints,
                                          body_joints25=body_joints25,)
            if cfg.model.model_name== 'so3smpl+uncert':
                # from the 6th index is the uncertainty
                uncertainty = pred[..., 6:] # B, T, 6
                pred6d = pred[..., :6]
                uncertainties.append(torch.mean(uncertainty, dim=-1, keepdim=True))
            else:
                pred6d = pred

            rot6d_preds.append(pred6d) # (B, T, 6)

            abs_poses.append(torch.stack(batch['abs_poses']).reshape(B, T, -1)) # (B, T, 4, 4)
            if pc_can is None:
                pts = batch['pclouds'][0]
                pc_can = torch.matmul(pts, batch['abs_poses'][0][0, :3, :3]).to('cuda')
            cam_K.append(torch.stack(batch['K']).reshape(B, T, -1))
            cam_trans.append(torch.stack(batch['T']).reshape(B, T, -1))

            # save image
            if cfg.run.sample_save_gt:
                # Save input images
                filestr = str(output_dir / '{dir}' / '{category}' / '{name}.{ext}')
                for i, file_list in enumerate(batch['image_path']):
                    for j, file in enumerate(file_list):
                        ss = file.split(os.sep)
                        sequence_category, sequence_name = ss[-3], ss[-2]
                        filename = filestr.format(dir='images', category=sequence_category, name=sequence_name, ext='png')
                        if not osp.isfile(filename):
                            os.makedirs(osp.dirname(filename), exist_ok=True)
                            TVF.to_pil_image(batch['images'][i][j]).save(filename)

        # convert it to a sequence
        rot6d_preds = torch.cat(rot6d_preds, 0)
        rot6d_seq = clips2seq_fast(rot6d_preds, 1, rot6d_preds.shape[1])
        rot_seq = rot6d_to_rotmat(rot6d_seq)
        assert len(rot6d_seq) == len(file_paths), f'length not match: {len(rot6d_seq)}!={len(file_paths)}!'
        # rot_seq2 = clips2seq_fast(torch.cat(rot6d_preds_real, 0), 1, rot6d_preds.shape[1])
        # rot_seq2 = rot6d_to_rotmat(rot_seq2)

        # save results
        abs_poses = torch.cat(abs_poses, 0)
        abs_poses_seq = clips2seq_fast(abs_poses, 1, abs_poses.shape[1])
        assert len(abs_poses_seq) == len(file_paths), f'length not match: {len(abs_poses_seq)}!={len(file_paths)}!'
        abs_rots = abs_poses_seq.reshape(len(file_paths), 4, 4)[:, :3, :3]

        cam_trans = torch.cat(cam_trans, 0)
        cam_K = torch.cat(cam_K, 0)
        cam_trans_seq = clips2seq_fast(cam_trans, 1, abs_poses.shape[1])
        cam_K_seq = clips2seq_fast(cam_K, 1, abs_poses.shape[1]).reshape(len(file_paths), 4, 4)

        # compute uncertainty
        if len(uncertainties) > 0:
            uncertainties = torch.cat(uncertainties, 0)
            uncertainties = clips2seq_fast(uncertainties, 1, uncertainties.shape[1])
            print('uncertainties shape after conversion:', uncertainties.shape)

        ss = str(file_paths[0]).split(os.sep)
        filestr = str(output_dir / '{dir}' / '{category}' / '{name}.{ext}')
        sequence_category = ss[-3]
        (output_dir / 'pred' / sequence_category).mkdir(exist_ok=True, parents=True)
        (output_dir / 'metadata' / sequence_category).mkdir(exist_ok=True, parents=True)

        align_file = osp.join(cfg.dataset.demo_data_path, 'behave2shapenet_alignment.pkl')
        if cfg.dataset.align_objav:
            print("Using only shapenet alignment, no objaverse!")
            # the prediction is already in shapenet space
            align_file = osp.join(cfg.dataset.demo_data_path, 'behave2shapenet_alignment_old.pkl')
        behave2shapenet = pkl.load(open(align_file, 'rb'))
        obj_name = sequence_category.split('_')[2]
        align2canonical = obj_name in behave2shapenet.keys() and ('proc15fps' in cfg.run.name or 'so3smpl' in cfg.run.name or 'proc5obj' in cfg.run.name) and 'real' not in cfg.run.name
        if cfg.dataset.all_shapenet_pose:
            align2canonical = True # always align for this setup
        # align2canonical = False
        if align2canonical:
            print(f"Computing alignment to canonical shapenet, alignment loaded from file {align_file}")
        else:
            print("Not computing alignment")
        for i, file in enumerate(file_paths):
            rot = rot_seq[i]
            # change the alignment for models that predict shapenet pose
            if align2canonical:
                align = torch.from_numpy(behave2shapenet[obj_name][:3, :3]).float().to('cuda') # this is from behave canonical to shapenet canonical
                rot = torch.matmul(rot, align) # first apply canonical to shapenet, and then the predicted rotation

            pc_i = torch.matmul(pc_can, rot.T)

            sequence_name = osp.basename(osp.dirname(file))
            filename = filestr.format(dir='pred', category=sequence_category, name=sequence_name, ext='ply')
            trimesh.PointCloud(pc_i.cpu().numpy()).export(filename)

            # save metadata as well
            camera = PerspectiveCameras(
                R=torch.tensor([[[-1, 0, 0.],
                                [0, -1., 0],
                                [0, 0, 1.]]]),
                T=cam_trans_seq[i:i+1],
                K=cam_K_seq[i:i+1]
            )
            metadata = dict(index=i, sequence_name=sequence_name,
                            sequence_category=sequence_category,
                            camera=camera,
                            image_path=file,
                            rotation=rot,
                            uncertainty=None if len(uncertainties) == 0 else uncertainties[i]
                            )
            filename = filestr.format(dir='metadata', category=sequence_category, name=sequence_name, ext='pth')
            torch.save(metadata, filename)

            # save gt as well
            if cfg.run.sample_save_gt:
                pc_i_gt = torch.matmul(pc_can, abs_rots[i].T)
                filename = filestr.format(dir='gt', category=sequence_category, name=sequence_name, ext='ply')
                if not osp.isfile(filename):
                    os.makedirs(osp.dirname(filename), exist_ok=True)
                    trimesh.PointCloud(pc_i_gt.cpu().numpy()).export(filename)

        # add synlink
        # for pat in ['images', 'gt']:
        #     if pat == 'gt' and cfg.run.sample_save_gt:
        #         continue
        #     syn_file = output_dir/pat
        #     if not osp.islink(str(syn_file)):
        #         cmd = f'ln -s /BS/xxie-2/work/pc2-diff/experiments/outputs/so3_reg-dinotune-16-scale0.5bemb/single/15fps/{pat} {str(syn_file)}'
        #         os.system(cmd)
        print(f"Saved to {output_dir.absolute()}, all done.")


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    cfg.run.job = 'sample' # make sure no shuffle!
    trainer = SamplerAvgRot(cfg)
    import traceback
    try:
        trainer.run_sample(cfg)
    except Exception as e:
        print(traceback.format_exc())


if __name__ == '__main__':
    main()