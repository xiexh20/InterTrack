"""
Main entry point for the CorrAE model
"""
import sys, os
import time
from typing import Iterable, Optional

import hydra
import torch
import trimesh, pickle

import wandb, imageio, cv2, json
import numpy as np
from accelerate import Accelerator
from pytorch3d.ops import knn_points
from torch.distributed.elastic.multiprocessing.errors import record
from torchvision.transforms import functional as TVF
from pathlib import Path
import os.path as osp

from configs.structured import ProjectConfig

sys.path.append(os.getcwd())

# from trainer import Trainer
from main import TrainerBehave
import training_utils


class TrainerAutoencoder(TrainerBehave):
    def compute_loss(self, batch, model):
        "return a loss "
        device = 'cuda'
        # model forward and then compute chamfer
        points = torch.stack([x.to('cuda') for x in batch['pclouds']], 0) # (B, N, 3)
        points_gt = torch.stack([x.to('cuda') for x in batch['pclouds_gt']], 0) if 'pclouds_gt' in batch else points
        pred = model(points)
        closest_dist_in_s2 = knn_points(points_gt, pred, K=1)
        closest_dist_in_s1 = knn_points(pred, points_gt, K=1)

        chamf = closest_dist_in_s2.dists.mean()*500 + closest_dist_in_s1.dists.mean()*500 # squared chamf distance

        # also v2v loss, very small
        if self.v2v_loss > 0.:
            gt = torch.stack([x.to('cuda') for x in batch['pclouds_ordered']], 0) # (B, N, 3)
            v2v = torch.sum((pred-gt)**2, -1).mean() * self.v2v_loss
            self.loss_sep = (chamf, v2v)
            loss = chamf + v2v
        else:
            loss = chamf
            self.loss_sep = (chamf, 0.)

        return loss

    def add_log_item(self, metric_logger):
        ""
        metric_logger.add_meter('train_loss_chamf', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('train_loss_v2v', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        return metric_logger

    def logging_addition(self, log_dict:dict):
        log_dict['train_loss_chamf'] = float(self.loss_sep[0])
        log_dict['train_loss_v2v'] = float(self.loss_sep[1])
        return log_dict

    @torch.no_grad()
    def visualize(
            self,
            cfg: ProjectConfig,
            model: torch.nn.Module,
            dataloader_vis: Iterable,
            accelerator: Accelerator,
            identifier: str = '',
            num_batches: Optional[int] = None,
            output_dir: str = 'vis',
    ):
        """
        render the video and save
        :param cfg:
        :param model:
        :param dataloader_vis:
        :param accelerator:
        :param identifier:
        :param num_batches:
        :param output_dir:
        :return:
        """

        from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras
        from render.pyt3d_wrapper import MeshRendererWrapper, get_kinect_camera, PcloudRenderer
        from pytorch3d.renderer import look_at_view_transform
        from pytorch3d.structures import Pointclouds
        import imageio

        # Eval mode
        model.eval()
        device = 'cuda'
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        progress_bar = metric_logger.log_every(dataloader_vis, cfg.run.print_step_freq, "Vis")

        output_dir: Path = Path(output_dir)
        (output_dir / 'videos').mkdir(exist_ok=True, parents=True)
        # step = int(identifier)
        video_file = str(output_dir / 'videos'/ f"step-{identifier}.mp4")
        video_writer = imageio.get_writer(video_file, format='FFMPEG', mode='I', fps=1)

        rend_size = cfg.model.image_size
        renderer = PcloudRenderer(image_size=rend_size, radius=0.0075)

        wandb_log_dict = {}
        fscores, chamfs = [], []
        vc_points = None
        for batch_idx, batch in enumerate(progress_bar):
            if num_batches is not None and batch_idx >= num_batches:
                break

            points = torch.stack([x.to('cuda') for x in batch['pclouds']], 0)  # (B, N, 3)
            pred = model(points)

            # compute fscore and chamf
            for i in range(len(pred)):
                cd, fscore = self.compute_fscore_chamf(points[i].cpu().numpy(), pred[i].cpu().numpy(), 0.1)
                fscores.append(fscore)
                chamfs.append(cd)

            # render pc and visualize
            if vc_points is None:
                vc_points = self.visu(pred[0].cpu().numpy())
            features = torch.from_numpy(vc_points[None]).repeat(len(pred), 1, 1).to(device)
            pc = Pointclouds(pred, features=features)
            cam = PerspectiveCameras(R=torch.stack(batch['R'], 0),
                                     T=torch.stack(batch['T'], 0),
                                     K=torch.stack(batch['K'], 0),
                                     in_ndc=True,
                                     device='cuda')
            # side camera
            at = torch.zeros(len(pred), 3)
            R, T = look_at_view_transform(2.5, 0, 80, up=((0, -1, 0),),
                                          at=at, device=device)
            side_camera = PerspectiveCameras(image_size=((rend_size, rend_size),),
                                             device=device,
                                             R=R, T=T,
                                             focal_length=rend_size * 1.5,
                                             principal_point=torch.tensor(((rend_size / 2., rend_size / 2.))).repeat(len(pred),1).to(device),
                                             in_ndc=False)
            rends, masks = [torch.stack(batch['images'], 0).permute(0, 2, 3, 1).cpu().numpy()], []
            for ii, cam in enumerate([cam, side_camera]):
                rend, mask = renderer.render(pc, cam, mode='mask')
                rends.append(rend)
            rend = np.concatenate(rends, 2) # (B, H, W*3, 3)
            for ii, img in enumerate(rend):
                image_path = batch["image_path"][ii]
                ss = str(image_path).split(os.sep)
                text = f'{ss[-3]}_{ss[-2]}_{ss[-1][:2]}'
                comb = (img * 255).astype(np.uint8).copy()
                # img_idx = int(batch['idx'][ii])
                cv2.putText(comb, text, (50, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 0, 0), 2)  # color (0, 255, 255)=bright blue, same as human color
                video_writer.append_data(comb)

        video_writer.close()
        print("Video saved to", video_file)
        return fscores, chamfs, np.zeros((len(chamfs), ))

    def visu(self, vertices):
        """
        compute a color map for all vertices
        Parameters
        ----------
        vertices

        Returns
        -------

        """
        min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
        cmap = (vertices - min_coord) / (max_coord - min_coord)
        return cmap

    @torch.no_grad()
    def sample(self, cfg: ProjectConfig,
                model: torch.nn.Module,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        """
        similar to visulize, but now we have to save the name based on batches
        same output format as pc2: save predicted pc with color
        :param cfg:
        :param model:
        :param dataloader:
        :param accelerator:
        :param output_dir:
        :return:
        """
        from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
        from pytorch3d.structures import Pointclouds
        from tqdm import tqdm

        # Eval mode
        model.eval()
        progress_bar: Iterable[FrameData] = tqdm(dataloader, disable=(not accelerator.is_main_process))

        # Output dir
        output_dir: Path = Path(output_dir)
        end_idx = cfg.run.batch_end if cfg.run.batch_end is not None else len(dataloader)
        vc_points = None

        # init SMPL fitter
        from smplfitter.pt import BodyModel, BodyFitter
        body_model = BodyModel('smplh', 'male',
                               model_root='/BS/xxie2020/static00/mysmpl/smplh').to('cuda')  # create the body model to be fitted
        fitter = BodyFitter(body_model, num_betas=10).to('cuda')

        # Visualize
        for batch_idx, batch in enumerate(progress_bar):
            progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d}')
            if cfg.run.num_sample_batches is not None and batch_idx >= cfg.run.num_sample_batches:
                break

            # for debug: save sampled frames
            filename = '{name}.{ext}'
            filestr = str(output_dir / '{dir}' / '{category}' / filename)
            sequence_category = self.get_seq_category(batch, 0)  # TODO: replace for different dataset

            file = filestr.format(dir='images', category=sequence_category, name=f"batch_{batch_idx:02d}",
                                  ext='json')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            json.dump(batch['image_path'], open(file, 'w'))
            print("sequence:", sequence_category, 'first image:', batch['image_path'][0])
            # continue

            # Optionally produce multiple samples for each point cloud
            for sample_idx in range(cfg.run.num_samples):
                if self.is_done(batch, output_dir) and not cfg.run.redo:
                    print(f"batch {batch_idx} already done, skipped")
                    continue

                # Filestring
                filename = f'{{name}}-{sample_idx}.{{ext}}' if cfg.run.num_samples > 1 else '{name}.{ext}'
                filestr = str(output_dir / '{dir}' / '{category}' / filename)

                start = time.time()
                key, points_gt, pred = self.forward_batch(batch, cfg, model, ret_latent=True)
                end = time.time()
                # Length of pred: 2, total time: 0.035, avg time: 0.001
                print(f"Length of pred: {len(pred)}, total time: {end-start:.3f}, avg time: {(end-start)/cfg.dataloader.batch_size:.5f}") # 16???
                pred, latent = pred # latent code: (B, D)
                if vc_points is None:
                    vc_points = self.visu(pred[0].cpu().numpy())

                # Feb13, 2025: do fast IK to obtain SMPLH parameters
                # normalize prediction
                if key == 'pred_hum':
                    cent = torch.mean(pred, 1, keepdim=True)
                    scale = torch.sqrt(torch.max(torch.sum((pred-cent)**2, -1), -1)[0]) # (B, )
                    vertices = 0.85 * (pred - cent) / scale[:, None, None]  # total 1.7m tall
                    fit_res = fitter.fit(vertices, num_iter=3, beta_regularizer=1,
                             requested_keys=['shape_betas', 'trans', 'vertices', 'pose_rotvecs'])
                    # now scale back
                    fit_res['vertices'] =  fit_res['vertices'] * scale[:, None, None] / 0.85 + cent

                # Save individual samples
                for i in range(len(pred)):
                    sequence_name = self.get_seq_name(batch, i) # + f'_{i}'  # this is the frame name
                    sequence_category = self.get_seq_category(batch, i) # folder name

                    # pred_key = 'aeout' if key == 'pred_hum' else 'pred'
                    pred_key = 'pred'
                    (output_dir / 'gt' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / pred_key / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'images' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'metadata' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'input' / sequence_category).mkdir(exist_ok=True, parents=True)

                    if key == 'pred_hum':
                        # undo normalization to place back to interaction space
                        pc = pred[i] * batch['radius_hum_pred'][i] * 2 + batch['cent_hum_pred'][i]

                        # save SMPL registration results
                        verts = fit_res['vertices'][i] * batch['radius_hum_pred'][i] * 2 + batch['cent_hum_pred'][i]
                        verts = verts.cpu().numpy()
                        outfile = filestr.format(dir='smplh', category=sequence_category, name=sequence_name, ext='ply')
                        os.makedirs(osp.dirname(outfile), exist_ok=True)
                        trimesh.PointCloud(verts, vc_points).export(outfile)
                        cent_i = batch['cent_hum_pred'][i] + cent[i, 0]*batch['radius_hum_pred'][i] * 2

                        params = {
                            "pose": fit_res['pose_rotvecs'][i].cpu().numpy(),
                            "betas": fit_res['shape_betas'][i].cpu().numpy(),
                            "trans": fit_res['trans'][i].cpu().numpy(),
                            # normalization parameters
                            "center": cent_i.cpu().numpy(),
                            'scale': batch['radius_hum_pred'][i].cpu().numpy() * scale[i].cpu().numpy()/0.85
                        }
                        with open(outfile.replace('.ply', '.pkl'), 'wb') as f:
                            pickle.dump(params, f)

                        # print("Undo normalization")
                        pgt = (points_gt[i] * batch['radius_hum_pred'][i] * 2 + batch['cent_hum_pred'][i]).cpu().numpy()
                    elif key == 'pred_obj':
                        # undo normalization to place back to interaction space
                        pc = pred[i] * batch['radius_obj_pred'][i] * 2 + batch['cent_obj_pred'][i]
                        # print("Undo normalization")
                        pgt = (points_gt[i] * batch['radius_obj_pred'][i] * 2 + batch['cent_obj_pred'][i]).cpu().numpy()
                    else:
                        pc = pred[i]
                        # verts = fit_res['vertices'][i]
                        pgt = points_gt[i].cpu().numpy()
                    pc = pc.cpu().numpy()
                    # Save ground truth and predicted pc
                    trimesh.PointCloud(pc, vc_points).export(filestr.format(dir=pred_key,
                                                                                       category=sequence_category,
                                                                                       name=sequence_name,
                                                                                       ext='ply'))


                    if cfg.run.sample_save_gt and key != 'pred_hum':
                        trimesh.PointCloud(pgt).export(filestr.format(dir='gt',
                                                                                    category=sequence_category,
                                                                                       name=sequence_name,
                                                                                       ext='ply'))

                    # save input
                    # if key == 'pred_hum':
                    #     pc_input = batch['pred_hum'][i] * batch['radius_hum_pred'][i] * 2 + batch['cent_hum_pred'][i]
                    #     trimesh.PointCloud(pc_input.cpu().numpy()).export(filestr.format(dir='input',
                    #                                                   category=sequence_category,
                    #                                                   name=sequence_name,
                    #                                                   ext='ply'))
                    # Save input images if not for AE inference
                    if key != 'pred_hum':
                        filename = filestr.format(dir='images', category=sequence_category, name=sequence_name,
                                                  ext='png')
                        TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)

                        # Save camera
                        filename = filestr.format(dir='metadata', category=sequence_category, name=sequence_name,
                                                  ext='pth')
                        metadata = self.get_metadata(batch, i)
                        metadata['latent'] = latent[i].cpu() # and also latent for later optimization
                        torch.save(metadata, filename)

        print('Saved samples to: ')
        print(output_dir.absolute())

    def forward_batch(self, batch, cfg, model, ret_latent=False):
        """
        one feedforward step
        :param batch:
        :param cfg:
        :param model:
        :return: points_gt, to be saved as GT points,
        """
        if cfg.run.sample_mode == 'sample':
            key = 'pclouds'
        elif cfg.run.sample_mode == 'interm-obj':
            key = 'pred_obj'
        elif cfg.run.sample_mode == 'interm-hum':
            key = 'pred_hum' # this is for humanae
        else:
            raise ValueError(f"Unknown sample mode {cfg.run.sample_mode}")
        points = torch.stack([x.to('cuda') for x in batch[key]], 0)  # (B, N, 3)
        if 'pclouds_ordered' not in batch:
            points_gt = torch.stack([x.to('cuda') for x in batch[key]], 0)  # input points
            print(f"Warning: no GT corr points found! using {key} as gt.")
        else:
            points_gt = torch.stack([x.to('cuda') for x in batch['pclouds_ordered']], 0)  # ordered GT points
        # points_gt = torch.stack([x.to('cuda') for x in batch[key]], 0)  # input points

        print("Inputing", key)
        pred = model(points, ret_latent=ret_latent)
        return key, points_gt, pred



@record
@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    trainer = TrainerAutoencoder(cfg) # for SMPL correspondence

    if cfg.run.job == 'sample':
        trainer.run_sample(cfg)
    else:
        trainer.train(cfg)


if __name__ == '__main__':
    main()