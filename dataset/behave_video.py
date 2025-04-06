"""
for behave video data
"""
import time

import cv2
import os, sys
import numpy as np
import torch
import pickle as pkl
from copy import deepcopy
import os.path as osp

import trimesh
from scipy.spatial.transform import Rotation
from .behave_dataset import BehaveDataset
from .behave_dataset import DataPaths
# from .behave_dataset import BehaveObjOnly
import joblib
import multiprocessing as mp
from tqdm import tqdm
from model.geometry_utils import axis_to_rot6D, numpy_axis_to_rot6D
from lib_smpl import get_smpl
from lib_smpl.body_landmark import BodyLandmarks
from .img_utils import resize, crop

scratch_path = "/scratch/inf0/user/xxie/behave/" # SSD file system
old_path_behave = "/BS/xxie-4/static00/behave-fps30/" # old 30fps data
old_path_icap = "/BS/xxie-6/static00/InterCap/"

class BehaveObjectVideoDataset(BehaveDataset):
    def __init__(self, data_paths, clip_len, window, 
                 num_samples, fix_sample=True,
                 input_size=(224, 224), split=None,
                 sample_ratio_hum=0.5,
                 normalize_type='comb', **kwargs):
        """
        
        Parameters
        ----------
        data_paths : a list of list of frames, different from frame based dataloader, here each element is a list
        clip_len : T, how many frames are used as one example
        window : sliding window, distance between two clips
        num_samples : 
        fix_sample : 
        input_size : 
        split : 
        sample_ratio_hum : 
        normalize_type : 
        kwargs : 
        """
        super().__init__(data_paths, num_samples, fix_sample, input_size, split, sample_ratio_hum, normalize_type, **kwargs)

        # generate clips
        self.clip_len = clip_len
        # self.init_paths(clip_len, data_paths, window)
        self.data_paths_seqs = data_paths # a list of seq lists, each element in the list in a list of image paths

        self.smpl_src = kwargs.get('smpl_src', 'gt')
        self.data_paths, self.start_inds, self.seq_inds = self.init_paths(clip_len, data_paths, window)
        # self.image_cache = mp.Manager().dict() # cache for images
        self.image_cache = {}
        self.init_others()
        self.align_objav = kwargs.get('align_objav', False) # align objaverse pose to shapenet pose
        self.all_shapenet_pose = kwargs.get('all_shapenet_pose', False) # the GT rotation is always from shapenet cannonical space or not
        print("Number of examples:", len(self.start_inds), ', align objaverse?', self.align_objav, 'all shapenet pose?', self.all_shapenet_pose)


    def preload_images(self, files):
        for rgb_file in tqdm(files):
            # rgb_file = self.data_paths[ind]
            Kroi, obj_mask, person_mask, rgb = self.load_images_from_files(rgb_file)
            mask = np.stack([person_mask, obj_mask], 0)
            img_key = self.get_cache_key(rgb_file)
            self.image_cache[img_key] = Kroi, (rgb * 255).astype(np.uint8), mask.astype(bool)

    def init_others(self):
        return

    @staticmethod
    def get_num_batches(seqs, clip_len, window):
        count = 0
        for seq in seqs:
            # data_paths.extend(seq)
            for i in range(0, len(seq) - clip_len + 1, window):
                count += 1
        return count

    def init_paths(self, clip_len, seqs, window):
        packed_data_path = self.behave_packed_dir
        complete_keys = ['frames', 'obj_angles',
                         'obj_trans', 'poses', 'betas', 'trans', 'gender', 'occ_ratios',
                            "joints_smpl",
                            "joints_body", # body 25 joints
                            # "joints_hand",
                            # "joints_face",
                         ]
        data_complete = True
        # self.gt_data = mp.Manager().dict() # is this very slow?? yes! super slow in cluster!
        self.gt_data = {}
        data_paths, start_inds, seq_index = [], [], []  # all paths, and index to start image

        offset = 0
        smpl_neutral = get_smpl('neutral', False)
        landmarks = BodyLandmarks(osp.join(self.demo_data_path, 'assets'))
        for sid, seq in enumerate(tqdm(seqs)):
            data_paths.extend(seq)
            for i in range(0, len(seq) - clip_len + 1, window):
                start_inds.append(i + offset)
                seq_index.append(sid)
            offset += len(seq)

            seq_name = DataPaths.get_seq_name(seq[0])
            if seq_name in self.gt_data:
                continue
            data = joblib.load(osp.join(packed_data_path, f'{seq_name}_GT-packed.pkl'))

            # for mocap, compute joints smpl and body online
            if self.smpl_src == 'mocap':
                kid = 1 if 'Date0' in seq_name else 0
                mocap_betas, mocap_poses = data['mocap_betas'][:, kid], data['mocap_poses'][:, kid]
                smpl_verts, smpl_jtrs, _, _ = smpl_neutral(torch.from_numpy(mocap_poses).float(),
                                                           torch.from_numpy(mocap_betas).float(),
                                                           torch.zeros(len(mocap_betas), 3))
                joints_body = landmarks.get_landmarks_batch(smpl_verts)[0]
                joints_smpl = np.zeros((len(joints_body), 52, 3)) # to have the same shape as original SMPL joints
                joints_smpl[:, :22] = smpl_jtrs.numpy()[:, :22]
                joints_smpl[:, 37:38] = smpl_jtrs.numpy()[:, 23:] # TODO: might be a bug here!
                data["joints_smpl"] = joints_smpl
                data["joints_body"] = joints_body.numpy()
                # also replace GT SMPL pose with predicted ones
                smplh_poses = np.zeros((len(joints_body), 156))
                smplh_poses[:, :69] = mocap_poses[:, :69]
                smplh_poses[:, 111:114] = mocap_poses[:, 69:72]
                assert len(smplh_poses) == len(data['poses']), f'mocap poses do not match for {seq_name}!'
                data['poses'] = smplh_poses # this is also a conditioning
                print(f'Using mocap joints for {seq_name}. {joints_body.shape}, {smpl_jtrs.shape}')

            # check data completion
            for k in complete_keys:
                if k not in data:
                    print(f'{k} does not exist in {seq_name}!')
                    data_complete = False
            self.gt_data[seq_name] = data

            if not data_complete:
                print('Data incomplete, exiting...')
                exit(-1)

        # sanity check
        for ind in start_inds[len(start_inds) // 2:]:
            assert ind + clip_len <= len(data_paths), f'ind={ind}, clip_len={clip_len}, total: {len(self.data_paths)}'

        return data_paths, np.array(start_inds), np.array(seq_index)

    def __len__(self):
        return len(self.start_inds)

    def load_masks(self, rgb_file, flip=False):
        old_path = old_path_behave if old_path_behave in rgb_file else old_path_icap
        person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.png").replace(old_path, scratch_path)
        obj_mask_file = rgb_file.replace('.color.jpg', ".obj_rend_mask.png").replace(old_path, scratch_path)
        if not osp.isfile(person_mask_file):
            person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.png")
        if not osp.isfile(obj_mask_file):
            obj_mask_file = rgb_file.replace('.color.jpg', ".obj_rend_mask.png")

        person_mask = cv2.imread(person_mask_file, cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(obj_mask_file, cv2.IMREAD_GRAYSCALE)

        return person_mask, obj_mask
    
    def get_item(self, idx):
        """
        return one template object points and corresponding object pose for each frame
        Parameters
        ----------
        idx : index of the data

        Returns
        -------

        """
        start_time = time.time()
        image_files = self.get_chunk_files(idx)

        # suggested by https://discuss.pytorch.org/t/training-crashes-due-to-insufficient-shared-memory-shm-nn-dataparallel/26396/41
        # to avoid out of shared memory
        L = len(image_files)
        rgbs, masks, samples, Krois = np.empty((L, 3, *self.input_size)), np.empty((L, 2, *self.input_size)), None, np.empty((L, 4, 4))
        transforms, can_pose = np.empty((L, 4, 4)), np.eye(4)
        poses_abs, centers, radius_all = np.empty((L, 4, 4)), np.empty((L, 3)), np.empty((L, ))
        camera_T, cent_0 = np.empty((L, 3)), None
        occ_ratios = np.empty((L, ))
        smpl_poses, smpl_joints, body_joints25 = np.empty((L, 24, 6)), np.empty((L, 23, 3)), np.empty((L, 25, 3))
        frame_indices = np.empty((L, )) # index for each frame used to compute positional encoding

        time_images = 0
        can_points = None
        for i, rgb_file in enumerate(image_files):
            ttstart = time.time()
            kroi, mask, rgb = self.load_rgb_masks(rgb_file)
            # Krois.append(kroi)
            # rgbs.append(rgb)
            # masks.append(mask)
            Krois[i] = kroi
            rgbs[i] = rgb
            masks[i] = mask
            time_images += time.time() - ttstart

            date, kid = DataPaths.get_seq_date(rgb_file), DataPaths.get_kinect_id(rgb_file)
            w2c = self.kin_transforms[date].world2local_mat(kid)
            c2w = np.linalg.inv(w2c)
            if i == 0:
                obj_category = DataPaths.rgb2object_name(rgb_file)
                rot, trans = self.get_obj_params(rgb_file)
                can_pose[:3, :3] = rot  # the coordinate of canonical object in this small clip
                can_pose[:3, 3] = trans  # always the canonical object space to k1 transform
                # if obj_category not in self.obj_samples_cache: # this is useless for synthetic data
                if can_points is None:
                    # print('reloading sample for file', rgb_file, 'category:', obj_category)
                    # load mesh and do sample, to obtain points in canonical space
                    samples_obj = self.sample_obj_gt(rgb_file)

                    # get object samples in canonical space
                    live2can = np.linalg.inv(can_pose)
                    # self.obj_samples_cache[obj_category] = np.matmul(samples_obj, live2can[:3, :3].T) + live2can[:3, 3]
                    can_points = np.matmul(samples_obj, live2can[:3, :3].T) + live2can[:3, 3]
                samples_can = can_points
                # samples_can = self.obj_samples_cache[obj_category]
                # randomly shuffle the point orders
                indices = np.random.choice(len(samples_can), len(samples_can), replace=False)
                samples_can = samples_can[indices]
                samples_obj = np.matmul(samples_can, rot.T) + trans # canonical to current frame
                # also consider the world to local camera transform
                samples_obj = np.matmul(samples_obj, w2c[:3, :3].T) + w2c[:3, 3]
                samples = samples_obj
                pose_i = np.eye(4)  # this is the relative pose

            else:
                # only load params, and then compute a transform
                rot, trans = self.get_obj_params(rgb_file)

                pose_i = np.eye(4)
                pose_i[:3, :3] = rot
                pose_i[:3, 3] = trans
                # first transform to world, then to object canonical space, then to current frame, finally to local camera
                pose_i = w2c @ pose_i @ np.linalg.inv(can_pose) @ c2w # relative pose

            # abs pose
            pose_iabs = np.eye(4)
            pose_iabs[:3, :3] = rot
            pose_iabs[:3, 3] = trans
            pose_iabs = np.matmul(w2c, pose_iabs)

            # compute camera center
            samples_i = np.matmul(samples, pose_i[:3, :3].T) + pose_i[:3, 3]
            cent, radius = self.unit_sphere_params(samples_i)
            cam_t = cent*np.array([-1, -1, 1.])/(2*radius)
            # camera_T.append(torch.from_numpy(cam_t).float())
            camera_T[i] = cam_t

            if i == 0:
                cent_0 = cent / (2*radius)
            # print(f'frame {self.get_cache_key(image_files[0])} {i} radius:', radius) # same object always have same radius
            pose_i[:3, 3] = pose_i[:3, 3]/(2*radius) # also update relative pose, mainly the translation
            # add translation of first frame cent_0 (from normalized to frame 0 space)
            # this would transform from normalized frame 0, to frame i in opencv space (assume camera translation=0)
            pose_i[:3, 3:4] = np.matmul(pose_i[:3, :3], cent_0.reshape((3, 1))) + pose_i[:3, 3].reshape((3, 1))
            # now center the points to origin again in frame i
            pose_i[:3, 3] = pose_i[:3, 3] - cent / (2*radius) # this will be all zeros!

            # transforms.append(torch.from_numpy(pose_i))
            # centers.append(cent / (2*radius))
            # radius_all.append(radius) # to compute relative transform later
            # poses_abs.append(pose_iabs)
            transforms[i] = pose_i # this is relative pose
            centers[i] = cent / (2*radius)
            radius_all[i] = radius
            poses_abs[i] = pose_iabs

            # object occlusion
            seq_name, frame = DataPaths.rgb2seq_frame(rgb_file)
            data_idx = self.gt_data[seq_name]['frames'].index(frame)
            occ_ratios[i] = self.gt_data[seq_name]['occ_ratios'][data_idx, kid]

            # SMPL body pose
            self.load_human_cond(body_joints25, data_idx, i, seq_name, smpl_joints, smpl_poses, w2c, kid)

            if 'Subxx' in seq_name:
                frame_indices[i] = data_idx * 2 # the synthetic ProciGen is 15fps, need to convert index in 30fps data
            else:
                frame_indices[i] = data_idx # index directly to 30fps data

        # compute distance transform
        ttstart = time.time()
        masks = torch.from_numpy(masks)
        dt, masks = self.compute_dt(masks)
        time_dt = time.time() - ttstart

        # normalize points
        cent, radius = self.unit_sphere_params(samples)
        samples = (samples - cent) / (2*radius)
        ss = image_files[0].split(os.sep)
        T = len(image_files)
        # also precompute the points for each image
        if self.split != 'train':
            pts_all = torch.from_numpy(samples).float()[None].repeat(T, 1, 1)
            rela_poses = torch.from_numpy(transforms).float()
            pts_all = torch.matmul(pts_all, rela_poses[:, :3, :3].transpose(1, 2)) + rela_poses[:, :3, 3].unsqueeze(-2)

            smpl_poses = torch.from_numpy(smpl_poses).float()
            # maybe we should not convert np to torch?
            data_dict = {
                "R": torch.from_numpy(self.opencv2py3d[:3, :3]).float()[None].repeat(T, 1, 1),
                'T': torch.from_numpy(camera_T).float(),
                'K': torch.from_numpy(Krois).float(),
                'image_path': image_files,
                'images': torch.from_numpy(rgbs).float(),
                'masks': masks,
                "dist_transform": dt,
                'pclouds': torch.from_numpy(samples).float(),
                'rela_poses': rela_poses,
                "abs_poses": torch.from_numpy(poses_abs).float(),
                'points_all': pts_all,
                'occ_ratios': torch.from_numpy(occ_ratios).float(),
                'smpl_poses': smpl_poses,
                'smpl_joints': torch.from_numpy(smpl_joints).float(),
                'body_joints25': torch.from_numpy(body_joints25).float(),

                # backward compatibility
                "sequence_name": ss[-2],# the frame name
                "synset_id":ss[-3],

                "centers": torch.from_numpy(centers).float(),
                'radius': torch.from_numpy(radius_all).float(),

                'frame_indices': torch.from_numpy(frame_indices).float(),
            }
        else:
            # for training: do not use torch
            pts_all = samples[None].repeat(T, 0)
            rela_poses = transforms
            pts_all = np.matmul(pts_all, rela_poses[:, :3, :3].transpose(0, 2, 1)) + rela_poses[:, :3, 3][:, None]

            # smpl_poses = torch.from_numpy(smpl_poses).float()
            data_dict = {
                "R": self.opencv2py3d[:3, :3].copy()[None].repeat(T, 0),
                'T': camera_T,
                'K': Krois,
                'image_path': image_files,
                'images': rgbs.astype(np.float32),
                'masks': masks.numpy().astype(np.float32),
                "dist_transform": dt.numpy().astype(np.float32),
                'pclouds': samples.astype(np.float32),
                'rela_poses': rela_poses.astype(np.float32),
                "abs_poses": poses_abs.astype(np.float32),
                'points_all': pts_all,
                'occ_ratios': occ_ratios,
                'smpl_poses': smpl_poses,
                'smpl_joints': smpl_joints,
                'body_joints25': body_joints25,

                # backward compatibility
                "sequence_name": ss[-2],  # the frame name
                "synset_id": ss[-3],

                "centers": centers,
                'radius': radius_all,
                'frame_indices': frame_indices, # abs frame indices
            }
        # also load predicted object pose
        if self.pred_obj_pose_path is not None:
            # print('loading object pose from', self.pred_obj_pose_path)
            # load predicted object rotation/translation
            pred_poses = self.load_pred_obj_poses(image_files)
            data_dict = {
                **data_dict,
                'pred_poses': np.stack(pred_poses, 0)
            }

        # load predicted object translation
        if self.ho_segm_pred_path is not None:
            # use by default: /BS/xxie-2/work/pc2-diff/experiments/outputs/sround3_plvis-1m/single/s1ddims2ddim-fv-exp2/pred
            T_obj_scaled_all = self.load_obj_norm_params(centers, image_files)
            data_dict = {
                **data_dict,
                'T_obj_scaled': np.stack(T_obj_scaled_all, 0)
            }
        return data_dict

    def load_obj_norm_params(self, centers, image_files):
        T_obj_scaled_all = []
        for i, rgb_file in enumerate(image_files):
            ss = rgb_file.split(os.sep)
            pred_file = osp.join(self.ho_segm_pred_path, ss[-3], ss[-2] + ".ply")
            pc = trimesh.load_mesh(pred_file, process=False)
            mask_hum = pc.colors[:, 2] > 0.5
            pc_hum, pc_obj = np.array(pc.vertices[mask_hum]), np.array(pc.vertices[~mask_hum])
            # Normalization
            cent_hum, cent_obj = np.mean(pc_hum, 0), np.mean(pc_obj, 0)
            # scale_hum = np.sqrt(np.max(np.sum((pc_hum - cent_hum) ** 2, -1)))
            scale_obj = np.sqrt(np.max(np.sum((pc_obj - cent_obj) ** 2, -1)))
            T_ho_scaled = centers[i]  # the translation for H+O space
            T_obj_scaled = (T_ho_scaled + cent_obj) / (2 * scale_obj)
            T_obj_scaled *= np.array([-1, -1, 1])  # final camera T used to project object points
            T_obj_scaled_all.append(T_obj_scaled)
        return T_obj_scaled_all

    def load_pred_obj_poses(self, image_files):
        pred_poses = []
        for i, rgb_file in enumerate(image_files):
            ss = rgb_file.split(os.sep)
            pred_file = osp.join(self.pred_obj_pose_path, ss[-3], ss[-2] + ".pth")
            if not osp.isfile(pred_file):
                print(f"Warning: {pred_file} does not exist! use dummy rotation")
                rot_pred = np.eye(3)
            else:
                rot_pred = torch.load(pred_file, map_location='cpu', weights_only=False)['rotation'].numpy()
            pred_pose = np.eye(4)
            pred_pose[:3, :3] = rot_pred
            pred_poses.append(pred_pose)
        return pred_poses

    def load_human_cond(self, body_joints25, data_idx, i, seq_name, smpl_joints, smpl_poses, w2c, kid):
        "load data for human conditioning"
        pose_smpl = self.gt_data[seq_name]['poses'][data_idx]
        pose_smpl = np.concatenate([pose_smpl[:69], pose_smpl[111:114]])
        # compute a world to local for the global orientation
        grot = np.matmul(w2c[:3, :3], Rotation.from_rotvec(pose_smpl[:3]).as_matrix())
        pose_smpl[:3] = Rotation.from_matrix(grot).as_rotvec()
        # smpl_poses.append(numpy_axis_to_rot6D(pose_smpl.reshape(-1, 3)).reshape(24, 6))
        smpl_poses[i] = numpy_axis_to_rot6D(pose_smpl.reshape(-1, 3)).reshape(24, 6)
        jtrs_smpl = self.gt_data[seq_name]['joints_smpl'][data_idx]  # (52, 3)
        jtrs_smpl = np.matmul(jtrs_smpl, w2c[:3, :3].T)
        # smpl_joints.append(np.concatenate([jtrs_smpl[:22], jtrs_smpl[37:38]], 0))
        smpl_joints[i] = np.concatenate([jtrs_smpl[:22], jtrs_smpl[37:38]],
                                        0)  # TODO: might be a bug here, should have 24 joints!
        jts25 = self.gt_data[seq_name]['joints_body'][data_idx]
        jts25 = np.matmul(jts25, w2c[:3, :3].T)
        # body_joints25.append(jts25)
        body_joints25[i] = jts25

    def get_chunk_files(self, idx):
        """
        get the list of files for this chunk in the batch
        Parameters
        ----------
        idx :

        Returns
        -------

        """
        assert self.test_transl_type == 'norm', f'type {self.test_transl_type} not implemented!'
        start, end = self.start_inds[idx], self.start_inds[idx] + self.clip_len
        image_files = deepcopy(self.data_paths[start:end])
        self.check_paths(image_files)
        if np.random.uniform() > 0.5 and self.split == 'train':
            image_files = image_files[::-1]  # flip the order
        return image_files

    def sample_obj_gt(self, rgb_file):
        smpl_name, obj_name = self.get_gt_fit_names(rgb_file)
        obj_path = self.get_obj_filepath(rgb_file, smpl_name, obj_name)
        obj = self.load_obj_gtmesh(obj_path)
        samples_obj = obj.sample(self.num_samples)
        return samples_obj

    def compute_dt(self, masks):
        "masks: torch tensor, shape (T, 2, H, W)"
        dts = []
        # masks = torch.stack(masks, 0)
        image_size = masks.shape[-1]
        for i in range(2):
            distance_transform = torch.stack([
                torch.from_numpy(cv2.distanceTransform(
                    (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
                ) / (image_size / 2))
                for m in masks[:, i].numpy().astype(np.uint8)
            ]).unsqueeze(1).clip(0, 1)
            dts.append(distance_transform)
        dt = torch.cat(dts, 1).float()  # (T, 2, H, W)
        return dt, masks

    def load_rgb_masks(self, rgb_file):
        # Typical loading of RGB + masks: this is however very slow for video model training
        # Kroi, obj_mask, person_mask, rgb = self.load_images_from_files(rgb_file)
        # mask = np.stack([person_mask, obj_mask], 0)
        # self.image_cache[img_key] = Kroi, (rgb * 255).astype(np.uint8), mask.astype(bool)

        # Load preprocessed images directly from npz file
        # the npz file stores RGB + masks in network input size, hence significantly improves the IO speed
        npz_file = self.get_npz_file(rgb_file) # TODO: use your own processed npz file path
        npz_data = np.load(npz_file)
        Kroi, mask = npz_data['Kroi'], npz_data['mask'].astype(float)
        rgb = npz_data['rgb'].astype(float)/255. # loading npz file is much faster than loading rgb file!
        # end of loading from processed npz file

        kroi, rgb = torch.from_numpy(Kroi).float(), torch.from_numpy(rgb).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()

        return kroi, mask, rgb

    def get_npz_file(self, rgb_file):
        # TODO: prepare your own npz file here
        ss = rgb_file.split(os.sep)
        npz_file = f'/scratch/inf0/user/xxie/behave-images/{ss[-3]}/{ss[-2]}_{ss[-1][:2]}_obj-crop.npz'
        if not osp.isfile(npz_file):
            npz_file = f'/scratch/inf0/user/xxie/behave-images/{ss[-3]}/{ss[-2]}_{ss[-1][:2]}_hum-obj.npz'
        return npz_file

    def load_images_from_files(self, rgb_file):
        mask_hum, mask_obj = self.load_masks(rgb_file)
        rgb_full = self.load_rgb(rgb_file)
        color_h, color_w = rgb_full.shape[:2]
        # print(f"Time to load images: {time.time()-ttstart:.4f}")
        bmax, bmin, crop_center, crop_size = self.get_crop_params(mask_hum, mask_obj)
        # crop
        rgb = resize(crop(rgb_full, crop_center, crop_size), self.input_size) / 255.
        person_mask = resize(crop(mask_hum, crop_center, crop_size), self.input_size) / 255.
        obj_mask = resize(crop(mask_obj, crop_center, crop_size), self.input_size) / 255.
        # mask bkg out
        mask_comb = (person_mask > 0.5) | (obj_mask > 0.5)
        rgb = rgb * np.expand_dims(mask_comb, -1)
        xywh = np.concatenate([crop_center - crop_size // 2, np.array([crop_size, crop_size])])
        Kroi = self.compute_K_roi(xywh, rgb_full.shape[1], rgb_full.shape[0])
        return Kroi, obj_mask, person_mask, rgb

    def get_crop_params(self, mask_hum, mask_obj, bbox_exp=1.0):
        ""
        area = np.sum(mask_obj > 127)
        # if area < 400:
        if area < 30:  # Feb24, 2024
            # use human + object
            bmax, bmin, crop_center, crop_size = super().get_crop_params(mask_hum, mask_obj, bbox_exp=1.0)
        else:
            # object only
            # bmax, bmin, crop_center, crop_size = super().get_crop_params(mask_obj, mask_obj, bbox_exp=1.5)
            bmax, bmin, crop_center, crop_size = super().get_crop_params(mask_obj, mask_obj, bbox_exp=1.5)
        return bmax, bmin, crop_center, crop_size

    def get_obj_params(self, rgb_file):
        seq_name, frame = DataPaths.rgb2seq_frame(rgb_file)
        data_idx = self.gt_data[seq_name]['frames'].index(frame)
        rot = Rotation.from_rotvec(self.gt_data[seq_name]['obj_angles'][data_idx]).as_matrix()
        trans = self.gt_data[seq_name]['obj_trans'][data_idx]

        # add an additional shapenet to objaverse alignment
        obj_cat = seq_name.split('_')[2]
        is_objav = obj_cat in ['backpack', 'suitcase', 'boxlarge', 'boxlong', 'boxmedium', 'boxsmall',
                               'boxtiny', 'yogaball', 'basketball', 'stool', 'obj01']

        if self.all_shapenet_pose and 'Subxx' not in rgb_file:
            # for real data, also transform the rotation to transform from shapenet
            if 'yoga' not in obj_cat:
                rot = np.matmul(rot, self.behave2shapenet[obj_cat][:3, :3].T) # multiply shapenet2behave transform
            # print('aligned with behave2shapenet!')

        return rot, trans

    def load_rgb(self, rgb_file):
        old_path = old_path_behave if old_path_behave in rgb_file else old_path_icap
        rgb_file_fast = rgb_file.replace(old_path, scratch_path)
        if not osp.isfile(rgb_file_fast):
            rgb_file_fast = rgb_file
        rgb_full = cv2.imread(rgb_file_fast)[:, :, ::-1]
        return rgb_full

    def check_paths(self, image_files):
        "sanity check, make sure all images are from the same sequence, same kinect"
        kid = DataPaths.get_kinect_id(image_files[0])
        seq_name = DataPaths.get_seq_name(image_files[0])
        for file in image_files:
            assert kid == DataPaths.get_kinect_id(file), f'{file} kid incompatible with {image_files[0]}'
            assert seq_name == DataPaths.get_seq_name(file), f'{file} seqname incompatible with {image_files[0]}'



class BehaveObjectVideoTestDataset(BehaveObjectVideoDataset):
    "do not load GT rotation etc."
    def init_paths(self, clip_len, seqs, window):
        "only load occlusion ratios, no thing else"
        # return # for test on in the wild, no GT data.
        vis_path = '/BS/xxie-2/work/pc2-diff/experiments/results/images/' # path to files that have visibility computed
        self.gt_data = {}
        data_paths, start_inds, seq_index = [], [], []
        offset = 0
        for sid, seq in enumerate(tqdm(seqs)):
            data_paths.extend(seq)
            for i in range(0, len(seq) - clip_len + 1, window):
                start_inds.append(i + offset)
                seq_index.append(sid)
            offset += len(seq)
            continue # for test on in the wild, no GT data.
            seq_name = DataPaths.get_seq_name(seq[0])
            if seq_name in self.gt_data:
                continue
            data = pkl.load(open(f'{vis_path}/{seq_name}_proj_vis_reso512.pkl', 'rb'))
            vis_pred = data['vis_hdm'] # N,
            kid = DataPaths.get_kinect_id(seq[0])
            if seq_name not in self.gt_data:
                self.gt_data[seq_name] = data
            if 'occ_ratios' not in self.gt_data[seq_name]:
                self.gt_data[seq_name]['occ_ratios'] = np.ones((len(data['frames']), 4))
            self.gt_data[seq_name]['occ_ratios'][:, kid] = vis_pred
        return data_paths, np.array(start_inds), np.array(seq_index)

    def get_item(self, idx):
        ""
        assert self.split != 'train'
        image_files = self.get_chunk_files(idx)
        L = len(image_files)
        rgbs, masks, samples, Krois = np.empty((L, 3, *self.input_size)), np.empty((L, 2, *self.input_size)), None, np.empty((L, 4, 4))
        samples = np.zeros((6000, 3))
        transforms, can_pose = np.empty((L, 4, 4)), np.eye(4)
        # poses_abs, centers, radius_all = np.empty((L, 4, 4)), np.empty((L, 3)), np.empty((L,))
        poses_abs, centers, radius_all = np.eye(4)[None].repeat(L, 0), np.zeros((L, 3)), np.ones((L,))/2.
        camera_T, cent_0 = np.zeros((L, 3)), None # dummy data
        occ_ratios = np.empty((L,))

        smpl_poses, smpl_joints, body_joints25 = np.zeros((L, 24, 6)), np.zeros((L, 23, 3)), np.zeros((L, 25, 3))
        frame_indices = np.empty((L,))  # index for each frame used to compute positional encoding

        for i, rgb_file in enumerate(image_files):
            npz_file = self.get_npz_file(rgb_file)
            if not osp.isfile(npz_file):
                Kroi, obj_mask, person_mask, rgb = self.load_images_from_files(rgb_file)
                rgbs[i] = rgb.transpose(2, 0, 1)
                masks[i] = np.stack([person_mask, obj_mask], 0)
            else:
                Kroi, mask, rgb = self.load_rgb_masks(rgb_file)
                rgbs[i] = rgb
                masks[i] = mask
            Krois[i] = Kroi # TODO: use axis to do visualization

            # Get object visibility ratio
            ss = rgb_file.split(os.sep)
            pred_file = osp.join(self.ho_segm_pred_path.replace('/pred', '/metadata'), ss[-3], ss[-2] + ".pth")
            occ_ratios[i] = float(torch.load(pred_file, map_location='cpu', weights_only=False)['obj_visibility'])


        rela_poses = torch.from_numpy(transforms).float()
        pts_all = torch.from_numpy(samples).float()[None].repeat(L, 1, 1)
        ss = image_files[0].split(os.sep)
        masks = torch.from_numpy(masks)
        dt, masks = self.compute_dt(masks)
        data_dict = {
            "R": torch.from_numpy(self.opencv2py3d[:3, :3]).float()[None].repeat(L, 1, 1),
            'T': torch.from_numpy(camera_T).float(),
            'K': torch.from_numpy(Krois).float(),
            'image_path': image_files,
            'images': torch.from_numpy(rgbs).float(),
            'masks': masks,
            "dist_transform": dt,
            'pclouds': torch.from_numpy(np.random.normal(0, 1, size=(6890, 3))).float(),
            'rela_poses': rela_poses,
            "abs_poses": torch.from_numpy(poses_abs).float(),
            'points_all': pts_all,
            'occ_ratios': torch.from_numpy(occ_ratios).float(),
            'smpl_poses': torch.from_numpy(smpl_poses).float(), # TODO: use estimated SMPL pose
            'smpl_joints': torch.from_numpy(smpl_joints).float(),
            'body_joints25': torch.from_numpy(body_joints25).float(),

            # backward compatibility
            "sequence_name": ss[-2],  # the frame name
            "synset_id": ss[-3],

            "centers": torch.from_numpy(centers).float(),
            'radius': torch.from_numpy(radius_all).float(),

            # 'frame_indices': torch.from_numpy(frame_indices).float(), # this is not used actually for pose estimation
        }

        # load predicted object pose
        if self.pred_obj_pose_path is not None:
            # print('loading object pose from', self.pred_obj_pose_path)
            # load predicted object rotation/translation
            pred_poses = self.load_pred_obj_poses(image_files)
            data_dict = {
                **data_dict,
                'pred_poses': np.stack(pred_poses, 0)
            }
        # load predicted object translation, for what purpose??
        if self.ho_segm_pred_path is not None:
            T_obj_scaled_all = self.load_obj_norm_params(centers, image_files)
            data_dict = {
                **data_dict,
                'T_obj_scaled': np.stack(T_obj_scaled_all, 0)
            }
        return data_dict

