"""
Evaluate InterTrack tracking results: Chamfer distance / F-score of human, object and combined points,
SMPL errors (v2v, MPJPE, MPJAE, translation), object rotation / translation errors,
acceleration and jitter errors, and hand contact accuracy.

Different from eval/eval_separate.py, this script does not need GT translation at inference time:
the results are obtained with estimated 2D translation (see scripts/demo_phone.sh), brought back to
metric camera space, and aligned to GT with one similarity transform computed from the SMPL vertices
of the first frame of each sequence. GT points are computed from the packed GT parameters.

Step 1 (once, GPU recommended): compute SMPL vertices/joints for prediction and GT, saved next to the
joint optimization checkpoints as checkpoint-<seq>_k<cam>_smpl.pth:

python eval/eval_rot_chamf.py --precompute -pr outputs/corrAE/single/opt-hoi-est2d-selviews/pred \
    -split configs/splits/intercap-seqs-21.json --packed_path <packed GT dir> --smpl_root <SMPL-H model dir>

Step 2: evaluation (CPU):

python eval/eval_rot_chamf.py -pr outputs/corrAE/single/opt-hoi-est2d-selviews/pred \
    -split configs/splits/intercap-seqs-21.json --packed_path <packed GT dir> \
    --icap_template_path <InterCap object template dir> --smpl_root <SMPL-H model dir> -i intercap

Required inputs:
- <pr_path>/<seq>/<frame>.ply: predicted human + object points (human points have blue color), normalized space.
- <ckpt_path>/checkpoint-<seq>_k<cam>.pth: checkpoint saved by the joint optimization (main_hoiopt_smplh.py),
  by default searched in the parent folder of pr_path.
- <packed_path>/<seq>_GT-packed.pkl: packed GT parameters (frames, poses, betas, trans, obj_angles, obj_trans).
- configs/splits/seqs/15fps-<seq>.pkl: frames used for inference, configs/splits/w2cs_all.pkl: world to camera
  transforms, configs/splits/selected-views-map.json: camera view used per sequence (InterCap: view 1).
"""
import sys, os

# avoid numpy threading hangs in the forked evaluation processes
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
sys.path.append(os.getcwd())
import json
import zlib
import os.path as osp
import multiprocessing as mp
import pickle as pkl

import numpy as np
import torch
import trimesh
import joblib
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from eval.eval_separate import SeparateEvaluator
from eval.chamfer_distance import chamfer_distance

NUM_SAMPLES = 8192  # number of points sampled from GT human and object surface
NUM_SMPL_JOINTS = 22  # body joints used for MPJPE/MPJAE, hand joints are excluded
HAND_JOINTS = np.array([22, 23 + 15])  # left and right hand joints of SMPL-H, used for contact accuracy


def geodesic_distance_batch(R1, R2):
    """
    geodesic distance between rotation matrices, in radians
    R1, R2: numpy array (..., 3, 3)
    """
    shape = R1.shape
    ones = [1 for _ in shape[:-2]]
    trace = (R1.astype(float) @ R2.astype(float).swapaxes(-2, -1)) * np.eye(3).reshape(*ones, 3, 3)
    shape = trace.shape
    trace = trace.reshape(*shape[:-2], 9).sum(axis=-1)
    cos = (trace - 1) / 2
    cos = cos.clip(-1, 1)
    return np.arccos(cos)


def get_smpl_layer(smpl_root):
    "male SMPL-H model, same as lib_smpl.get_smpl('male', True) but with configurable model path"
    from lib_smpl.smplpytorch import SMPL_Layer
    return SMPL_Layer(model_root=smpl_root, gender='male', hands=True)


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_seq_frames(args, seq_name):
    "names of the frames used for inference, e.g. t0000.067"
    frames = pkl.load(open(osp.join(args.seq_frames_path, f'15fps-{seq_name}.pkl'), 'rb'))['test']
    return [osp.basename(osp.dirname(x)) for x in frames]


def get_ckpt_file(args, seq_name, smpl=False):
    ckpt_path = args.ckpt_path if args.ckpt_path is not None else osp.dirname(osp.normpath(args.pr_path))
    suffix = '_smpl' if smpl else ''
    return osp.join(ckpt_path, f'checkpoint-{seq_name}_k{args.cam_id}{suffix}.pth')


def load_object_template(args, obj_name):
    "vertices (centered at the vertex mean) and faces of the object template"
    if obj_name.startswith('obj'):
        # InterCap objects: obj01 -> 01.ply
        path = osp.join(args.icap_template_path, f'{obj_name[3:]}.ply')
    else:
        from behave.utils import get_template_path
        path = get_template_path(args.behave_template_path, obj_name)
    assert osp.isfile(path), f'object template {path} does not exist, please check --icap_template_path/--behave_template_path'
    mesh = trimesh.load(path, process=False)
    verts = np.array(mesh.vertices)
    return verts - np.mean(verts, 0), np.array(mesh.faces)


class RotationChamfEvaluator(SeparateEvaluator):
    def evaluate(self, args):
        "one process per sequence, all frames of a sequence are evaluated together for the temporal metrics"
        self.args = args
        self.smpl_faces = get_smpl_layer(args.smpl_root).faces
        seqs = json.load(open(args.split))['seqs']
        jobs = []
        print("evaluating on {} seqs".format(len(seqs)))
        th_list = [args.th] if args.th is not None else self.threshold_list
        for seq in seqs:
            pred_files = self.glob_files(args.pr_path, seq)
            if len(pred_files) == 0:
                print("No files found in folder", args.pr_path, seq)  # do not evaluate this seq
                continue
            p = mp.Process(target=self.eval_seq, args=(None, pred_files, th_list, seq))
            p.start()
            jobs.append(p)
        for job in jobs:
            job.join()
        missing = [seq for seq in seqs if seq not in self.errors_dict.keys()]
        if len(missing) > 0:
            print(f"Warning: {len(missing)} of {len(seqs)} sequences were not evaluated: {missing}")
        self.collect_results(args, th_list)

    def eval_files(self, files_gt, files_pred, th_list):
        "all files belong to one sequence, files_gt is not used: GT is computed from the packed parameters"
        args = self.args
        N = NUM_SAMPLES
        NJ = NUM_SMPL_JOINTS
        seq_name = osp.basename(osp.dirname(files_pred[0]))
        # fixed seed per sequence, so that surface sampling and hence the numbers are reproducible
        np.random.seed((zlib.crc32(seq_name.encode()) + args.seed) % (2 ** 32))

        # frames and the optimization checkpoint with precomputed SMPL results
        frames = load_seq_frames(args, seq_name)
        ckpt_file = get_ckpt_file(args, seq_name, smpl=True)
        assert osp.isfile(ckpt_file), f'{ckpt_file} does not exist, please run this script with --precompute first'
        ckpt = torch.load(ckpt_file, map_location='cpu')
        assert len(ckpt['frame_indices']) == len(frames), f'unequal frames: {len(frames)} != {len(ckpt["frame_indices"])}'
        frames_chunk = [osp.splitext(osp.basename(x))[0] for x in files_pred]
        ckpt_indices = to_numpy(ckpt['frame_indices'])
        frame_indices = np.array([ckpt_indices[frames.index(x)] for x in frames_chunk])

        # predicted object rotation: TOPNet prediction + optimized residual
        rot_pred = to_numpy(ckpt['obj_pose_pred'])[frame_indices]  # (T, 3, 3)
        rot_opt = Rotation.from_rotvec(to_numpy(ckpt['obj_rot_axis'])[frame_indices]).as_matrix()
        rot_pr = np.matmul(rot_opt, rot_pred)

        # GT object pose, transformed to the camera view used for inference
        pack_data = joblib.load(osp.join(args.packed_path, f'{seq_name}_GT-packed.pkl'))
        indices_gt = np.array([pack_data['frames'].index(x) for x in frames_chunk])
        rot_gt = Rotation.from_rotvec(pack_data['obj_angles'][indices_gt]).as_matrix()
        trans_gt = pack_data['obj_trans'][indices_gt]
        w2cs_all = pkl.load(open(args.w2c_file, 'rb'))  # world to camera transforms of all recording dates
        seq2views = json.load(open(args.views_map))
        sel_view = int(seq2views[seq_name][1]) if seq_name in seq2views else args.cam_id  # InterCap: default view
        w2c = w2cs_all[seq_name.split('_')[0]][sel_view]  # new pose = w2c @ old pose
        rot_gt = np.matmul(w2c[:3, :3][None], rot_gt)
        trans_gt = (np.ascontiguousarray(trans_gt, dtype=np.float64) @ np.ascontiguousarray(w2c[:3, :3].T, dtype=np.float64)
                    + np.ascontiguousarray(w2c[:3, 3], dtype=np.float64))
        print(f"Using view {sel_view} for evaluation of {seq_name}, {len(frames_chunk)} frames")

        rot_errors = geodesic_distance_batch(rot_pr, rot_gt)

        # GT object points
        obj_name = seq_name.split('_')[2]
        temp_verts, temp_faces = load_object_template(args, obj_name)
        temp_pts = trimesh.Trimesh(temp_verts, temp_faces, process=False).sample(N)
        verts_all = temp_pts[None].repeat(len(rot_gt), 0)
        overts_gt = np.einsum('bij,bkj->bik', verts_all, rot_gt) + trans_gt[:, None, :]

        # predicted human: SMPL vertices and joints in the CorrAE input space
        hum_pts_aeout = to_numpy(ckpt['hum_verts_pred'])[frame_indices]
        jts_pr = to_numpy(ckpt['jts_pred'])[frame_indices]
        jtrs_rot_pr = to_numpy(ckpt['jtrs_rot_pred'])[frame_indices]
        cent_hum = to_numpy(ckpt['centers_smplh_aligned'])[frame_indices]  # (T, 3)
        radius_hum = to_numpy(ckpt['scales_smplh_aligned'])[frame_indices]  # (T, 1)
        hum_scale = to_numpy(ckpt['hum_scale'])[frame_indices][:, None]  # (T, 1, 1)
        hum_trans = to_numpy(ckpt['hum_trans'])[frame_indices][:, None]  # (T, 1, 3)
        transl_smplh_pr = to_numpy(ckpt['transl_smpl'])[frame_indices]
        transl_smplh_gt = pack_data['trans'][indices_gt]
        # to the normalized human-object space
        hum_verts_pr = hum_pts_aeout * radius_hum[:, None] * hum_scale + cent_hum[:, None] + hum_trans
        transl_smplh_pr = transl_smplh_pr * radius_hum * hum_scale[:, 0] + cent_hum + hum_trans[:, 0]
        hum_jtrs_pr = jts_pr * radius_hum[:, None] * hum_scale + cent_hum[:, None] + hum_trans

        # undo the crop normalization with the estimated translation: back to metric camera space
        trans_pr = to_numpy(ckpt['gt_trans'])[frame_indices]  # (T, 3), translation used at inference time
        radius_pr = to_numpy(ckpt['radius'])[frame_indices]  # (T, )
        hum_verts_pr = hum_verts_pr * radius_pr[:, None, None] * 2 + trans_pr[:, None, :]
        hum_jtrs_pr = hum_jtrs_pr * radius_pr[:, None, None] * 2 + trans_pr[:, None, :]
        transl_smplh_pr = transl_smplh_pr * radius_pr[:, None] * 2 + trans_pr

        # GT human, transformed to the camera view used for inference
        hum_verts_gt = to_numpy(ckpt['hum_verts_gt'])[frame_indices]
        jts_gt = to_numpy(ckpt['jts_gt'])[frame_indices]
        jtrs_rot_gt = to_numpy(ckpt['jtrs_rot_gt'])[frame_indices].copy()
        hum_verts_gt = np.matmul(hum_verts_gt, w2c[:3, :3][None].transpose(0, 2, 1)) + w2c[:3, 3][None]
        jts_gt = np.matmul(jts_gt, w2c[:3, :3][None].transpose(0, 2, 1)) + w2c[:3, 3][None]
        jtrs_rot_gt[:, 0] = np.matmul(w2c[:3, :3][None], jtrs_rot_gt[:, 0])  # only the global root rotation changes

        # align prediction to GT: per-frame scale to a constant body shape, and
        # one similarity transform computed from the SMPL vertices of the first frame
        scales2same_betas = to_numpy(ckpt['hum_scale_betas'])[frame_indices]  # (T, 1)
        align_to_gt, _, _ = trimesh.registration.procrustes(hum_verts_pr[0] * scales2same_betas[0], hum_verts_gt[0], scale=True)
        transl_smplh_pr = np.matmul(transl_smplh_pr, align_to_gt[:3, :3].T) + align_to_gt[:3, 3]
        hum_jtrs_pr = np.matmul(hum_jtrs_pr, align_to_gt[:3, :3].T) + align_to_gt[:3, 3]
        jtrs_rot_pr[:, 0] = np.matmul(align_to_gt[:3, :3][None], jtrs_rot_pr[:, 0])
        hum_verts_pr = np.matmul(hum_verts_pr, align_to_gt[:3, :3][None].transpose(0, 2, 1)) + align_to_gt[:3, 3][None, None]

        hum_pts_gt = np.stack([trimesh.Trimesh(v, self.smpl_faces, process=False).sample(N) for v in hum_verts_gt])
        pts_gt = np.concatenate([hum_pts_gt, overts_gt], 1)

        # SMPL errors in metric space
        v2v = np.sqrt(np.sum((hum_verts_pr - hum_verts_gt) ** 2, -1)).mean(-1)
        # MPJPE: root-relative, https://github.com/xiexh20/behave-dataset/blob/main/challenges/lib/metrics.py#L180
        jtrs_pr_rela = hum_jtrs_pr[:, :NJ] - hum_jtrs_pr[:, 0:1]
        jtrs_gt_rela = jts_gt[:, :NJ] - jts_gt[:, 0:1]
        mpjpe = np.sqrt(np.sum((jtrs_pr_rela - jtrs_gt_rela) ** 2, -1)).reshape((-1, NJ)).mean(-1)
        smpl_t = np.sqrt(np.sum((transl_smplh_pr - transl_smplh_gt) ** 2, -1))
        mpjae = geodesic_distance_batch(jtrs_rot_gt[:, :NJ].reshape((-1, 3, 3)),
                                        jtrs_rot_pr[:, :NJ].reshape((-1, 3, 3))).reshape((-1, NJ)).mean(-1)

        dist_chamf, chamf_hum, chamf_obj = [], [], []
        fscores_comb, fscores_hum, fscores_obj = [], [], []
        obj_trans_errs, objt_aligned, objt_gt_aligned, cont_acc = [], [], [], []
        for i, fpred in enumerate(tqdm(files_pred)):
            pc_hum, pc_obj = self.load_pred(fpred)
            gt_hum, gt_obj = pts_gt[i, :N], pts_gt[i, N:]

            # predicted points: normalized space -> metric camera space -> aligned to GT
            pc_hum = pc_hum * radius_pr[i] * 2 + trans_pr[i][None]
            pc_obj = pc_obj * radius_pr[i] * 2 + trans_pr[i][None]
            pc_hum = np.matmul(pc_hum * scales2same_betas[i], align_to_gt[:3, :3].T) + align_to_gt[:3, 3]
            pc_obj = np.matmul(pc_obj * scales2same_betas[i], align_to_gt[:3, :3].T) + align_to_gt[:3, 3]

            # object translation error: distance between the centers of predicted and GT object points
            objt_pr, objt_gt = np.mean(pc_obj, 0), np.mean(gt_obj, 0)
            obj_trans_errs.append(np.sqrt(np.sum((objt_pr - objt_gt) ** 2)))
            objt_aligned.append(objt_pr)
            objt_gt_aligned.append(objt_gt)

            # human + object, then human/object separately
            scores, cd = self.compute_fscores(np.concatenate([gt_hum, gt_obj], 0),
                                              np.concatenate([pc_hum, pc_obj], 0), th_list)
            dist_chamf.append(cd)
            fscores_comb.append(scores[0])
            scores, cd_h = self.compute_fscores(gt_hum, pc_hum, th_list)
            fscores_hum.append(scores[0])
            scores, cd_o = self.compute_fscores(gt_obj, pc_obj, th_list)
            fscores_obj.append(scores[0])
            chamf_hum.append(cd_h)
            chamf_obj.append(cd_o)

            # hand contact accuracy: is the hand-object contact state the same as GT?
            _, dist_hand_gt = chamfer_distance(jts_gt[i, HAND_JOINTS], gt_obj, direction='x_to_y', ret_intermediate=True)
            _, dist_hand_pr = chamfer_distance(hum_jtrs_pr[i, HAND_JOINTS], pc_obj, direction='x_to_y', ret_intermediate=True)
            acc = (dist_hand_gt < args.cont_thres) == (dist_hand_pr < args.cont_thres)
            cont_acc.append(acc.astype(np.float32).mean())

        # acceleration errors: object center, SMPL translation and SMPL joints
        objt_aligned = np.stack(objt_aligned, 0)
        objt_gt_aligned = np.stack(objt_gt_aligned, 0)
        acc_ot_err = self.acceleration_error(objt_aligned, objt_gt_aligned)
        acc_ht_err = self.acceleration_error(transl_smplh_pr, transl_smplh_gt)
        acc_hj_err = self.acceleration_error(hum_jtrs_pr[:, :NJ], jts_gt[:, :NJ]).mean(-1)
        # pad first and last value, so that (T-2) -> (T)
        acc_ot_err, acc_ht_err, acc_hj_err = [np.concatenate([x[0:1], x, x[-1:]], 0) for x in [acc_ot_err, acc_ht_err, acc_hj_err]]

        # jitter (third order difference) of human joints and object center
        jitter_hj = self.jitter(hum_jtrs_pr[:, :NJ]).mean(-1)
        jitter_ot = self.jitter(objt_aligned)
        # pad two values in front and one in the end, so that (T-3) -> (T)
        jitter_hj, jitter_ot = [np.concatenate([x[0:1].repeat(2), x, x[-1:]], 0) for x in [jitter_hj, jitter_ot]]

        return (files_pred, rot_errors, v2v, mpjpe, mpjae, smpl_t, fscores_comb, fscores_hum, fscores_obj, dist_chamf,
                chamf_hum, chamf_obj, obj_trans_errs, acc_ht_err, acc_ot_err, acc_hj_err, jitter_hj, jitter_ot, np.array(cont_acc))

    @staticmethod
    def acceleration_error(pred, gt):
        "pred, gt: (T, ..., 3), return (T-2, ...)"
        acc_pr = pred[2:] - 2 * pred[1:-1] + pred[:-2]
        acc_gt = gt[2:] - 2 * gt[1:-1] + gt[:-2]
        return np.sqrt(np.sum((acc_pr - acc_gt) ** 2, -1))

    @staticmethod
    def jitter(pred):
        "pred: (T, ..., 3), return (T-3, ...)"
        jitter = pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]
        return np.sqrt(np.sum(jitter ** 2, -1))

    def load_pred(self, fpred):
        "human points are saved with blue color"
        pc = trimesh.load_mesh(fpred, process=False)
        mask_hum = pc.colors[:, 2] > 0.5
        return np.array(pc.vertices[mask_hum]), np.array(pc.vertices[~mask_hum])

    def format_errors(self, files_pred, ret_values):
        (files_pred, rot_errors, v2v, mpjpe, mpjae, smpl_t, fscores_comb, fscores_hum, fscores_obj, dist_chamf, chamf_hum,
         chamf_obj, obj_trans_errs, acc_ht_err, acc_ot_err, acc_hj_err, jitter_hj, jitter_ot, cont_acc) = ret_values
        ret = {
            "files": files_pred,
            'chamf': np.array(dist_chamf),
            'mpjpe_errors': mpjpe,
            'mpjae_errors': mpjae,
            'v2v_errors': v2v,
            'smpl_t_errors': smpl_t,
            'fscores_comb': np.array(fscores_comb),
            'fscores_hum': np.array(fscores_hum),
            'fscores_obj': np.array(fscores_obj),
            'rot_errors': rot_errors,
            'trans_errors': np.array(obj_trans_errs),
            'chamf_hum': np.array(chamf_hum),
            'chamf_obj': np.array(chamf_obj),
            'acc-ht': np.array(acc_ht_err),
            'acc-ot': np.array(acc_ot_err),
            'acc-hj': np.array(acc_hj_err),
            'jitter-hj': np.array(jitter_hj),
            'jitter-ot': np.array(jitter_ot),
            'cont_hand': np.array(cont_acc),
        }
        return ret

    def print_summary(self, errors_all, th_list, category_name='All'):
        mean_cd = np.mean(errors_all['chamf'])
        res = (
            f"{category_name} {len(errors_all['rot_errors'])} images: "
            f"CD={mean_cd:.4f}, CD_hum={np.mean(errors_all['chamf_hum']):.4f}, CD_obj={np.mean(errors_all['chamf_obj']):.4f}, "
            f"rot={np.mean(errors_all['rot_errors']) * 180 / np.pi:.2f}, trans={np.mean(errors_all['trans_errors']):.4f}, "
            f"v2v={np.mean(errors_all['v2v_errors']):.4f}, mpjpe={np.mean(errors_all['mpjpe_errors']):.4f}, "
            f"mpjae={np.mean(errors_all['mpjae_errors']) * 180 / np.pi:.2f}, smpl_t={np.mean(errors_all['smpl_t_errors']):.4f}, "
            f"acc_ht={np.mean(errors_all['acc-ht']):.4f}, acc_ot={np.mean(errors_all['acc-ot']):.4f}, "
            f"acc_hj={np.mean(errors_all['acc-hj']):.4f}, cont_hand={np.mean(errors_all['cont_hand']):.4f}  ")
        mean_fscores = None
        for name in ['hum', 'obj', 'comb']:
            mean_fscores = np.mean(errors_all[f'fscores_{name}'], 0)
            for s, t in zip(mean_fscores, th_list):
                if t == 0.01:
                    res += f"{name}_F-score@{t}m={s:.4f}  "
        print(res)
        return mean_cd, mean_fscores

    @staticmethod
    def get_parser():
        from lib_smpl.const import SMPL_MODEL_ROOT
        parser = SeparateEvaluator.get_parser()
        parser.set_defaults(split='configs/splits/intercap-seqs-21.json')
        parser.add_argument('--ckpt_path', default=None,
                            help='folder of the joint optimization checkpoints, default: parent folder of pr_path')
        parser.add_argument('--cam_id', default=1, type=int, help='camera id in the checkpoint name, also the default camera view')
        parser.add_argument('--packed_path', required=True, help='folder with <seq>_GT-packed.pkl files')
        parser.add_argument('--icap_template_path', default=None, help='folder with InterCap object templates: 01.ply, 02.ply...')
        parser.add_argument('--behave_template_path', default=None, help='path to BEHAVE objects folder')
        parser.add_argument('--smpl_root', default=SMPL_MODEL_ROOT, help='folder with SMPLH_male.pkl')
        parser.add_argument('--seq_frames_path', default='configs/splits/seqs', help='folder with 15fps-<seq>.pkl files')
        parser.add_argument('--w2c_file', default='configs/splits/w2cs_all.pkl')
        parser.add_argument('--views_map', default='configs/splits/selected-views-map.json')
        parser.add_argument('--seed', default=0, type=int, help='seed for surface point sampling')
        parser.add_argument('--precompute', default=False, action='store_true',
                            help='compute and save SMPL vertices and joints for prediction and GT, needed once before evaluation')
        return parser


@torch.no_grad()
def precompute_smpl(args):
    """
    for each sequence in the split, save checkpoint-<seq>_k<cam>_smpl.pth: the optimization checkpoint plus
    predicted and GT SMPL vertices, joints, global joint rotations, and a per-frame scale that brings
    the prediction to a constant body shape (shape of the first frame).
    """
    from smplfitter.pt import BodyModel, BodyFitter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    body_model = BodyModel('smplh', 'male', model_root=args.smpl_root).to(device)  # the body model to be fitted
    fitter = BodyFitter(body_model, num_betas=10).to(device)
    smpl_layer = get_smpl_layer(args.smpl_root).to(device)

    seqs = json.load(open(args.split))['seqs']
    for seq_name in tqdm(seqs):
        file, outfile = get_ckpt_file(args, seq_name), get_ckpt_file(args, seq_name, smpl=True)
        if osp.isfile(outfile):
            continue
        frames_file = osp.join(args.seq_frames_path, f'15fps-{seq_name}.pkl')
        skip = False
        for f in [file, frames_file]:
            if not osp.isfile(f):
                print(f'{f} not found, skipped {seq_name}')
                skip = True
        if skip:
            continue  # skip this sequence only, the other sequences are still processed
        ckpt = torch.load(file, map_location='cpu')
        frame_indices = ckpt['frame_indices']
        T = len(frame_indices)
        hum_pts_aeout, jts_pr, _, _, jtrs_rot_pr = smpl_layer(ckpt['poses_hum'][frame_indices].to(device),
                                                              ckpt['betas_hum'].repeat(T, 1).to(device),
                                                              ckpt['transl_smpl'][frame_indices].to(device),
                                                              ret_glb_rot=True)

        # predicted human in metric camera space: CorrAE space -> normalized human-object space -> camera space
        cent_hum = ckpt['centers_smplh_aligned'][frame_indices].to(device)  # (T, 3)
        radius_hum = ckpt['scales_smplh_aligned'][frame_indices].to(device)  # (T, 1)
        hum_scale = ckpt['hum_scale'][frame_indices].to(device)  # (T, 1)
        hum_trans = ckpt['hum_trans'][frame_indices].to(device)  # (T, 3)
        hum_verts_pr = hum_pts_aeout * radius_hum[:, None] * hum_scale[:, None] + cent_hum[:, None] + hum_trans[:, None]
        trans_pr = torch.from_numpy(to_numpy(ckpt['gt_trans'])[to_numpy(frame_indices)]).float().to(device)  # (T, 3)
        radius_pr = torch.from_numpy(to_numpy(ckpt['radius'])[to_numpy(frame_indices)]).float().to(device)  # (T, )
        hum_verts_pr = hum_verts_pr * radius_pr[:, None, None] * 2 + trans_pr[:, None, :]

        # fit SMPL to the predicted humans, then compute the scale from each prediction
        # to the body with the shape of the first frame and frame-specific pose
        fit_res = fitter.fit(hum_verts_pr, num_iter=3, beta_regularizer=1,
                             requested_keys=['shape_betas', 'trans', 'vertices', 'pose_rotvecs'])
        verts_1st_betas = smpl_layer(fit_res['pose_rotvecs'], fit_res['shape_betas'][0:1].repeat(T, 1), fit_res['trans'])[0]
        verts_1st_betas_np, verts_pr_np = verts_1st_betas.cpu().numpy(), hum_verts_pr.cpu().numpy()
        scales = []
        for i in range(T):
            mat, _, _ = trimesh.registration.procrustes(verts_pr_np[i], verts_1st_betas_np[i], scale=True)
            scales.append(np.linalg.svd(mat[:3, :3])[1][0])
        ckpt['hum_scale_betas'] = np.array(scales)[:, None]

        # GT SMPL, in world coordinate
        pack_data = joblib.load(osp.join(args.packed_path, f'{seq_name}_GT-packed.pkl'))
        frames = load_seq_frames(args, seq_name)
        indices_gt = np.array([pack_data['frames'].index(x) for x in frames])
        hum_verts_gt, jts_gt, _, _, jtrs_rot_gt = smpl_layer(torch.from_numpy(pack_data['poses'][indices_gt]).float().to(device),
                                                             torch.from_numpy(pack_data['betas'][indices_gt]).float().to(device),
                                                             torch.from_numpy(pack_data['trans'][indices_gt]).float().to(device),
                                                             ret_glb_rot=True)
        ckpt['hum_verts_pred'] = hum_pts_aeout.cpu()
        ckpt['jts_pred'] = jts_pr.cpu()
        ckpt['jtrs_rot_pred'] = jtrs_rot_pr.cpu()
        ckpt['hum_verts_gt'] = hum_verts_gt.cpu()
        ckpt['jts_gt'] = jts_gt.cpu()
        ckpt['jtrs_rot_gt'] = jtrs_rot_gt.cpu()
        # alignment of the first frame from reconstruction to GT (in world coordinate, for reference only)
        ckpt['align_to_gt'] = trimesh.registration.procrustes(verts_pr_np[0], hum_verts_gt[0].cpu().numpy(), scale=True)[0]
        torch.save(ckpt, outfile)
        print('saved', outfile)


def main(args):
    if args.precompute:
        precompute_smpl(args)
    else:
        evaluator = RotationChamfEvaluator()
        evaluator.evaluate(args)
    print("All done")


if __name__ == '__main__':
    parser = RotationChamfEvaluator.get_parser()
    args = parser.parse_args()

    main(args)
