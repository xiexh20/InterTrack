"""
transform between different kinect cameras, sequence specific
Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
sys.path.append(os.getcwd())
import numpy as np
from behave.seq_utils import SeqInfo
from behave.utils import load_intrinsics, load_kinect_poses_back, load_kinect_poses
from psbody.mesh import Mesh

def inverse_Rt(r, t):
    "compute inverse transformation for the given R and t"
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    trans_back = np.linalg.inv(trans)  # now the y-axis point down
    r_back = trans_back[:3, :3]
    t_back = trans_back[:3, 3]
    return r_back, t_back

class KinectTransform:
    "transform between different kinect cameras, sequence specific"
    def __init__(self, seq, kinect_count=4, no_intrinsic=False):
        self.seq_info = SeqInfo(seq)
        self.kids = [x for x in range(self.seq_info.kinect_count())]
        if no_intrinsic:
            self.intrinsics = None
        else:
            self.intrinsics = load_intrinsics(self.seq_info.get_intrinsic(), self.kids)
        # rot, trans = load_kinect_poses(self.seq_info.get_config(), self.kids)
        rot, trans = self.load_local2world()
        self.local2world_R, self.local2world_t = rot, trans
        # rot, trans = load_kinect_poses_back(self.seq_info.get_config(), self.kids)
        # Compute world to local camera transformation
        Rt = [inverse_Rt(r, t) for r, t in zip(rot, trans)]
        rot, trans = [x[0] for x in Rt], [x[1] for x in Rt]
        self.world2local_R, self.world2local_t = rot, trans

    def world2color_mesh(self, mesh:Mesh, kid):
        "world coordinate to local color coordinate, assume mesh world coordinate is in k1 color camera coordinate"
        m = self.copy_mesh(mesh)
        m.v = np.matmul(mesh.v, self.world2local_R[kid].T) + self.world2local_t[kid]
        return m

    def flip_mesh(self, mesh:Mesh):
        "flip the mesh along x axis"
        m = self.copy_mesh(mesh)
        m.v[:, 0] = -m.v[:, 0]
        return m

    def copy_mesh(self, mesh:Mesh):
        m = Mesh(v=mesh.v)
        if hasattr(mesh, 'f'):
            m.f = mesh.f.copy()
        if hasattr(mesh, 'vc'):
            m.vc = np.array(mesh.vc)
        return m

    def world2local_meshes(self, meshes, kid):
        transformed = []
        for m in meshes:
            transformed.append(self.world2color_mesh(m, kid))
        return transformed

    def local2world_mesh(self, mesh, kid):
        m = self.copy_mesh(mesh)
        m.v = self.local2world(m.v, kid)
        return m

    def world2local(self, points, kid):
        return np.matmul(points, self.world2local_R[kid].T) + self.world2local_t[kid]

    def project2color(self, p3d, kid):
        "project 3d points to local color image plane"
        p2d = self.intrinsics[kid].project_points(self.world2local(p3d, kid))
        return p2d

    def kpts2center(self, kpts, depth:np.ndarray, kid):
        "kpts: (N, 2), x y format"
        kinect = self.intrinsics[kid]
        pc = kinect.pc_table_ext * depth[..., np.newaxis]
        kpts_3d = pc[kpts[:, 1], kpts[:, 0]]
        return kpts_3d

    def local2world(self, points, kid):
        R, t = self.local2world_R[kid], self.local2world_t[kid]
        return np.matmul(points, R.T) + t

    def dmap2pc(self, depth, kid):
        kinect = self.intrinsics[kid]
        pc = kinect.dmap2pc(depth)
        return pc

    def load_local2world(self):
        """
        For some sequences, the camera extrinsics are packed inside info.json file
        :return:
        """
        if 'extrinsic_params' in self.seq_info.info:
            pose_calibs = self.seq_info.info['extrinsic_params']
            rot = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in self.seq_info.kids]
            trans = [np.array(pose_calibs[x]['translation']) for x in self.seq_info.kids]
        else:
            rot, trans = load_kinect_poses(self.seq_info.get_config(), self.kids)
        return rot, trans









