"""
adapted by Xianghui from psbody-mesh library: https://github.com/MPI-IS/mesh/blob/master/mesh/mesh.py
"""
import os 
import numpy as np
from . import serialization


def recompute_landmark_indices_outer(self, landmark_fname=None, safe_mode=True):
    filtered_landmarks = dict(
        filter(
            lambda e, : e[1] != [0.0, 0.0, 0.0],
            self.landm_raw_xyz.items()
        ) if (landmark_fname and safe_mode) else self.landm_raw_xyz.items())
    if len(filtered_landmarks) != len(self.landm_raw_xyz):
        print("WARNING: %d landmarks in file %s are positioned at (0.0, 0.0, 0.0) and were ignored" % (len(self.landm_raw_xyz) - len(filtered_landmarks), landmark_fname))

    self.landm = {}
    self.landm_regressors = {}
    if filtered_landmarks:
        landmark_names = list(filtered_landmarks.keys())
        closest_vertices, _ = self.closest_vertices(np.array(list(filtered_landmarks.values())))
        self.landm = dict(zip(landmark_names, closest_vertices))
        if len(self.f):
            face_indices, closest_points = self.closest_faces_and_points(np.array(list(filtered_landmarks.values())))
            vertex_indices, coefficients = self.barycentric_coordinates_for_points(closest_points, face_indices)
            self.landm_regressors = dict([(name, (vertex_indices[i], coefficients[i])) for i, name in enumerate(landmark_names)])
        else:
            self.landm_regressors = dict([(name, (np.array([closest_vertices[i]]), np.array([1.0]))) for i, name in enumerate(landmark_names)])


class Mesh(object):
    """3d Triangulated Mesh class

    Attributes:
        v: Vx3 array of vertices
        f: Fx3 array of faces

    Optional attributes:
        fc: Fx3 array of face colors
        vc: Vx3 array of vertex colors
        vn: Vx3 array of vertex normals
        segm: dictionary of part names to triangle indices

    """

    def __init__(self,
                 v=None,
                 f=None,
                 segm=None,
                 filename=None,
                 basename=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 landmarks=None):
        """
        :param v: vertices
        :param f: faces
        :param filename: a filename from which a mesh is loaded
        """

        if filename is not None:
            self.load_from_file(filename)
            if hasattr(self, 'f'):
                self.f = np.require(self.f, dtype=np.uint32)
            self.v = np.require(self.v, dtype=np.float64)
            self.filename = filename
            if vscale is not None:
                self.v *= vscale
        if v is not None:
            self.v = np.array(v, dtype=np.float64)
            if vscale is not None:
                self.v *= vscale
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

        self.basename = basename
        if self.basename is None and filename is not None:
            self.basename = os.path.splitext(os.path.basename(filename))[0]

        if segm is not None:
            self.segm = segm
        # if landmarks is not None:
        #     self.set_landmark_indices_from_any(landmarks)

        if vc is not None:
            self.set_vertex_colors(vc)

        if fc is not None:
            self.set_face_colors(fc)

    def set_vertex_colors(self, vc, vertex_indices=None):
        if vertex_indices is not None:
            self.vc[vertex_indices] = self.colors_like(vc, self.v[vertex_indices])
        else:
            self.vc = self.colors_like(vc, self.v)
        return self

    def set_face_colors(self, fc):
        self.fc = self.colors_like(fc, self.f)
        return self

    def load_from_file(self, filename):
        serialization.load_from_file(self, filename)

    def load_from_ply(self, filename):
        serialization.load_from_ply(self, filename)

    def load_from_obj(self, filename):
        serialization.load_from_obj(self, filename)

    def write_json(self, filename, header="", footer="", name="", include_faces=True, texture_mode=True):
        serialization.write_json(self, filename, header, footer, name, include_faces, texture_mode)

    def write_three_json(self, filename, name=""):
        serialization.write_three_json(self, filename, name)

    def write_ply(self, filename, flip_faces=False, ascii=False, little_endian=True, comments=[]):
        serialization.write_ply(self, filename, flip_faces, ascii, little_endian, comments)

    def write_mtl(self, path, material_name, texture_name):
        """Serializes a material attributes file"""
        serialization.write_mtl(self, path, material_name, texture_name)

    def write_obj(self, filename, flip_faces=False, group=False, comments=None):
        serialization.write_obj(self, filename, flip_faces, group, comments)

    def recompute_landmark_indices(self, landmark_fname=None, safe_mode=True):
        # dummy data
        recompute_landmark_indices_outer(self, landmark_fname, safe_mode)



