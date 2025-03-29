import numpy as np
from plyfile import PlyData
from PIL import Image
from pathlib import Path
from numpy.linalg import norm
from time import time
from utils import compute_normal, compute_normals

class mesh:
    """
    A class to load and process 3D mesh data from PLY files for the SUM dataset.

    Args:
        data_path (str): Path to the directory containing mesh data.
        file (str): Filename of the PLY file to be processed.

    Attributes:
        v (np.ndarray): Vertex coordinates.
        coor (np.ndarray): UV texture coordinates for faces.
        f (np.ndarray): Face indices.
        l (np.ndarray): Labels for each face.
        c (np.ndarray): Centers of faces.
        tex (np.ndarray): Texture applied to the vertices.
        n (np.ndarray): Weighted vertex normals.
        areas (np.ndarray): Areas of faces.
    """
    def __init__(self, data_path, file):
        self.v, self.coor, self.f, self.l = self.load_ply_data(data_path, file)
        self.c = self.compute_centers(self.f, self.v)
        self.tex = self.load_texture(Path(data_path) / (file[:-18] + '0.jpg'))
        self.tex = self.compute_vertex_color(self.f, self.coor, self.v, self.tex)
        self.n, self.areas = self.compute_weighted_normals(self.v, self.f)
           
    def load_ply_data(self, data_path, file):
        """
        Load PLY data from a file.

        Args:
            data_path (str): Path to the directory containing the PLY file.
            file (str): Filename of the PLY file.

        Returns:
            tuple: Vertex coordinates, UV texture coordinates, face indices, and face labels.
        """
        point_dim = 3
        normal_dim = 3
        face_idx_dim = 3
        face_coor_dim = 6
        var = Path(data_path) / file
        with open(var) as ply_file:
            ply_file = ply_file.readlines()
            for j, line in enumerate(ply_file):
                if 'element vertex' in line:
                    num_points = int(line.split()[2])
                elif 'element face' in line:
                    num_faces = int(line.split()[2])
                elif 'end_header' in line:
                    start = j
                    break

            data_vertex = np.zeros((num_points, point_dim), dtype=np.float64)
            data_face_idx = np.zeros((num_faces, face_idx_dim), dtype=np.int64)
            data_face_coor = np.zeros((num_faces, face_coor_dim), dtype=np.float64)
            data_face_label = np.zeros((num_faces, ), dtype=np.int64)

            for j, line in enumerate(ply_file[start + 1:]):
                if j < num_points:
                    data_vertex[j] = np.array(line.split()[:3], dtype=np.float64)
                else:
                    line = line.split()
                    data_face_idx[j - num_points] = np.array(line[1:4], dtype=np.int64)
                    data_face_label[j - num_points] = np.array(line[4], dtype=np.int64)
                    data_face_coor[j - num_points] = np.array(line[9:15], dtype=np.float64)
        return data_vertex, data_face_coor, data_face_idx, data_face_label
        
    def load_ply_data_slow(self, ply_file_path):
        """
        Load PLY file data using "PlyData".

        Args:
            ply_file_path (str): Path to the PLY file to be loaded.

        Returns:
            tuple:
                - vertices (np.ndarray): An array of vertex coordinates.
                - uv_coords (np.ndarray): UV texture coordinates for each face.
                - faces (np.ndarray): An array of face indices.
                - labels (np.ndarray): Labels associated with each face.
        """
        plydata = PlyData.read(ply_file_path)

        vertex_data = plydata['vertex']
        vertices = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z']))
        faces_data = plydata['face']
        faces = np.vstack(faces_data.data['vertex_indices'])
        uv_coords = np.vstack(faces_data.data['texcoord'])
        labels = faces_data['label']
        return vertices, uv_coords, faces, labels

    def compute_vertex_color(self, faces, coor, v, tex):
        """
        Compute vertex colors using texture information.

        Args:
            faces (np.ndarray): Face indices.
            coor (np.ndarray): UV texture coordinates for each face.
            v (np.ndarray): Vertex coordinates.
            tex (np.ndarray): Texture image.

        Returns:
            np.ndarray: Normalized vertex colors.
        """
        v_color = np.zeros_like(v)
        width = tex.shape[1]
        height = tex.shape[0]
        for j, t in enumerate(faces):
            for i, p in enumerate(t):
                x = int(width*self.coor[j][i*2]+0.5)
                y = height - int(height*self.coor[j][i*2+1]+0.5)
                v_color[p] = tex[y][x]
        v_color/=255.
        return v_color

    def compute_centers(self, faces, vertices):
        """
        Compute the centers of faces.

        Args:
            faces (np.ndarray): Face indices.
            vertices (np.ndarray): Vertex coordinates.

        Returns:
            np.ndarray: Centers of gravity.
        """
        centers = np.zeros((faces.shape[0], 3))
        for i, idx in enumerate(faces):
            centers[i] = (vertices[idx[0]] + vertices[idx[1]] + vertices[idx[2]]) / 3.
        return centers

    def load_texture(self, texture_file_path):
        """
        Load texture image as a numpy array.

        Args:
            texture_file_path (str): Path to the texture file.

        Returns:
            np.ndarray: Texture image data.
        """
        img = Image.open(texture_file_path)
        texture_data = np.array(img)
        return texture_data
        
    def compute_weighted_normals(self, vertices, faces):
        """
        Compute weighted vertex normals and face areas.

        Args:
            vertices (np.ndarray): Vertex coordinates.
            faces (np.ndarray): Face indices.

        Returns:
            tuple: Weighted vertex normals and face areas.
        """
        vertex_normals = np.zeros_like(vertices)

        face_normals = compute_normals(vertices, faces)
        face_normal_lengths = np.linalg.norm(face_normals, axis=1)

        areas = face_normal_lengths / 2.

        face_normal_lengths_nonzero = np.where(face_normal_lengths > 0, face_normal_lengths, 1)
        face_normals_normalized = face_normals / face_normal_lengths_nonzero[:, np.newaxis]

        weighted_normals = face_normals_normalized * areas[:, np.newaxis]

        vertex_indices = faces.flatten()
        repeated_normals = np.repeat(weighted_normals, 3, axis=0)
        np.add.at(vertex_normals, vertex_indices, repeated_normals)

        vertex_normal_lengths = np.linalg.norm(vertex_normals, axis=1)
        vertex_normal_lengths_nonzero = np.where(vertex_normal_lengths > 0, vertex_normal_lengths, 1)
        vertex_normals = vertex_normals / vertex_normal_lengths_nonzero[:, np.newaxis]

        return vertex_normals, areas
        