import numpy as np
from torch.utils.data import Dataset
import h5py
import os
from utils import rotate_up, scale_point_cloud
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class SUM_Dataset(Dataset):
    """
    Dataset class to load and process mesh data from HDF5 files.

    Args:
        mode (str): Mode of the dataset, e.g., 'train', 'test', or 'val'.
        data_path (str): Path to the dataset directory.

    Attributes:
        mode (str): Mode of operation ('train', 'test', 'val').
        data_path (str): Path to the data for the specified mode.
        data_look_up (list): A list of loaded data arrays for each file.
        labels (np.ndarray): A numpy array containing labels for all samples.
        labels_look_up (list): A lookup table mapping global indices to file-specific indices.
    """
    def __init__(self, mode, data_path=None):
        self.mode=mode
        if data_path is None:
            self.data_path = BASE_DIR / 'data' / mode
        else:
            self.data_path = Path(data_path)

        print('Load {} data'.format(mode))
        
        files = [f for f in self.data_path.iterdir() if f.is_file()]
        
        self.data_look_up = []
        self.labels = []
        self.labels_look_up = []
        
        for i, file in enumerate(files):
            data, label = self.load_h5py(file)
            self.data_look_up.append(data)
            self.labels.extend(label)
            zip_list = [list(pair) for pair in zip([i]*len(label), list(range(0, len(label))))]
            self.labels_look_up.extend(zip_list)
        self.labels = np.array(self.labels).astype('int8')
            
    def load_h5py(self, path):
        """
        Load data from an HDF5 file.

        Args:
            path (str): Path to the HDF5 file.

        Returns:
            tuple: A tuple containing the mesh data (vertices, normals, center, colors, area, patch) 
                   and their labels as numpy arrays.
        """
        with h5py.File(path, 'r') as hdf:
            v = np.array(hdf.get('vertex'))
            n = np.array(hdf.get('vertex_normal'))
            color = np.array(hdf.get('vertex_color'))
            c = np.array(hdf.get('center'))
            area = np.array(hdf.get('area'))
            label = np.array(hdf.get('label'))
            p = np.array(hdf.get('mep'), dtype=np.int)
        return [v, n, c, color, area, p] , label
        
    def get_met(self, item, vertex, n, c, color, area, patch):
        """
        Generate the MET for a specific index

        Args:
            item (int): Index of the triangle to retrieve.
            vertex (np.ndarray): Vertex coordinates.
            n (np.ndarray): Vertex normals.
            c (np.ndarray): Point cloud center.
            color (np.ndarray): Vertex colors.
            area (np.ndarray): Area values.
            patch (np.ndarray): Patch indices.

        Returns:
            tuple: A tuple containing the feature matrix and area.
        """
        p = patch[item]
        met = np.zeros((p.shape[0], p.shape[1], 9))
        a = area[item]
        for i, r in enumerate(p):
            for j, e in enumerate(r):
                met[i][j][0:3] = vertex[e]
                met[i][j][3:6] = n[e]
                met[i][j][6:9] = color[e]

        met[:,:,0:3]-=c[item]
        return met, a

    def __getitem__(self, item):
        """
        Retrieve a specific data sample.

        Args:
            item (int): Index of the sample.

        Returns:
            tuple: A tuple containing the MET, label, and the area of the corresponding triangle.
        """
        label = self.labels[item]
        idx = self.labels_look_up[item]
        data = self.data_look_up[idx[0]]
        data, area = self.get_met(idx[1], *data)

        if self.mode=='train':
            # Apply rotation augmentation for training data
            data[:,:,:6] = rotate_up(data[:,:,:6]).reshape((data.shape[0],data.shape[1],6))

        return data, label, area

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.labels.shape[0]
