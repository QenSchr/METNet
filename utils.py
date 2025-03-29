import numpy as np
from numpy.linalg import norm

class AverageTracker():
    """
    A utility class to track and compute running averages of a metric (e.g., loss, accuracy).

    Methods:
        reset(): Resets all stored values to their initial state.
        add(value, size): Updates the tracker with a new value and its corresponding size.

    Attributes:
        value (float): The most recently added value.
        sum (float): The weighted sum of all values added so far.
        count (int): The total weight of all values added.
        average (float): The current average of the tracked metric.
    """
    def __init__(self):
        """Initialize the tracker by resetting all attributes."""
        self.reset()

    def reset(self):
        """Reset all tracked values to their initial state."""
        self.value = 0
        self.sum = 0
        self.count = 0
        self.average = 0
        
    def add(self, value, size):
        """
        Add a new value to the tracker.

        Args:
            value (float): The value to be added (e.g., a loss or accuracy value).
            size (int): The weight associated with the value (e.g., batch size).
        """
        self.value = value
        self.sum += value * size
        self.count += size
        self.average = self.sum / self.count

def compute_normal(v0, v1, v2):
    """
    Compute the normal vector of a triangle given its three vertices.

    Args:
        v0 (np.ndarray): Vertex 0 of the triangle.
        v1 (np.ndarray): Vertex 1 of the triangle.
        v2 (np.ndarray): Vertex 2 of the triangle.

    Returns:
        np.ndarray: Normal vector of the triangle, normalized to unit length if possible.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0

    face_normal = np.cross(edge1, edge2)
    face_normal = face_normal / norm(face_normal) if norm(face_normal) > 0 else face_normal
    return face_normal

def compute_normals(vertices, faces):
    """
    Compute face normals for a given set of vertices and face indices.

    Args:
        vertices (np.ndarray): An array of vertex coordinates.
        faces (np.ndarray): An array of face indices.

    Returns:
        np.ndarray: An array of face normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    face_normals = np.cross(edge1, edge2)
    return face_normals
    
def rotate_up(data):
    """
    Apply a random rotation around the z-axis to the data.

    Args:
        data (np.ndarray): Input data.
                           The first 3 columns are coordinates, and the next 3 columns are normals.

    Returns:
        np.ndarray: Rotated data.
    """
    data = data.reshape((-1, 6))
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    data[:,:3] = np.dot(data[:,:3], rotation_matrix)
    data[:,3:6] = np.dot(data[:,3:6], rotation_matrix)
    return data

def scale_point_cloud(data, u_limit=1.2, l_limit=0.8):
    """
    Scale the data by a random factor within a specified range.

    Args:
        data (np.ndarray): Input data
        u_limit (float, optional): Upper limit for the scaling factor. Default is 1.2.
        l_limit (float, optional): Lower limit for the scaling factor. Default is 0.8.

    Returns:
        np.ndarray: Scaled data, same shape as input.
    """
    scale = np.random.rand(1, 1, data.shape[-1]) * (u_limit - l_limit) + l_limit
    return data*scale
	