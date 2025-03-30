import numpy as np
from numpy.linalg import norm
import multiprocessing as mp
from load_mesh import mesh
from heapq import heappop, heappush
from copy import deepcopy
import argparse
import h5py
import time
import os
from utils import compute_normal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def parse_option():
    parser = argparse.ArgumentParser('Arguments for MEP generation')

    parser.add_argument('--N', type=int, default=16,
                        help='number of points')
    parser.add_argument('--M', type=int, default=16,
                        help='number of paths')
    parser.add_argument('--data_usage', type=float, default=1.,
                        help='training data usage. 1 means 100% data usage')
    parser.add_argument('--mode', type=str, default='test',
                        help='train/validate/test')
                        
    config = parser.parse_args()
    return config

class MEPGenerater:
    def __init__(self, path, file, config):
        self.config = config
        self.mesh = mesh(path, file)

        self.edges = [(str(vertex[i % 3]), str(vertex[(i + 1) % 3]),
                       self.calc_dist(self.mesh.v[vertex[i % 3]], self.mesh.v[vertex[(i + 1) % 3]]))
                      for i in range(3)
                      for vertex in self.mesh.f]
        self.edges = [self.create_edge(*e) for e in self.edges]
        self.vertices = self.all_vertices()
        self.start = [e['start'] for e in self.edges]
        self.neighbors = self.get_neighbors()

        self.distances = {v: float('inf') for v in self.vertices}
        self.prev_v = {v: None for v in self.vertices}
        self.steps = {v: 0 for v in self.vertices}
        self.count = np.zeros((self.config.N + 1,))
        self.num_vertices = str(len(self.vertices)+1)

    def create_edge(self, start, end, cost):
        """Creates a dictionary representing an edge in the graph."""
        return {'start':start, 'end':end, 'cost':cost}
        
    def calc_dist(self, vert1, vert2):
        """Calculates the Euclidean distance between two points."""
        if vert1.shape != vert2.shape:
            raise ValueError("Input points must have the same dimensions.")
        return norm(vert1-vert2)

    def all_vertices(self):
        """Retrieves all unique vertices in the graph."""
        return set(
            e['start'] for e in self.edges
        ).union(e['end'] for e in self.edges)

    def get_neighbors(self):
        """Generates a neighbor dict for all vertices in the graph."""
        neighbors = {v: set() for v in self.vertices}
        for e in self.edges:
            neighbors[e['start']].add((e['end'], e['cost']))
            neighbors[e['end']].add((e['start'], e['cost']))
        return neighbors

    def compute_mep(self, face):
        """Compute MEP"""
        distances = self.distances.copy()
        prev_v = self.prev_v.copy()
        steps = self.steps.copy()
        count = self.count.copy()
        N = self.config.N + 1
        M = self.config.M + 1
        num_vertices = self.num_vertices
        neighbors = self.neighbors.copy()

        distances[num_vertices], prev_v[num_vertices], steps[num_vertices], neighbors[num_vertices] = 0, None, 0, set()

        for e in self.mesh.f[face]:
            neighbors[num_vertices].add((str(e), self.calc_dist(self.mesh.c[face], self.mesh.v[e])))
            neighbors[str(e)].add((str(num_vertices), self.calc_dist(self.mesh.c[face], self.mesh.v[e])))

        pq, current_steps = [(0, num_vertices)], [0]
        potential_paths = []
        while (len(pq) > 0):
            current_distance, v = heappop(pq)

            if prev_v[v] is not None:
                steps[v] = steps[prev_v[v]] + 1
            else:
                steps[v] += 1

            if steps[v] > self.config.N:
                potential_paths.append(v)
                continue

            if not current_steps or min(current_steps) > N:
                break

            if prev_v[v] is not None:
                if count[steps[v]] == 0:
                    current_steps.append(steps[v])
                    count[steps[v]] += 1
                if count[steps[prev_v[v]]] == 1:
                    try:
                        current_steps.remove(steps[prev_v[v]])
                    except:
                        return np.zeros((M-1,N-1), dtype=np.int32)
                count[steps[prev_v[v]]] -= 1
            else:
                current_steps.append(steps[v])
                count[steps[v]] += 1
                current_steps.remove(0)

            for neighbor, cost in neighbors[v]:
                path_cost = current_distance + cost

                if path_cost < distances[neighbor]:
                    distances[neighbor], prev_v[neighbor] = path_cost, v
                    heappush(pq, (path_cost, neighbor))

        max_steps_tmp = N
        if len(potential_paths)<M:
            max_steps_tmp = int(N*0.7)
            potential_paths = [key for key, _ in distances.items() if steps[key] == max_steps_tmp]
            
            if len(potential_paths)<M*0.7 and max_steps_tmp>3:
                max_steps_tmp = int(max_steps_tmp*0.7)
                potential_paths = [key for key, _ in distances.items() if steps[key] == max_steps_tmp]
                
                if len(potential_paths)==0:
                    max_steps_tmp = 2
                    potential_paths = [key for key, _ in distances.items() if steps[key] == max_steps_tmp]

        face_normal = compute_normal(*self.mesh.v[self.mesh.f[face]])

        paths = self.select_paths(potential_paths, face, M-1, face_normal)
        out = np.zeros((M-1,N-1), dtype=np.int32)

        for i, v in enumerate(paths):
            cur = v
            for j in range(max_steps_tmp-1):
                out[i][j] = int(cur)
                cur = prev_v[cur]

        return out
        
    def fill_incomplete_paths(self, P):
        for path in P:
            reversed_path = path[::-1]

            valid_indices = np.where(reversed_path != -1)[0]
            if len(valid_indices) > 0:
                first_valid_index = valid_indices[0]
                first_valid_point = reversed_path[first_valid_index]

                reversed_path[:first_valid_index] = first_valid_point
            path[:] = reversed_path[::-1]

        return P

    def select_paths(self, potential_paths, start, M, face_normal):
        """Selects paths based on the shortest distance of the last points in the paths to the plane."""
        keys = np.array([int(key) for key in potential_paths])
        points = self.mesh.v[keys]
        start_point = self.mesh.c[start]
        points -= start_point

        norm_ = np.linalg.norm(face_normal)
        norm_face_normal = face_normal / norm_ if norm_ > 0 else np.array([0,1,0])

        ortho = np.array([1., 1., -1. * (norm_face_normal[0] + norm_face_normal[1]) / max(norm_face_normal[2], 1e-5)])
        ortho /= np.linalg.norm(ortho)

        rotation_matrices = self._generate_rotation_matrices(norm_face_normal, M)

        ortho = ortho.reshape(1, 3)
        directions = rotation_matrices @ ortho.T 
        directions = directions.squeeze(axis=2)

        plane_normal = np.cross(norm_face_normal, directions)

        norm_ = np.linalg.norm(plane_normal, axis=1, keepdims=True)
        norm_plane_normal = np.divide(plane_normal, norm_, out=np.zeros_like(plane_normal), where=norm_ > 0)

        dist_matrix = norm_plane_normal @ points.T
        dist_matrix = np.abs(dist_matrix)

        ortho_proj = directions @ points.T

        mask = ortho_proj > 0.0
        dist_matrix = np.where(mask, dist_matrix, np.inf)

        indices = np.argmin(dist_matrix, axis=1)

        sel_keys = keys[indices].tolist()
        sel_keys = [str(idx) for idx in sel_keys]

        return sel_keys

    def _generate_rotation_matrices(self, normal, M):
        """Generates M rotation matrices based on the normal direction."""
        angle = 2. * np.pi / M
        cos_vals = np.cos(np.arange(M) * angle)
        sin_vals = np.sin(np.arange(M) * angle)
        one_minus_cos = 1.0 - cos_vals

        nx, ny, nz = normal
        rotation_matrices = np.zeros((M, 3, 3))

        rotation_matrices[:, 0, 0] = cos_vals + nx * nx * one_minus_cos
        rotation_matrices[:, 0, 1] = nx * ny * one_minus_cos - nz * sin_vals
        rotation_matrices[:, 0, 2] = nx * nz * one_minus_cos + ny * sin_vals
        rotation_matrices[:, 1, 0] = ny * nx * one_minus_cos + nz * sin_vals
        rotation_matrices[:, 1, 1] = cos_vals + ny * ny * one_minus_cos
        rotation_matrices[:, 1, 2] = ny * nz * one_minus_cos - nx * sin_vals
        rotation_matrices[:, 2, 0] = nz * nx * one_minus_cos - ny * sin_vals
        rotation_matrices[:, 2, 1] = nz * ny * one_minus_cos + nx * sin_vals
        rotation_matrices[:, 2, 2] = cos_vals + nz * nz * one_minus_cos

        return rotation_matrices

    def loop_part(self, data, queue):
        name = data[0]
        data = data[1]
        out = np.zeros((data.shape[0],self.config.M,self.config.N), dtype=np.int32)
        labels = np.zeros((data.shape[0],), dtype=np.int32)-1
        for i, e in enumerate(data):
            if self.mesh.l[e]==0:
                continue
            elif self.config.mode=='train' and np.random.uniform()>self.config.data_usage:
                continue
            labels[i] = self.mesh.l[e]
            out[i] = self.compute_mep(e)
        queue.put({name: [out, labels]})

def create_h5(path, file, config):
    num_cores = mp.cpu_count()
    print("Load Mesh")
    start = time.time()
    scene = MEPGenerater(path=Path(path) / 'raw_data' / config.mode, file=file, config=config)
    print("Finish: ", time.time()-start)

    length = len(scene.mesh.c)

    pakage_size = int(length/(num_cores-1))

    print('Start MEP generation: ', file)

    queue = mp.Queue()
    procs = []

    start = time.time()

    for i in range(num_cores):
        if i==num_cores-1:
            p = mp.Process(target=scene.loop_part, args=([i,np.arange(i*pakage_size,length)], queue))
        else:
            p = mp.Process(target=scene.loop_part, args=([i,np.arange(i*pakage_size,(i+1)*pakage_size)], queue))
        procs.append(p)
        p.start()

    results = {}
    for i in range(num_cores):
        results.update(queue.get())

    for i in procs:
        i.join()

    out = np.zeros((length, config.M, config.N))
    labels = np.zeros((length,))

    pos = 0

    for i in range(num_cores):
        cur_data = results[i][0]
        out[pos:pos+cur_data.shape[0]] = cur_data
        labels[pos:pos+cur_data.shape[0]] = results[i][1]
        pos += cur_data.shape[0]

    args_delete = np.argwhere(labels == -1)
    out = np.delete(out, args_delete,0)
    centers = np.delete(scene.mesh.c, args_delete,0)
    areas = np.delete(scene.mesh.areas, args_delete,0)
    labels = np.delete(labels, args_delete,0)
    labels-=1

    with h5py.File(Path(path) / 'data' / config.mode / (file[:-3] + "h5"), 'w') as hdf:
        hdf.create_dataset('vertex', data=np.array(scene.mesh.v))
        hdf.create_dataset('vertex_normal', data=np.array(scene.mesh.n))
        hdf.create_dataset('vertex_color', data=np.array(scene.mesh.tex))
        hdf.create_dataset('center', data=np.array(centers))
        hdf.create_dataset('area', data=np.array(areas))
        hdf.create_dataset('label', data=np.array(labels))
        hdf.create_dataset('mep', data=np.array(out))

    print('End File ', file, 'Time: ', time.time()-start)
    
def main():
    config = parse_option()
    
    raw_data_dir = BASE_DIR / 'raw_data' / config.mode
    data_dir = BASE_DIR / 'data' / config.mode

    for i, file in enumerate(raw_data_dir.iterdir()):
        if file.suffix.lower() != '.ply' or (data_dir / (file.stem + "h5")).exists():
            continue
        print(i, file.name)
        create_h5(str(BASE_DIR), file.name, config)


if __name__=='__main__':
    main()
