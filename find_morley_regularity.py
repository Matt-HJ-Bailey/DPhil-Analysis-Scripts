#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:33:03 2022

@author: matthew-bailey
"""
import glob
import os
import numpy as np
import scipy.spatial
from collections import defaultdict
import pandas as pd
import networkx as nx
from draw_and_colour import draw_periodic_coloured, to_tikz, colour_graph
from rings import PeriodicRingFinder

from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt

DO_PLOTS = True
class Shape:
    def __init__(self, coords, indices):
        self.coords = coords
        self.hull = scipy.spatial.ConvexHull(self.coords)
        self.indices = indices
        
    def to_node_list(self):
        return self.indices
        
    def edges(self):
        return [(self.indices[i], self.indices[(i + 1) % len(self.indices)]) for i in range(len(self.indices))]
        
    def balanced_repartition(self) -> float:
        stds = np.std(self.coords, axis=0, ddof=1)
        return min(np.sqrt(np.min(stds) / np.max(stds)), 1.0)

    def area(self) -> float:
        x = self.coords[:,0]
        y = self.coords[:,1]

        S1 = np.sum(x*np.roll(y,-1))
        S2 = np.sum(y*np.roll(x,-1))

        return 0.5*np.abs(S1 - S2)

    def perimeter(self) -> float:
        perimeter = 0.0
        for i in range(len(self)):
            j = (i + 1) % len(self)
            edge_len = np.hypot(*(self.coords[j, :] - self.coords[i, :]))
            perimeter += edge_len
        return perimeter
    
    def convexity(self) -> float:
        return max(min(self.hull.area / self.perimeter() , 1.0), 0.0)
    
    def solidity(self) -> float:
        return max(min(self.area() / self.hull.volume, 1.0), 0.0)
    
    def __len__(self):
        return self.coords.shape[0]
    
    def src(self):
        return self.solidity() * self.convexity() * self.balanced_repartition()

def fix_shape_pb(shape_coords, periodic_cell):
    mic_x, mic_y = (periodic_cell[:, 1] - periodic_cell[:, 0]) / 2.0
    
    for i in range(shape_coords.shape[0]):
        for j in range(i):
            diff = shape_coords[j] - shape_coords[i]
            if diff[0] > mic_x:
                shape_coords[j, 0] -= mic_x * 2
            elif diff[0] < -mic_x:
                shape_coords[j, 0] += mic_x * 2
    
            if diff[1] > mic_y:
                shape_coords[j, 1] -= mic_y * 2
            elif diff[1] < -mic_y:
                shape_coords[j, 1] += mic_y * 2
                
    return shape_coords


def bootstrap(arr: np.ndarray, samples=100, lo=2.5, hi=97.5, seed=None):
    """
    Calculate the bootstrap confidence interval for this array.
    """
    arr = np.asarray(arr)
    means = np.empty([samples], dtype=float)
    rng = np.random.default_rng(seed=seed)
    for idx in range(samples):    
        data = rng.choice(arr, size=arr.shape, replace=True)
        means[idx] = np.mean(data)
    return np.percentile(means, [lo, hi])
    
def find_morley_rings(prefix: str):
    with open(f"{prefix}_A_aux.dat", "r") as fi:
        _ = fi.readline()
        _ = fi.readline()
        _ = fi.readline()
        pb_x, pb_y = [float(item) for item in fi.readline().split()]
    periodic_cell = np.array([[0.0, pb_x], [0.0, pb_y]])
    
    coords = []
    with open(f"{prefix}_A_crds.dat", "r") as fi:
        for line in fi:
            coords.append(np.array([float(item) for item in line.split()]))
    coords = np.vstack(coords)
    edges = []
    with open(f"{prefix}_A_net.dat", "r") as fi:
        for idx, line in enumerate(fi.readlines()):
            for item in line.split():
                edges.append(frozenset([idx, int(item)]))
    edges = frozenset(edges)

    shapes = []
    with open(f"{prefix}_B_dual.dat", "r") as fi:
        for line in fi:
            ring = [int(item) for item in line.split()]
            if len(ring) >= 2:
                shape_coords = np.array([coords[idx] for idx in ring])
                shape_coords = fix_shape_pb(shape_coords, periodic_cell)
                shapes.append(Shape(shape_coords, indices=ring))
    return shapes, periodic_cell

def find_job_directory(directory):
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            try:
                dirname = int(subdir)
                return os.path.join(directory, subdir)
            except ValueError:
                pass
    return None
                

def find_repeats(prefix: str, bond: str, angle: str):
    return glob.glob(f"{prefix}_{bond}_{angle}_*/")

def main():
    for bond in ["0.01", "1", "10", "100"]:
        for angle in ["0.01", "1", "10", "100"]:
            shapes = []
            for idx, repeat in enumerate(find_repeats("./NTMC", bond, angle), 1):
                print(idx, repeat)
                job_dir = find_job_directory(repeat)
                if job_dir is None:
                    continue
                try:
                    found_shapes, periodic_cell = find_morley_rings(os.path.join(job_dir, f"out_{bond}_{angle}"))
                    srcs = [shape.src() for shape in found_shapes]
                    print(repeat, np.mean(srcs), np.mean(np.abs(bootstrap(srcs) - np.mean(srcs))))
                    shapes.extend(found_shapes)
                except FileNotFoundError:
                    continue
                
                G = nx.Graph()
                pos_dict = {}
                for shape in found_shapes:
                    G.add_edges_from(shape.edges())
                    for n_idx, node in enumerate(shape.indices):
                        pos_dict[node] = np.array(shape.coords[n_idx, :])
                G = colour_graph(G)
                nx.set_node_attributes(G, pos_dict, "pos")
                
                to_tikz(filename=f"./Figures/NETMC_{bond}_{angle}_{idx}.tex", graph=G, rings=found_shapes, periodic_box=periodic_cell, pos=pos_dict, vmin=0.5, vmax=1, cmap="coolwarm_r", colour_lut=10)
            
            
            br_dict = defaultdict(list)
            solidity_dict = defaultdict(list)
            convexity_dict = defaultdict(list)
            src_dict = defaultdict(list)
            for shape in shapes:
                br_dict[len(shape)].append(shape.balanced_repartition())
                solidity_dict[len(shape)].append(shape.solidity())
                convexity_dict[len(shape)].append(shape.convexity())
                src_dict[len(shape)].append(shape.src())
            ring_sizes = list(range(3, 21))
            df = pd.DataFrame({"ring_sizes": ring_sizes,
                               "convexities": [np.mean(convexity_dict[ring_size]) for ring_size in ring_sizes],
                               "solidities": [np.mean(solidity_dict[ring_size]) for ring_size in ring_sizes],
                               "balanced_repartitions": [np.mean(br_dict[ring_size]) for ring_size in ring_sizes],
                               "srcs": [np.mean(src_dict[ring_size]) for ring_size in ring_sizes]})
            df.to_csv(f"./Results/shapestats-{bond}-{angle}.csv", index=False)
if __name__ == "__main__":
    main()
