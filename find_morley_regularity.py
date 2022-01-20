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

from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt

DO_PLOTS = False
class Shape:
    def __init__(self, coords):
        self.coords = coords
        self.hull = scipy.spatial.ConvexHull(self.coords)
        
    def balanced_repartition(self) -> float:
        stds = np.std(self.coords, axis=1, ddof=1)
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
            coords.append([float(item) for item in line.split()])
    coords = np.asarray(coords)
    
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
                shapes.append(Shape(shape_coords))
    return shapes
    

def find_job_directory(directory):
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            try:
                dirname = int(subdir)
                return os.path.join(directory, subdir)
            except ValueError:
                pass
    return None
                

def find_repeats(prefix: str, temperature: str):
    return glob.glob(f"{prefix}_{temperature}_*/")

def main():
    for temperature in ["-2.00", "-2.50", "-3.00", "-3.50", "-4.00", "-5.00"]:
        shapes = []
        for idx, repeat in enumerate(find_repeats("./MINT", temperature)):
            
            job_dir = find_job_directory(repeat)
            if job_dir is None:
                continue
            try:
                found_shapes = find_morley_rings(os.path.join(job_dir, f"out_{temperature}"))
                shapes.extend(found_shapes)
            except FileNotFoundError:
                continue
            
            if DO_PLOTS:
                fig, ax = plt.subplots()
                pc = PolyCollection([shape.coords for shape in found_shapes])
                pc.set_array([len(shape) for shape in found_shapes])
                ax.add_collection(pc)
                
                lc = LineCollection(segments=[[np.array([x, y]) for (x, y) in shape.coords] + [(shape.coords[0, 0], shape.coords[0, 1])]
                                              for shape in found_shapes],
                                    colors="black")
                ax.add_collection(lc)
                
                ax.set_xlim(-5, 60)
                ax.set_ylim(-5, 55)
                ax.axis("off")
                fig.savefig(f"./Figures/MINT_{temperature}_{idx}.pdf")
                plt.close(fig)
        
        
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
        df.to_csv(f"./Results/shapestats-{temperature}.csv", index=False)
if __name__ == "__main__":
    main()