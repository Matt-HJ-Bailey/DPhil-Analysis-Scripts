#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:12:07 2021

@author: matthew-bailey
"""

import clustering
import numpy as np
import MDAnalysis as mda
import sys
import pandas as pd
import matplotlib.pyplot as plt

def find_all_clusters(positions, ids, cutoff: float, cell: np.array):
    pairs = clustering.find_lj_pairs(positions, ids, cutoff, cell)
    clusters = clustering.find_lj_clusters(pairs)
    return clusters


def parse_lammps_log(filename):
    with open(filename) as fi:
        data = []
        read_mode = False
        for idx, line in enumerate(fi.readlines()):
            if line.startswith("Per MPI rank memory allocation") and idx > 200:
                read_mode = True
                continue
            if line.startswith("Nlocal:"):
                read_mode = False
                continue
            if read_mode and line.strip():
                data.append(line.split())
        df = pd.DataFrame(data=[line for line in data if not line[0] == "Step"], columns=data[0])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        df.rename({"f_5[2]": "N_broken"}, axis=1, inplace=True)
        df.set_index("Step", inplace=True, verify_integrity=False)
        df = df[~df.index.duplicated(keep='first')]
    return df

def analyse_node_fractures(position_file, topology_file):
    # Node cluster analysis
    universe = mda.Universe(position_file, topology=topology_file, format="LAMMPSDUMP",
                            dt=10000)
    terminals = universe.select_atoms("type 2 or type 3")
    cluster_dict = dict()
    for timestep in universe.trajectory[::25]:
        # MDAnalysis has an awful habit of re-scaling positions.
        # Counteract that here.
        timestep._pos[:, 2] = 0.0
        terminals.positions *= 1.0 / np.array(
            [timestep.dimensions[0], timestep.dimensions[1], 1.0]
        )
        cell = np.array(
            [[0, timestep.dimensions[0]], [0, timestep.dimensions[1]]]
        )
        clusters = find_all_clusters(terminals.positions,
                                     terminals.ids,
                                     cutoff=137.5,
                                     cell=cell)
        cluster_sizes = np.array(sorted([len(cluster) for cluster in clusters]))
        cluster_dict[universe.trajectory.time] = cluster_sizes
    return cluster_dict


def parse_lammps_custom(filename: str):
    cu = CustomUniverse()
    curr_timestep = None
    with open(filename) as fi:
        for line in fi:
            if line.startswith("ITEM: TIMESTEP"):
                
                if curr_timestep is not None:
                    if "xs" in data_dict:
                        x_size= curr_timestep.dimensions[0, 1] - curr_timestep.dimensions[0, 0] 
                        data_dict["x"] = [float(item) * x_size for item in data_dict["xs"]]
                    
                    if "ys" in data_dict:
                        x_size= curr_timestep.dimensions[1, 1] - curr_timestep.dimensions[1, 0] 
                        data_dict["y"] = [float(item) * x_size for item in data_dict["ys"]] 
                    
                    if "zs" in data_dict:
                        x_size= curr_timestep.dimensions[2, 1] - curr_timestep.dimensions[2, 0] 
                        data_dict["z"] = [float(item) * x_size for item in data_dict["zs"]]
                    
                    curr_timestep.positions = np.array([[float(item) for item in data_dict["x"]],
                                                       [float(item) for item in data_dict["y"]]]).T
                    if "vx" in data_dict and "vy" in data_dict:
                        curr_timestep.velocities = np.array([[float(item) for item in data_dict["vx"]],
                                                       [float(item) for item in data_dict["vy"]]]).T
                    
                    cu.trajectory.append(curr_timestep)
                    
                read_mode = False
                curr_timestep = Timestep()
                curr_timestep.time = float(fi.readline())
            
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                curr_timestep.num = int(fi.readline())
                
            if line.startswith("ITEM: BOX BOUNDS"):
                xlo, xhi = [float(item) for item in fi.readline().split()]
                ylo, yhi = [float(item) for item in fi.readline().split()]
                zlo, zhi = [float(item) for item in fi.readline().split()]
                curr_timestep.dimensions = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])
            
            
            if line.startswith("ITEM: ATOMS"):
                cols = line.removeprefix("ITEM: ATOMS ").strip().split(" ")
                read_mode = True
                data_dict = {col: [] for col in cols}
                continue
                
            if read_mode:
                splitline = line.strip().split(" ")
                for idx, item in enumerate(splitline):
                    print(idx, item, cols, splitline)
                    col_name = cols[idx]
                    data_dict[col_name].append(item)
    return cu
    
def main():
    if len(sys.argv) == 4:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
        log_file = sys.argv[3]
    else:
        position_file = "output-stretch.lammpstrj.gz"
        topology_file = "hexagonal-net.dat"
        log_file = "log.polymer_total.txt"
    
    df = parse_lammps_log(log_file)
    node_dict = analyse_node_fractures(position_file, topology_file)
    average_cluster_size = {int(key): np.mean(val) for key, val in node_dict.items()}
    
    avg_cluster_series = np.zeros_like(df["Pxy"])
    for key, val in sorted(average_cluster_size.items(), key=lambda x: x[0]):
        avg_cluster_series[df.index >= key] = val
        

    df["Avg_K"] = avg_cluster_series
    df["Xratio"] = (df["Xhi"] - df["Xlo"])/(df["Xhi"].iloc[0] - df["Xlo"].iloc[0])
    df["Yratio"] = (df["Yhi"] - df["Ylo"])/(df["Yhi"].iloc[0] - df["Ylo"].iloc[0])   
   
    df.drop(["Atoms", "Xlo", "Xhi", "Ylo", "Yhi"], axis=1, inplace=True) 
    df.to_csv("results.csv")
        
if __name__ == "__main__":
    main()
