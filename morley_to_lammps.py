#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:03:04 2021

@author: matthew-bailey
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from morley_parser import load_morley
from graph_to_molecules import graph_to_molecules


def morley_to_lammps(morley_prefix:str, lammps_name:str, desired_box =None):
    pos_dict, graph, periodic_box = load_morley(morley_prefix)
    
    if desired_box is not None:
        current_x_len = periodic_box[0, 1] -  periodic_box[0, 0]
        current_y_len = periodic_box[1, 1] -  periodic_box[1, 0]
        
        desired_x_len = desired_box[0, 1] -  desired_box[0, 0]
        desired_y_len = desired_box[1, 1] -  desired_box[1, 0]
        
        matrix = np.array([[desired_x_len / current_x_len, 0.0],
                           [0.0, desired_y_len / current_y_len]])    
        print(matrix, current_x_len, desired_x_len)          
        for key, pos in pos_dict.items():
            pos_dict[key] = matrix @ pos
        periodic_box[:, 1] = matrix @ periodic_box[:, 1]
           
    curves = graph_to_molecules(graph=graph, pos=pos_dict, periodic_box=periodic_box)
    curves.rescale(300)
    periodic_box *= 300
    print(f"Writing to {lammps_name}")
    curves.to_lammps(lammps_name, periodic_box=periodic_box, mass=0.5 / 6)
    fig, ax = plt.subplots()
    curves.plot_onto(ax)
    ax.axis("equal")
    plt.show()

def find_morley_prefixes(in_directory: str="./"):
    """
    Find the prefixes of all morley files in all subdirectories.
    """
    
    for directory in os.listdir(in_directory):
        directory = os.path.join(in_directory, directory)
        if not os.path.isdir(directory):
            continue
        for subdir in os.listdir(directory):
            subdir = os.path.join(directory, subdir)
            if not os.path.isdir(subdir):
                continue
            subdir_files = [
                item for item in os.listdir(subdir) if item.endswith(".dat")
            ]
            if not subdir_files:
                continue

            prefixes = set(item.rsplit("_", 2)[0] for item in subdir_files)
            if len(prefixes) != 1:
                print("Got multiple prefixes")
            prefix = prefixes.pop()
            morley_prefix = os.path.join(subdir, prefix) + "_A"
            print(morley_prefix)
            yield morley_prefix
            
def find_morley_timesteps(in_directory: str="./"):
    """
    Find the timesteps of morley outputs in this directory
    """
    fnames = glob.glob(f"{in_directory}*_t0_*.dat")
    temperatures = set()
    times = set()
    prefixes = set()
    for fname in fnames:
        prefix, temperature, time, aorb, suffix = fname.rsplit("_", maxsplit=4)
        prefixes.add(prefix)
        temperatures.add(temperature)
        times.add(time)
    for prefix in sorted(prefixes):
        for temperature in sorted(temperatures, key=lambda s: int(s[1:])):
            for time in sorted(times, key=lambda s:int(s)):
                yield f"{prefix}_{temperature}_{time}_A"   

def main():
    for morley_prefix in find_morley_timesteps():
        #morley_prefix = "STRETCH_NETMC_-2.0_25_A"
        morley_to_lammps(morley_prefix,
                         os.path.basename(morley_prefix) + ".data",
                         desired_box=np.array([[0.0, 18.0*2.0/np.sqrt(3)],
                                               [0.0, 18.0]]))

if __name__ == "__main__":
    main()
