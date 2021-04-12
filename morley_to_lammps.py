#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:03:04 2021

@author: matthew-bailey
"""

import sys
import os

from morley_parser import load_morley
from graph_to_molecules import graph_to_molecules


def morley_to_lammps(morley_prefix, lammps_name):
    pos_dict, graph, periodic_box = load_morley(morley_prefix)
    curves = graph_to_molecules(graph=graph, pos=pos_dict, periodic_box=periodic_box)
    curves.rescale(300)
    periodic_box *= 300
    curves.to_lammps(lammps_name, periodic_box=periodic_box, mass=0.5 / 6)


def find_morley_prefixes(in_directory="./"):
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
            morley_to_lammps(morley_prefix, f"{prefix}.data")


if __name__ == "__main__":
    find_morley_prefixes()
