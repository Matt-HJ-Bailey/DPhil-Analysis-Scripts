#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:01:58 2021

@author: matthew-bailey
"""
import subprocess
from lammps_parser import parse_molecule_topology, write_molecule_topology


def main():
    epsilon = 0.01
    periodic_box, masses, atoms, molecules, bonds, angles = parse_molecule_topology(
        "./single-defect.data"
    )

    for atom_id in atoms:
        for dim in range(2):
            atoms[atom_id]["pos"][dim] += epsilon
            write_molecule_topology(
                "./offset.data", periodic_box, masses, atoms, molecules, bonds, angles
            )
            res = subprocess.run(["lmp", "-in", "hessian.inpt"], capture_output=True)
            print(res)
            atoms[atom_id]["pos"][dim] -= epsilon


if __name__ == "__main__":
    main()
