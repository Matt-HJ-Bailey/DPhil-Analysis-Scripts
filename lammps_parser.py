#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:48:26 2019

@author: matthew-bailey
"""

from collections import defaultdict

import numpy as np
from typing import Tuple, Dict, List, Union


def parse_molecule_topology(
    filename: str,
) -> Tuple[
    Dict[int, Dict[str, Union[str, np.array]]],
    Dict[int, List[int]],
    List[Tuple[int, int]],
]:
    """
    Extracts atom, molecule and position information
    from a LAMMPS data file.
    :param filename: the name of the lammps file to open
    :return atoms: a dictionary of atoms, with atom ids as keys
    and the values are a dictionary of the type and position.
    :return molecs: a dictionary of molecules, with molecule ids
    as keys and values a list of atoms in that molecule.
    :return bonds: a list of pairs, representing atom ids
    at each end of the bonds.
    """
    bonds_mode = False
    atoms_mode = False
    angles_mode = False
    masses_mode = False
    molecules = defaultdict(list)
    atoms = defaultdict(dict)
    bonds = defaultdict(dict)
    angles = defaultdict(dict)
    masses = dict()
    periodic_box = np.zeros([3, 2], dtype=float)
    with open(filename, "r") as fi:
        for line in fi.readlines():
            if "xlo" in line or "xhi" in line:
                xlo, xhi, _, _ = line.split()
                periodic_box[0, :] = float(xlo), float(xhi)
            if "ylo" in line or "yhi" in line:
                ylo, yhi, _, _ = line.split()
                periodic_box[1, :] = float(ylo), float(yhi)
            if "zlo" in line or "zhi" in line:
                zlo, zhi, _, _ = line.split()
                periodic_box[2, :] = float(zlo), float(zhi)
            if not line:
                continue
            if "Atoms" in line:
                atoms_mode = True
                bonds_mode = False
                angles_mode = False
                masses_mode = False
                continue
            if "Bonds" in line:
                atoms_mode = False
                bonds_mode = True
                angles_mode = False
                masses_mode = False
                continue
            if "Angles" in line:
                atoms_mode = False
                bonds_mode = False
                angles_mode = True
                masses_mode = False
                continue
            if "Masses" in line:
                atoms_mode = False
                bonds_mode = False
                angles_mode = False
                masses_mode = True
                continue

            if masses_mode:
                try:
                    atom_type, mass = line.split()
                    masses[int(atom_type)] = float(mass)
                except ValueError:
                    if line == "\n":
                        continue
                    print(
                        "Could not read line:",
                        line,
                        "expected form: atom_id, molec_id, type, x, y, z",
                    )
                    continue
            if atoms_mode:
                try:
                    atom_id, molec_id, atom_type, x, y, z = line.split()
                except ValueError:
                    if line == "\n":
                        continue
                    print(
                        "Could not read line:",
                        line,
                        "expected form: atom_id, molec_id, type, x, y, z",
                    )
                    continue
                atom_id = int(atom_id)
                molec_id = int(molec_id)
                atom_type = int(atom_type)
                x, y, z = float(x), float(y), float(z)
                atoms[atom_id] = {"type": atom_type, "pos": np.array([x, y, z])}
                molecules[molec_id].append(atom_id)
            if bonds_mode:
                try:
                    bond_id, bond_type, atom_a, atom_b = line.split()
                except ValueError:
                    if line == "\n":
                        continue
                    print(
                        "Could not read bond line:",
                        line,
                        "Expected form: bond_id, bond_type, a, b",
                    )
                    continue
                bonds[bond_id] = {
                    "type": bond_type,
                    "atoms": (int(atom_a), int(atom_b)),
                }
            if angles_mode:
                try:
                    angle_id, angle_type, atom_a, atom_b, atom_c = line.split()
                except ValueError:
                    if line == "\n":
                        continue
                    print(
                        "Could not read bond line:",
                        line,
                        "Expected form: bond_id, bond_type, a, b",
                    )
                    continue
                angles[angle_id] = {
                    "type": angle_type,
                    "atoms": (int(atom_a), int(atom_b), int(atom_c)),
                }
    return periodic_box, masses, atoms, molecules, bonds, angles


def write_molecule_topology(
    filename: str, periodic_box, masses, atoms, molecules, bonds, angles
):
    with open(filename, "w") as fi:
        fi.write("Generated by write_molecule_topology\n")
        fi.write("\n")
        fi.write(f"\t{len(atoms)}\t atoms\n")
        fi.write(f"\t{len(bonds)}\t bonds\n")
        fi.write(f"\t{len(angles)}\t angles\n")
        fi.write("\n")
        atom_types = set(atom["type"] for key, atom in atoms.items())
        fi.write(f"\t{len(atom_types)}\t atom types\n")
        bond_types = set(bond["type"] for key, bond in bonds.items())
        fi.write(f"\t{len(bond_types)}\t bond types\n")
        angle_types = set(angle["type"] for key, angle in angles.items())
        fi.write(f"\t{len(angle_types)}\t angle types\n")
        fi.write("\n")
        fi.write(f"\t{periodic_box[0, 0]}\t{periodic_box[0, 1]}\t xlo xhi\n")
        fi.write(f"\t{periodic_box[1, 0]}\t{periodic_box[1, 1]}\t ylo yhi\n")
        fi.write(f"\t{periodic_box[2, 0]}\t{periodic_box[2, 1]}\t zlo zhi\n")
        fi.write("\n")
        fi.write("Masses\n")
        for atom_type, mass in masses.items():
            fi.write(f"\t{atom_type}\t{mass}\n")
        fi.write("\n")
        fi.write("Atoms\n")
        fi.write("\n")
        for atom_id, data in atoms.items():
            atom_type = data["type"]
            x, y, z = data["pos"]
            fi.write(f"\t{atom_id}\t{atom_type}\t{x}\t{y}\t{z}\n")
        fi.write("\n")
        fi.write("Bonds\n")
        fi.write("\n")
        for bond_id, data in bonds.items():
            bond_type = data["type"]
            u, v = data["atoms"]
            fi.write(f"\t{bond_id}\t{bond_type}\t{u}\t{v}\n")
        fi.write("\n")
        fi.write("Angles\n")
        fi.write("\n")
        for angle_id, data in angles.items():
            angle_type = data["type"]
            u, v, w = data["atoms"]
            fi.write(f"\t{angle_id}\t{angle_type}\t{u}\t{v}\t{w}\n")


def main():
    periodic_box, masses, atoms, molecules, bonds, angles = parse_molecule_topology(
        "./single-defect.data"
    )
    write_molecule_topology(
        "./single-defect-rewritten.data",
        periodic_box,
        masses,
        atoms,
        molecules,
        bonds,
        angles,
    )


if __name__ == "__main__":
    main()
