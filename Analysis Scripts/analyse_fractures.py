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
import networkx as nx
import gzip


def find_all_clusters(positions, ids, cutoff: float, cell: np.array):
    pairs = clustering.find_lj_pairs(positions, ids, cutoff, cell)
    clusters = clustering.find_lj_clusters(pairs)
    return clusters


def reconstruct_molecules_from_bonds(bonds):
    """
    Given a list of MDA bonds, find the molecules they correspond to.

    A molecule is a single strongly connected component of the overall graph.
    """
    mol_graph = nx.Graph()
    mol_graph.add_edges_from([tuple(atom.id for atom in bond.atoms) for bond in bonds])
    return [
        frozenset(component)
        for component in nx.algorithms.connected_components(mol_graph)
    ]


def calculate_velocity_ipr(velocities: np.ndarray) -> float:
    """
    Calculate the inverse participation ratio to measure localisation.
    """

    kinetic_energies = 0.5 * np.sum(velocities ** 2, axis=1)
    ke_fractions = kinetic_energies / np.sum(kinetic_energies)
    ipr = ke_fractions ** 2
    return np.sum(ipr)


def find_edge_molecules(universe):
    """
    Find the molecules on the edge of a hole.

    An edge molecule is defined as having at least one head group
    in a cluster of two molecules.
    """

    terminals = universe.select_atoms("type 2 or type 3")
    cell = np.array(
        [
            [0, universe.trajectory[0].dimensions[0]],
            [0, universe.trajectory[0].dimensions[1]],
        ]
    )
    clusters = find_all_clusters(
        terminals.positions, terminals.ids, cutoff=137.5, cell=cell
    )
    two_clusters = [cluster for cluster in clusters if len(cluster) == 2]

    molecs = reconstruct_molecules_from_bonds(universe.bonds)

    edge_molecs = []
    for molec in molecs:
        for cluster in two_clusters:
            overlap = cluster.intersection(molec)
            if overlap:
                edge_molecs.append(molec)
                break
    return edge_molecs


def parse_lammps_log(filename):
    with open(filename) as fi:
        data = []
        read_mode = False
        for idx, line in enumerate(fi.readlines()):
            if line.startswith("Per MPI rank memory allocation") and idx > 190:
                read_mode = True
                continue
            if line.startswith("Nlocal:"):
                read_mode = False
                continue
            if read_mode and line.strip():
                data.append(line.split())
        df = pd.DataFrame(
            data=[line for line in data if not line[0] == "Step"], columns=data[0]
        )

        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        df.rename({"f_5[2]": "N_broken"}, axis=1, inplace=True)
        df.set_index("Step", inplace=True, verify_integrity=False)
        df = df[~df.index.duplicated(keep="first")]
    return df


def analyse_node_fractures(position_file, topology_file):
    # Node cluster analysis
    universe = mda.Universe(topology_file, position_file, format="LAMMPSDUMP", dt=10000)
    terminals = universe.select_atoms("type 2 or type 3")
    cluster_dict = dict()
    for timestep in universe.trajectory[::25]:
        # MDAnalysis has an awful habit of re-scaling positions.
        # Counteract that here.
        timestep._pos[:, 2] = 0.0
        terminals.positions *= 1.0 / np.array(
            [timestep.dimensions[0], timestep.dimensions[1], 1.0]
        )
        cell = np.array([[0, timestep.dimensions[0]], [0, timestep.dimensions[1]]])
        clusters = find_all_clusters(
            terminals.positions, terminals.ids, cutoff=137.5, cell=cell
        )
        cluster_sizes = np.array(sorted([len(cluster) for cluster in clusters]))
        cluster_dict[universe.trajectory.time] = cluster_sizes
    return cluster_dict


class CustomUniverse:
    def __init__(self):
        self.trajectory = []


class Timestep:
    def __init__(self):
        self.positions = None
        self.time = None
        self.velocities = None
        self.dimensions = None


def parse_lammps_custom(filename: str):
    cu = CustomUniverse()
    curr_timestep = None
    data_dict = None

    if filename.endswith("gz"):
        open_func = gzip.open
        args = "rt"
    else:
        open_func = open
        args = "r"
    with open_func(filename, args) as fi:
        for line in fi:
            if line.startswith("ITEM: TIMESTEP"):

                if curr_timestep is not None:
                    if "xs" in data_dict:
                        x_size = (
                            curr_timestep.dimensions[0, 1]
                            - curr_timestep.dimensions[0, 0]
                        )
                        data_dict["x"] = [
                            float(item) * x_size for item in data_dict["xs"]
                        ]

                    if "ys" in data_dict:
                        x_size = (
                            curr_timestep.dimensions[1, 1]
                            - curr_timestep.dimensions[1, 0]
                        )
                        data_dict["y"] = [
                            float(item) * x_size for item in data_dict["ys"]
                        ]

                    if "zs" in data_dict:
                        x_size = (
                            curr_timestep.dimensions[2, 1]
                            - curr_timestep.dimensions[2, 0]
                        )
                        data_dict["z"] = [
                            float(item) * x_size for item in data_dict["zs"]
                        ]

                    curr_timestep.positions = np.array(
                        [
                            [float(item) for item in data_dict["x"]],
                            [float(item) for item in data_dict["y"]],
                        ]
                    ).T
                    if "vx" in data_dict and "vy" in data_dict:
                        curr_timestep.velocities = np.array(
                            [
                                [float(item) for item in data_dict["vx"]],
                                [float(item) for item in data_dict["vy"]],
                            ]
                        ).T

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
                curr_timestep.dimensions = np.array(
                    [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
                )

            if line.startswith("ITEM: ATOMS"):
                cols = line.removeprefix("ITEM: ATOMS ").strip().split(" ")
                read_mode = True
                data_dict = {col: [] for col in cols}
                continue

            if read_mode:
                splitline = line.strip().split(" ")
                for idx, item in enumerate(splitline):
                    col_name = cols[idx]
                    data_dict[col_name].append(item)
    return cu


def track_broken_bonds(
    position_file, topology_file, break_len=137.5, filename: str = "./edge-vs-bulk.csv"
):
    universe = mda.Universe(
        topology_file,
        position_file,
        format="LAMMPSDUMP",
        dt=10000,
        topology_format="DATA",
    )
    step = 50
    is_broken_arr = np.zeros(
        [len(universe.trajectory[::step]), len(universe.bonds)], dtype=bool
    )

    # Find bond indices of the edges
    edge_molecs = find_edge_molecules(universe)
    edge_b_idxs = []
    for b_idx, bond in enumerate(universe.bonds):
        bond_ids = frozenset(atom.id for atom in bond.atoms)
        is_on_edge = any([bool(bond_ids.intersection(molec)) for molec in edge_molecs])
        if is_on_edge:
            edge_b_idxs.append(b_idx)
    edge_b_idxs = np.array(edge_b_idxs)

    bulk_b_idxs = np.array(
        [idx for idx in range(len(universe.bonds)) if idx not in edge_b_idxs]
    )
    times = []
    for idx, timestep in enumerate(universe.trajectory[::step]):
        print(timestep.time)
        times.append(timestep.time)
        for b_idx, bond in enumerate(universe.bonds):
            bond_len = bond.length(pbc=True)
            if bond_len > break_len:
                is_broken_arr[idx:, b_idx] = True

    fraction_edge_broken = np.mean(is_broken_arr[:, edge_b_idxs], axis=1)
    fraction_bulk_broken = np.mean(is_broken_arr[:, bulk_b_idxs], axis=1)
    df = pd.DataFrame(
        {"time": times, "edge": fraction_edge_broken, "bulk": fraction_bulk_broken}
    )
    df.to_csv(filename, index=False)


def calculate_vacf(cu):
    """
    Calculate the velocity autocorrelation function from a custom universe
    """
    vacf_data = []
    for timediff in range(0, len(cu.trajectory), 1):
        samples = len(cu.trajectory) - timediff
        vac = 0
        for i, j in zip(
            range(0, len(cu.trajectory) - timediff), range(timediff, len(cu.trajectory))
        ):
            vels_i, vels_j = cu.trajectory[i].velocities, cu.trajectory[j].velocities
            dots = np.sum(vels_i * vels_j, axis=1)
            vac += np.mean(dots)
        vac /= samples
        vacf_data.append(vac)
    return np.array(vacf_data)


def plot_velocity_arrows(cu):

    vel_magnitudes = np.array(
        [np.linalg.norm(timestep.velocities, axis=1) for timestep in cu.trajectory]
    )

    this_cm = cm.get_cmap("coolwarm")
    normaliser = matplotlib.colors.SymLogNorm(
        vmin=np.min(vel_magnitudes), vmax=np.max(vel_magnitudes), linthresh=1.0
    )

    for idx, timestep in enumerate(cu.trajectory[:]):
        print(timestep.time)
        fig, ax = plt.subplots()
        pos_x = timestep.positions[:, 0]
        pos_y = timestep.positions[:, 1]
        arrow_u = timestep.velocities[:, 0]
        arrow_v = timestep.velocities[:, 1]

        velocities = np.linalg.norm(timestep.velocities, axis=1)
        colors = this_cm(normaliser(velocities))
        ax.scatter(pos_x, pos_y, s=5.0, color=colors, edgecolor="black", linewidths=0.5)
        ax.quiver(
            pos_x,
            pos_y,
            arrow_u,
            arrow_v,
            color=colors,
            scale_units="xy",
            scale=0.1,
            width=0.005,
        )
        ax.axis("equal")
        ax.axis("off")
        fig.savefig(f"./Images/quiver-{idx}.png", dpi=276, bbox_inches="tight")
        plt.close(fig)


def main():
    if len(sys.argv) == 4:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
        log_file = sys.argv[3]
    else:
        position_file = "output-stretch.lammpstrj.gz"
        topology_file = "hexagonal-net.dat"
        log_file = "log.polymer_total.txt"

    cu = parse_lammps_custom(position_file)
    mass = 0.08333333333333333e-21
    boltz = 1.38064852e-23
    kes = []
    iprs = []
    eff_ts = []
    for idx, timestep in enumerate(cu.trajectory):
        ke = 0.5 * mass * np.sum(timestep.velocities ** 2)
        kes.append(ke)
        eff_t = ke / (timestep.velocities.shape[0] * boltz)
        eff_ts.append(eff_t)
        ipr = calculate_velocity_ipr(timestep.velocities)
        iprs.append(ipr)

    vacf_data = calculate_vacf(cu)
    out_df = pd.DataFrame(
        {
            "time": [timestep.time for timestep in cu.trajectory],
            "vacf": vacf_data,
            "ipr": iprs,
            "ke": kes,
            "eff_ts": eff_ts,
        }
    )
    out_df.to_csv("velocity-data.csv")

    df = parse_lammps_log(log_file)
    node_dict = analyse_node_fractures(position_file, topology_file)
    average_cluster_size = {int(key): np.mean(val) for key, val in node_dict.items()}

    avg_cluster_series = np.zeros_like(df["Pxy"])
    for key, val in sorted(average_cluster_size.items(), key=lambda x: x[0]):
        avg_cluster_series[df.index >= key] = val

    df["Avg_K"] = avg_cluster_series
    df["Xratio"] = (df["Xhi"] - df["Xlo"]) / (df["Xhi"].iloc[0] - df["Xlo"].iloc[0])
    df["Yratio"] = (df["Yhi"] - df["Ylo"]) / (df["Yhi"].iloc[0] - df["Ylo"].iloc[0])

    df.drop(["Atoms", "Xlo", "Xhi", "Ylo", "Yhi"], axis=1, inplace=True)
    df.to_csv("results.csv")


if __name__ == "__main__":
    main()
