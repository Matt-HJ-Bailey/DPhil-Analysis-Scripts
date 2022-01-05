#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:26:50 2019

@author: matthew-bailey
"""

import sys
import copy
from collections import Counter, defaultdict

from typing import Iterable, Any, Dict, List, Tuple, Set, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects

import scipy.optimize
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform

import MDAnalysis as mda
import networkx as nx
import numpy as np

from clustering import (
    find_lj_pairs,
    find_lj_clusters,
    find_cluster_centres,
    connect_clusters,
    apply_minimum_image_convention,
)
from lammps_parser import parse_molecule_topology
from rings.periodic_ring_finder import PeriodicRingFinder
from rings.ring_finder import RingFinderError, convert_to_ring_graph
from draw_and_colour import draw_periodic_coloured

from analysis_files import AnalysisFiles

LJ_BOND = 137.5
FIND_BODIES = True
STEP_SIZE = 1000
DT = 0.001 * 20000


def bootstrap(arr: np.ndarray, samples=10, lo=2.5, hi=97.5, seed=None):
    """
    Calculate the bootstrap confidence interval for this array.
    """
    means = np.empty([samples], dtype=float)
    rng = np.random.default_rng(seed=seed)
    for idx in range(samples):
        data = rng.choice(arr, size=arr.shape, replace=True)
        means[idx] = np.mean(data)
    return np.percentile(means, [lo, hi])


def calculate_roughness(positions: np.ndarray, periodic_cell: np.ndarray, bins=100):
    """
    Calculate a roughness coefficient
    """
    x_mic = (periodic_cell[0, 1] - periodic_cell[0, 0]) / 2
    y_mic = (periodic_cell[1, 1] - periodic_cell[1, 0]) / 2
    z_mic = (periodic_cell[2, 1] - periodic_cell[2, 0]) / 2

    x_seps = squareform(pdist(positions[:, 0].reshape(-1, 1)))
    y_seps = squareform(pdist(positions[:, 1].reshape(-1, 1)))
    z_seps = squareform(pdist(positions[:, 2].reshape(-1, 1)), "sqeuclidean")

    x_seps[x_seps > x_mic] -= x_mic
    x_seps[x_seps < -x_mic] += x_mic
    y_seps[y_seps > y_mic] -= y_mic
    y_seps[y_seps < -y_mic] += y_mic
    z_seps[z_seps > z_mic] -= z_mic
    z_seps[z_seps < -z_mic] += z_mic

    linear_distances = np.sqrt(x_seps ** 2 + y_seps ** 2)
    # Now need to separate into bins
    bin_edges = np.linspace(
        np.min(linear_distances), np.max(linear_distances), num=bins
    )
    bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(bins - 1)]
    hist = np.zeros_like(bin_edges)
    hist_lo = np.zeros_like(bin_edges)
    hist_hi = np.zeros_like(bin_edges)
    for idx, (lo, hi) in enumerate(bins):
        mask = np.logical_and(linear_distances >= lo, linear_distances < hi)
        hist[idx] = np.mean(z_seps[mask])
        hist_lo[idx], hist_hi[idx] = bootstrap(z_seps[mask].ravel())
    return bin_edges, hist, hist_lo, hist_hi


def roughness_func(rs, alpha, w, xi):
    predicted = np.empty_like(rs)
    predicted[rs < xi] = 2.0 * w ** 2 * (rs[rs < xi] / xi) ** (2 * alpha)
    predicted[rs >= xi] = 2.0 * w ** 2
    return predicted


def analyse_roughness(bin_edges, hist):
    """
    Calculate the three critical roughness parameters

    Parameters
    ----------
    bin_edges
        x values of the roughness
    hist
        Roughness values at each x

    Returns
    -------
        alpha is the local roughness parameter
        xi is the local roughness length scale
        w is the width of the surface
    """

    size = bin_edges.shape[0]
    lo_idx = int(size * 0.25)
    hi_idx = int(size * 0.75)

    width = np.mean(hist[lo_idx:hi_idx])

    first_above_width = np.argmax(hist > width)

    xi = bin_edges[first_above_width]

    w = np.sqrt(width / 2.0)
    res = scipy.optimize.minimize(
        lambda xs: np.sum(
            (hist - roughness_func(rs=bin_edges, alpha=xs[0], w=xs[1], xi=xs[2])) ** 2
        ),
        x0=[1.0, w, xi],
        bounds=[(0.0, 1.0), (0.8 * w, 1.2 * w), (0.5 * xi, 1.5 * xi)],
    )
    print(res)
    alpha, width, xi = res.x

    return alpha, width, xi


def calculate_existence_matrix(
    ring_trajectory: Iterable[Iterable[Any]],
    graph_trajectory: Iterable[nx.Graph],
    all_atoms: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate when a given ring can be said to exist.

    Parameters
    ----------
    ring_trajectory
        An iterable containing a set of rings, one for each step
    graph_trajectory
        An iterable containing graphs at each step
    all_atoms
        A dictionary containing atom data

    Returns
    -------
    np.ndarray
        The existence matrix of size [num rings, steps], True if ring i exists at step j
    np.ndarray
        The sizes of each unique ring
    """
    assert len(ring_trajectory) == len(graph_trajectory), (
        f"Ring trajectory {len(ring_trajectory)}"
        + "and graph trajectory {len(graph_trajectory)} must have the same size."
    )
    # Find all unique rings
    molecs_trajectory = []
    ring_sizes = {}
    for ring_step, graph_step in zip(ring_trajectory, graph_trajectory):
        molec_step = []
        for ring in ring_step:
            size = len(ring)
            molecs = shape_to_molecs(ring, graph_step)
            ring_sizes[molecs] = size
            molec_step.append(molecs)
        molecs_trajectory.append(frozenset(molec_step))

    # Get a unique list of all the rings, but in a consistent order
    all_rings = sorted(list(frozenset().union(*molecs_trajectory)), key=tuple)
    existence_matrix = []
    for ring in all_rings:
        is_in_step = [ring in step for step in molecs_trajectory]
        existence_matrix.append(is_in_step)
    existence_matrix = np.array(existence_matrix, dtype=bool)
    return existence_matrix, np.array([ring_sizes[ring] for ring in all_rings])


def shape_to_atoms(shape, graph):
    """
    Get the atoms corresponding to this shape.
    """

    nodes = shape.to_node_list()
    clusters_data = nx.get_node_attributes(graph, "cluster")
    true_nodes = [node % len(graph) for node in nodes]
    clusters_in_shape = [clusters_data[true_node] for true_node in true_nodes]
    atoms_in_shape = frozenset().union(*clusters_in_shape)
    return atoms_in_shape


def shape_to_molecs(shape, graph):
    """
    Get the atoms corresponding to this shape.
    """

    nodes = shape.to_node_list()
    true_nodes = [node % len(graph) for node in nodes]
    molec_data = nx.get_node_attributes(graph, "molec")
    molec_counts = Counter()
    for true_node in true_nodes:
        assert (
            true_node in graph
        ), f"Node {true_node} not in graph with {len(graph)} nodes."
        molec_counts.update(molec_data[true_node])
    molecs_in_shape = frozenset(
        [molec for molec, count in molec_counts.items() if count == 2]
    )
    return molecs_in_shape


def find_broken_bonds(
    bonds: Iterable[Tuple[int, int]], positions: np.ndarray, periodic_box: np.ndarray
) -> Set[Set[int]]:
    """
    Find which bonds have been broken this step.

    A broken bond is one that is over LJ_BOND length in this step,
    taking the minimum image convention into account.
    Watch out for bond indices not matching positions data.

    Parameters
    ----------
    bonds
        Iterable of pairs of atomic ids.
    positions
        Nx3 array of positions
    periodic_box
        3x2 array in the form [[0, x_max], [0, y_max], [0, z_max]]

    Returns
    -------
        Frozenset of frozensets, where each subset is a (u, v) pair.
    """
    broken_bonds = []
    for u, v in bonds:
        x_mic, y_mic, z_mic = periodic_box[:, 1] / 2
        # -1 is for LAMMPS offset
        bond_vec = positions[u - 1, :] - positions[v - 1, :]
        bond_vec = apply_minimum_image_convention(bond_vec, x_mic, y_mic)
        # All bonds over 137.5 are permanently broken
        if np.linalg.norm(bond_vec) > LJ_BOND:
            broken_bonds.append(frozenset([u, v]))
    return frozenset(broken_bonds)


def draw_rings(ring_finder, timestep, graph, ax=None):
    """
    Draw the rings, with labels if need be.

    Parameters
    ----------
    ring_finder

    timestep
        An mdanalysis timestep
    graph
        A graph of edges that the ring finder used.
    title
        The name to save the graph to
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlim(0, timestep.dimensions[0])
    ax.set_ylim(0, timestep.dimensions[1])
    ring_finder.draw_onto(ax, cmap_name="tab20b", min_ring_size=4, max_ring_size=30)

    draw_periodic_coloured(
        graph,
        periodic_box=np.array(
            [[0, timestep.dimensions[0]], [0, timestep.dimensions[1]]]
        ),
        ax=ax,
    )
    # for node, data in graph.nodes(data=True):
    #    text = ax.text(
    #        data["pos"][0],
    #        data["pos"][1],
    #        ",".join(str(item) for item in data["molec"]),
    #        horizontalalignment="center",
    #        verticalalignment="center",
    #        color="white",
    #        fontsize=6,
    #    )
    #    text.set_path_effects(
    #        [
    #            path_effects.Stroke(linewidth=1, foreground="black"),
    #            path_effects.Normal(),
    #        ]
    #    )

    ax.axis("off")


def plot_lifetimes(existence_matrix, ring_sizes, fig=None, ax=None):
    """
    Plot lifetime bars, coloured by ring size.

    Parameters
    ----------
    existence_matrix
        NxT bool matrix, with N unique rings over T timesteps.
    ring_sizes
        Nx1 int matrix containing the size of each ring
    ax
        The ax to plot onto

    Returns
    -------
        ax with lifetime bars on it

    """
    if ax is None and ax is None:
        fig, ax = plt.subplots()
    lines = []
    for i in range(existence_matrix.shape[0]):
        current_run = []
        is_true = np.where(existence_matrix[i, :])[0]
        runs = []
        current_run = [is_true[0]]
        for j in range(1, is_true.shape[0]):
            if is_true[j - 1] == is_true[j] - 1:
                current_run.append(is_true[j])
            else:
                runs.append(current_run)
                current_run = [is_true[j]]
        runs.append(current_run)
        for run in runs:

            if len(run) > 1:
                lines.append(
                    (
                        i,
                        min(run) * STEP_SIZE * DT * 1e-3,
                        max(run) * STEP_SIZE * DT * 1e-3,
                    )
                )
            else:
                lines.append(
                    (
                        i,
                        run[0] * STEP_SIZE * DT * 1e-3,
                        (run[0] + 0.5) * STEP_SIZE * DT * 1e-3,
                    )
                )

    # Now find the correct order for these...
    lowest_ymins = []
    for unique_x in range(existence_matrix.shape[0]):
        lines_at_x = [line for line in lines if line[0] == unique_x]
        lowest_ymin = min([line[1] for line in lines_at_x])
        highest_ymax = max([line[2] for line in lines_at_x])
        lowest_ymins.append((lines_at_x[0][0], lowest_ymin, highest_ymax))

    ymins_sorting_arr = np.array(
        [(item[1], item[2]) for item in lowest_ymins],
        dtype=[("ymin", float), ("ymax", float)],
    )
    # ymins_order = np.argsort([item[1] for item in lowest_ymins])
    ymins_order = np.argsort(ymins_sorting_arr, order=("ymin", "ymax"))
    xmapper = {old_idx: new_idx for new_idx, old_idx in enumerate(ymins_order)}

    all_xs = []
    all_ymins = []
    all_ymaxs = []
    all_sizes = []
    for line_idx in range(len(lines)):
        x, y_min, y_max = lines[line_idx]
        all_xs.append(xmapper[x])
        all_ymins.append(y_min)
        all_ymaxs.append(y_max)
        all_sizes.append(ring_sizes[x])

    x_order = np.argsort(all_xs)
    all_xs = np.asarray(all_xs)[x_order]
    all_ymins = np.asarray(all_ymins)[x_order]
    all_ymaxs = np.asarray(all_ymaxs)[x_order]
    all_sizes = np.asarray(all_sizes)[x_order]

    cmapper = cm.ScalarMappable(
        norm=colors.Normalize(vmin=3, vmax=20, clip=True), cmap="coolwarm"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(cmapper, cax=cax)
    cbar.ax.set_ylabel("Ring Size", rotation=270)
    ring_colors = cmapper.to_rgba(all_sizes)
    ax.vlines(all_xs, all_ymins, all_ymaxs, colors=ring_colors)
    ax.set_ylabel("Time / microsecond")
    ax.set_ylim(-5, 100)
    # ax.axhline(100.0, color="black", linestyle="dotted")

    # Find the index of the ring that was first born after
    # the mode switch by starting at the end
    # and working backwardss
    first_born_after = all_ymins.shape[0] - 1
    while all_ymins[first_born_after] >= 100.0:
        first_born_after -= 1
    # ax.axvline(all_xs[first_born_after], color="black", linestyle="dotted")

    ax.set_xlabel("Ring ID")
    ax.set_xlim(0, existence_matrix.shape[0])
    return ax


def write_output_files(output_files, universe, graph, ring_finder):
    """
    Write a series of useful outputs to files.

    Parameters
    ----------
    output_files
        Output file caching object
    universe
        The MDA universe we're looking at
    graph
        The edge graph we have analysed
    ring_finder
        The ring finder, already run on graph
    """
    output_files.write_coordinations(universe.trajectory.time, graph)
    output_files.write_areas(universe.trajectory.time, ring_finder.current_rings)
    output_files.write_sizes(universe.trajectory.time, ring_finder.current_rings)
    output_files.write_regularity(universe.trajectory.time, ring_finder.current_rings)

    output_files.write_maximum_entropy(
        universe.trajectory.time, ring_finder.current_rings
    )
    output_files.write_edge_lengths(
        universe.trajectory.time, ring_finder.analyse_edges()
    )
    try:
        ring_graph = convert_to_ring_graph(ring_finder.current_rings)
        assortativity = nx.numeric_assortativity_coefficient(ring_graph, "size")
    except ValueError:
        assortativity = np.nan
    output_files.write_assortativity(universe.trajectory.time, assortativity)


def main():
    # Parsing section -- read the files, and split the atoms
    # and molecules into a few types. This could probably
    # be neatened with more use of MDA.
    if len(sys.argv) == 4:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
        output_prefix = sys.argv[3]
    else:
        topology_file = "./removed-edge-1.data"
        output_prefix = "./test"
        position_file = "output-equilibrate.lammpstrj"

    universe = mda.Universe(
        topology_file,
        ["output-stretch.lammpstrj"],
        format="LAMMPSDUMP",
        dt=0.001 * 10000,
    )
    _, _, atoms, molecs, bonds, _ = parse_molecule_topology(topology_file)
    bonds = [val["atoms"] for _, val in bonds.items()]
    total_graph = nx.Graph()
    total_graph.add_edges_from(bonds)

    bond_molecs = {}
    for u, v in bonds:
        molec_u, molec_v = atoms[u]["molec"], atoms[v]["molec"]
        assert molec_u == molec_v, "Bonded atoms must be in same molecule"
        bond_molecs[u, v] = molec_u
    nx.set_edge_attributes(total_graph, bond_molecs, name="molec")

    atom_types = {atom_id: atom["type"] for atom_id, atom in atoms.items()}
    nx.set_node_attributes(total_graph, atom_types, name="atom_types")
    molec_types = {atom_id: atom["molec"] for atom_id, atom in atoms.items()}

    nx.set_node_attributes(total_graph, molec_types, name="molec")
    ring_trajectory, graph_trajectory, bond_trajectory = [], [], []
    output_files = AnalysisFiles(output_prefix)
    for timestep in universe.trajectory[::STEP_SIZE]:
        print(timestep, "out of", len(universe.trajectory), timestep.time)
        periodic_box = np.array(
            [
                [0, timestep.dimensions[0]],
                [0, timestep.dimensions[1]],
                [0, timestep.dimensions[2]],
            ]
        )
        # find the terminal atoms, and group them into clusters.
        all_atoms = universe.select_atoms("all")
        all_atoms.positions -= np.min(all_atoms.positions, axis=0)

        bin_edges, hist, hist_lo, hist_hi = calculate_roughness(
            all_atoms.positions, periodic_box
        )
        alpha, width, xi = analyse_roughness(bin_edges, hist)
        fig, ax = plt.subplots()
        ax.plot(bin_edges, hist)
        ax.fill_between(bin_edges, hist_lo, hist_hi, alpha=0.6)
        ax.plot(
            bin_edges,
            roughness_func(rs=bin_edges, alpha=alpha, xi=xi, w=width),
            color="black",
            linestyle="dashed",
        )
        ax.set_xlim(0, 2100)
        ax.set_xlabel("Separation")
        ax.set_ylim(0, 150)
        ax.set_ylabel("Roughness")
        ax.set_title(f"Xi = {xi:.2f}, alpha = {alpha:.2f}, w = {width:.2f}")
        fig.savefig(f"./roughness-{int(timestep.time)}.pdf")
        plt.close(fig)

        terminals = universe.select_atoms("type 2 or type 3")
        terminal_pairs = find_lj_pairs(
            terminals.positions, terminals.ids, LJ_BOND, cell=periodic_box[:2, :]
        )
        terminal_clusters = find_lj_clusters(terminal_pairs)

        broken_bonds_step = find_broken_bonds(bonds, all_atoms.positions, periodic_box)
        if not bond_trajectory:
            bond_trajectory = [frozenset(broken_bonds_step)]
        else:
            bond_trajectory.append(
                frozenset(bond_trajectory[-1].union(broken_bonds_step))
            )

        # Now remove those edges from the in_graph
        total_graph.remove_edges_from([(u, v) for u, v in bond_trajectory[-1]])

        body_clusters = [
            frozenset([item]) for item in universe.select_atoms("type 4").ids
        ]
        all_clusters = sorted(list(terminal_clusters.union(body_clusters)))
        # sort the list of clusters into a consistent list so
        # we can index them.
        cluster_positions = find_cluster_centres(
            all_clusters, all_atoms.positions, cutoff=50.0
        )

        ring_finder_successful = True
        try:
            G = connect_clusters(in_graph=total_graph, clusters=all_clusters)
            nx.set_node_attributes(G, cluster_positions, "pos")
            colours = dict()
            for i, cluster in enumerate(all_clusters):
                cluster_atom_types = [universe.atoms[atom - 1].type for atom in cluster]
                modal_type = Counter(cluster_atom_types).most_common(1)[0][0]
                colours[i] = (int(modal_type),)
            nx.set_node_attributes(G, colours, name="color")

            ring_finder = PeriodicRingFinder(G, cluster_positions, periodic_box[:2, :])
            # Convert each ring into atoms which have a persistent ID
            # between steps
            graph_trajectory.append(copy.deepcopy(G))
            ring_trajectory.append(ring_finder.current_rings)
        except RingFinderError as ex:
            print("failed with code: ", ex)
            ring_finder_successful = False
            graph_trajectory.append(copy.deepcopy(G))
            ring_trajectory.append([])
        except ValueError as ex:
            print("failed with value code: ", ex)
            ring_finder_successful = False
            graph_trajectory.append(copy.deepcopy(G))
            ring_trajectory.append([])
        except nx.exception.NetworkXError as ex:
            print("Failed with networkx error: ", ex)
            ring_finder_successful = False
            graph_trajectory.append(copy.deepcopy(G))
            ring_trajectory.append([])
        except RuntimeError as ex:
            print("failed with RuntimeError: ", ex)
            ring_finder_successful = False
            graph_trajectory.append(copy.deepcopy(G))
            ring_trajectory.append([])
        if ring_finder_successful:
            fig, ax = plt.subplots()
            draw_rings(ring_finder, timestep, G, ax=ax)
            fig.savefig(f"{output_prefix}_{timestep.time}.pdf")
            plt.close(fig)

            write_output_files(output_files, universe, G, ring_finder)

    output_files.flush()
    existence_matrix, ring_sizes = calculate_existence_matrix(
        ring_trajectory, graph_trajectory, atoms
    )

    births, deaths = [], []
    lifespans = defaultdict(list)
    for i in range(existence_matrix.shape[0]):
        is_true = np.where(existence_matrix[i, :])[0]
        births.append(np.min(is_true) * STEP_SIZE * DT * 1e-3)
        deaths.append(np.max(is_true) * STEP_SIZE * DT * 1e-3)
        lifespans[ring_sizes[i]].append(np.sum(is_true) * STEP_SIZE * DT * 1e-3)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cmapper = cm.ScalarMappable(
        norm=colors.Normalize(vmin=3, vmax=10, clip=True), cmap="coolwarm"
    )
    ax.scatter(births, deaths, c=cmapper.to_rgba(ring_sizes))
    ax.plot([0, 0], [110, 110], linestyle="dotted", color="black")
    # ax.axvline(100.0, linestyle="dotted", color="black")
    # ax.axhline(100.0, linestyle="dotted", color="black")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    cbar = fig.colorbar(cmapper, cax=cax)
    cbar.ax.set_ylabel("Ring Size", rotation=270)
    ax.set_xlabel("Birth / microsecond")
    ax.set_ylabel("Death / microsecond")
    fig.savefig("birth-death.pdf")
    plt.close(fig)

    ring_size_list = list(range(min(lifespans.keys()), max(lifespans.keys())))
    fig, ax = plt.subplots()
    mean_lifespans = [np.nanmean(lifespans[ring_size]) for ring_size in ring_size_list]
    std_lifespans = [
        np.nanstd(lifespans[ring_size], ddof=1) for ring_size in ring_size_list
    ]
    ax.errorbar(ring_size_list, mean_lifespans, std_lifespans)
    ax.set_ylabel("Lifespan / microsecond")
    ax.set_xlabel("Ring Size")
    ax.set_ylim(0, np.nanmax(mean_lifespans) + np.nanmax(std_lifespans))
    ax.set_xlim(0, np.max(ring_size_list))
    fig.savefig("./lifespan.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_lifetimes(existence_matrix, ring_sizes, fig=fig, ax=ax)
    fig.savefig("lifetimes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
