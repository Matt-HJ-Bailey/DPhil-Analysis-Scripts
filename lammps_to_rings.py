#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:26:50 2019

@author: matthew-bailey
"""

import sys
from collections import Counter, defaultdict

from typing import Iterable, Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import MDAnalysis as mda
import networkx as nx
import numpy as np

from clustering import (
    find_lj_pairs,
    find_lj_clusters,
    find_cluster_centres,
    connect_clusters,
)
from lammps_parser import parse_molecule_topology
from rings.periodic_ring_finder import PeriodicRingFinder
from rings.ring_finder import RingFinderError, convert_to_ring_graph
from draw_and_colour import draw_periodic_coloured
from nodeme import NodeME

LJ_BOND = 137.5
FIND_BODIES = True
STEP_SIZE = 25
DT = 0.001 * 20000


class AnalysisFiles:
    def __init__(self, prefix: str):

        self.edges_prefixes = []
        self.edges_data = []
        self.edges_file = f"{prefix}_edges.dat"

        self.areas_prefixes = []
        self.areas_data = []
        self.areas_file = f"{prefix}_areas.dat"

        self.rings_prefixes = []
        self.rings_data = []
        self.rings_file = f"{prefix}_rings.dat"

        self.coordinations_prefixes = []
        self.coordinations_data = []
        self.coordinations_file = f"{prefix}_coordinations.dat"

        self.assortativity_prefixes = []
        self.assortativity_data = []
        self.assortativity_file = f"{prefix}_assortativity.dat"

        self.maxent_prefixes = []
        self.maxent_data = []
        self.maxent_file = f"{prefix}_maxent.dat"

        self.regularity_prefixes = []
        self.regularity_data = []
        self.regularity_file = f"{prefix}_regularity.dat"

    def write_coordinations(self, prefix: str, graph: nx.Graph):
        """
        Buffer coordination data for later writing.

        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        coordination_counter = Counter([x[1] for x in graph.degree])
        self.coordinations_data.append(coordination_counter)
        self.coordinations_prefixes.append(str(prefix))

    def write_areas(self, prefix: str, ring_list):
        """
        Buffer sorted area data for later writing
        ----------
        ring_list : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        data = sorted([ring.area for ring in ring_list])
        self.areas_data.append(data)
        self.areas_prefixes.append(str(prefix))

    def write_edge_lengths(self, prefix, edge_length_list):
        self.edges_data.append(sorted(edge_length_list))
        self.edges_prefixes.append(str(prefix))

    def write_assortativity(self, prefix, assort):
        self.assortativity_data.append(str(assort))
        self.assortativity_prefixes.append(str(prefix))

    def write_sizes(self, prefix, ring_list):
        ring_sizes = Counter(len(ring) for ring in ring_list)
        self.rings_data.append(ring_sizes)
        self.rings_prefixes.append(str(prefix))

    def write_maximum_entropy(self, prefix, ring_list):
        ring_sizes = Counter(len(ring) for ring in ring_list)
        modal_ring_size, number_modal = ring_sizes.most_common(1)[0]
        me = NodeME(k_mean=modal_ring_size)(
            target_pk=number_modal / len(ring_list), k=modal_ring_size
        )
        if me is not None:
            self.maxent_data.append(me)
        else:
            self.maxent_data.append(np.array([np.nan for _ in range(3, 20)]))
        self.maxent_prefixes.append(str(prefix))

    def write_regularity(self, prefix, ring_list):
        self.regularity_data.append([ring.regularity_metric() for ring in ring_list])
        self.regularity_prefixes.append(str(prefix))

    def flush(self):
        # Write out coordinations
        with open(self.coordinations_file, "w") as fi:
            all_coordinations = set()
            for row in self.coordinations_data:
                all_coordinations = all_coordinations.union(row.keys())
            all_coordinations = sorted(list(all_coordinations))
            fi.write(
                "# Timestep, "
                + ", ".join([str(coordination) for coordination in all_coordinations])
                + "\n"
            )
            for i, row in enumerate(self.coordinations_data):
                fi.write(
                    self.coordinations_prefixes[i]
                    + ",  "
                    + ", ".join(
                        [str(row[coordination]) for coordination in all_coordinations]
                    )
                    + "\n"
                )

        with open(self.areas_file, "w") as fi:
            fi.write("# Timestep, Ring Areas...\n")
            for i, row in enumerate(self.areas_data):
                fi.write(
                    self.areas_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )

        with open(self.rings_file, "w") as fi:
            all_sizes = set()
            for row in self.rings_data:
                all_sizes = all_sizes.union(row.keys())
            all_sizes = sorted(list(all_sizes))
            fi.write(
                "# Timestep, " + ", ".join([str(size) for size in all_sizes]) + "\n"
            )
            for i, row in enumerate(self.rings_data):
                fi.write(
                    self.rings_prefixes[i]
                    + ",  "
                    + ", ".join([str(row.get(size, 0)) for size in all_sizes])
                    + "\n"
                )

        with open(self.edges_file, "w") as fi:
            fi.write("# Timestep, Edge Lengths...\n")
            for i, row in enumerate(self.edges_data):
                fi.write(
                    self.edges_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )

        with open(self.assortativity_file, "w") as fi:
            fi.write("# Timestep, Ring Assortativity\n")
            for i, row in enumerate(self.assortativity_data):
                fi.write(self.assortativity_prefixes[i] + ",  " + row + "\n")

        with open(self.regularity_file, "w") as fi:
            fi.write("# Timestep, Regularities...\n")
            for i, row in enumerate(self.regularity_data):
                fi.write(
                    self.regularity_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )

        with open(self.maxent_file, "w") as fi:
            fi.write(
                "# Timestep, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20\n"
            )
            for i, row in enumerate(self.maxent_data):
                fi.write(
                    self.maxent_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )


def calculate_existence_matrix(
    ring_trajectory: Iterable[Iterable[Any]], graph_trajectory: Iterable[nx.Graph]
):
    """
    Calculate when a given ring can be said to exist.
    """

    # Find all unique rings
    atoms_trajectory = []
    ring_sizes = {}
    for ring_step, graph_step in zip(ring_trajectory, graph_trajectory):
        atom_step = []
        for ring in ring_step:
            size = len(ring)
            atoms = shape_to_atoms(ring, graph_step)
            ring_sizes[atoms] = size
            atom_step.append(atoms)

        atoms_trajectory.append(frozenset(atom_step))

    # Get a unique list of all the rings, but in a consistent order
    all_rings = sorted(list(frozenset().union(*atoms_trajectory)))
    existence_matrix = []
    for ring in all_rings:
        is_in_step = [ring in step for step in atoms_trajectory]
        existence_matrix.append(is_in_step)
    existence_matrix = np.array(existence_matrix, dtype=bool)
    return existence_matrix, np.array([ring_sizes[ring] for ring in all_rings])


def shape_to_atoms(shape, graph):
    """
    Get the atoms corresponding to this shape.
    """

    nodes = shape.to_node_list()
    clusters_data = nx.get_node_attributes(graph, "cluster")
    clusters_in_shape = [clusters_data.get(node, frozenset()) for node in nodes]
    atoms_in_shape = frozenset().union(*clusters_in_shape)
    return atoms_in_shape


def main():
    # Parsing section -- read the files, and split the atoms
    # and molecules into a few types. This could probably
    # be neatened with more use of MDA.
    if len(sys.argv) == 4:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
        output_prefix = sys.argv[3]
    else:
        topology_file = "../polymer_total.data"
        output_prefix = "./test"
        position_file = "out.lammpstrj"

    universe = mda.Universe(
        "../polymer_total.data",
        ["output-equilibrate.lammpstrj.gz", "output-stretch.lammpstrj.gz"],
        format="LAMMPSDUMP",
        dt=0.001 * 20000,
    )
    _, _, atoms, molecs, bonds, _ = parse_molecule_topology(topology_file)
    bonds = [val["atoms"] for _, val in bonds.items()]
    total_graph = nx.Graph()
    total_graph.add_edges_from(bonds)
    atom_types = {atom_id: atom["type"] for atom_id, atom in atoms.items()}
    nx.set_node_attributes(total_graph, atom_types, name="atom_types")
    molec_types = {
        molec_id: [atom_types[atom_id] for atom_id in molec]
        for molec_id, molec in molecs.items()
    }

    ring_trajectory, graph_trajectory = [], []
    output_files = AnalysisFiles(output_prefix)
    for timestep in universe.trajectory[::STEP_SIZE]:
        print(timestep, "out of", len(universe.trajectory), timestep.time)
        periodic_box = np.array(
            [[0, timestep.dimensions[0]], [0, timestep.dimensions[1]]]
        )
        # find the terminal atoms, and group them into clusters.
        all_atoms = universe.select_atoms("all")
        all_atoms.positions -= np.min(all_atoms.positions, axis=0)
        terminals = universe.select_atoms("type 2 or type 3")
        terminal_pairs = find_lj_pairs(
            terminals.positions, terminals.ids, LJ_BOND, cell=periodic_box
        )
        terminal_clusters = find_lj_clusters(terminal_pairs)

        body_clusters = [
            frozenset([item]) for item in universe.select_atoms("type 4").ids
        ]
        all_clusters = sorted(list(terminal_clusters.union(body_clusters)))
        # sort the list of clusters into a consistent list so
        # we can index them.
        cluster_positions = find_cluster_centres(
            all_clusters, all_atoms.positions, cutoff=50.0
        )
        G = connect_clusters(in_graph=total_graph, clusters=all_clusters)
        colours = dict()
        for i, cluster in enumerate(all_clusters):
            cluster_atom_types = [universe.atoms[atom - 1].type for atom in cluster]
            modal_type = Counter(cluster_atom_types).most_common(1)[0][0]
            colours[i] = (int(modal_type),)
        # nx.draw(g, pos=cluster_positions)
        nx.set_node_attributes(G, colours, name="color")
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, timestep.dimensions[0])
        # ax.set_ylim(0, timestep.dimensions[1])
        ring_finder_successful = True
        try:
            ring_finder = PeriodicRingFinder(G, cluster_positions, periodic_box)
            # ring_finder.draw_onto(
            #    ax, cmap_name="tab20b", min_ring_size=4, max_ring_size=30
            # )

            ring_graph = convert_to_ring_graph(ring_finder.current_rings)
            # Convert each ring into atoms which have a persistent ID
            # between steps
            graph_trajectory.append(G)
            ring_trajectory.append(ring_finder.current_rings)
        except RingFinderError as ex:
            print("failed with code: ", ex)
            ring_finder_successful = False
            ring_trajectory.append([])
        except ValueError as ex:
            print("failed with value code: ", ex)
            ring_finder_successful = False
            ring_trajectory.append([])
        except nx.exception.NetworkXError as ex:
            print("Failed with networkx error: ", ex)
            ring_finder_successful = False
            ring_trajectory.append([])
        # draw_periodic_coloured(
        #    G, pos=cluster_positions, periodic_box=periodic_box, ax=ax
        # )

        # ax.axis("off")
        # fig.savefig(f"{output_prefix}_{universe.trajectory.time}.pdf")
        # plt.close(fig)
        if ring_finder_successful:

            output_files.write_coordinations(universe.trajectory.time, G)
            output_files.write_areas(
                universe.trajectory.time, ring_finder.current_rings
            )
            output_files.write_sizes(
                universe.trajectory.time, ring_finder.current_rings
            )
            output_files.write_regularity(
                universe.trajectory.time, ring_finder.current_rings
            )

            output_files.write_maximum_entropy(
                universe.trajectory.time, ring_finder.current_rings
            )
            output_files.write_edge_lengths(
                universe.trajectory.time, ring_finder.analyse_edges()
            )
            try:
                assortativity = nx.numeric_assortativity_coefficient(ring_graph, "size")
            except ValueError:
                assortativity = np.nan
            output_files.write_assortativity(universe.trajectory.time, assortativity)

    output_files.flush()
    existence_matrix, ring_sizes = calculate_existence_matrix(
        ring_trajectory, graph_trajectory
    )
    print(existence_matrix.shape)
    births, deaths = [], []
    lifespans = defaultdict(list)
    for i in range(existence_matrix.shape[0]):
        is_true = np.where(existence_matrix[i, :])[0]
        births.append(np.min(is_true) * STEP_SIZE * DT * 1e-3)
        deaths.append(np.max(is_true) * STEP_SIZE * DT * 1e-3)
        lifespans[ring_sizes[i]].append(
            max(deaths[-1] - births[-1], STEP_SIZE * DT * 1e-3)
        )

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cmapper = cm.ScalarMappable(
        norm=colors.Normalize(vmin=3, vmax=10, clip=True), cmap="coolwarm"
    )
    ax.scatter(births, deaths, c=cmapper.to_rgba(ring_sizes))
    ax.axvline(10.0, linestyle="dotted", color="black")
    ax.axhline(10.0, linestyle="dotted", color="black")
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
    mean_lifespans = [np.mean(lifespans[ring_size]) for ring_size in ring_size_list]
    std_lifespans = [
        np.std(lifespans[ring_size], ddof=1) for ring_size in ring_size_list
    ]
    ax.errorbar(ring_size_list, mean_lifespans, std_lifespans)
    ax.set_ylabel("Lifespan / microsecond")
    ax.set_xlabel("Ring Size")
    ax.set_ylim(0, 20)
    ax.set_xlim(0, 20)
    fig.savefig("./lifespan.pdf")
    plt.close(fig)

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
        lowest_ymins.append((lines_at_x[0][0], lowest_ymin))

    ymins_order = np.argsort([item[1] for item in lowest_ymins])
    xmapper = {old_idx: new_idx for new_idx, old_idx in enumerate(ymins_order)}

    all_xs = []
    all_ymins = []
    all_ymaxs = []
    for line_idx in range(len(lines)):
        x, y_min, y_max = lines[line_idx]
        all_xs.append(xmapper[x])
        all_ymins.append(y_min)
        all_ymaxs.append(y_max)

    x_order = np.argsort(all_xs)
    all_xs = np.asarray(all_xs)[x_order]
    all_ymins = np.asarray(all_ymins)[x_order]
    all_ymaxs = np.asarray(all_ymaxs)[x_order]
    ax.vlines(all_xs, all_ymins, all_ymaxs, colors=cmapper.to_rgba(ring_sizes))
    ax.set_ylabel("Time / microsecond")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Ring ID")
    ax.set_xlim(0, existence_matrix.shape[0])
    fig.savefig("./lifetimes.pdf")


if __name__ == "__main__":
    main()
