#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:32:46 2021

@author: matthew-bailey
"""

import networkx as nx
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from typing import Optional

from graph_to_molecules import hexagonal_lattice_graph, graph_to_molecules
from draw_and_colour import draw_periodic_coloured, colour_graph

from rings.periodic_ring_finder import PeriodicRingFinder, RingFinder


def energy_func(distances_sq: np.array):
    """
    Harmonic energy function with K = 1
    """
    return 1.0 * (np.sqrt(distances_sq) - 1.0) ** 2


def angle_func(angles: np.array):
    """
    Cosine angle function with K = 1
    """
    return 1.0 * (np.cos(angles) - np.cos(2 * np.pi / 3.0)) ** 2


def calculate_graph_energy(pos: np.array, G: nx.Graph, periodic_cell=None):
    """
    Calculate the energy of a graph with harmonic bonds

    Parameters
    ----------
    pos
        An [N, 2] or [N*2] shaped numpy array representing 2D coordinates.
        If [N*2], it is reshaped.
    G
        A networkx graph to calculate the energy of
    periodic_cell
        If this graph is periodic, what are the min x, max x it is contained in

    Returns
    -------
    energy
        The harmonic energy of this graph
    """
    if len(pos.shape) == 1:
        pos = pos.reshape([len(G), 2])
    separations = np.array([pos[v] - pos[u] for u, v in sorted(G.edges())])
    # Apply the minimum image convention to all parts
    if periodic_cell is not None:
        for axis in (0, 1):
            mic_length = np.abs(periodic_cell[axis, 1] - periodic_cell[axis, 0]) / 2.0
            mic_mask = separations[:, axis] > mic_length
            separations[:, axis][mic_mask] -= 2 * mic_length

            mic_mask = separations[:, axis] < -mic_length
            separations[:, axis][mic_mask] += 2 * mic_length

    u_v_to_sep = {}
    for i, (u, v) in enumerate(sorted(G.edges())):
        u_v_to_sep[u, v] = separations[i]
        u_v_to_sep[v, u] = -separations[i]

    # angles = []
    # for a in G.nodes():
    #     angles_around_a = []
    #     neighbors = list(G.neighbors(a))
    #     for i in range(len(neighbors)):
    #         b = neighbors[i]
    #         a_b_sep = u_v_to_sep[a, b]
    #         a_b_sep /= np.linalg.norm(a_b_sep)
    #         for j in range(i):
    #             c = neighbors[j]
    #             a_c_sep = u_v_to_sep[a, c]
    #             a_c_sep /= np.linalg.norm(a_c_sep)
    #             angle = np.abs(np.arccos(np.dot(a_b_sep, a_c_sep)))
    #             angles_around_a.append(angle)
    #     angles_around_a.sort()
    #     # Keep the smallest n angles
    #     angles_around_a = angles_around_a[:len(neighbors)]
    #     angles.extend(angles_around_a)
    distances_sq = np.sum(separations ** 2, axis=1)
    energies = energy_func(distances_sq)
    # angle_energies = angle_func(angles)
    energy = np.sum(energies)  # + np.sum(angle_energies)
    return energy


def optimise_graph_positions(G: nx.Graph, periodic_cell: Optional[np.array] = None):
    """
    Move the nodes in this graph to optimise the energy.

    Parameters
    ----------
    G
        The networkx graph to optimise, with node attribute "pos"
    periodic_cell
        If this graph is periodic, what are the min x max x etc. of its cell

    Returns
    -------
    None
        Modifies the graph G
    """
    pos_dict = nx.get_node_attributes(G, "pos")
    positions = np.array([pos_dict[u] for u in sorted(G.nodes())])
    res = scipy.optimize.minimize(
        x0=np.ravel(positions),
        fun=calculate_graph_energy,
        args=(G, periodic_cell),
        jac=False,
        options={"disp": True},
    )
    new_pos = res.x.reshape([len(G), 2])
    pos_dict = {u: new_pos[u] for u in sorted(G.nodes())}
    nx.set_node_attributes(G, pos_dict, name="pos")


def do_bond_switch(G: nx.Graph, periodic_cell: Optional[np.array] = None):
    """
    Do a single Wooten-Winer-Weaire bond switch to the graph G.
    Requires ring-finding to make sure we pick two edges in different rings.

    Parameters
    ----------
    G
        The networkx graph to switch an edge in
    periodic_cell
        If this graph is periodic, what are the min x max x etc. of its cell

    Returns
    -------
    None
        Modifies G
    """
    if periodic_cell is not None:
        rf = PeriodicRingFinder(
            G, coords_dict=nx.get_node_attributes(G, "pos"), cell=periodic_cell
        )
    else:
        rf = RingFinder(G, coords_dict=nx.get_node_attributes(G, "pos"))
    rng = np.random.default_rng()

    rings_with_uv = []
    while len(rings_with_uv) != 2:
        u, v = rng.choice(G.edges())
        rings_with_uv = [ring for ring in rf.current_rings if frozenset([u, v]) in ring]

    u_neighbors = list(G.neighbors(u))
    u_neighbors.remove(v)
    u_neighbor = u_neighbors.pop()
    ring_with_u_un = [
        ring for ring in rings_with_uv if frozenset([u, u_neighbor]) in ring
    ][0]
    for v_neighbor in G.neighbors(v):
        if v_neighbor == u:
            continue
        if frozenset([v, v_neighbor]) not in ring_with_u_un:
            break

    G.add_edge(v_neighbor, u)
    G.add_edge(u_neighbor, v)
    G.remove_edge(v, v_neighbor)
    G.remove_edge(u, u_neighbor)
    return G


def main():
    num_nodes = 6
    G = hexagonal_lattice_graph(num_nodes, num_nodes, periodic=True)
    G = nx.relabel.convert_node_labels_to_integers(G)
    periodic_cell = np.array(
        [
            [0.0, 1.5 * num_nodes],
            [0.0, num_nodes * np.sqrt(3)],
        ]
    )
    G = do_bond_switch(G, periodic_cell)
    G = colour_graph(G)
    optimise_graph_positions(G, periodic_cell)
    pos = nx.get_node_attributes(G, "pos")
    curves = graph_to_molecules(G, pos=pos, periodic_box=periodic_cell)
    scale_factor = 50.0 / np.mean(curves[0].vectors[:, 0])
    curves.rescale(scale_factor)
    for key, val in pos.items():
        pos[key] *= scale_factor
    periodic_cell *= scale_factor
    curves.to_lammps("./single-defect.data", periodic_box=periodic_cell, mass=0.5 / 6)


if __name__ == "__main__":
    main()
