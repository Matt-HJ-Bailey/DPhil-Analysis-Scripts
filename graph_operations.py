#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:32:46 2021

@author: matthew-bailey
"""

import sys
import random
import copy

import networkx as nx
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from typing import Optional, Union, Dict, Any, Tuple, Set

from graph_to_molecules import hexagonal_lattice_graph, graph_to_molecules
from draw_and_colour import draw_periodic_coloured, colour_graph

from rings.periodic_ring_finder import PeriodicRingFinder, RingFinder


def systematic_perturbation(arr: np.array, epsilon: float):
    for i in range(len(arr.ravel())):
        new_arr = copy.deepcopy(arr.ravel())
        new_arr[i] += epsilon
        yield new_arr.reshape(arr.shape)


def edges_at_topo_distance(G: nx.Graph, u, v, distance: int) -> Set[frozenset[Any]]:
    """
    Find the edges in a graph at a given topological distance.

    The topological distance between two edges is measured as the number
    of nodes visited on the shortest path between (u, v) and (a, b),
    starting at the closest of (u, v) and ending at the closest of (a, b)
    For example, two neighbouring edges have a topological distance of 1.
    Two edges that are separated by 1 intermediate have a topological distance of 2.

    Parameters
    ----------
    G
        Networkx graph to find edges in
    u
        Label of one edge, (u, v) must be in G
    v
        Label of the other edge, (u, v) must be in G
    distance
        Topological depth to find edges at

    Return
    ------
    edges
        A set of frozensets, each representing an order-independent edge.
    """
    assert distance > 0, f"Distance must be positive, not {distance}"
    assert G.has_edge(u, v), f"{u}, {v} not in the graph."
    paths_u = {
        key: val
        for key, val in nx.single_source_shortest_path(G, u, cutoff=distance).items()
        if len(val) == distance + 1
    }
    paths_v = {
        key: val
        for key, val in nx.single_source_shortest_path(G, v, cutoff=distance).items()
        if len(val) == distance + 1
    }
    of_interest = dict()
    of_interest.update(paths_u)
    of_interest.update(paths_v)
    if u in of_interest:
        del of_interest[u]
    if v in of_interest:
        del of_interest[v]

    return set(frozenset(val[-2:]) for val in of_interest.values())


def generate_u_v_to_sep(
    G: nx.Graph,
    pos: Union[np.array, Dict[Any, np.array]],
    periodic_cell: Optional[np.array],
) -> Dict[Tuple[Any, Any], np.array]:
    """
    Create a dictionary mapping (u, v) pairs to separations

    Also applies the minimum image convention in the process.

    Parameters
    ----------
    G
        Graph with nodes that have positions
    pos
        Positions of the nodes in G
    periodic_cell
        If G is periodic, the cell its contained in

    Returns
    -------
    u_v_to_sep
        a dictionary mapping (u, v) edges in the graph to separations
    """

    if isinstance(pos, np.ndarray) and len(pos.shape) == 1:
        pos = pos.reshape([len(G), 2])

    assert len(pos) == len(
        G
    ), f"G and pos must be the same length, got {len(pos)} and {len(G)}"

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

    return u_v_to_sep


def calculate_angle_energy(
    pos: np.array, G: nx.Graph, periodic_cell: Optional[np.array]
) -> Tuple[float, np.array]:
    """
    Calculate the angular energy and the force according to the Dreiding-like

    U = K (cos(theta) - cos(theta_eqm))^2

    Parameters
    ----------
    G
        Graph to analyse
    pos
        Positions of nodes in the graph
    periodic_cell
        If this graph is periodic, what are the min x max x etc. of its cell

    Returns
    -------
    energy
        The angular energy in this graph
    forces
        The first derivative of the energy with respect to x1, y1, x2, etc
    """

    if len(pos.shape) == 1:
        pos = pos.reshape([len(G), 2])

    u_v_to_sep = generate_u_v_to_sep(G, pos, periodic_cell)

    K_ang = 1.0
    angle_energy = 0.0
    angle_forces = np.zeros_like(pos)
    for a in G.nodes():
        neighbors = list(G.neighbors(a))
        cos_eqm = np.cos(2.0 * np.pi / len(neighbors))
        for i in range(len(neighbors)):
            b = neighbors[i]
            a_b_sep = u_v_to_sep[a, b]
            a_b_sq = np.sum(a_b_sep ** 2)
            a_b_len = np.sqrt(a_b_sq)
            for j in range(i):
                c = neighbors[j]
                a_c_sep = u_v_to_sep[a, c]
                a_c_sq = np.sum(a_c_sep ** 2)
                a_c_len = np.sqrt(a_c_sq)
                cos_abc = np.dot(a_b_sep, a_c_sep)
                cos_abc /= a_b_len * a_c_len

                dcostheta = cos_abc - cos_eqm
                a_fac = 2.0 * K_ang * dcostheta
                a_bb = a_fac * cos_abc / a_b_sq
                a_bc = -a_fac / (a_b_len * a_c_len)
                a_cc = a_fac * cos_abc / a_c_sq
                force_b = np.array(
                    [
                        (a_bb * a_b_sep[i]) + (a_bc * a_c_sep[i])
                        for i in range(len(a_b_sep))
                    ]
                )
                force_c = np.array(
                    [
                        (a_cc * a_c_sep[i]) + (a_bc * a_b_sep[i])
                        for i in range(len(a_b_sep))
                    ]
                )

                angle_forces[b] -= force_b
                angle_forces[a] += force_b + force_c
                angle_forces[c] -= force_c
                this_angle_energy = K_ang * (dcostheta) ** 2
                angle_energy += this_angle_energy

    angle_forces[np.isclose(angle_forces, 0.0)] = 0.0
    return angle_energy, angle_forces


def calculate_bond_energy(
    pos: np.array, G: nx.Graph, periodic_cell: Optional[np.array]
):
    """
    Calculate the angular energy and the force according to the Harmonic

    U = k(r-r_eqm)^2

    Parameters
    ----------
    G
        Graph to analyse
    pos
        Positions of nodes in the graph
    periodic_cell
        If this graph is periodic, what are the min x max x etc. of its cell

    Returns
    -------
    energy
        The angular energy in this graph
    forces
        The first derivative of the energy with respect to x1, y1, x2, etc
    """
    if len(pos.shape) == 1:
        pos = pos.reshape([len(G), 2])

    u_v_to_sep = generate_u_v_to_sep(G, pos, periodic_cell)

    bond_forces = np.zeros_like(pos)
    bond_energy = 0.0
    bond_k = 1.0
    for u, v in G.edges():
        separation = u_v_to_sep[(u, v)]
        distance_sq = np.sum(separation ** 2)
        distance = np.sqrt(distance_sq)
        sep_normed = separation / distance

        force = bond_k * (distance - 1.0) * sep_normed
        # Watch out for the definition of +/- here
        bond_forces[u] += force
        bond_forces[v] -= force
        this_bond_energy = 0.5 * bond_k * (distance - 1.0) ** 2
        bond_energy += this_bond_energy
    return bond_energy, bond_forces


def calculate_graph_energy(
    pos: np.array, G: nx.Graph, periodic_cell=None, do_angles=True
):
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
    do_angles
        Should we calculate the angles?
    Returns
    -------
    energy
        The total energy in this graph
    forces
        The first derivative of the energy

    """
    if len(pos.shape) == 1:
        pos = pos.reshape([len(G), 2])
    energy, forces = 0.0, np.zeros_like(pos)
    bond_energy, bond_forces = calculate_bond_energy(pos, G, periodic_cell)
    energy += bond_energy
    forces += bond_forces
    if do_angles:
        angle_energy, angle_forces = calculate_angle_energy(pos, G, periodic_cell)
        energy += angle_energy
        forces += bond_forces
    # Make sure you correctly use +forces or -forces
    return energy, -forces.ravel()


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
    assert all(
        degree > 1 for node, degree in G.degree()
    ), "All nodes must be 2 coordinate or greater"
    G = nx.relabel.convert_node_labels_to_integers(G)
    pos_dict = nx.get_node_attributes(G, "pos")
    positions = np.array([pos_dict[u] for u in sorted(G.nodes())])
    # Optimise bonds first
    res = scipy.optimize.minimize(
        x0=np.ravel(positions),
        fun=calculate_graph_energy,
        args=(G, periodic_cell, False),
        jac=True,
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
    rng = np.random.default_rng(seed=1)

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


def steepest_descent(
    G: nx.Graph,
    periodic_cell: Optional[np.array],
    timestep: float = 0.01,
    iterations: int = 1000,
):
    """
    Minimise the positions of this graph by doing crude steepest descent.

    Applies the velocity verlet algorithm resetting when the change in
    velocity is small or every N steps.

    Parameters
    ----------
    G
        The graph to minimise with attribute "pos" containing positions
    periodic_cell
        Optional, the cell this graph is containined in if it's periodic
    timestep
        The timestep of each velocity verlet step
    iterations
        How many iterations to do
    """
    pos = nx.get_node_attributes(G, "pos")
    pos = np.array([pos[u] for u in sorted(G.nodes())])

    velocities = np.zeros_like(pos)
    forces = np.zeros_like(pos)
    last_velocities = np.zeros_like(pos)
    for it in range(iterations):

        pos += (velocities * timestep) + (0.5 * forces * timestep ** 2)
        energy, next_forces = calculate_graph_energy(pos, G, periodic_cell)
        # Remember the forces are negative
        next_forces = -next_forces.reshape([len(G), -1])
        velocities += 0.5 * (forces + next_forces) * timestep
        forces = next_forces

        delta_v = velocities - last_velocities
        delta_v_sum = np.sum(np.abs(delta_v))
        if delta_v_sum < 1e-5 * timestep or it % 50 == 0:
            print("Resetting v at step", it, delta_v_sum)
            velocities[:] = 0.0
            last_velocities[:] = 0.0
    pos_dict = {u: pos[u] for u in sorted(G.nodes())}
    nx.set_node_attributes(G, pos_dict, name="pos")


def remove_n_edges(G: nx.Graph, n: int):
    """
    Remove n edges from G.

    This generates a procrystal-like graph with all nodes being
    at least two coordinate.
    If n is larger than the maximum number of edges to remove (m),
    only remove m edges.

    Parameters
    ----------
    G
        The graph to remove edges from
    n
        The number of edges to remove

    Returns
    -------
    G
        G with up to n edges removed.
    """
    edges_removed = 0
    edges = list(G.edges())
    random.shuffle(edges)
    while edges_removed < n:
        u_neighbours = -1
        v_neighbours = -1
        while (u_neighbours < 3 or v_neighbours < 3) and len(edges) >= 1:
            u, v = edges.pop()
            u_neighbours, v_neighbours = len(G[u]), len(G[v])
        if not edges:
            break
        G.remove_edge(u, v)
        edges_removed += 1

    return G, edges_removed


def remove_single_coordinate(graph: nx.Graph) -> nx.Graph:
    """
    Remove all 1-coordinate nodes from the graph.

    Parameters
    ----------
    graph
        the graph to remove k=1 nodes from
    """
    k_1_nodes = [node for node, degree in graph.degree() if degree == 1]
    while k_1_nodes:
        graph.remove_nodes_from(k_1_nodes)
        k_1_nodes = [node for node, degree in graph.degree() if degree == 1]
    return graph


def remove_nodes_around(graph: nx.Graph, centre, radius: int) -> nx.Graph:
    """
    Remove nodes in a given radius around a centre node.

    A node is removed if the shortest path from it to the centre is
    radius edges or fewer.

    Parameters
    ----------
    graph
        The graph to modify
    centre
        A node in the graph to remove
    radius
        Remove nodes this many edges or fewer away from the centre

    Returns
    -------
    graph
        A graph with a hole in the middle.
    """
    nodes = nx.single_source_shortest_path(graph, centre, radius)
    graph.remove_nodes_from(nodes)
    return graph



def open_on_one_side(graph, periodic_box):
    """
    Open the network up on one side
    """
    
    pos_dict = nx.get_node_attributes(graph, "pos")
    x_cutoff = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
    to_remove = []
    for u, v in graph.edges():
        pos_u, pos_v = pos_dict[u], pos_dict[v]
        diff = np.abs(pos_v - pos_u)
        if diff[0] > x_cutoff:
            to_remove.append((u, v))
            
    graph.remove_edges_from(to_remove)
    return graph, periodic_box
    
def main():
    for item in sys.argv[1:]:
        if "help" in item.lower() or "h" in item.lower():
            print("Usage: python ./graph_operations.py SIZE TO_REMOVE SEED")
            print("SIZE -- the number of rings on one edge, must be even")
            print("TO_REMOVE -- the number of edges to remove")
            print("SEED -- random seed")
            break

    RANDOM_SEED = int(np.pi * 1e16)
    if len(sys.argv) >= 4:
        RANDOM_SEED = int(sys.argv[3])
        random.seed(RANDOM_SEED)

    # Set a default number of nodes and allow it to be overridden
    num_nodes = 6
    if len(sys.argv) >= 2:
        num_nodes = int(sys.argv[1])

    G = hexagonal_lattice_graph(num_nodes, num_nodes, periodic=True)
    G = nx.relabel.convert_node_labels_to_integers(G)
    periodic_cell = np.array(
        [
            [0.0, 1.5 * num_nodes],
            [0.0, num_nodes * np.sqrt(3)],
        ]
    )

    if len(sys.argv) >= 3:
        num_edges_to_remove = int(sys.argv[2])
        remove_n_edges(G, num_edges_to_remove)

    G = colour_graph(G)
    #G = remove_nodes_around(G, 30, 2)
    G = remove_single_coordinate(G)
    print(periodic_cell)
    open_on_one_side(G, periodic_cell)
    print(periodic_cell)
    optimise_graph_positions(G, periodic_cell)
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots()
    ax.axis("equal")
    draw_periodic_coloured(G, pos, periodic_cell, with_labels=False, ax=ax)
    fig.savefig("./graph-optimized.pdf", bbox_inches="tight")
    plt.close(fig)

    curves = graph_to_molecules(G, pos=pos, periodic_box=periodic_cell)
    scale_factor = 50.0 / np.mean(curves[0].vectors[:, 0])
    curves.rescale(scale_factor)
    for key, val in pos.items():
        pos[key] *= scale_factor
    periodic_cell *= scale_factor
    curves.to_lammps("./removed-edge.data", periodic_box=periodic_cell, mass=0.5 / 6)


if __name__ == "__main__":
    main()
