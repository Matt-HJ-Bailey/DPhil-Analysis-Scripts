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

from typing import Optional, Union, Dict, Any, Tuple

from graph_to_molecules import hexagonal_lattice_graph, graph_to_molecules
from draw_and_colour import draw_periodic_coloured, colour_graph

from rings.periodic_ring_finder import PeriodicRingFinder, RingFinder


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
    G.remove_edge(69, 70)
    G = colour_graph(G)
    fig, ax = plt.subplots()
    draw_periodic_coloured(
        G, nx.get_node_attributes(G, "pos"), periodic_cell, with_labels=True, ax=ax
    )
    fig.savefig("./graph-unoptimized.pdf")
    plt.close(fig)
    optimise_graph_positions(G, periodic_cell)
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots()
    ax.axis("equal")
    draw_periodic_coloured(G, pos, periodic_cell, with_labels=True, ax=ax)
    fig.savefig("./graph-optimized.pdf")
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
