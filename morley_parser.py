#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:34:20 2020

@author: newc4592
"""

from collections import defaultdict
from typing import Dict, Tuple

import networkx as nx
import numpy as np

try:
    from rings import PeriodicRingFinder

    RING_FINDER_ENABLED = True
except ImportError as RING_FINDER_EX:
    RING_FINDER_ENABLED = False

from draw_and_colour import colour_graph

import sys


def load_morley(
    prefix: str, reset_origin: bool = False
) -> Tuple[Dict[int, np.array], nx.Graph, np.array]:
    """
    Load a graph in the Morley output format.
    
    Parameters
    ----------
    prefix
        A string used to load {prefix}_crds.dat, {prefix}_net.dat and {prefix}_aux.dat
    reset_origin
        Should we move the bottom left coordinate to be (0, 0)
    
    Returns
    -------
    positions, graph, box
        A node: pos dictionary of positions, a loaded graph and the periodic box.
    """
    coords_file = prefix + "_crds.dat"
    network_file = prefix + "_net.dat"
    aux_file = prefix + "_aux.dat"

    graph = nx.Graph()
    pos_dict = dict()
    with open(coords_file) as fi:
        for i, line in enumerate(fi.readlines()):
            coords = [float(item) for item in line.split()]
            pos_dict[i] = np.array(coords)

    with open(network_file) as fi:
        for u, line in enumerate(fi.readlines()):
            connections = [int(item) for item in line.split()]
            for v in connections:
                graph.add_edge(u, v)

    dual_file = prefix + "_dual.dat"
    dual_connections = dict()
    with open(dual_file) as fi:
        for node_id, line in enumerate(fi.readlines()):
            dual_connections[node_id] = [int(item) for item in line.split()]

    nx.set_node_attributes(graph, dual_connections, "dual_connections")

    with open(aux_file) as fi:
        num_atoms = int(fi.readline())
        _, _ = [int(item) for item in fi.readline().split()]
        geometry_code = fi.readline().strip()
        box_max_x, box_max_y = [float(item) for item in fi.readline().split()]
        inv_box_max_x, inv_box_max_y = [float(item) for item in fi.readline().split()]

        if not np.isclose(box_max_x, 1.0 / inv_box_max_x, 1e-4):
            raise RuntimeError(
                f"Inverse periodic box side does not match periodic box size in x. Got {box_max_x}, {1.0 / inv_box_max_x}."
            )

        if not np.isclose(box_max_y, 1.0 / inv_box_max_y, 1e-4):
            raise RuntimeError(
                f"Inverse periodic box side does not match periodic box size in y. Got {box_max_y}, {1.0 / inv_box_max_y}."
            )
        periodic_box = np.array([[0.0, box_max_x], [0.0, box_max_y]], dtype=float)

    if reset_origin:
        min_x = min(val[0] for val in pos_dict.values())
        min_y = min(val[1] for val in pos_dict.values())
        for key in pos_dict.keys():
            pos_dict[key] += np.array([min_x, min_y])
    graph = colour_graph(graph)
    return pos_dict, graph, periodic_box


def construct_morley_dual(
    graph: nx.Graph, pos: Dict[int, np.array], periodic_box: np.array
):
    """
    Construct the dual graph of this Morley graph.
    The dual graph connects centres of all polygons.
    
    Doesn't work if the PeriodicRingFinder couldn't be imported
    """

    # Bail out if we can't find the RingFinder.
    if not RING_FINDER_ENABLED:
        raise RING_FINDER_EX

    ring_finder = PeriodicRingFinder(graph, pos, periodic_box[:, 1])

    num_nodes = len(graph)
    dual_connections = defaultdict(list)
    current_rings = list(ring_finder.current_rings)
    for ring_id, ring in enumerate(current_rings):
        for node in ring.to_node_list():
            real_node = node % num_nodes
            dual_connections[real_node].append(ring_id)

    # Now we must order the dual connections clockwise from +y
    sorted_dual_connections = dict()
    for node, node_duals in dual_connections.items():
        node_pos = pos[node]
        duals_pos = [current_rings[ring_id].centroid() for ring_id in node_duals]
        duals_vectors = [pos - node_pos for pos in duals_pos]
        # Apply the minimum image convention
        for vec in duals_vectors:
            minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
            minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
            if vec[0] > minimum_image_x:
                vec -= np.array([2 * minimum_image_x, 0.0])
            elif vec[0] < -minimum_image_x:
                vec += np.array([2 * minimum_image_x, 0.0])

            if vec[1] > minimum_image_y:
                vec -= np.array([0, 2 * minimum_image_y])
            elif vec[1] < -minimum_image_y:
                vec += np.array([0, 2 * minimum_image_y])
        duals_vectors = [vec / np.linalg.norm(vec) for vec in duals_vectors]
        angles_with_y = [np.sign(vec[0]) * np.arccos(vec[1]) for vec in duals_vectors]
        angles_with_y = [
            2 * np.pi + angle if angle < 0 else angle for angle in angles_with_y
        ]
        sorted_indices = np.argsort(angles_with_y)
        sorted_node_duals = [node_duals[i] for i in sorted_indices]
        sorted_dual_connections[node] = sorted_node_duals

    dual_connections = sorted_dual_connections
    morley_connections = nx.get_node_attributes(graph, "dual_connections")
    return dual_connections


def write_out_morley(
    graph: nx.Graph, pos: Dict[int, np.array], periodic_box: np.array, prefix: str
):
    """
    Write out into a netmc readable file.
    
    Parameters
    ----------
    graph
        The graph of the network to write out
    pos
        A node: position dictionary
    periodic_box
        Lengths of the periodic box in x and y dimensions.
    prefix
        The prefix of the files to write to
    """
    coordinate_file = prefix + "_crds.dat"
    aux_file = prefix + "_aux.dat"
    network_file = prefix + "_net.dat"
    dual_file = prefix + "_dual.dat"

    with open(coordinate_file, "w") as fi:
        for i in range(len(graph)):
            fi.write(f"{pos[i][0]}\t{pos[i][1]}\n")

    all_dual_connections = nx.get_node_attributes(graph, "dual_connections")
    with open(aux_file, "w") as fi:
        fi.write("f{len(graph)}\n")  # Number of nodes
        max_net_connections = max(graph.degree())[1]
        dual_connections = [len(val) for key, val in all_dual_connections.items()]
        max_dual_connections = max(dual_connections)
        fi.write(f"{max_net_connections}\t{max_dual_connections}\n")
        fi.write("2DE\n")  # Geometry code
        x_size = np.abs(periodic_box[0, 1] - periodic_box[0, 0])
        y_size = np.abs(periodic_box[1, 1] - periodic_box[1, 0])
        fi.write(f"{x_size} \t {y_size} \n")  # Periodic box
        fi.write(f"{1.0/x_size} \t {1.0 / y_size} \n")  # Reciprocal box

    with open(network_file, "w") as fi:
        for i in range(len(graph)):
            neighbours = graph.neighbors(i)
            fi.write("\t".join([str(item) for item in neighbours]) + "\n")

    with open(dual_file, "w") as fi:
        for i in range(len(graph)):
            dual_connections = all_dual_connections[i]
            fi.write("\t".join([str(item) for item in dual_connections]) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        MORLEY_PREFIX = sys.argv[1]
    else:
        MORLEY_PREFIX = "./Data/hexagon_network_A"
    HEX_POS, HEX_GRAPH, HEX_BOX = load_morley(MORLEY_PREFIX)
    print("HEX_POS has", len(HEX_POS), "items at the start.")
    DUAL_CNXS = construct_morley_dual(HEX_GRAPH, pos=HEX_POS, periodic_box=HEX_BOX)
    nx.set_node_attributes(HEX_GRAPH, DUAL_CNXS, name="dual_connections")
    write_out_morley(HEX_GRAPH, HEX_POS, HEX_BOX, prefix="./Data/hexagon_rewritten_A")
