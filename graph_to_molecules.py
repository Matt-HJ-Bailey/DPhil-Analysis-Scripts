#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:01:29 2020

@author: matthew-bailey
"""

import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from draw_and_colour import (
    CORRESPONDING_COLOURS,
    colour_graph,
)
from WormLikeCurve.Bonds import AngleBond, HarmonicBond
from WormLikeCurve.CurveCollection import CurveCollection
from WormLikeCurve.WormLikeCurve import WormLikeCurve


def calculate_edge_factor(num_sides: int, side_length: float = 2 ** (1 / 6)) -> float:
    """
    Calculate the gap from the central point of a shape to its vertex.

       x
      ____.
    x / .
     /.    <- 2^(1/6)
    Assumes the side lengths of the regular shape are the minima
    of a Lennard-Jones potential with sigma = 1
    Parameters
    ----------
    num_sides
        the number of sides of this shape
    side_length
        the length of one side of this shape

    Returns
    -------
    edge_factor
        a fraction of the total length that the gap here should be.

    """
    if num_sides == 0 or num_sides == 1:
        return 0.0
    return side_length / (2.0 * np.sin(np.pi / num_sides))


def graph_to_molecules(
    graph: nx.Graph,
    pos,
    edge_factors=calculate_edge_factor,
    num_segments: int = 5,
    periodic_box=None,
):
    """
    Convert a graph to a set of molecules.

    Parameters
    ----------
    graph
        the graph to convert to molecules
    edge_factors
        A function that calculates how much of the molecule should be set aside
        for LJ interactions
    num_segments
        The number of segments within each molecule
    periodic_box
        Is this graph periodic? If so, what's the repeat?
    Returns
    -------
    curves
        A curvecollection where each molecule occupies the same length it did in the graph. Needs rescaling.
    """
    molecules = []
    for u, v in graph.edges():
        gradient = pos[v] - pos[u]
        # Calculate the edge factors generally, assuming that the
        # nodes are a regular polyhedron with n sides each of
        # length 2^(1/6), where n is the coordination number
        coordination_u = len(graph.edges(u))
        coordination_v = len(graph.edges(v))
        # This is now the total length of one molecule *after scaling*
        # (which we do elsewhere)

        total_length = (
            num_segments + edge_factors(coordination_u) + edge_factors(coordination_v)
        )
        # Now normalise to the fraction of the molecule that the edge
        # pieces represent.
        edge_factor_u = edge_factors(coordination_u) / total_length
        edge_factor_v = edge_factors(coordination_v) / total_length
        has_changed = False
        if periodic_box is not None:
            # If we're in a periodic box, we have to apply the
            # minimum image convention. Do this by creating
            # a virtual position for v, which is a box length away.
            minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
            minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2

            # We need the += and -= to cope with cases where we're out in
            # both x and y.
            new_pos_v = pos[v]
            if gradient[0] > minimum_image_x:
                new_pos_v -= np.array([2 * minimum_image_x, 0.0])
                has_changed = True
            elif gradient[0] < -minimum_image_x:
                new_pos_v += np.array([2 * minimum_image_x, 0.0])
                has_changed = True

            if gradient[1] > minimum_image_y:
                new_pos_v -= np.array([0, 2 * minimum_image_y])
                has_changed = True
            elif gradient[1] < -minimum_image_y:
                new_pos_v += np.array([0, 2 * minimum_image_y])
                has_changed = True

        if has_changed:
            gradient = new_pos_v - pos[u]

        angle = np.arctan2(gradient[1], gradient[0])

        starting_point = pos[u] + (edge_factor_u * gradient)
        segment_length = (1 - edge_factor_u - edge_factor_v) * gradient / num_segments
        segment_length = np.hypot(*segment_length)
        harmonic_bond = HarmonicBond(k=1, length=segment_length)
        angle_bond = AngleBond(k=100, angle=np.pi)
        curve = WormLikeCurve(
            num_segments=num_segments + 1,
            harmonic_bond=harmonic_bond,
            angle_bond=angle_bond,
            start_pos=starting_point,
        )
        # Check if we know either of the end colours
        u_colour = graph.nodes(data=True)[u]["color"]
        v_colour = graph.nodes(data=True)[v]["color"]
        if len(u_colour) == 1:
            curve.atom_types[0] = u_colour[0]
            curve.atom_types[-1] = CORRESPONDING_COLOURS[u_colour[0]]
        elif len(v_colour) == 1:
            curve.atom_types[-1] = v_colour[0]
            curve.atom_types[0] = CORRESPONDING_COLOURS[v_colour[0]]
        else:
            # We don't know either!
            random_u_choice = random.choice(u_colour)
            curve.atom_types[0] = random_u_choice
            curve.atom_types[-1] = CORRESPONDING_COLOURS[random_u_choice]

        curve.start_pos = starting_point
        curve.vectors = np.array([[segment_length, angle] for i in range(num_segments)])
        curve.vectors_to_positions()
        molecules.append(curve)
    return CurveCollection(molecules)


def hexagonal_lattice_graph(
    m: int,
    n: int,
    periodic: bool = False,
    with_positions: bool = True,
    create_using: nx.Graph = None,
) -> nx.Graph:
    """
    Return an `m` by `n` hexagonal lattice graph.

    Taken from networkx source code and modified by me, to remove the offset.

    The *hexagonal lattice graph* is a graph whose nodes and edges are
    the `hexagonal tiling`_ of the plane.

    The returned graph will have `m` rows and `n` columns of hexagons.
    `Odd numbered columns`_ are shifted up relative to even numbered columns.

    Positions of nodes are computed by default or `with_positions is True`.
    Node positions creating the standard embedding in the plane
    with sidelength 1 and are stored in the node attribute 'pos'.
    `pos = nx.get_node_attributes(G, 'pos')` creates a dict ready for drawing.

    .. _hexagonal tiling: https://en.wikipedia.org/wiki/Hexagonal_tiling
    .. _Odd numbered columns: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/

    Parameters
    ----------
    : param m : The number of rows of hexagons in the lattice.

    :param n : The number of columns of hexagons in the lattice.

    : periodic : Whether to make a periodic grid by joining the boundary vertices.
        For this to work `n` must be odd and both `n > 1` and `m > 1`.
        The periodic connections create another row and column of hexagons
        so these graphs have fewer nodes as boundary nodes are identified.

    : with_positions : (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with vertical columns of hexagons
        offset to interleave and cover the plane.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    :param create_using : NetworkX graph
        If specified, this must be an instance of a NetworkX graph
        class. It will be cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
        If graph is directed, edges will point up or right.

    :return: The *m* by *n* hexagonal lattice graph.
    """
    G = create_using if create_using is not None else nx.Graph()
    G.clear()
    if m == 0 or n == 0:
        return G
    if periodic and (n % 2 == 1 or m < 2 or n < 2):
        msg = "periodic hexagonal lattice needs m > 1, n > 1 and even n"
        raise nx.NetworkXError(msg)

    M = 2 * m  # twice as many nodes as hexagons vertically
    rows = range(M + 2)
    cols = range(n + 1)
    # make lattice
    col_edges = (((i, j), (i, j + 1)) for i in cols for j in rows[: M + 1])
    row_edges = (((i, j), (i + 1, j)) for i in cols[:n] for j in rows if i % 2 == j % 2)
    G.add_edges_from(col_edges)
    G.add_edges_from(row_edges)
    # Remove corner nodes with one edge
    G.remove_node((0, M + 1))
    G.remove_node((n, (M + 1) * (n % 2)))

    # identify boundary nodes if periodic
    if periodic:
        for i in cols[:n]:
            G = nx.contracted_nodes(G, (i, 0), (i, M))
        for i in cols[1:]:
            G = nx.contracted_nodes(G, (i, 1), (i, M + 1))
        for j in rows[1:M]:
            G = nx.contracted_nodes(G, (0, j), (n, j))
        G.remove_node((n, M))

    # calc position in embedded space
    ii = (i for i in cols for j in rows)
    jj = (j for i in cols for j in rows)
    xx = (0.5 + i + i // 2 + (j % 2) * ((i % 2) - 0.5) for i in cols for j in rows)
    h = np.sqrt(3) / 2
    yy = (h * j for i in cols for j in rows)
    # exclude nodes not in G
    pos = {
        (i, j): np.array([x, y]) for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in G
    }
    nx.set_node_attributes(G, pos, "pos")
    return G


def construct_hex_lattice(num_nodes: int, bond_length: float = 1.0) -> CurveCollection:
    """
    Construct a hexagonal lattice of molecules.

    :param num_nodes: the number of hexagons per side
    :param bond_length: the length of a single segment in the molecule.
    :return: a CurveCollection of wormlike curves
    """
    hex_graph = hexagonal_lattice_graph(num_nodes, num_nodes, periodic=True)
    pos = dict(nx.get_node_attributes(hex_graph, "pos"))
    for key, val in pos.items():
        pos[key] = np.asarray(val)
    periodic_box = np.array(
        [
            [0.0, 1.5 * num_nodes],
            [0.0, num_nodes * np.sqrt(3)],
            [-0.5 * num_nodes * np.sqrt(3), 0.5 * num_nodes * np.sqrt(3)],
        ]
    )
    hex_graph = colour_graph(hex_graph)
    curves = graph_to_molecules(hex_graph, pos, periodic_box=periodic_box)
    scale_factor = bond_length / np.mean(curves[0].vectors[:, 0])
    curves.rescale(scale_factor)
    for key, val in pos.items():
        pos[key] *= scale_factor
    periodic_box *= scale_factor
    return curves, periodic_box


def construct_alt_sq_lattice(
    num_squares: int, bond_length: float = 1.0
) -> CurveCollection:
    """
    Construct a lattice of alternating 4 and 2 coordination.

    :param num_squares: the number of squares per side
    :param bond_length: the length of a single segment in the molecule.
    :return: a CurveCollection of wormlike curves
    """
    periodic_box = np.array(
        [[0.0, num_squares * 2], [0.0, num_squares * 2], [-num_squares, num_squares]],
        dtype=float,
    )
    G = nx.Graph()
    pos = dict()

    for row in range(num_squares):
        for col in range(num_squares):
            # Repeat a series of |_ triples across the grid,
            # and connect them into their own L shape
            pos[(2 * row, 2 * col)] = np.array([2 * row, 2 * col], dtype=float)
            pos[(2 * row) + 1, 2 * col] = np.array(
                [(2 * row) + 1, 2 * col], dtype=float
            )
            pos[2 * row, (2 * col) + 1] = np.array(
                [2 * row, (2 * col) + 1], dtype=float
            )
            G.add_edge((2 * row, 2 * col), ((2 * row) + 1, 2 * col))
            G.add_edge((2 * row, 2 * col), ((2 * row), (2 * col) + 1))

            # Now connect the L shapes to each other, disregarding
            # the outer edge.
            if row != num_squares - 1:
                G.add_edge(((2 * row) + 1, 2 * col), ((2 * row) + 2, (2 * col)))
            if col != num_squares - 1:
                G.add_edge((2 * row, (2 * col) + 1), (2 * row, (2 * col) + 2))

    # Finally, add the periodic edges
    for index in range(num_squares):
        G.add_edge((0, 2 * index), ((2 * num_squares) - 1, 2 * index))
        G.add_edge((2 * index, 0), (2 * index, (2 * num_squares) - 1))
    G = colour_graph(G)
    curves = graph_to_molecules(G, pos, periodic_box=periodic_box, num_segments=5)

    scale_factor = bond_length / curves[0].vectors[0, 0]
    curves.rescale(scale_factor)
    for key, val in pos.items():
        pos[key] *= scale_factor
    periodic_box *= scale_factor
    return curves, periodic_box


def rotate_one_curve(curves, curve_id: int, rotation: float, correct_com: bool = False):
    """
    Rotate a single curve around its end point.

    Parameters
    ----------
    curves
        An indexable of wormlike curves
    curve_id
        The index of the curve to rotate
    rotation
        How much to rotate it by in radians
    correct_com
        if true, keep the same centre of mass

    Returns
    -------
        curves with one rotated
    """
    if correct_com:
        old_com = curves[curve_id].centroid

    curves[curve_id].rotate(rotation, recentre=False)
    if correct_com:
        com_offset = curves[curve_id].centroid - old_com
        print("COM offset is", com_offset)
        curves[curve_id].translate(-com_offset)
    return curves


if __name__ == "__main__":
    # FIG, AX = plt.subplots()
    SCALE_FACTOR = 1.0
    TRANSFORMATION_MATRIX = None
    MORLEY_PREFIX = "./Data/hex_t30_2000_A"
    if len(sys.argv) == 2:
        # Apply a uniform scaling
        SCALE_FACTOR = float(sys.argv[1])
    elif len(sys.argv) == 3:
        SCALE_FACTOR = float(sys.argv[1])
        MORLEY_PREFIX = sys.argv[2]
    elif len(sys.argv) == 5:
        # Apply a whole transformation matrix in form
        # [[argv[1], argv[2]], [argv[3], argv[4]]]
        TRANSFORMATION_MATRIX = np.array(
            [
                [float(sys.argv[1]), float(sys.argv[2])],
                [float(sys.argv[3]), float(sys.argv[4])],
            ]
        )
    elif len(sys.argv) == 6:
        # Apply a whole transformation matrix in form
        # [[argv[1], argv[2]], [argv[3], argv[4]]]
        TRANSFORMATION_MATRIX = np.array(
            [
                [float(sys.argv[1]), float(sys.argv[2])],
                [float(sys.argv[3]), float(sys.argv[4])],
            ]
        )
        MORLEY_PREFIX = sys.argv[5]
    FIG, AX = plt.subplots()
    AX.axis("equal")

    CURVES, PERIODIC_BOX = construct_hex_lattice(6, bond_length=50)
    kwarg_list = [
        {"end_size": 0.8 * 50, "linewidths": 0, "colors": "black"}
        for _ in range(len(CURVES))
    ]
    rotate_one_curve(CURVES, 0, np.pi / 2.0, correct_com=True)
    CURVES.plot_onto(AX, kwarg_list, label_nodes=False)
    AX.set_axis_off()

    plt.savefig("./initial.pdf")
    plt.close(FIG)
    CURVES.to_lammps("./polymer_total.data", periodic_box=PERIODIC_BOX, mass=0.5 / 6)
