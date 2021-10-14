#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:51:06 2019

@author: matthew-bailey
"""

from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.spatial.distance


def apply_minimum_image_convention(
    vector: np.array, x_mic: float, y_mic: float
) -> np.array:
    """
    Apply the minimum image convention to a separation vector, given two half box lengths.
    That is, no two points can be separated by more than half a box length.

    Parameters
    ----------
    vector
        A separation vector between two periodic points
    x_mic
        Half box length in x
    y_mic
        Half box length in y
    Returns
    -------
    mic_vector
        A vector no more than half a box length long in either dimension.
    """
    if vector[0] < -x_mic:
        vector[0] += x_mic * 2
    elif vector[0] > x_mic:
        vector[0] -= x_mic * 2

    if vector[1] < -y_mic:
        vector[1] += y_mic * 2
    elif vector[1] > y_mic:
        vector[1] -= y_mic * 2
    return vector


def find_lj_pairs(positions, ids, cutoff: float, cell=None):
    """
    Find a set of mini-clusters based around each atom. A mini
    cluster contains all the other atoms within a pairwise cutoff
    of the central atom. An atom is regarded as being in a mini-cluster
    with itself, so atoms with no nearby neighbours have a mini-cluster
    size of 1.
    Accepts periodic boundary conditions, but it is much slower
    because I can't use scipy wizardry.

    :param positions: an Nx3 numpy array of atomic positions
    :param ids: If these positions have come from a larger array, which what were their original indicies?
    :param cutoff: the distance below which they can be considered bonded
    :param cell: a [[min_x, max_x], [min_y, max_y], [min_z, max_z]] periodic cell, which is used for periodic boundary conditions. Defaults to None, in which case no periodic boundary conditions are used.
    :return lj_pairs: a dictionary of sets, each set representing all of the atoms in a mini-cluster.

    """
    distances = scipy.spatial.distance.pdist(positions)
    distances = scipy.spatial.distance.squareform(distances)
    if cell is not None:
        assert cell.shape == (3, 2) or cell.shape == (2, 2)

        x_mic = np.abs(cell[0, 1] - cell[0, 0]) / 2
        y_mic = np.abs(cell[1, 1] - cell[1, 0]) / 2
        # Precalculate some cell offsets to save time.
        min_mic = min([x_mic, y_mic])
        outside_mic = np.argwhere(distances > min_mic)
        for i, j in outside_mic:
            # Here are a few optimisations to make this routine a little faster.
            if i > j:
                # Don't double count.
                continue

            vector = positions[i, :] - positions[j, :]
            vector = apply_minimum_image_convention(vector, x_mic, y_mic)

            new_distance = np.hypot(vector[0], vector[1])

            distances[i, j] = new_distance
            distances[j, i] = new_distance
    within_cutoff = np.argwhere(distances < cutoff)

    # Construct a dictionary that holds which pairs
    # are lj bonded. Does not currently take
    # periodic boundary conditions into account.
    lj_pairs = defaultdict(set)
    for atom_1, atom_2 in within_cutoff:
        if atom_1 != atom_2:
            atom_1_id, atom_2_id = ids[atom_1], ids[atom_2]
            lj_pairs[atom_1_id].add(atom_2_id)
            lj_pairs[atom_2_id].add(atom_1_id)

    # If an atom has no neighbours, the within_cutoff
    # array doesn't have an entry for it. Make sure
    # that we explicitly give this a pair of itself.
    for atom_1 in range(positions.shape[0]):
        atom_1_id = ids[atom_1]
        lj_pairs[atom_1_id].add(atom_1_id)

    return lj_pairs


def find_lj_clusters(pair_dict, max_depth=10):
    """
    Find clusters of lennard jones pairs. Groups
    each set of Lennard-Jones pairs into a single set.

    :param pair_dict: a dictionary of sets, each set representing all of the atoms in a cluster.
    :param max_depth: the maximum number of pair steps that count as one cluster.
    :return clusters: a set of frozensets, with each entry containing a list of real molecules that make up a cluster.
    """
    clusters = set()
    for key, value in pair_dict.items():
        # This node should be in a cluster with not only
        # its neighbours, but its neighbours' neighbours.
        # and so on recursively.
        old_set = value
        for _ in range(max_depth):
            set_neighbours = set(
                item for neighbour in old_set for item in pair_dict[neighbour]
            )
            set_neighbours.update(old_set)
            if old_set != set_neighbours:
                old_set = set_neighbours
            else:
                break
        clusters.add(frozenset(set_neighbours))
    return clusters


def find_cluster_centres(
    clusters, atom_positions, offset: int = 1, cutoff: float = None
):
    """
    Finds the centroid of all clusters of Lennard-Jones atoms.
    Can take periodic boundary conditions into account.

    :param clusters: an ordered collection of frozensets, with each entry containing a list of real molecules that make up a cluster.
    :param atom_positions: a numpy array of atomic positions in order of id.
    :param offset: how the ids are offset from positions.
    :param cutoff: if there are any distances greater than this cutoff within a cluster, it indicates that a cluster spans a periodic boundary. In that case, the mean algorithm won't work so we just pick the first atom. Can be None, in which case this will naively perform a mean of the positions.
    :return cluster_positions: a dictionary of positions, with keys of atom ids and entries 2D positions.
    """
    cluster_positions = dict()
    # Make sure the clusters are ordered otherwise this will not
    # be helpful
    for i, cluster in enumerate(clusters):
        # LAMMPS indexes from 1, but positions doesn't!
        positions = [atom_positions[atom - offset] for atom in cluster]
        positions = np.vstack(positions)
        # If we've constructed this cell across multiple periodic
        # images, the mean method won't work. Filthy hack in
        # the mean time: just pick one point arbitrarily.
        if cutoff is not None:
            distances = scipy.spatial.distance.pdist(positions)
            if np.any(distances > cutoff):
                cluster_positions[i] = positions[0][0:2]
            else:
                cluster_positions[i] = np.mean(positions, axis=0)[0:2]
        else:
            cluster_positions[i] = np.mean(positions, axis=0)[0:2]
        # print(f"Placing {i} at {cluster_positions[i]}")
    return cluster_positions


def find_molecule_terminals(molecules, atom_types, type_connections):
    """
    Finds the sticky ends of a LAMMPS molecule.
    Connects each node to
    :param molecules: a dictionary of molecules, with molecule ids as keys and values a list of atoms in that molecule.
    :param atom_types: a dictionary of molecules, with molecule ids as keys and the values a list of atom types.
    :param type_connections: a dictionary of molecule ids, containing a tuple of the types of atoms we wish to connect this to.
    :return molec_terminals: a dict, with keys being the ids of one terminal and the values being the ids of the other terminals in this molecule.
    """
    molec_terminals = defaultdict(list)
    for i, molec in molecules.items():
        molec_atom_types = atom_types[i]
        for j, atom in enumerate(molec):
            try:
                to_connect = type_connections[molec_atom_types[j]]
            except KeyError:
                # This molecule isn't in the connection dict,
                # so don't connect it.
                continue
            for k, other_atom in enumerate(molec):
                if molec_atom_types[k] in to_connect:
                    molec_terminals[atom].append(other_atom)
    return molec_terminals


def connect_clusters(in_graph, clusters, out_graph=None, body_types=frozenset([4])):
    """
    Connect the clusters to one another via the molecules
    that make them up.
    :param terminals:  a dict, with keys being the ids of one terminal and the values being the ids of the other terminals in this molecule.
    :param clusters: an ordered list of frozensets, with each entry containing a list of real molecules that make up a cluster.
    :param out_graph: an empty networkx graph to add these edges to.
    :param body_type: a set of atom_types that count as body nodes
    :return graph: a filled networkx graph with these edges in.
    """
    if out_graph is None:
        out_graph = nx.Graph()

    node_connected_components = {
        node: nx.node_connected_component(in_graph, node) for node in in_graph
    }
    added_edges = set()
    atom_types = nx.get_node_attributes(in_graph, name="atom_types")
    if not atom_types:
        raise AttributeError("in_graph must have node attribute 'atom_types'")

    # Calculate the body atom id -> cluster id link in advance
    body_cluster_ids = {}
    for atom in in_graph:
        if atom_types[atom] in body_types:
            body_cluster_id = [atom in body_cluster for body_cluster in clusters].index(
                True
            )
            body_cluster_ids[atom] = body_cluster_id

    if not atom_types:
        raise RuntimeError('Must provide a graph attribute with name="atom_types".')
    # Each cluster is made up of atoms which each belong to distinct
    # connected components of the graph. Find those connected components,
    # and then find the other clusters in each one.
    for i, cluster in enumerate(clusters):
        components_in_cluster = [node_connected_components[atom] for atom in cluster]
        all_nodes = set().union(*components_in_cluster)
        # Awful dict comprehension to find which other clusters are in this
        # set of connected components, with entries only where there are
        # nodes in common and not counting ourselves. This can probably
        # be accelerated to not do all operations twice.
        clusters_in_common = {
            j: all_nodes.intersection(other_cluster)
            for j, other_cluster in enumerate(clusters)
            if len(all_nodes.intersection(other_cluster)) != 0 and i != j
        }
        # Fast path for isolated molecules
        if len(clusters_in_common) == 1:
            for key in clusters_in_common.keys():
                added_edges.add((i, key))
        else:
            # We have to be clever here. Find the shortest path between
            # each set of nodes, and if it goes through a body node,
            # add the edge to that instead. This is an O(N^3) horror,
            # however, depending on the size of the clusters.
            for atom in cluster:
                for other_cluster_id in clusters_in_common.keys():
                    for other_atom in clusters[other_cluster_id]:
                        # Skip the self-interactions
                        if other_cluster_id == i:
                            continue
                        if other_atom == atom:
                            continue
                        # Thankfully, we can rule out atoms that aren't
                        # in the same connected component as we can't ever
                        # find a path to them!
                        if (
                            node_connected_components[atom]
                            != node_connected_components[other_atom]
                        ):
                            continue

                        # Find the shortest path, and the types along that path
                        shortest_path_between = nx.shortest_path(
                            in_graph, atom, other_atom
                        )

                        bodies_atom_path = [
                            k
                            for k in shortest_path_between
                            if atom_types[k] in body_types
                        ]
                        # hang on, what the fuck?
                        if not bodies_atom_path:
                            added_edges.add((i, other_cluster_id))
                            continue
                        bodies_cluster_path = [
                            body_cluster_ids[k] for k in bodies_atom_path
                        ]
                        # Now zip up the pairs along this path to connect the bodies
                        body_path_pairs = [
                            (bodies_cluster_path[k], bodies_cluster_path[k + 1])
                            for k in range(len(bodies_cluster_path) - 1)
                        ]

                        # and finally, add the terminal-body links, making
                        # sure not to add any self-loops
                        if i != bodies_cluster_path[0]:
                            body_path_pairs.append((i, bodies_cluster_path[0]))
                        if other_cluster_id != bodies_cluster_path[-1]:
                            body_path_pairs.append(
                                (other_cluster_id, bodies_cluster_path[-1])
                            )
                        added_edges.update(body_path_pairs)

    if len(added_edges) == 0:
        raise RuntimeError(
            "Did not connect any clusters together. Double check type_connections"
            + " in find_molecule_terminals, or your cutoff radius."
        )
    out_graph.add_edges_from(added_edges)
    cluster_data = {i: cluster for i, cluster in enumerate(clusters)}
    nx.set_node_attributes(out_graph, cluster_data, name="cluster")
    return out_graph


def cluster_molecule_bodies(molecs, molec_types, types_to_cluster):
    """
    Clusters all the sites of the same type within a single molecule, so they act
    as a unified 'body' cluster.
    :param molecs: a dictionary of molecules, with molecule ids as keys and values a list of atoms in that molecule.
    :param molec_types: a dictionary of molecules, with molecule ids as keys and the values a list of atom types.
    :param types_to_cluster: an iterable of molecule types, which get goruped together.
    :return molec_clusters: a set of the clusters of same types within each molecule.
    """
    lj_clusters = defaultdict(set)
    for molec_id, molec in molecs.items():
        atoms_of_type = [
            atom_type in types_to_cluster for atom_type in molec_types[molec_id]
        ]
        pairs = [
            (molec[i], molec[j])
            for j in range(len(atoms_of_type))
            if atoms_of_type[j]
            for i in range(len(atoms_of_type))
            if atoms_of_type[i]
        ]
        for atom_1, atom_2 in pairs:
            lj_clusters[atom_1].add(atom_2)
            lj_clusters[atom_2].add(atom_1)
    return lj_clusters
