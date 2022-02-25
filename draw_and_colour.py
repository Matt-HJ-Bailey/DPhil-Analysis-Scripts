#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:59:05 2021

@author: matthew-bailey
"""

from collections import defaultdict
from typing import Dict, Optional
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import copy
import matplotlib.pyplot as plt

UNKNOWN_COLOUR = (2,)
COLOUR_TO_TYPE = defaultdict(lambda: UNKNOWN_COLOUR)
COLOUR_TO_TYPE[0] = (2,)
COLOUR_TO_TYPE[1] = (3,)
COLOUR_TO_TYPE[2] = (4,)
COLOUR_TO_TYPE[3] = (5,)
CORRESPONDING_COLOURS = {2: 3, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
COLOUR_LUT = {
    None: "blue",
    0: "white",
    1: "purple",
    2: "blue",
    3: "green",
    4: "orange",
    5: "red",
    6: "purple",
    7: "pink",
    8: "brown",
    9: "cyan",
    10: "magenta",
    11: "yellow",
}


def colour_graph(graph: nx.Graph, colour_to_type: Dict = COLOUR_TO_TYPE) -> nx.Graph:
    """
    Assign a type to each node of a graph.

    Proceeds recursively, assigning a type to each node on a graph.
    Then, assigns the corresponding type according to the type dictionary.
    In the case of odd rings, this can't be done, so we instead assign
    a set of types to that node.

    :param graph: the graph to colour
    :param corresponding_types: a dictionary with a set of types in it,
    each of which corresponds to one other.
    """
    colours = nx.algorithms.coloring.greedy_color(
        graph, strategy="smallest_last", interchange=True
    )
    for key, value in colours.items():
        colours[key] = colour_to_type.get(value, UNKNOWN_COLOUR)
    nx.set_node_attributes(graph, colours, "color")
    return graph


def draw_periodic_coloured(
    graph: nx.Graph,
    periodic_box: np.array,
    pos: Optional[Dict[int, np.array]] = None,
    ax=None,
    with_labels=False,
    **kwargs,
):
    """
    Draw an aperiodic graph with the nodes coloured correctly.

    Parameters
    -----------
    graph
        the graph we wish to draw with node attributes of 'color'
    pos
        dictionary keyed by node id with values being positions
    periodic_box
        the periodic box to wrap around
    ax
        the axis to draw on. Can be none for a fresh axis.
    Returns
    -------
        an axis with the drawn graph on.
    """
    if ax is None:
        _, ax = plt.subplots()
    if pos is None:
        pos = nx.get_node_attributes(graph, "pos")

    edge_list = []
    periodic_edge_list = []
    for u, v in graph.edges():
        distance = np.abs(pos[v] - pos[u])
        if (
            distance[0] < periodic_box[0, 1] / 2
            and distance[1] < periodic_box[1, 1] / 2
        ):
            edge_list.append((u, v))
        else:
            periodic_edge_list.append((u, v))
    nodes_in_edge_list = set([item for edge_pair in edge_list for item in edge_pair])
    nodes_in_edge_list = list(nodes_in_edge_list)

    periodic_nodes = set(
        [item for edge_pair in periodic_edge_list for item in edge_pair]
    )
    periodic_nodes = list(periodic_nodes)

    try:
        node_colours = {
            node_id: (COLOUR_LUT[colour[0]] if colour is not None else "blue")
            for node_id, colour in graph.nodes(data="color")
        }
    except TypeError:
        node_colours = {
            node_id: (COLOUR_LUT[colour] if colour is not None else "blue")
            for node_id, colour in graph.nodes(data="color")
        }
    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        node_size=10,
        node_color=[node_colours[node_id] for node_id in nodes_in_edge_list],
        edgelist=edge_list,
        nodelist=nodes_in_edge_list,
        # font_size=8,
        edgecolors="black",
        linewidths=0.5,
        **kwargs,
    )

    node_colours_list = []
    new_edge_list = []
    new_node_list = []
    new_pos = {key: value for key, value in pos.items()}
    temporary_edges = []
    temporary_nodes = []
    # We often encounter an edge periodic in more than one
    # way. Keep track, and give each one a virtual position.
    encounters = defaultdict(lambda: 0)
    for u, v in periodic_edge_list:
        gradient = new_pos[v] - new_pos[u]

        # If we're in a periodic box, we have to apply the
        # minimum image convention. Do this by creating
        # a virtual position for v, which is a box length away.
        minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
        minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
        # print(pos[v], pos[u])
        # print(gradient, minimum_image_x, minimum_image_y)
        # We need the += and -= to cope with cases where we're out in
        # both x and y.
        new_pos_v = np.array([item for item in new_pos[v]])
        if gradient[0] > minimum_image_x:
            new_pos_v -= np.array([2 * minimum_image_x, 0.0])
        elif gradient[0] < -minimum_image_x:
            new_pos_v += np.array([2 * minimum_image_x, 0.0])

        if gradient[1] > minimum_image_y:
            new_pos_v -= np.array([0, 2 * minimum_image_y])
        elif gradient[1] < -minimum_image_y:
            new_pos_v += np.array([0, 2 * minimum_image_y])

        encounters[v] += 1
        new_v = f"{v}_p{encounters[v]}"
        new_pos[new_v] = new_pos_v
        node_colours[new_v] = node_colours[v]
        node_colours_list.extend([node_colours[node_id] for node_id in (u, new_v)])
        new_edge_list.append((u, new_v))
        new_node_list.extend([u, new_v])
        temporary_edges.append((u, new_v))
        temporary_nodes.append(new_v)

    graph.add_edges_from(temporary_edges)

    nx.draw(
        graph,
        pos=new_pos,
        ax=ax,
        node_size=10,
        node_color=node_colours_list,
        edgelist=new_edge_list,
        nodelist=new_node_list,
        style="dashed",
        edgecolors="black",
        linewidths=0.5,
    )

    if with_labels:
        nx.draw_networkx_labels(graph, pos=new_pos, ax=ax)

    graph.remove_edges_from(temporary_edges)
    graph.remove_nodes_from(temporary_nodes)
    return ax

def map_periodic_ring(ring, periodic_box):
    coords = ring.coords
    minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
    minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
    
    centre = coords[np.argmin(coords[:, 0] - (minimum_image_x + periodic_box[0, 0]))]
    xs, ys = coords[:, 0], coords[:, 1]
    xs[(xs - centre[0]) < -minimum_image_x] += 2 * minimum_image_x
    ys[(ys - centre[1]) < -minimum_image_y] += 2 * minimum_image_y
    
    xs[(xs - centre[0]) > minimum_image_x] -= 2 * minimum_image_x
    ys[(ys - centre[1]) > minimum_image_y] -= 2 * minimum_image_y
    
    return np.array([xs, ys]).T
    
def to_tikz(filename: str, graph, rings=None, cmap: str="coolwarm", periodic_box=None,vmin=None, vmax=None, scale=10, pos=None, color_lut=None, color_data=None):

    if pos is None:
        pos = nx.get_node_attributes(graph, "pos")

    # If we're in a periodic box, we have to apply the
    # minimum image convention. Do this by creating
    # a virtual position for v, which is a box length away.
    minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
    minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
        
    edge_list = []
    periodic_edge_list = []
    for u, v in graph.edges():
        distance = pos[v] - pos[u]
        if distance[0] < -minimum_image_x:
            periodic_edge_list.append((u, v))
        elif distance[0] > minimum_image_x:
            periodic_edge_list.append((u, v))
        elif distance[1] < -minimum_image_y:
            periodic_edge_list.append((u, v))
        elif distance[1] > minimum_image_y:
            periodic_edge_list.append((u, v))
        else:
            edge_list.append((u, v))
    nodes_in_edge_list = set([item for edge_pair in edge_list for item in edge_pair])
    nodes_in_edge_list = list(nodes_in_edge_list)

    periodic_nodes = set(
        [item for edge_pair in periodic_edge_list for item in edge_pair]
    )
    periodic_nodes = list(periodic_nodes)
    
    node_colours = {node_id: colour[0] for node_id, colour in graph.nodes(data="color")}

    node_colours_list = []
    new_edge_list = []
    new_node_list = []
    new_pos = {key: value for key, value in pos.items()}
    temporary_edges = []
    temporary_nodes = []
    # We often encounter an edge periodic in more than one
    # way. Keep track, and give each one a virtual position.
    encounters = defaultdict(lambda: 0)
    
    for u, v in periodic_edge_list:
        gradient = new_pos[v] - new_pos[u]

        
        # print(pos[v], pos[u])
        # print(gradient, minimum_image_x, minimum_image_y)
        # We need the += and -= to cope with cases where we're out in
        # both x and y.
        new_pos_v = np.array([item for item in new_pos[v]])
        if gradient[0] > minimum_image_x:
            new_pos_v -= np.array([2 * minimum_image_x, 0.0])
        elif gradient[0] < -minimum_image_x:
            new_pos_v += np.array([2 * minimum_image_x, 0.0])

        if gradient[1] > minimum_image_y:
            new_pos_v -= np.array([0, 2 * minimum_image_y])
        elif gradient[1] < -minimum_image_y:
            new_pos_v += np.array([0, 2 * minimum_image_y])

        encounters[v] += 1
        new_v = f"{v}_p{encounters[v]}"
        new_pos[new_v] = new_pos_v
        node_colours[new_v] = node_colours[v]
        new_edge_list.append((u, new_v))
        new_node_list.extend([u, new_v])
        #temporary_edges.append((u, new_v))
        #temporary_nodes.append(new_v)
        
    
    for key, val in new_pos.items():
        new_pos[key] = np.array([scale*val[0] / (periodic_box[0, 1] - periodic_box[0, 0]), scale*val[1] / (periodic_box[1, 1] - periodic_box[1, 0])])
        
    with open(filename, "w") as fi:
        fi.write(r"""\begin{tikzpicture}[
    netmcnode1/.style={circle, draw=black, thin, inner sep=0pt,minimum size=4pt, fill=brewer1},
    netmcnode2/.style={circle, draw=black, thin, inner sep=0pt,minimum size=4pt, fill=brewer2},
    netmcnode3/.style={circle, draw=black, thin, inner sep=0pt,minimum size=4pt, fill=brewer3},
    netmcnode4/.style={circle, draw=black, thin, inner sep=0pt,minimum size=4pt, fill=brewer4},]""" + "\n")
        
        if rings is not None:
            if color_data is None:
                color_data = [len(ring) for ring in rings]
                
            if vmin is None:
                vmin = min(color_data)
            if vmax is None:
                vmax = max(color_data)
            if color_lut is None:
                color_lut = int(vmax - vmin)
                
            color_data = [int( color_lut * (item - vmin) / (vmax - vmin)) for item in color_data]
            
            colors = cm.get_cmap(cmap)(np.linspace(0, 1, color_lut))
            for idx, color in enumerate(colors):
                fi.write("\definecolor{" + f"{cmap}{vmax-vmin}v{idx}" "}{RGB}{" + f"{int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}" + "}\n")
                
            for j, ring in enumerate(rings):
                color_idx = color_data[j]               
                fi.write(r"\fill [" + f"{cmap}{vmax-vmin}v{color_idx}] ")
                coords = map_periodic_ring(ring, periodic_box)
                coords[:, 0] *= scale / (periodic_box[0, 1] - periodic_box[0, 0])
                coords[:, 1] *= scale / (periodic_box[1, 1] - periodic_box[1, 0])
                for coord in coords:
                    fi.write(f"({coord[0]:.2f}, {coord[1]:.2f}) -- ")
                fi.write("cycle;\n")
        
        for u, v in edge_list:
            if (u, v) in periodic_edge_list or (v, u) in periodic_edge_list:
                continue
            fi.write(r"\draw[thick, black] " +f"({new_pos[u][0]}, {new_pos[u][1]}) -- ({new_pos[v][0]}, {new_pos[v][1]});\n")
        
        for u, v in new_edge_list:
            fi.write(r"\draw[thick, black, dotted] " +f"({new_pos[u][0]}, {new_pos[u][1]}) -- ({new_pos[v][0]}, {new_pos[v][1]});\n")
        
        for node in graph.nodes:
            fi.write(r"\node [ " + f"netmcnode{node_colours[node]}" + "] " + f"(node{node}) at ({new_pos[node][0]}, {new_pos[node][1]})" + "{};\n")
        
        for node in new_node_list:
            fi.write(r"\node [ " + f"netmcnode{node_colours[node]}" + "] " + f"(node{node}) at ({new_pos[node][0]}, {new_pos[node][1]})" + "{};\n")
        fi.write(r"\end{tikzpicture}")

def draw_nonperiodic_coloured(graph: nx.Graph, pos: Dict[int, np.array], ax=None):
    """
    Draw an aperiodic graph with the nodes coloured correctly.

    Parameters
    -----------
    graph
        the graph we wish to draw with node attributes of 'color'
    pos
        dictionary keyed by node id with values being positions
    ax
        the axis to draw on. Can be none for a fresh axis.
    Returns
    -------
        an axis with the drawn graph on.
    """
    if ax is None:
        _, ax = plt.subplots()
    edge_list = []
    for u, v in graph.edges():
        edge_list.append((u, v))
    nodes_in_edge_list = set([item for edge_pair in edge_list for item in edge_pair])
    nodes_in_edge_list = list(nodes_in_edge_list)

    node_colours = {
        node_id: (COLOUR_LUT[colour[0]] if colour is not None else "blue")
        for node_id, colour in graph.nodes(data="color")
    }
    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        node_size=10,
        node_color=[node_colours[node_id] for node_id in nodes_in_edge_list],
        edgelist=edge_list,
        nodelist=nodes_in_edge_list,
        font_size=8,
        edgecolors="black",
        linewidths=0.5,
    )

    return ax
