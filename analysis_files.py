#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:12:37 2021

@author: matthew-bailey
"""

import networkx as nx
from nodeme import NodeME
from collections import Counter
import numpy as np


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
