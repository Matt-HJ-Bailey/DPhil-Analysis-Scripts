#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:51:06 2019

@author: matthew-bailey

A set of tests for turning ImageJ output into networks
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from plot_ridges import Line

class TestLine:
    """
    Test the line subclass
    """

    def test_identify_ends(self):
        line = Line(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert line.junction_points == [0, 1]

        line = Line(np.array([[0.0, 1.0], [1.0, 1.0]]))
        assert line.junction_points == [0, 1]
        
    def test_within_line(self):
        """
        Test that we correctly identify that some points have
        perpendicular projections within the line
        """
        line = Line(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert line.projection_within_line(np.array([0.5, 1.0])) == True
        assert line.projection_within_line(np.array([0.5, -2.0])) == True

        assert line.projection_within_line(np.array([0.0, 1.0])) == True
        assert line.projection_within_line(np.array([1.0, -2.0])) == True

        assert line.projection_within_line(np.array([-0.5, 1.0])) == False
        assert line.projection_within_line(np.array([-0.5, -2.0])) == False

        assert line.projection_within_line(np.array([1.5, 1.0])) == False
        assert line.projection_within_line(np.array([1.5, -2.0])) == False

    def test_distance_to_line_within(self):
        """
        Test that we correctly calculate the distance to a 
        line for a number of points in the perpendicular regime.
        """
        line = Line(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert line.distance_from(np.array([0.5, 1.0])) == 1.0
        assert line.distance_from(np.array([0.5, -2.0])) == 2.0

        assert line.distance_from(np.array([0.0, 1.0])) == 1.0
        assert line.distance_from(np.array([1.0, -2.0])) == 2.0


class TestIntersection:
    """
    Test if two lines intersect
    """
    def test_obvious_intersect(self):
        """
        Test that we correctly calculate that two perpendicular lines intersect
        """
        line_a = Line(np.array([[0.0, 0.0], [1.0, 0.0]]))
        line_b = Line(np.array([[0.5, -1.0], [0.5, 1.0]]))

        assert line_a.test_intersect(line_b) == True
        assert line_b.test_intersect(line_a) == True

