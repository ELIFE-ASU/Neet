# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

import sys
import os
import random as ran
import networkx as nx
import numpy as np
import copy
from .wtnetwork import WTNetwork



def rewiring_fixed_degree(net):
    '''
    Generate a random network by rewiring a given network
    with fixed (weighted) degree sequence, threshold and self-loops.

    :param net: a network
    :returns: a random network
    '''
    arr = copy.copy(net.weights)
    length_arr = np.shape(arr)[0]
    for i in range(length_arr):
        for j in range(length_arr):
            if arr[i, j] == 0:
                continue
            if i == j:   # to preserve self-loops
                continue
            edges_swiped = False
            for k in range(length_arr):
                if edges_swiped:
                    break
                for l in range(length_arr):
                    if edges_swiped:
                        break
                    if k == l: # to perserve self-loops
                        continue
                    if arr[i, j] != arr[k, l]: # edge-swipe is allowed only between two edges sharing the same weight
                        continue
                    if k == i or l == j: # this edge-swipe doesn't make difference
                        continue
                    if arr[i, l] != 0 or arr[k, j] != 0: # this edge-swipe will result in double edges.
                        continue
                    if ran.random() < 0.5:
                        arr[i, l] = arr[i, j]
                        arr[k, j] = arr[k, l]
                        arr[i, j] = 0
                        arr[k, l] = 0
                        edges_swiped = True

    return WTNetwork(arr, copy.copy(net.thresholds), theta=net.theta)


def rewiring_fixed_size(net):
    '''
    Generate a random network by rewiring a given network
    with fixed size (the number of nodes and edges for each weight), threshold and self-loops.
    
    :param net: a network
    :returns: a random network
    '''
    arr = copy.copy(net.weights)
    length_arr = np.shape(arr)[0]
    for i in range(length_arr):
        for j in range(length_arr):
            if arr[i, j] == 0:
                continue
            if i == j:   # to preserve self-loops
                continue
            edges_swiped = False
            for k in range(length_arr):
                if edges_swiped:
                    break
                for l in range(length_arr):
                    if edges_swiped:
                        break
                    if k == l: # to perserve self-loops
                        continue
                    if ran.random() < 0.5:
                        temp = arr[i, j]
                        arr[i, j] = arr[k, l]
                        arr[k, l] = temp
                        edges_swiped = True

    return WTNetwork(arr, copy.copy(net.thresholds), theta=net.theta)
