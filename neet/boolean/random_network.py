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



def random_preserving_degree(net):
    #arr = net.weights ==> add this line after chaning
    arr = copy.copy(net.weights)
    print np.shape(arr)
    if np.shape(arr)[0] != np.shape(arr)[1]:
        sys.exit() # assert ==> change syntex if needed and print error message
    length_arr = np.shape(arr)[0]
    for i in range(length_arr):
        for j in range(length_arr):
            if arr[i, j] == 0:
                continue
            if i == j:   #preserve self-loops
                continue
            #edge_swipping = False
            for k in range(length_arr):
                if arr[i, j] == 0: #edge_swipping: #arr[i, j] == 0: # this breaks k-loop after first edge-swipping is operated on arr[i, j]
                    break
                for l in range(length_arr):
                    if arr[i, j] == 0: #edge_swipping: # this breaks l-loop after first edge-swipping is operated on arr[i, j]
                        break
                    if arr[i, j] != arr[k, l]: # preserving weight == > edge-swip is allowed only between two edges sharing the same weight
                        continue
                    if k == i or l == j:
                        continue
                    #print "check swipe ", i, j, "with", k, l
                    if arr[i, l] != 0 or arr[k, j] != 0:
                        continue
                    if ran.random() < 0.5:
                        #print "swipe", i, j, "with", k, l
                        #print "results", i, l, "and ",  k, j
                        arr[i, l] = arr[i, j]
                        arr[k, j] = arr[k, l]
                        arr[i, j] = 0
                        arr[k, l] = 0
                        #print arr
                        #edge_swipping = True
    return WTNetwork(arr, copy.copy(net.thresholds), theta=net.theta)


def random_preserving_size(arr):
    #arr = net.weights ==> add this line after chaning
    print np.shape(arr)
    if np.shape(arr)[0] != np.shape(arr)[1]:
        sys.exit() # assert ==> change syntex if needed and print error message
    length_arr = np.shape(arr)[0]


# #a = np.array( [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]] )
# a = np.random.randint(low = -1, high = 2, size = (10,10))
# #print a
# b = random_preserving_degree(a)
# #print b
#
# G = nx.from_numpy_matrix(a, create_using = nx.DiGraph())
# ranG = nx.from_numpy_matrix(b, create_using = nx.DiGraph())
#
#
# print G.out_degree(weight = 'weight').values()
# print ranG.out_degree(weight = 'weight').values()
# ##
# print G.in_degree(weight = 'weight').values()
# print ranG.in_degree(weight = 'weight').values()
