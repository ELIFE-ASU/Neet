# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
The examples module provides a collection of pre-loaded model networks such
as :py:attr:`s_pombe` (fission yeast) and :py:attr:`s_cerevisiae` (budding yeast).
"""
from neet.boolean import WTNetwork
from os.path import dirname, abspath, realpath, join

# Determine the path to the "data" directory of the neet.boolean module
DATA_PATH = join(dirname(abspath(realpath(__file__))), "data")

# Get the path of the nodes and edges files for the fission yeast cell cycle
S_POMBE_NODES = join(DATA_PATH, "s_pombe-nodes.txt")
S_POMBE_EDGES = join(DATA_PATH, "s_pombe-edges.txt")

"""
The cell cycle network for *S. pombe* (fission yeast).
"""
s_pombe = WTNetwork.read(S_POMBE_NODES, S_POMBE_EDGES)

# Get the path of the nodse and edges files for the budding yeast cell cycle
S_CEREVISIAE_NODES = join(DATA_PATH, "s_cerevisiae-nodes.txt")
S_CEREVISIAE_EDGES = join(DATA_PATH, "s_cerevisiae-edges.txt")
"""
The cell cycle network for *S. cerevisiae* (budding yeast).
"""
s_cerevisiae = WTNetwork.read(S_CEREVISIAE_NODES, S_CEREVISIAE_EDGES)
