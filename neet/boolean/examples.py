# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
The examples module provides a collection of pre-loaded model networks such
as :py:attr:`s_pombe` (fission yeast) and :py:attr:`s_cerevisiae` (budding yeast).
"""
from neet.boolean import WTNetwork, LogicNetwork
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

# Get the path of the nodes and edges files for the budding yeast cell cycle
S_CEREVISIAE_NODES = join(DATA_PATH, "s_cerevisiae-nodes.txt")
S_CEREVISIAE_EDGES = join(DATA_PATH, "s_cerevisiae-edges.txt")
"""
The cell cycle network for *S. cerevisiae* (budding yeast).
"""
s_cerevisiae = WTNetwork.read(S_CEREVISIAE_NODES, S_CEREVISIAE_EDGES)


# Get the path of the nodes and edges files for C. elegans
C_ELEGANS_NODES = join(DATA_PATH, "c_elegans-nodes.dat")
C_ELEGANS_EDGES = join(DATA_PATH, "c_elegans-edges.dat")

"""
    The cell cycle network for *C. elegans*.
"""
c_elegans = WTNetwork.read(C_ELEGANS_NODES, C_ELEGANS_EDGES)

# Get the path of the nodes and edges files for the p53 GRN w/no damage
P53_NO_DMG_NODES = join(DATA_PATH, "p53_no_dmg-nodes.txt")
P53_NO_DMG_EDGES = join(DATA_PATH, "p53_no_dmg-edges.txt")
"""
The p53 GRN with no damage present.
"""

p53_no_dmg = WTNetwork.read(P53_NO_DMG_NODES, P53_NO_DMG_EDGES)

# Get the path of the nodes and edges files for the p53 GRN w/damage
P53_DMG_NODES = join(DATA_PATH, "p53_dmg-nodes.txt")
P53_DMG_EDGES = join(DATA_PATH, "p53_dmg-edges.txt")
"""
The p53 GRN with damage present.
"""
p53_dmg = WTNetwork.read(P53_DMG_NODES, P53_DMG_EDGES)

# Get the path of the nodes and edges files for the mouse cortical gene regulatory network
# Edges are from either figure 7B or 7C (Giacomantonio, 2010)

MOUSE_CORTICAL_ANT_INIT_NODES = join(
    DATA_PATH, "mouse_cortical_anterior_init-nodes.txt")
MOUSE_CORTICAL_ANT_FINAL_NODES = join(
    DATA_PATH, "mouse_cortical_anterior_final-nodes.txt")
MOUSE_CORTICAL_POST_INIT_NODES = join(
    DATA_PATH, "mouse_cortical_posterior_init-nodes.txt")
MOUSE_CORTICAL_POST_FINAL_NODES = join(
    DATA_PATH, "mouse_cortical_posterior_final-nodes.txt")
MOUSE_CORTICAL_7B_EDGES = join(DATA_PATH, "mouse_cortical_fig_7B-edges.txt")
MOUSE_CORTICAL_7C_EDGES = join(DATA_PATH, "mouse_cortical_fig_7C-edges.txt")

# Anterior mouse cortical networks ---------------------------------------------------------------
"""
The gene regulatory network for *mouse cortical* (anterior, initial, 7B edges).
"""
mouse_cortical_ant_init_7B = WTNetwork.read(
    MOUSE_CORTICAL_ANT_INIT_NODES, MOUSE_CORTICAL_7B_EDGES)

"""
The gene regulatory network for *mouse cortical* (anterior, final, 7B edges).
"""
mouse_cortical_ant_final_7B = WTNetwork.read(
    MOUSE_CORTICAL_ANT_FINAL_NODES, MOUSE_CORTICAL_7B_EDGES)

"""
The gene regulatory network for *mouse cortical* (anterior, initial, 7C edges).
"""
mouse_cortical_ant_init_7C = WTNetwork.read(
    MOUSE_CORTICAL_ANT_INIT_NODES, MOUSE_CORTICAL_7C_EDGES)

"""
The gene regulatory network for *mouse cortical* (anterior, final, 7C edges).
"""
mouse_cortical_ant_final_7C = WTNetwork.read(
    MOUSE_CORTICAL_ANT_FINAL_NODES, MOUSE_CORTICAL_7C_EDGES)

# Posterior mouse coritical networks ---------------------------------------------------------------
"""
The gene regulatory network for *mouse cortical* (posterior, initial, 7B edges).
"""
mouse_cortical_post_init_7B = WTNetwork.read(
    MOUSE_CORTICAL_POST_INIT_NODES, MOUSE_CORTICAL_7B_EDGES)

"""
The gene regulatory network for *mouse cortical* (posterior, final, 7B edges).
"""
mouse_cortical_post_final_7B = WTNetwork.read(
    MOUSE_CORTICAL_POST_FINAL_NODES, MOUSE_CORTICAL_7B_EDGES)

"""
The gene regulatory network for *mouse cortical* (posterior, initial, 7C edges).
"""
mouse_cortical_post_init_7C = WTNetwork.read(
    MOUSE_CORTICAL_POST_INIT_NODES, MOUSE_CORTICAL_7C_EDGES)

"""
The gene regulatory network for *mouse cortical* (posterior, final, 7C edges).
"""
mouse_cortical_post_final_7C = WTNetwork.read(
    MOUSE_CORTICAL_POST_FINAL_NODES, MOUSE_CORTICAL_7C_EDGES)

"""
Differentiation control network for *myeloid* progenitors.
"""
MYELOID_TRUTH_TABLE = join(DATA_PATH, "myeloid-truth_table.txt")
MYELOID_LOGIC_EXPRESSIONS = join(DATA_PATH, "myeloid-logic_expressions.txt")

myeloid = LogicNetwork.read_table(MYELOID_TRUTH_TABLE)
myeloid_from_expr = LogicNetwork.read_logic(MYELOID_LOGIC_EXPRESSIONS)
