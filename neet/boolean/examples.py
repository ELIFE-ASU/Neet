"""
.. currentmodule:: neet.boolean.examples

Example Networks
================

The examples module provides a collection of pre-loaded model networks such as
:attr:`s_pombe` (fission yeast) and :attr:`s_cerevisiae` (budding yeast).
"""
from neet.boolean import WTNetwork, LogicNetwork
from os.path import dirname, abspath, realpath, join

DATA_PATH = join(dirname(abspath(realpath(__file__))), "data")

S_POMBE_NODES = join(DATA_PATH, "s_pombe-nodes.txt")
S_POMBE_EDGES = join(DATA_PATH, "s_pombe-edges.txt")

#: The cell cycle network for *S. pombe* (fission yeast).
s_pombe = WTNetwork.read(S_POMBE_NODES, S_POMBE_EDGES)
s_pombe.metadata.update({
    'name': 's_pombe',
    'description': 'The cell cycle network for *S. pombe* (fission yeast).',
    'citation': '',
})

S_CEREVISIAE_NODES = join(DATA_PATH, "s_cerevisiae-nodes.txt")
S_CEREVISIAE_EDGES = join(DATA_PATH, "s_cerevisiae-edges.txt")

#: The cell cycle network for *S. cerevisiae* (budding yeast).
s_cerevisiae = WTNetwork.read(S_CEREVISIAE_NODES, S_CEREVISIAE_EDGES)
s_cerevisiae.metadata.update({
    'name': 's_cerevisiae',
    'description': 'The cell cycle network for *S. cerevisiae*'
                   ' (budding yeast).',
    'citation': '',
})

C_ELEGANS_NODES = join(DATA_PATH, "c_elegans-nodes.dat")
C_ELEGANS_EDGES = join(DATA_PATH, "c_elegans-edges.dat")

#: The cell cycle network for *C. elegans*.
c_elegans = WTNetwork.read(C_ELEGANS_NODES, C_ELEGANS_EDGES)
c_elegans.metadata.update({
    'name': 'c_elegans',
    'description': 'The cell cycle network for *C. elegans*.',
    'citation': '',
})

P53_NO_DMG_NODES = join(DATA_PATH, "p53_no_dmg-nodes.txt")
P53_NO_DMG_EDGES = join(DATA_PATH, "p53_no_dmg-edges.txt")

#: The p53 GRN with no damage present.
p53_no_dmg = WTNetwork.read(P53_NO_DMG_NODES, P53_NO_DMG_EDGES)
p53_no_dmg.metadata.update({
    'name': 'p53_no_dmg',
    'description': 'The p53 GRN with no damage present.',
    'citation': 'Choi, Minsoo, Jue Shi, Sung Hoon Jung, Xi Chen, and'
                ' Kwang-Hyun Cho. Attractor Landscape Analysis Reveals'
                ' Feedback Loops in the p53 Network That Control the'
                ' Cellular Response to DNA Damage. Science Signaling 5,'
                ' no. 251 (2012): ra83. doi:10.1126/scisignal.2003363.',
})

P53_DMG_NODES = join(DATA_PATH, "p53_dmg-nodes.txt")
P53_DMG_EDGES = join(DATA_PATH, "p53_dmg-edges.txt")

#: The p53 GRN with damage present.
p53_dmg = WTNetwork.read(P53_DMG_NODES, P53_DMG_EDGES)
p53_dmg.metadata.update({
    'name': 'p53_dmg',
    'description': 'The p53 GRN with damage present.',
    'citation': 'Choi, Minsoo, Jue Shi, Sung Hoon Jung, Xi Chen, and'
                ' Kwang-Hyun Cho. Attractor Landscape Analysis Reveals'
                ' Feedback Loops in the p53 Network That Control the'
                ' Cellular Response to DNA Damage. Science Signaling 5,'
                ' no. 251 (2012): ra83. doi:10.1126/scisignal.2003363.',
})

MOUSE_CORTICAL_7B_TRUTH_TABLE = join(
    DATA_PATH, "mouse_cortical_fig_7B-truth_table.txt")
MOUSE_CORTICAL_7B_EXPRESSIONS = join(
    DATA_PATH, "mouse_cortical_fig_7B-logic_expressions.txt")

#: The gene regulatory network for *mouse cortical* (7B edges)
mouse_cortical_7B = LogicNetwork.read_table(MOUSE_CORTICAL_7B_TRUTH_TABLE)
mouse_cortical_7B.metadata.update({
    'name': 'mouse_cortical_fig_7B',
    'description': 'The gene regulatory network for *mouse cortical*'
                   ' (Taken from figure 7B in citation below).',
    'citation': 'Giacomantonio, Clare E., and Geoffrey J. Goodhill. "A'
                ' Boolean model of the gene regulatory network underlying'
                ' Mammalian cortical area development." PLoS'
                ' computational biology 6, no. 9 (2010):'
                ' e1000936. doi:10.1371/journal.pcbi.1000936',

})
mouse_cortical_7B_from_expr = LogicNetwork.read_logic(
    MOUSE_CORTICAL_7B_EXPRESSIONS)

MOUSE_CORTICAL_7C_TRUTH_TABLE = join(
    DATA_PATH, "mouse_cortical_fig_7C-truth_table.txt")
MOUSE_CORTICAL_7C_EXPRESSIONS = join(
    DATA_PATH, "mouse_cortical_fig_7C-logic_expressions.txt")

#: The gene regulatory network for *mouse cortical* (7C edges)
mouse_cortical_7C = LogicNetwork.read_table(MOUSE_CORTICAL_7C_TRUTH_TABLE)
mouse_cortical_7C.metadata.update({
    'name': 'mouse_cortical_fig_7C',
    'description': 'The gene regulatory network for *mouse cortical*'
                   ' (Taken from figure 7C in citation below).',
    'citation': 'Giacomantonio, Clare E., and Geoffrey J. Goodhill. "A'
                ' Boolean model of the gene regulatory network underlying'
                ' Mammalian cortical area development." PLoS'
                ' computational biology 6, no. 9 (2010):'
                ' e1000936. doi:10.1371/journal.pcbi.1000936',

})
mouse_cortical_7C_from_expr = LogicNetwork.read_logic(
    MOUSE_CORTICAL_7C_EXPRESSIONS)

MYELOID_TRUTH_TABLE = join(DATA_PATH, "myeloid-truth_table.txt")
MYELOID_LOGIC_EXPRESSIONS = join(DATA_PATH, "myeloid-logic_expressions.txt")

#: Differentiation control network for *myeloid* progenitors.
myeloid = LogicNetwork.read_table(MYELOID_TRUTH_TABLE)
myeloid.metadata.update({
    'name': 'myeloid',
    'description': 'Differentiation control network for *myeloid*'
                   ' progenitors.',
    'citation': '',
})
myeloid_from_expr = LogicNetwork.read_logic(MYELOID_LOGIC_EXPRESSIONS)

IL_6_SIGNALING_EXPRESSIONS = join(DATA_PATH, 'il-6-signaling-expressions.txt')
IL_6_SIGNALING_EXTERNAL = join(DATA_PATH, 'il-6-signaling-external.txt')
il_6_signaling = LogicNetwork.read_logic(IL_6_SIGNALING_EXPRESSIONS, IL_6_SIGNALING_EXTERNAL)

HGF_SIGNALING_EXPRESSIONS = join(DATA_PATH, 'hgf-signaling-expressions.txt')
HGF_SIGNALING_EXTERNAL = join(DATA_PATH, 'hgf-signaling-external.txt')
hgf_signaling_in_keratinocytes = LogicNetwork.read_logic(HGF_SIGNALING_EXPRESSIONS,
                                                         HGF_SIGNALING_EXTERNAL)
