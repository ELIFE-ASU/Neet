"""
.. currentmodule:: neet.boolean.examples

.. testsetup:: examples

    from neet.boolean.examples import *
"""
from neet.boolean import WTNetwork, LogicNetwork
from os.path import dirname, abspath, realpath, join

DATA_PATH = join(dirname(abspath(realpath(__file__))), "data")

S_POMBE_NODES = join(DATA_PATH, "s_pombe-nodes.txt")
S_POMBE_EDGES = join(DATA_PATH, "s_pombe-edges.txt")

#: A gene regulatory network model of the *S. pombe* (fission yeast) cell
#: cycle, as described in [Davidich2008]_.
s_pombe = WTNetwork.read(S_POMBE_NODES, S_POMBE_EDGES, metadata={
    'name': 'Fission Yeast Cell Cycle',
    'description': 'Cell cycle network for *S. pombe* (fission yeast)',
    'citation': 'M. I. Davidich and S. Bornholdt, "Boolean network model '
                'predicts cell cycle sequence of fission yeast," PLoS One, '
                'vol. 3, no. 2, p. e1672, Feb. 2008.'
                'doi:10.1371/journal.pone.0001672',
})

S_CEREVISIAE_NODES = join(DATA_PATH, "s_cerevisiae-nodes.txt")
S_CEREVISIAE_EDGES = join(DATA_PATH, "s_cerevisiae-edges.txt")

#: A gene regulatory network model of the *S. cerevisiae* (budding yeast) cell
#: cycle, as described in [Li2004]_.
s_cerevisiae = WTNetwork.read(S_CEREVISIAE_NODES, S_CEREVISIAE_EDGES, metadata={
    'name': 'Budding Yeast Cell Cycle',
    'description': 'Cell cycle network for *S. cerevisiae* (budding yeast)',
    'citation': 'F. Li, T. Long, Y. Lu, Q. Ouyang, and C. Tang, "The yeast '
                'cell-cycle network is robustly designed," Proc. Natl. Acad. '
                'Sci. U. S. A., vol. 101, no. 14, pp. 4781-4786, Apr. 2004. '
                'doi:10.1073/pnas.0305937101',
})

C_ELEGANS_NODES = join(DATA_PATH, "c_elegans-nodes.dat")
C_ELEGANS_EDGES = join(DATA_PATH, "c_elegans-edges.dat")

#: A gene regulatory network model of the *S. elegans* cell cycle, as described
#: in [Huang2013]_.
c_elegans = WTNetwork.read(C_ELEGANS_NODES, C_ELEGANS_EDGES, metadata={
    'name': 'C. elegans Cell Cycle',
    'description': 'Cell cycle network for *C. elegans*.',
    'citation': 'X. Huang, L. Chen, H. Chim, L. L. H. Chan, Z. Zhao, and '
                'H. Yan, "Boolean genetic network model for the control of '
                'C. elegans early embryonic cell cycles," Biomed. Eng. Online, '
                'vol. 12 Suppl 1, p. S1, Dec. 2013. '
                'doi:10.1186/1475-925X-12-S1-S1',
})

P53_NO_DMG_NODES = join(DATA_PATH, "p53_no_dmg-nodes.txt")
P53_NO_DMG_EDGES = join(DATA_PATH, "p53_no_dmg-edges.txt")

#: A simplified gene regulatory network model of p53 signaling *without*
#: damage, as described in [Choi2012]_.
p53_no_dmg = WTNetwork.read(P53_NO_DMG_NODES, P53_NO_DMG_EDGES, metadata={
    'name': 'p53 Signaling Without Damage',
    'description': 'A simplified model of p53 signaling without damage',
    'citation': 'M. Choi, J. Shi, S. H. Jung, X. Chen, and K. H. Cho, '
                '"Attractor landscape analysis reveals feedback loops in the '
                'p53 network that control the cellular response to DNA '
                'damage," Sci. Signal., vol. 5, no. 251, p. ra83, Nov. 2012.'
                'doi:10.1126/scisignal.2003363'
})

P53_DMG_NODES = join(DATA_PATH, "p53_dmg-nodes.txt")
P53_DMG_EDGES = join(DATA_PATH, "p53_dmg-edges.txt")

#: A simplified gene regulatory network model of p53 signaling *with*
#: damage, as described in [Choi2012]_.
p53_dmg = WTNetwork.read(P53_DMG_NODES, P53_DMG_EDGES, metadata={
    'name': 'p53 Signaling With Damage',
    'description': 'A simplified model of p53 signaling with damage',
    'citation': 'M. Choi, J. Shi, S. H. Jung, X. Chen, and K. H. Cho, '
                '"Attractor landscape analysis reveals feedback loops in the '
                'p53 network that control the cellular response to DNA '
                'damage," Sci. Signal., vol. 5, no. 251, p. ra83, Nov. 2012.'
                'doi:10.1126/scisignal.2003363'
})

MOUSE_CORTICAL_7B_TRUTH_TABLE = join(
    DATA_PATH, "mouse_cortical_fig_7B-truth_table.txt")
MOUSE_CORTICAL_7B_EXPRESSIONS = join(
    DATA_PATH, "mouse_cortical_fig_7B-logic_expressions.txt")

#: A gene regulatory network model for cortical area development in mice, as
#: described in fig. 7B of [Giacomantonio2010]_.
mouse_cortical_7B = LogicNetwork.read_table(MOUSE_CORTICAL_7B_TRUTH_TABLE, metadata={
    'name': 'Mouse Cortical Area Development (fig. 7B)',
    'description': 'The gene regulatory network for cortical area development '
                   'in mice. Network taken from fig. 7B.',
    'citation': 'C. E. Giacomantonio and G. J. Goodhill, "A Boolean model of '
                'the gene regulatory network underlying Mammalian cortical '
                'area development," PLoS Comput. Biol., vol. 6, no. 9, Sep. '
                '2010. doi:10.1371/journal.pcbi.1000936'
})
mouse_cortical_7B_from_expr = LogicNetwork.read_logic(MOUSE_CORTICAL_7B_EXPRESSIONS, metadata={
    'name': 'Mouse Cortical Area Development (fig. 7B)',
    'description': 'The gene regulatory network for cortical area development '
                   'in mice. Network taken from fig. 7B.',
    'citation': 'C. E. Giacomantonio and G. J. Goodhill, "A Boolean model of '
                'the gene regulatory network underlying Mammalian cortical '
                'area development," PLoS Comput. Biol., vol. 6, no. 9, Sep. '
                '2010. doi:10.1371/journal.pcbi.1000936'
})

MOUSE_CORTICAL_7C_TRUTH_TABLE = join(
    DATA_PATH, "mouse_cortical_fig_7C-truth_table.txt")
MOUSE_CORTICAL_7C_EXPRESSIONS = join(
    DATA_PATH, "mouse_cortical_fig_7C-logic_expressions.txt")

#: A gene regulatory network model for cortical area development in mice, as
#: described in fig. 7C of [Giacomantonio2010]_.
mouse_cortical_7C = LogicNetwork.read_table(MOUSE_CORTICAL_7C_TRUTH_TABLE, metadata={
    'name': 'Mouse Cortical Area Development (fig. 7C)',
    'description': 'The gene regulatory network for cortical area development '
                   'in mice. Network taken from fig. 7C.',
    'citation': 'C. E. Giacomantonio and G. J. Goodhill, "A Boolean model of '
                'the gene regulatory network underlying Mammalian cortical '
                'area development," PLoS Comput. Biol., vol. 6, no. 9, Sep. '
                '2010. doi:10.1371/journal.pcbi.1000936'
})
mouse_cortical_7C_from_expr = LogicNetwork.read_logic(MOUSE_CORTICAL_7C_EXPRESSIONS, metadata={
    'name': 'Mouse Cortical Area Development (fig. 7C)',
    'description': 'The gene regulatory network for cortical area development '
                   'in mice. Network taken from fig. 7C.',
    'citation': 'C. E. Giacomantonio and G. J. Goodhill, "A Boolean model of '
                'the gene regulatory network underlying Mammalian cortical '
                'area development," PLoS Comput. Biol., vol. 6, no. 9, Sep. '
                '2010. doi:10.1371/journal.pcbi.1000936'
})

MYELOID_TRUTH_TABLE = join(DATA_PATH, "myeloid-truth_table.txt")
MYELOID_LOGIC_EXPRESSIONS = join(DATA_PATH, "myeloid-logic_expressions.txt")

#: A gene regulatory network for the differentiation of myeloid progenitors, as
#: described in [Krumsiek2011]_.
myeloid = LogicNetwork.read_table(MYELOID_TRUTH_TABLE, metadata={
    'name': 'Differentiation of Myeloid Progenitors',
    'description': 'Differentiation control network for myeloid progenitors.',
    'citation': 'J. Krumsiek, C. Marr, T. Schroeder, and F. J. Theis, '
                '"Hierarchical differentiation of myeloid progenitors is '
                'encoded in the transcription factor network," PLoS One, '
                'vol. 6, no. 8, p. e22649, Aug. 2011.'
                'doi:10.1371/journal.pone.0022649',
})
myeloid_from_expr = LogicNetwork.read_logic(MYELOID_LOGIC_EXPRESSIONS)

IL_6_SIGNALING_EXPRESSIONS = join(DATA_PATH, 'il-6-signaling-expressions.txt')
IL_6_SIGNALING_EXTERNAL = join(DATA_PATH, 'il-6-signaling-external.txt')

#: A gene regulatory model of interleukin 6 signaling, as described in
#: [Ryll2011]_.
il_6_signaling = LogicNetwork.read_logic(IL_6_SIGNALING_EXPRESSIONS, IL_6_SIGNALING_EXTERNAL, metadata={
    'name': 'IL-6 Signaling',
    'description': 'A model of interleukin 6 signaling',
    'citation': 'A. Ryll, R. Samaga, F. Schaper, L. G. Alexopoulos, and '
                'S. Klamt, "Large-scale network models of IL-1 and IL-6 '
                'signalling and their hepatocellular specification," '
                'Mol. Biosyst., vol. 7, no. 12, pp. 3253-3270, Dec. 2011.'
                'doi:10.1039/c1mb05261f',
})

HGF_SIGNALING_EXPRESSIONS = join(DATA_PATH, 'hgf-signaling-expressions.txt')
HGF_SIGNALING_EXTERNAL = join(DATA_PATH, 'hgf-signaling-external.txt')

#: A gene regulatory network model of hepatocyte growth-factor induced
#: migration of primary human keratinocytes, as described in [Singh2012]_.
hgf_signaling_in_keratinocytes = LogicNetwork.read_logic(HGF_SIGNALING_EXPRESSIONS,
                                                         HGF_SIGNALING_EXTERNAL, metadata={
                                                             'name': 'HGF Signaling in Keratinocytes',
                                                             'description': 'A model of hepatocyte growth-factor induced migration of '
                                                                            'primary human keratinocytes',
                                                             'citation': 'A. Singh, J. M. Nascimento, S. Kowar, H. Busch, and '
                                                                         'M. Boerries, "Boolean approach to signalling pathway '
                                                                         'modelling in HGF-induced keratinocyte migration," '
                                                                         'Bioinformatics, vol. 28, no. 18, pp. i495-i501, Sep. 2012.'
                                                                         'doi:10.1093/bioinformatics/bts410',
                                                         })
