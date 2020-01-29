"""Unit test for LogicNetwork"""
import unittest
import numpy as np
from neet.python import long
from neet.boolean import BooleanNetwork, LogicNetwork
from neet.exceptions import FormatError
from neet import Network
from os.path import dirname, abspath, realpath, join


class TestLogicNetwork(unittest.TestCase):
    def setUp(self):
        self.TABLE_PATH = join(dirname(abspath(realpath(__file__))), 'data', 'table')
        self.LOGIC_PATH = join(dirname(abspath(realpath(__file__))), 'data', 'logic')

    def test_is_network(self):
        self.assertTrue(isinstance(LogicNetwork([([0], {'0'})]), Network))
        self.assertTrue(isinstance(LogicNetwork([([0], {'0'})]), BooleanNetwork))

    def test_init(self):
        net = LogicNetwork([((0,), {'0'})])
        self.assertEqual(1, net.size)
        self.assertEqual([(1, {0})], net._encoded_table)

        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})],
                           names=['A', 'B'])
        self.assertEqual(2, net.size)
        self.assertEqual(['A', 'B'], net.names)
        self.assertEqual([(2, {0, 2}), (1, {1})], net._encoded_table)

    def test_init_fail(self):
        with self.assertRaises(TypeError):
            LogicNetwork("not a list or tuple")

        with self.assertRaises(TypeError):
            LogicNetwork([((0,), {'0'})], names=5)

        with self.assertRaises(ValueError):
            LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})], names=['A'])

        with self.assertRaises(ValueError):
            LogicNetwork([((0,), {'0'})], names=["A", "B"])

        with self.assertRaises(TypeError):
            LogicNetwork([((0,),)])

        with self.assertRaises(IndexError):
            LogicNetwork([((1,), {'0'})])

        with self.assertRaises(TypeError):
            LogicNetwork([((0,), '0')])

    def test_init_long(self):
        table = [((), set()) for _ in range(65)]
        table[0] = ((np.int64(64),), set('1'))

        mask = long(2)**64

        net = LogicNetwork(table)
        self.assertEqual(net.table, table)
        self.assertEqual(net._encoded_table[0], (mask, set([mask])))

    def test_inplace_update(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        state = [0, 1]
        self.assertEqual(net.update(state), [1, 0])
        self.assertEqual([1, 0], state)

    def test_update(self):
        net = LogicNetwork([((0,), {'0'})])
        self.assertEqual(net.update([0], 0), [1])
        self.assertEqual(net.update([1], 0), [0])
        self.assertEqual(net.update([0]), [1])
        self.assertEqual(net.update([1]), [0])

        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        self.assertEqual(net.update([0, 0], 0), [1, 0])
        self.assertEqual(net.update([0, 0], 1), [0, 0])
        self.assertEqual(net.update([0, 0]), [1, 0])
        self.assertEqual(net.update([0, 1], 0), [1, 1])
        self.assertEqual(net.update([0, 1], 1), [0, 0])
        self.assertEqual(net.update([0, 1]), [1, 0])
        self.assertEqual(net.update([1, 0], 0), [1, 0])
        self.assertEqual(net.update([1, 0], 1), [1, 1])
        self.assertEqual(net.update([1, 0]), [1, 1])
        self.assertEqual(net.update([1, 1]), [1, 1])

        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), {(0, 1), '10', (1, 1)}),
                            ((0, 1), {'11'})])
        self.assertEqual(net.update([0, 1, 0]), [1, 0, 0])
        self.assertEqual(net.update([1, 1, 1], 1), [1, 1, 1])
        self.assertEqual(net.update([0, 0, 1]), [1, 1, 0])
        self.assertEqual(net.update([0, 0, 1], pin=[1]), [1, 0, 0])
        self.assertEqual(net.update([0, 0, 1], pin=[0, 1]), [0, 0, 0])

        self.assertEqual(net.update([0, 0, 1], values={0: 0}), [0, 1, 0])
        self.assertEqual(net.update(
            [0, 0, 1], pin=[1], values={0: 0}), [0, 0, 0])

    def test_update_exceptions(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), {(0, 1), '10', (1, 1)}),
                            ((0, 1), {'11'})])
        with self.assertRaises(ValueError):
            net.update([0, 0])

        with self.assertRaises(ValueError):
            net.update([0, 0, 1], values={0: 2})

        with self.assertRaises(ValueError):
            net.update([0, 0, 1], pin=[0], values={0: 1})

    def test_has_metadata(self):
        net = LogicNetwork([((0,), {'0'})])
        self.assertTrue(hasattr(net, 'metadata'))
        self.assertEqual(type(net.metadata), dict)

    def test_logic_simple_read(self):
        """
        Read a network from a truth table file
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "simple_table.txt")
        simple = LogicNetwork.read_table(TRUTH_TABLE)

        self.assertEqual(simple.names, ['A', 'B', 'C', 'D'])
        self.assertEqual(simple.table, [((1, 2), set(['11', '10'])),
                                        ((0,), set(['1'])),
                                        ((1, 2, 0), set(
                                            ['010', '011', '101'])),
                                        ((3,), set(['1']))])

    def test_logic_simple_read_no_commas(self):
        """
        Test simple network read in (no commas)
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "no_commas.txt")
        simple = LogicNetwork.read_table(TRUTH_TABLE)

        self.assertEqual(simple.names, ['A', 'B', 'C', 'D'])
        self.assertEqual(simple.table, [((1, 2), set(['11', '10'])),
                                        ((0,), set(['1'])),
                                        ((1, 2, 0), set(
                                            ['010', '011', '101'])),
                                        ((3,), set(['1']))])

    def test_logic_simple_read_no_header(self):
        """
        Test simple network read in (no header)
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "no_header.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_table(TRUTH_TABLE)

    def test_logic_simple_read_no_node_headers(self):
        """
        Test simple network read in (no node headers)
        """
        SIMPLE_TRUTH_TABLE = join(self.TABLE_PATH, "no_node_headers.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_table(SIMPLE_TRUTH_TABLE)

    def test_logic_simple_read_empty(self):
        """
        Test simple network read in (empty table)
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "empty_table.txt")
        simple = LogicNetwork.read_table(TRUTH_TABLE)

        self.assertEqual(simple.names, ['A', 'B', 'C', 'D'])
        self.assertEqual(simple.table, [((0,), set(['1'])),
                                        ((1,), set(['1'])),
                                        ((2,), set(['1'])),
                                        ((3,), set(['1']))])

    def test_logic_simple_read_custom_comment(self):
        """
        Test simple network read in with custom comments
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "custom_comment.txt")
        simple = LogicNetwork.read_table(TRUTH_TABLE)

        self.assertEqual(simple.names, ['A', 'B', 'C', 'D'])
        self.assertEqual(simple.table, [((1, 2), set(['11', '10'])),
                                        ((0,), set(['1'])),
                                        ((1, 2, 0), set(
                                            ['010', '011', '101'])),
                                        ((3,), set(['1']))])

    def test_read_table_missing_node(self):
        """
        Raise FormatError if a node is missing from the header
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "missing_node.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_table(TRUTH_TABLE)

    def test_read_table_skips_empty(self):
        """
        Empty lines are skipped when reading truth tables
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "empty_lines.txt")
        simple = LogicNetwork.read_table(TRUTH_TABLE)
        self.assertEqual(simple.names, ['A', 'B', 'C'])
        self.assertEqual(simple.table, [((1, 2), set(['10', '11'])),
                                        ((0,), set(['1'])),
                                        ((1, 2, 0), set(['101', '010', '011']))])

    def test_read_table_condition_len_mismatch(self):
        """
        Raise FormatError if a condition has too many or too few bits in it
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "missing_condition.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_table(TRUTH_TABLE)

        TRUTH_TABLE = join(self.TABLE_PATH, "extra_condition.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_table(TRUTH_TABLE)

    def test_read_table_invalid_condition(self):
        """
        Raise FormatError if a condition has an invalid state
        """
        TRUTH_TABLE = join(self.TABLE_PATH, "invalid_condition.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_table(TRUTH_TABLE)

    def test_read_logic_missing_equation(self):
        """
        Raise ValueError if an equation is missing
        """
        LOGIC = join(self.LOGIC_PATH, "missing_equation.txt")
        with self.assertRaises(FormatError):
            LogicNetwork.read_logic(LOGIC)

    def test_neighbors_in(self):

        net = LogicNetwork([((1, 2), set(['11', '10'])),
                            ((0,), set(['1'])),
                            ((0, 1, 2), set(['010', '011', '101'])),
                            ((3,), set(['1']))])

        self.assertEqual(net.neighbors_in(2), set([0, 1, 2]))

        with self.assertRaises(TypeError):
            net.neighbors_in(2.0)

        with self.assertRaises(TypeError):
            net.neighbors_in('2')

    def test_neighbors_out(self):

        net = LogicNetwork([((1, 2), set(['11', '10'])),
                            ((0,), set(['1'])),
                            ((0, 1, 2), set(['010', '011', '101'])),
                            ((3,), set(['1']))])

        self.assertEqual(net.neighbors_out(2), set([0, 2]))

    def test_neighbors_both(self):

        net = LogicNetwork([((1, 2), set(['11', '10'])),
                            ((0,), set(['1'])),
                            ((0, 1, 2), set(['010', '011', '101'])),
                            ((3,), set(['1']))])

        self.assertEqual(net.neighbors(2), set([0, 1, 2]))

    def test_node_dependency(self):
        net = LogicNetwork([((1, 2), {'11', '10'}),
                            ((0,), {'1'}),
                            ((0, 1, 2), {'010', '011', '101', '100'})])

        self.assertTrue(net.is_dependent(0, 1))
        self.assertFalse(net.is_dependent(0, 2))

        self.assertTrue(net.is_dependent(1, 0))
        self.assertFalse(net.is_dependent(1, 2))

        self.assertFalse(net.is_dependent(2, 2))
        self.assertTrue(net.is_dependent(2, 0))
        self.assertTrue(net.is_dependent(2, 1))

    def test_reduce_table(self):
        table = [((1, 2), {'11', '10'}),
                 ((0,), {'1'}),
                 ((0, 1, 2), {'010', '011', '101', '100'})]
        net = LogicNetwork(table, reduced=True)

        reduced_table = [((1,), {'1'}),
                         ((0,), {'1'}),
                         ((0, 1), {'01', '10'})]
        self.assertEqual(net.table, reduced_table)

        net = LogicNetwork(table)
        self.assertEqual(net.table, table)
        self.assertEqual(net._encoded_table,
                         [(6, {6, 2}), (1, {1}), (7, {2, 6, 5, 1})])

        net.reduce_table()
        self.assertEqual(net.table, reduced_table)
        self.assertEqual(net._encoded_table,
                         [(2, {2}), (1, {1}), (3, {2, 1})])

        net.reduce_table()
        self.assertEqual(net.table, reduced_table)

        net = LogicNetwork([((0, 1), {'00', '01', '10', '11'}),
                            ((1,), {'1'})],
                           reduced=True)

        self.assertEqual(net.table,
                         [((0,), {'0', '1'}),
                          ((1,), {'1'})])

        net = LogicNetwork([((1,), {'0', '1'}),
                            ((0,), set())],
                           reduced=True)

        self.assertEqual(net.table,
                         [((0,), {'0', '1'}),
                          ((1,), set())])

    def test_network_graph_names(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ((0, 1), '10', [1, 1])),
                            ((0, 1), {'11'})],
                           names=['A', 'B', 'C'])

        nx_net = net.network_graph(labels='names')
        self.assertEqual(set(nx_net), set(['A', 'B', 'C']))

    def test_network_graph_names_fail(self):
        net = LogicNetwork([((0,), {'0'})])

        with self.assertRaises(ValueError):
            net.network_graph(labels='names')

    def test_network_graph_metadata(self):
        net = LogicNetwork([((0,), {'0'})])
        net.metadata['name'] = 'net_name'

        nx_net = net.network_graph(labels='indices')

        self.assertEqual(nx_net.graph['name'], net.metadata['name'])

    def test_network_graph_has_all_nodes(self):
        net = LogicNetwork([((), set())])
        g = net.network_graph()
        self.assertEqual(g.number_of_nodes(), net.size)

        net = LogicNetwork([((), set()), ((2,), {'0'}), ((1,), {'1'})])
        g = net.network_graph()
        self.assertEqual(g.number_of_nodes(), net.size)
