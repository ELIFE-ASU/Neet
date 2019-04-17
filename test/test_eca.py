import unittest
import neet.automata as ca
import numpy as np


class TestECA(unittest.TestCase):
    def test_is_network(self):
        from neet.interfaces import Network
        self.assertTrue(isinstance(ca.ECA(23, 3), Network))

    def test_fail_init(self):
        with self.assertRaises(ValueError):
            ca.ECA(-1, 5)

        with self.assertRaises(ValueError):
            ca.ECA(256, 5)

        with self.assertRaises(TypeError):
            ca.ECA([1, 1, 0, 1, 1, 0, 0, 1])

        with self.assertRaises(TypeError):
            ca.ECA("30", 5)

        with self.assertRaises(TypeError):
            ca.ECA(30, "5")

        with self.assertRaises(ValueError):
            ca.ECA(30, -1)

        with self.assertRaises(ValueError):
            ca.ECA(30, 0)

        with self.assertRaises(TypeError):
            ca.ECA(30, 5, boundary=[1, 2])

        with self.assertRaises(ValueError):
            ca.ECA(30, 5, boundary=(1, 0, 1))

        with self.assertRaises(ValueError):
            ca.ECA(30, 5, boundary=(1, 2))

    def test_init(self):
        for code in range(256):
            for size in range(1, 5):
                for left in range(2):
                    for right in range(2):
                        eca = ca.ECA(code, size, (left, right))
                        self.assertEqual(code, eca.code)
                        self.assertEqual(size, eca.size)
                        self.assertEqual((left, right), eca.boundary)

    def test_invalid_code(self):
        eca = ca.ECA(30, 5)

        eca.code = 45

        with self.assertRaises(ValueError):
            eca.code = -1

        with self.assertRaises(ValueError):
            eca.code = 256

        with self.assertRaises(TypeError):
            eca.code = "30"

    def test_invalid_size(self):
        eca = ca.ECA(30, 5)

        eca.size = 8

        with self.assertRaises(ValueError):
            eca.size = -1

        with self.assertRaises(ValueError):
            eca.size = 0

        with self.assertRaises(TypeError):
            eca.size = "5"

    def test_invalid_boundary(self):
        eca = ca.ECA(30, 5)

        eca.boundary = (0, 0)
        eca.boundary = None

        with self.assertRaises(ValueError):
            eca.boundary = (1, 1, 1)

        with self.assertRaises(ValueError):
            eca.boundary = (1, 2)

        with self.assertRaises(TypeError):
            eca.boundary = 1

        with self.assertRaises(TypeError):
            eca.boundary = [0, 1]

    def test_state_space(self):
        self.assertEqual(2, len(list(ca.ECA(30, 1).state_space())))
        self.assertEqual(4, len(list(ca.ECA(30, 2).state_space())))
        self.assertEqual(8, len(list(ca.ECA(30, 3).state_space())))

    def test_invalid_lattice_state_update(self):
        eca = ca.ECA(30, 3)
        with self.assertRaises(ValueError):
            eca.update([-1, 0, 1])

        with self.assertRaises(ValueError):
            eca.update([1, 0, -1])

        with self.assertRaises(ValueError):
            eca.update([2, 0, 0])

        with self.assertRaises(ValueError):
            eca.update([1, 0, 2])

        with self.assertRaises(ValueError):
            eca.update([[1], [0], [2]])

        with self.assertRaises(ValueError):
            eca.update("101")

    def test_update_closed(self):
        eca = ca.ECA(30, 1)

        lattice = [0]

        eca.update(lattice)
        self.assertEqual([0], lattice)

        eca.size = 2
        lattice = [0, 0]

        eca.update(lattice)
        self.assertEqual([0, 0], lattice)

        eca.size = 5
        lattice = [0, 0, 1, 0, 0]

        eca.update(lattice)
        self.assertEqual([0, 1, 1, 1, 0], lattice)

        eca.update(lattice)
        self.assertEqual([1, 1, 0, 0, 1], lattice)

    def test_update_open(self):
        eca = ca.ECA(30, 1, (0, 1))

        lattice = [0]

        eca.update(lattice)
        self.assertEqual([1], lattice)

        eca.size = 2
        lattice = [0, 0]

        eca.update(lattice)
        self.assertEqual([0, 1], lattice)

        eca.size = 5
        lattice = [0, 0, 1, 0, 0]

        eca.update(lattice)
        self.assertEqual([0, 1, 1, 1, 1], lattice)

        eca.update(lattice)
        self.assertEqual([1, 1, 0, 0, 0], lattice)

    def test_update_long_time_closed(self):
        eca = ca.ECA(45, 14)
        lattice = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        expected = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        state_space = eca.state_space()
        if lattice in state_space:
            for n in range(1000):
                eca._unsafe_update(lattice)
        self.assertEqual(expected, lattice)

    def test_update_long_time_open(self):
        eca = ca.ECA(45, 14, (0, 1))
        lattice = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        expected = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        state_space = eca.state_space()
        if lattice in state_space:
            for n in range(1000):
                eca._unsafe_update(lattice)
        self.assertEqual(expected, lattice)

    def test_update_numpy(self):
        eca = ca.ECA(30, 1, (0, 1))

        lattice = np.asarray([0])

        eca.update(lattice)
        self.assertTrue(np.array_equal([1], lattice))

        eca.size = 2
        lattice = np.asarray([0, 0])

        eca.update(lattice)
        self.assertTrue(np.array_equal([0, 1], lattice))

        eca.size = 5
        lattice = [0, 0, 1, 0, 0]

        eca.update(lattice)
        self.assertTrue(np.array_equal([0, 1, 1, 1, 1], lattice))

        eca.update(lattice)
        self.assertTrue(np.array_equal([1, 1, 0, 0, 0], lattice))

    def test_update_index_error(self):
        eca = ca.ECA(30, 2)
        with self.assertRaises(IndexError):
            eca.update([0, 0], index=2)

        with self.assertRaises(IndexError):
            eca.update([0, 0], index=-3)

    def test_update_index(self):
        eca = ca.ECA(30, 5, (1, 1))

        lattice = [0, 0, 0, 0, 0]
        eca.update(lattice, index=0)
        self.assertEqual([1, 0, 0, 0, 0], lattice)

        lattice = [0, 0, 0, 0, 0]
        eca.update(lattice, index=1)
        self.assertEqual([0, 0, 0, 0, 0], lattice)

        lattice = [0, 0, 0, 0, 0]
        eca.update(lattice, index=-1)
        self.assertEqual([0, 0, 0, 0, 1], lattice)

        lattice = [0, 0, 1, 0, 0]
        eca.update(lattice, index=1)
        self.assertEqual([0, 1, 1, 0, 0], lattice)

    def test_update_index_numpy(self):
        eca = ca.ECA(30, 5, (1, 1))

        lattice = np.asarray([0, 0, 0, 0, 0])
        eca.update(lattice, index=0)
        self.assertTrue(np.array_equal([1, 0, 0, 0, 0], lattice))

        lattice = np.asarray([0, 0, 0, 0, 0])
        eca.update(lattice, index=1)
        self.assertTrue(np.array_equal([0, 0, 0, 0, 0], lattice))

        lattice = np.asarray([0, 0, 0, 0, 0])
        eca.update(lattice, index=-1)
        self.assertTrue(np.array_equal([0, 0, 0, 0, 1], lattice))

        lattice = np.asarray([0, 0, 1, 0, 0])
        eca.update(lattice, index=1)
        self.assertTrue(np.array_equal([0, 1, 1, 0, 0], lattice))

    def test_update_pin_none(self):
        eca = ca.ECA(30, 5)

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 1, 1, 1, 0], eca.update(xs, pin=None))
        self.assertEqual([1, 1, 0, 0, 1], eca.update(xs, pin=[]))

    def test_update_pin_index_clash(self):
        eca = ca.ECA(30, 2)
        with self.assertRaises(ValueError):
            eca.update([0, 0], index=0, pin=[1])
        with self.assertRaises(ValueError):
            eca.update([0, 0], index=1, pin=[1])
        with self.assertRaises(ValueError):
            eca.update([0, 0], index=1, pin=[0, 1])

    def test_update_pin(self):
        eca = ca.ECA(30, 5)

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 0, 1, 1, 0], eca.update(xs, pin=[1]))
        self.assertEqual([0, 0, 1, 0, 1], eca.update(xs, pin=[1]))
        self.assertEqual([1, 0, 1, 0, 1], eca.update(xs, pin=[1]))

        eca.boundary = (1, 1)
        xs = [0, 0, 0, 0, 0]
        self.assertEqual([1, 0, 0, 0, 0], eca.update(xs, pin=[-1]))
        self.assertEqual([1, 1, 0, 0, 0], eca.update(xs, pin=[0, -1]))

    def test_update_values_none(self):
        eca = ca.ECA(30, 5)

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 1, 1, 1, 0], eca.update(xs, values=None))
        self.assertEqual([1, 1, 0, 0, 1], eca.update(xs, values={}))

    def test_update_invalid_values(self):
        eca = ca.ECA(30, 5)
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], values={0: 2})
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], values={0: -1})

    def test_update_values_index_clash(self):
        eca = ca.ECA(30, 5)
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], index=0, values={0: 1})
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], index=1, values={1: 0})
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], index=1, values={0: 0, 1: 0})

    def test_update_values_pin_clash(self):
        eca = ca.ECA(30, 5)
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], pin=[0], values={0: 1})
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], pin=[1], values={1: 0})
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], pin=[1], values={0: 0, 1: 0})
        with self.assertRaises(ValueError):
            eca.update([0, 0, 0, 0, 0], pin=[1, 0], values={0: 0})

    def test_update_values(self):
        eca = ca.ECA(30, 5)

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 1, 0, 1, 0], eca.update(xs, values={2: 0}))
        self.assertEqual([1, 0, 0, 0, 1], eca.update(xs, values={1: 0, 3: 0}))
        self.assertEqual([0, 1, 0, 1, 0], eca.update(xs, values={-1: 0}))

        eca.boundary = (1, 1)
        xs = [0, 0, 0, 0, 0]
        self.assertEqual([1, 0, 1, 0, 1], eca.update(xs, values={2: 1}))

    def test_neighbors_in(self):
        net = ca.ECA(30, 3)

        self.assertEqual(net.neighbors_in(2), set([0, 1, 2]))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors_in(3))

        net.boundary = (1, 1)

        self.assertEqual(net.neighbors_in(2), set([1, 2, 3]))

        with self.assertRaises(ValueError):
            net.neighbors_in(3)

        with self.assertRaises(ValueError):
            net.neighbors_in(5)

        with self.assertRaises(TypeError):
            net.neighbors_in('2')

        with self.assertRaises(ValueError):
            net.neighbors_in(-1)

    def test_neighbors_out(self):
        net = ca.ECA(30, 3)

        self.assertEqual(net.neighbors_out(2), set([0, 1, 2]))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors_out(3))

        net.boundary = (1, 1)

        self.assertEqual(net.neighbors_out(2), set([1, 2]))

        with self.assertRaises(ValueError):
            net.neighbors_out(5)

        with self.assertRaises(TypeError):
            net.neighbors_out('2')

        with self.assertRaises(ValueError):
            net.neighbors_out(-1)

    def test_neighbors_both(self):
        net = ca.ECA(30, 4)

        self.assertEqual(net.neighbors(2), set([1, 2, 3]))

    def test_to_networkx_metadata(self):
        net = ca.ECA(30, 3)
        net.boundary = (1, 0)

        nx_net = net.to_networkx_graph()

        self.assertEqual(nx_net.graph['code'], 30)
        self.assertEqual(nx_net.graph['size'], 3)
        self.assertEqual(nx_net.graph['boundary'], (1, 0))
