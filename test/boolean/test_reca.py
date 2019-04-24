import unittest
import numpy as np
from neet.boolean import ECA, RewiredECA
from neet.network import Network
from neet.boolean.network import BooleanNetwork


class TestRewiredECA(unittest.TestCase):
    """
    Unit tests of the RewiredECA class
    """

    def test_is_network(self):
        """
        Ensure that RewiredECA meets the requirement of a network
        """
        self.assertTrue(isinstance(RewiredECA(23, size=3), Network))
        self.assertTrue(isinstance(RewiredECA(23, size=3), BooleanNetwork))
        self.assertTrue(isinstance(RewiredECA(
            30, wiring=[[-1, 0, 1], [0, 1, 2], [1, 2, 3]]), Network))
        self.assertTrue(isinstance(RewiredECA(
            30, wiring=[[-1, 0, 1], [0, 1, 2], [1, 2, 3]]), BooleanNetwork))

    def test_invalid_code(self):
        """
        Ensure that init fails when an invalid Wolfram code is provided
        """
        with self.assertRaises(ValueError):
            RewiredECA(-1, size=3)
        with self.assertRaises(ValueError):
            RewiredECA(256, size=3)
        with self.assertRaises(TypeError):
            RewiredECA("30", size=3)

    def test_invalid_boundary(self):
        """
        Ensure that init fails when an invalid boundary condition is provided
        """
        with self.assertRaises(TypeError):
            RewiredECA(30, boundary=[1, 2], size=3)
        with self.assertRaises(ValueError):
            RewiredECA(30, boundary=(1, 0, 1), size=3)
        with self.assertRaises(ValueError):
            RewiredECA(30, boundary=(1, 2), size=3)

    def test_invalid_size(self):
        """
        Ensure that init fails when an invalid size is provided
        """
        with self.assertRaises(TypeError):
            RewiredECA(30, size="3")
        with self.assertRaises(ValueError):
            RewiredECA(30, size=-1)
        with self.assertRaises(ValueError):
            RewiredECA(30, size=0)

    def test_invalid_wiring(self):
        """
        Ensure that init fails when an invalid wiring is provided
        """
        with self.assertRaises(TypeError):
            RewiredECA(30, wiring=5)
        with self.assertRaises(TypeError):
            RewiredECA(30, wiring="apples")
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=[])
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=[-1, 0, 1])
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=np.asarray([-1, 0, 1]))
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=[[0], [0]])
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=[[0, 0], [0], [0]])
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=[[-2], [0], [0]])
        with self.assertRaises(ValueError):
            RewiredECA(30, wiring=[[2], [0], [0]])

    def test_invalid_size_wiring(self):
        """
        Ensure that size and wiring are not both provided, but at least one is
        """
        with self.assertRaises(ValueError):
            RewiredECA(30, size=3, wiring=[])
        with self.assertRaises(ValueError):
            RewiredECA(30)
        with self.assertRaises(ValueError):
            RewiredECA(30, boundary=(0, 0))

    def test_size_init(self):
        """
        Ensure that size initialization is working correctly
        """
        eca = RewiredECA(30, size=2)
        self.assertEqual(30, eca.code)
        self.assertEqual(2, eca.size)
        self.assertTrue(np.array_equal([[-1, 0], [0, 1], [1, 2]], eca.wiring))

        eca = RewiredECA(23, boundary=(1, 0), size=5)
        self.assertEqual(23, eca.code)
        self.assertEqual(5, eca.size)
        self.assertTrue(
            np.array_equal([[-1, 0, 1, 2, 3], [0, 1, 2, 3, 4],
                            [1, 2, 3, 4, 5]],
                           eca.wiring))

    def test_wiring_init(self):
        """
        Ensure that wiring initialization is working correctly
        """
        eca = RewiredECA(30, wiring=[[0], [0], [0]])
        self.assertEqual(30, eca.code)
        self.assertEqual(1, eca.size)
        self.assertTrue(np.array_equal([[0], [0], [0]], eca.wiring))

        eca = RewiredECA(23, boundary=(1, 0), wiring=[
                         [-1, -1, 0], [0, 1, 1], [2, 0, -1]])
        self.assertEqual(23, eca.code)
        self.assertEqual(3, eca.size)
        self.assertTrue(
            np.array_equal([[-1, -1, 0], [0, 1, 1], [2, 0, -1]],
                           eca.wiring))

    def test_setting_wiring(self):
        """
        Ensure that we cannot reshape the wiring
        """
        eca = RewiredECA(30, size=2)
        with self.assertRaises(AttributeError):
            eca.wiring = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(-1, eca.wiring[0, 0])
        eca.wiring[0, 0] = 0
        self.assertEqual(0, eca.wiring[0, 0])

    def test_invalid_lattice_size(self):
        """
        Ensure that update fails when the lattice is the wrong size
        """
        eca = RewiredECA(30, size=3)
        with self.assertRaises(ValueError):
            eca.update([])
        with self.assertRaises(ValueError):
            eca.update([0])
        with self.assertRaises(ValueError):
            eca.update([0, 0])

    def test_invalid_lattice_state(self):
        """
        Ensure that the states of the lattice are binary states
        """
        eca = RewiredECA(30, size=3)
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

    def test_reproduce_closed_ecas(self):
        """
        Ensure that RewiredECAs can reproduce closed ECAs
        """
        reca = RewiredECA(30, size=7)
        eca = ECA(30, size=7)
        state = [0, 0, 0, 1, 0, 0, 0]
        for _ in range(10):
            expect = eca.update(np.copy(state))
            got = reca.update(state)
            self.assertTrue(np.array_equal(expect, got))

    def test_reproduce_open_ecas(self):
        """
        Ensure that RewiredECAs can reproduce open ECAs
        """
        reca = RewiredECA(30, boundary=(1, 0), size=7)
        eca = ECA(30, size=7, boundary=(1, 0))
        state = [0, 0, 0, 1, 0, 0, 0]
        for _ in range(10):
            expect = eca.update(np.copy(state))
            got = reca.update(state)
            self.assertTrue(np.array_equal(expect, got))

    def test_rewired_network(self):
        """
        Test a non-trivially rewired network
        """
        reca = RewiredECA(30, wiring=[
            [-1, 0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]
        ])
        state = [0, 0, 0, 0, 1]
        self.assertEqual([1, 0, 0, 1, 1], reca.update(state))

        reca.wiring[:, :] = [
            [0, 4, 1, 2, 3], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ]
        state = [0, 0, 0, 0, 1]
        self.assertEqual([0, 1, 0, 1, 1], reca.update(state))
        self.assertEqual([0, 0, 0, 1, 0], reca.update(state))
        self.assertEqual([0, 0, 1, 1, 1], reca.update(state))
        self.assertEqual([0, 0, 1, 0, 0], reca.update(state))
        self.assertEqual([0, 1, 1, 1, 0], reca.update(state))
        self.assertEqual([0, 1, 0, 0, 1], reca.update(state))
        self.assertEqual([0, 0, 1, 1, 1], reca.update(state))

    def test_reca_invalid_index(self):
        """
        Test for invalid index arguments
        """
        reca = RewiredECA(30, wiring=[
            [0, 4, 1, 2, 3], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        with self.assertRaises(IndexError):
            reca.update([0, 0, 0, 0, 1], index=6)

        with self.assertRaises(IndexError):
            reca.update([0, 0, 0, 0, 1], index=-1)

    def test_reca_index(self):
        """
        Test the index argument
        """
        reca = RewiredECA(30, wiring=[
            [0, 4, 1, 2, 3], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        self.assertEqual([0, 0, 0, 1, 1], reca.update([0, 0, 0, 0, 1], index=3))
        self.assertEqual([0, 0, 1, 1, 1], reca.update([0, 0, 0, 1, 1], index=2))
        self.assertEqual([0, 1, 1, 1, 0], reca.update([0, 0, 1, 1, 0], index=1))
        self.assertEqual([0, 1, 0, 1, 0], reca.update([0, 1, 0, 1, 0], index=0))

    def test_reca_pin_none(self):
        """
        Ensure that pin behaves correctly for nil arguments
        """
        reca = RewiredECA(30, size=5)

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 1, 1, 1, 0], reca.update(xs, pin=None))
        self.assertEqual([1, 1, 0, 0, 1], reca.update(xs, pin=[]))

    def test_reca_pin_index_clash(self):
        """
        Ensure ValueError is raised when index and pin are provided
        """
        reca = RewiredECA(30, size=5)

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], index=0, pin=[1])
        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], index=1, pin=[1])
        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], index=1, pin=[0, 1])

    def test_reca_pin(self):
        """
        Test the pin argument
        """
        reca = RewiredECA(30, wiring=[
            [-1, 4, 1, 2, -1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 0, 1, 1, 0], reca.update(xs, pin=[1]))
        self.assertEqual([0, 1, 1, 1, 0], reca.update(xs, pin=[3]))
        self.assertEqual([0, 1, 1, 1, 0], reca.update(xs, pin=[3, 2]))
        self.assertEqual([0, 1, 0, 1, 0], reca.update(xs, pin=[-2]))

        reca.boundary = (1, 1)
        xs = [0, 0, 1, 0, 0]
        self.assertEqual([1, 0, 1, 0, 0], reca.update(xs, pin=[1, 3]))
        self.assertEqual([1, 1, 1, 0, 0], reca.update(xs, pin=[-2, -5]))
        self.assertEqual([1, 1, 1, 1, 0], reca.update(xs, pin=[0, 2]))

    def test_reca_values_none(self):
        """
        Ensure that values behaves correctly for nil arguments
        """
        reca = RewiredECA(30, size=5)

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 1, 1, 1, 0], reca.update(xs, values=None))
        self.assertEqual([1, 1, 0, 0, 1], reca.update(xs, values={}))

    def test_reca_invalid_values(self):
        """
        Ensure ValueError is raised for invalid values
        """
        reca = RewiredECA(30, size=5)

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], values={0: 2})

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], values={0: -1})

    def test_reca_values_index_clash(self):
        """
        Ensure ValueError is raised when index and values are both provided
        """
        reca = RewiredECA(30, size=5)

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], index=0, values={0: 1})

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], index=1, values={1: 0})

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], index=1, values={0: 0, 1: 0})

    def test_reca_values_pin_clash(self):
        """
        Ensure ValueError is raised when pin and values are both provided
        """
        reca = RewiredECA(30, size=5)

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], pin=[0], values={0: 1})

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], pin=[1], values={1: 0})

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], pin=[1], values={0: 0, 1: 0})

        with self.assertRaises(ValueError):
            reca.update([0, 0, 0, 0, 0], pin=[1, 0], values={0: 0, 1: 0})

    def test_reca_values(self):
        """
        Test the values argument
        """
        reca = RewiredECA(30, wiring=[
            [-1, 4, 1, 2, -1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        xs = [0, 0, 1, 0, 0]
        self.assertEqual([1, 1, 1, 1, 0], reca.update(xs, values={0: 1}))
        self.assertEqual([1, 1, 0, 0, 0], reca.update(xs, values={-1: 0}))
        self.assertEqual([1, 1, 1, 1, 1], reca.update(xs, values={-2: 1}))
        self.assertEqual([1, 0, 1, 0, 0], reca.update(
            xs, values={2: 1, -5: 1}))

        reca.boundary = (1, 1)
        xs = [0, 0, 1, 0, 0]
        self.assertEqual([0, 1, 1, 1, 0], reca.update(xs, values={0: 0}))
        self.assertEqual([1, 1, 0, 0, 1], reca.update(xs, values={-1: 1}))
        self.assertEqual([0, 0, 1, 0, 0], reca.update(xs, values={-2: 0}))
        self.assertEqual([0, 1, 0, 1, 0], reca.update(
            xs, values={2: 0, -5: 0}))


class TestRewiredECAIntegerUpdate(unittest.TestCase):
    def test_reproduce_closed_ecas(self):
        reca = RewiredECA(30, size=7)
        eca = ECA(30, size=7)
        state = 8
        for _ in range(10):
            expect = eca.update(state)
            got = reca.update(state)
            self.assertTrue(np.array_equal(expect, got))
            state = expect

    def test_reproduce_open_ecas(self):
        reca = RewiredECA(30, boundary=(1, 0), size=7)
        eca = ECA(30, size=7, boundary=(1, 0))
        state = 8
        for _ in range(10):
            expect = eca.update(state)
            got = reca.update(state)
            self.assertTrue(np.array_equal(expect, got))
            state = expect

    def test_rewired_network(self):
        reca = RewiredECA(30, wiring=[
            [-1, 0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]
        ])
        self.assertEqual(25, reca.update(16))

        reca.wiring[:, :] = [
            [0, 4, 1, 2, 3], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ]
        self.assertEqual(26, reca.update(16))
        self.assertEqual(8, reca.update(26))
        self.assertEqual(28, reca.update(8))
        self.assertEqual(4, reca.update(28))
        self.assertEqual(14, reca.update(4))
        self.assertEqual(18, reca.update(14))
        self.assertEqual(28, reca.update(18))

    def test_reca_index(self):
        reca = RewiredECA(30, wiring=[
            [0, 4, 1, 2, 3], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        self.assertEqual(24, reca.update(16, index=3))
        self.assertEqual(28, reca.update(24, index=2))
        self.assertEqual(14, reca.update(12, index=1))
        self.assertEqual(10, reca.update(10, index=0))

    def test_reca_pin_none(self):
        reca = RewiredECA(30, size=5)

        self.assertEqual(14, reca.update(4, pin=None))
        self.assertEqual(19, reca.update(14, pin=[]))

    def test_reca_pin(self):
        reca = RewiredECA(30, wiring=[
            [-1, 4, 1, 2, -1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        self.assertEqual(12, reca.update(4, pin=[1]))
        self.assertEqual(14, reca.update(12, pin=[3]))
        self.assertEqual(14, reca.update(14, pin=[3, 2]))
        self.assertEqual(10, reca.update(14, pin=[-2]))

        reca.boundary = (1, 1)
        self.assertEqual(5, reca.update(4, pin=[1, 3]))
        self.assertEqual(7, reca.update(5, pin=[-2, -5]))
        self.assertEqual(15, reca.update(7, pin=[0, 2]))

    def test_reca_values_none(self):
        reca = RewiredECA(30, size=5)

        self.assertEqual(14, reca.update(4, values=None))
        self.assertEqual(19, reca.update(14, values={}))

    def test_reca_values(self):
        reca = RewiredECA(30, wiring=[
            [-1, 4, 1, 2, -1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]
        ])

        self.assertEqual(15, reca.update(4, values={0: 1}))
        self.assertEqual(3, reca.update(15, values={-1: 0}))
        self.assertEqual(31, reca.update(3, values={-2: 1}))
        self.assertEqual(5, reca.update(31, values={2: 1, -5: 1}))

        reca.boundary = (1, 1)
        self.assertEqual(14, reca.update(4, values={0: 0}))
        self.assertEqual(19, reca.update(14, values={-1: 1}))
        self.assertEqual(4, reca.update(19, values={-2: 0}))
        self.assertEqual(10, reca.update(4, values={2: 0, -5: 0}))
