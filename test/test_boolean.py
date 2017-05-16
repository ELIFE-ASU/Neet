# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet.boolean as bnet
import numpy as np

class TestWTNetwork(unittest.TestCase):
    def test_is_network(self):
        from neet.interfaces import is_network
        self.assertTrue(is_network(bnet.WTNetwork))
        self.assertTrue(is_network(bnet.WTNetwork([[1]])))


    def test_is_fixed_sized(self):
        from neet.interfaces import is_fixed_sized
        self.assertTrue(is_fixed_sized(bnet.WTNetwork))
        self.assertTrue(is_fixed_sized(bnet.WTNetwork([[1]])))


    def test_init_failed(self):
        with self.assertRaises(ValueError):
            bnet.WTNetwork(None)

        with self.assertRaises(ValueError):
            bnet.WTNetwork('a')

        with self.assertRaises(ValueError):
            bnet.WTNetwork(0)

        with self.assertRaises(ValueError):
            bnet.WTNetwork(-1)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], 'a')

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], 0)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], -1)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([])

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[]])

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]],[])

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]],[1,2])


    def test_init_weights(self):
        net = bnet.WTNetwork([[1]])
        self.assertEqual(1, net.size)
        self.assertTrue(np.array_equal([[1]], net.weights))
        self.assertTrue(np.array_equal([0], net.thresholds))
        self.assertIsNone(net.names)

        net = bnet.WTNetwork([[1,0],[0,1]])
        self.assertEqual(2, net.size)
        self.assertTrue(np.array_equal([[1,0],[0,1]], net.weights))
        self.assertTrue(np.array_equal([0,0], net.thresholds))
        self.assertIsNone(net.names)

        net = bnet.WTNetwork([[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(3, net.size)
        self.assertTrue(np.array_equal([[1,0,0],[0,1,0],[0,0,1]], net.weights))
        self.assertTrue(np.array_equal([0,0,0], net.thresholds))
        self.assertIsNone(net.names)


    def test_init_weights_thresholds(self):
        net = bnet.WTNetwork([[1]], [1])
        self.assertEqual(1, net.size)
        self.assertTrue(np.array_equal([[1]], net.weights))
        self.assertTrue(np.array_equal([1], net.thresholds))
        self.assertIsNone(net.names)

        net = bnet.WTNetwork([[1,0],[0,1]], [1,2])
        self.assertEqual(2, net.size)
        self.assertTrue(np.array_equal([[1,0],[0,1]], net.weights))
        self.assertTrue(np.array_equal([1,2], net.thresholds))
        self.assertIsNone(net.names)

        net = bnet.WTNetwork([[1,0,0],[0,1,0],[0,0,1]], [1,2,3])
        self.assertEqual(3, net.size)
        self.assertTrue(np.array_equal([[1,0,0],[0,1,0],[0,0,1]], net.weights))
        self.assertTrue(np.array_equal([1,2,3], net.thresholds))
        self.assertIsNone(net.names)


    def test_init_names(self):
        with self.assertRaises(TypeError):
            bnet.WTNetwork([[1]], names=5)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], names=["A","B"])

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], names=["A","B"])

        net = bnet.WTNetwork([[1]], names=["A"])
        self.assertEqual(['A'], net.names)

        net = bnet.WTNetwork([[1,0],[1,1]], names=['A','B'])
        self.assertEqual(['A','B'], net.names)

        net = bnet.WTNetwork([[1]], names="A")
        self.assertEqual(['A'], net.names)

        net = bnet.WTNetwork([[1,0],[1,1]], names="AB")
        self.assertEqual(['A','B'], net.names)

    def test_init_thresholds(self):
        with self.assertRaises(TypeError):
            bnet.WTNetwork([[1]], theta = 5)

        net = bnet.WTNetwork([[1]])
        self.assertEqual(bnet.WTNetwork.split_threshold, net.theta)

        net = bnet.WTNetwork([[1]], theta = bnet.WTNetwork.negative_threshold)
        self.assertEqual(bnet.WTNetwork.negative_threshold, net.theta)

        net = bnet.WTNetwork([[1]], theta = bnet.WTNetwork.positive_threshold)
        self.assertEqual(bnet.WTNetwork.positive_threshold, net.theta)


    def test_state_space(self):
        net = bnet.WTNetwork([[1]])
        self.assertEqual(2, len(list(net.state_space().states())))
        net = bnet.WTNetwork([[1,0],[1,1]])
        self.assertEqual(4, len(list(net.state_space().states())))
        net = bnet.WTNetwork([[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(8, len(list(net.state_space().states())))


    def test_check_states_nonsense(self):
        net = bnet.WTNetwork([[1,0],[1,1]])

        with self.assertRaises(TypeError):
            net.check_states(5)

        with self.assertRaises(ValueError):
            net.check_states("elife")


    def test_check_states_list(self):
        net = bnet.WTNetwork([[1,0],[1,1]])

        with self.assertRaises(ValueError):
            net.check_states([])

        with self.assertRaises(ValueError):
            net.check_states([[]])

        with self.assertRaises(ValueError):
            net.check_states([0])

        with self.assertRaises(ValueError):
            net.check_states([0,2])

        self.assertTrue(net.check_states([0,0]))
        self.assertTrue(net.check_states([1,1]))


    def test_check_states_numpy(self):
        net = bnet.WTNetwork(np.asarray([[1,0],[1,1]]))

        with self.assertRaises(ValueError):
            net.check_states(np.asarray([]))

        with self.assertRaises(ValueError):
            net.check_states(np.asarray([[]]))

        with self.assertRaises(ValueError):
            net.check_states(np.asarray([0]))

        with self.assertRaises(ValueError):
            net.check_states(np.asarray([0,2]))

        self.assertTrue(net.check_states(np.asarray([0,0])))
        self.assertTrue(net.check_states(np.asarray([1,1])))


    def test_update_empty_states(self):
        net = bnet.WTNetwork([[1,0],[1,1]])
        with self.assertRaises(ValueError):
            net.update([])


    def test_update_invalid_states(self):
        net = bnet.WTNetwork([[1,0],[1,1]])
        with self.assertRaises(ValueError):
            net.update([-1,0])

        with self.assertRaises(ValueError):
            net.update([1,-1])

        with self.assertRaises(ValueError):
            net.update([2,0])

        with self.assertRaises(ValueError):
            net.update([1,2])

        with self.assertRaises(ValueError):
            net.update([[1],[0]])

        with self.assertRaises(ValueError):
            net.update("101")


    def test_update_invalid_index(self):
        net = bnet.WTNetwork([[1,0],[1,1]])
        with self.assertRaises(IndexError):
            net.update([0,0], 2)


    def test_update(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0])

        self.assertEqual(bnet.WTNetwork.split_threshold, net.theta)

        xs = [0,0]
        self.assertEqual([0,0], net.update(xs))
        self.assertEqual([0,0], xs)

        xs = [1,0]
        self.assertEqual([1,0], net.update(xs))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([0,1], net.update(xs))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs))
        self.assertEqual([1,1], xs)

        net.theta = bnet.WTNetwork.negative_threshold

        xs = [0,0]
        self.assertEqual([0,0], net.update(xs))
        self.assertEqual([0,0], xs)

        xs = [1,0]
        self.assertEqual([1,0], net.update(xs))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([0,1], net.update(xs))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,0], net.update(xs))
        self.assertEqual([1,0], xs)

        net.theta = bnet.WTNetwork.positive_threshold

        xs = [0,0]
        self.assertEqual([0,1], net.update(xs))
        self.assertEqual([0,1], xs)

        xs = [1,0]
        self.assertEqual([1,0], net.update(xs))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([0,1], net.update(xs))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs))
        self.assertEqual([1,1], xs)


    def test_update_index(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0])

        self.assertEqual(bnet.WTNetwork.split_threshold, net.theta)

        xs = [0,0]
        self.assertEqual([0,0], net.update(xs, 0))
        self.assertEqual([0,0], net.update(xs, 1))
        self.assertEqual([0,0], xs)

        xs = [1,0]
        self.assertEqual([1,0], net.update(xs, 0))
        self.assertEqual([1,0], net.update(xs, 1))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([0,1], net.update(xs, 0))
        self.assertEqual([0,1], net.update(xs, 1))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs, 0))
        self.assertEqual([1,1], net.update(xs, 1))
        self.assertEqual([1,1], xs)

        net.theta = bnet.WTNetwork.negative_threshold

        xs = [0,0]
        self.assertEqual([0,0], net.update(xs, 0))
        self.assertEqual([0,0], net.update(xs, 1))
        self.assertEqual([0,0], xs)

        xs = [1,0]
        self.assertEqual([1,0], net.update(xs, 0))
        self.assertEqual([1,0], net.update(xs, 1))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([0,1], net.update(xs, 0))
        self.assertEqual([0,1], net.update(xs, 1))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs, 0))
        self.assertEqual([1,0], net.update(xs, 1))
        self.assertEqual([1,0], xs)

        net.theta = bnet.WTNetwork.positive_threshold

        xs = [0,0]
        self.assertEqual([0,0], net.update(xs, 0))
        self.assertEqual([0,1], net.update(xs, 1))
        self.assertEqual([0,1], xs)

        xs = [1,0]
        self.assertEqual([1,0], net.update(xs, 0))
        self.assertEqual([1,0], net.update(xs, 1))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([0,1], net.update(xs, 0))
        self.assertEqual([0,1], net.update(xs, 1))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs, 0))
        self.assertEqual([1,1], net.update(xs, 1))
        self.assertEqual([1,1], xs)


    def test_user_defined_threshold(self):
        def reverse_negative(values, states):
            if isinstance(values, list) or isinstance(values, np.ndarray):
                for i, x in enumerate(values):
                    if x <= 0:
                        states[i] = 1
                    else:
                        states[i] = 0
                return states
            else:
                if values <= 0:
                    return 1
                else:
                    return 0

        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0], theta=reverse_negative)
        xs = [0,0]
        self.assertEqual([1,1], net.update(xs))
        self.assertEqual([1,1], xs)

        xs = [0,0]
        self.assertEqual([1,0], net.update(xs, 0))
        self.assertEqual([1,0], xs)

        xs = [0,0]
        self.assertEqual([0,1], net.update(xs, 1))
        self.assertEqual([0,1], xs)

        xs = [1,0]
        self.assertEqual([0,1], net.update(xs))
        self.assertEqual([0,1], xs)

        xs = [1,0]
        self.assertEqual([0,0], net.update(xs, 0))
        self.assertEqual([0,0], xs)

        xs = [1,0]
        self.assertEqual([1,1], net.update(xs, 1))
        self.assertEqual([1,1], xs)

        xs = [0,1]
        self.assertEqual([1,0], net.update(xs))
        self.assertEqual([1,0], xs)

        xs = [0,1]
        self.assertEqual([1,1], net.update(xs, 0))
        self.assertEqual([1,1], xs)

        xs = [0,1]
        self.assertEqual([0,0], net.update(xs, 1))
        self.assertEqual([0,0], xs)

        xs = [1,1]
        self.assertEqual([0,1], net.update(xs))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([0,1], net.update(xs, 0))
        self.assertEqual([0,1], xs)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs, 1))
        self.assertEqual([1,1], xs)


    def test_fission_yeast(self):
        net = bnet.WTNetwork(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [ 0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0,-1.0, 1.0, 0.0],
             [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
             [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0],
             [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,-1.0]],
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

        self.assertEqual(9, net.size)
        self.assertEqual(512, len(list(net.state_space().states())))

        init = [1,0,1,1,0,0,1,0,0]
        bio_sequence = [[0,0,0,0,0,0,1,0,0],
                        [0,1,0,0,0,0,1,0,0],
                        [0,1,0,0,0,0,0,1,0],
                        [0,1,0,0,0,1,0,1,0],
                        [0,1,0,0,1,1,0,1,0],
                        [0,0,0,0,1,0,0,1,1],
                        [0,0,1,1,0,0,1,0,1],
                        [0,0,1,1,0,0,1,0,0]]

        for expected in bio_sequence:
            self.assertEqual(expected, net.update(init))


    def test_fission_yeast_numpy(self):
        net = bnet.WTNetwork(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [ 0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0,-1.0, 1.0, 0.0],
             [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
             [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0],
             [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,-1.0]],
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

        self.assertEqual(9, net.size)
        self.assertEqual(512, len(list(net.state_space().states())))

        init = np.asarray([1,0,1,1,0,0,1,0,0])
        bio_sequence = np.asarray([[0,0,0,0,0,0,1,0,0],
                                   [0,1,0,0,0,0,1,0,0],
                                   [0,1,0,0,0,0,0,1,0],
                                   [0,1,0,0,0,1,0,1,0],
                                   [0,1,0,0,1,1,0,1,0],
                                   [0,0,0,0,1,0,0,1,1],
                                   [0,0,1,1,0,0,1,0,1],
                                   [0,0,1,1,0,0,1,0,0]])

        for expected in bio_sequence:
            self.assertTrue(np.array_equal(expected, net.update(init)))


    def test_split_threshold(self):
        xs = [0,0,0]
        self.assertEqual([1,0,0], bnet.WTNetwork.split_threshold([1, -1, 0], xs))
        self.assertEqual([1,0,0], xs)

        xs = [1,1,1]
        self.assertEqual([1,0,1], bnet.WTNetwork.split_threshold([1, -1, 0], xs))
        self.assertEqual([1,0,1], xs)


    def test_split_threshold_scalar(self):
        test = {
            ( 1, 0) : 1,
            ( 0, 0) : 0,
            (-1, 0) : 0,
            ( 1, 1) : 1,
            ( 0, 1) : 1,
            (-1, 1) : 0,
        }
        for x, s in test:
            self.assertEqual(test[(x,s)], bnet.WTNetwork.split_threshold(x,s))


    def test_negative_threshold(self):
        xs = [0,0,0]
        self.assertEqual([1,0,0], bnet.WTNetwork.negative_threshold([1, -1, 0], xs))
        self.assertEqual([1,0,0], xs)

        xs = [1,1,1]
        self.assertEqual([1,0,0], bnet.WTNetwork.negative_threshold([1, -1, 0], xs))
        self.assertEqual([1,0,0], xs)


    def test_negative_threshold_scalar(self):
        test = {
            ( 1, 0) : 1,
            ( 0, 0) : 0,
            (-1, 0) : 0,
            ( 1, 1) : 1,
            ( 0, 1) : 0,
            (-1, 1) : 0,
        }
        for x, s in test:
            self.assertEqual(test[(x,s)], bnet.WTNetwork.negative_threshold(x,s))


    def test_positive_threshold(self):
        xs = [0,0,0]
        self.assertEqual([1,0,1], bnet.WTNetwork.positive_threshold([1, -1, 0], xs))
        self.assertEqual([1,0,1], xs)

        xs = [1,1,1]
        self.assertEqual([1,0,1], bnet.WTNetwork.positive_threshold([1, -1, 0], xs))
        self.assertEqual([1,0,1], xs)


    def test_positive_threshold_scalar(self):
        test = {
            ( 1, 0) : 1,
            ( 0, 0) : 1,
            (-1, 0) : 0,
            ( 1, 1) : 1,
            ( 0, 1) : 1,
            (-1, 1) : 0,
        }
        for x, s in test:
            self.assertEqual(test[(x,s)], bnet.WTNetwork.positive_threshold(x,s))

    def test_update_pin_none(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        xs = [0,0]
        self.assertEqual([0,1], net.update(xs, pin=None))
        xs = [0,0]
        self.assertEqual([0,1], net.update(xs, pin=[]))

    def test_update_pin(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        xs = [1,1]
        self.assertEqual([1,0], net.update(xs, pin=[0]))
        xs = [0,0]
        self.assertEqual([0,0], net.update(xs, pin=[1]))
