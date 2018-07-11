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

        net = bnet.WTNetwork(1)
        self.assertEqual(1, net.size)
        self.assertTrue(np.array_equal([[0]], net.weights))
        self.assertTrue(np.array_equal([0], net.thresholds))
        self.assertIsNone(net.names)

        net = bnet.WTNetwork(2)
        self.assertEqual(2, net.size)
        self.assertTrue(np.array_equal([[0,0],[0,0]], net.weights))
        self.assertTrue(np.array_equal([0,0], net.thresholds))
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
        self.assertEqual(2, len(list(net.state_space())))
        net = bnet.WTNetwork([[1,0],[1,1]])
        self.assertEqual(4, len(list(net.state_space())))
        net = bnet.WTNetwork([[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(8, len(list(net.state_space())))


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
        self.assertEqual(512, len(list(net.state_space())))

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
        self.assertEqual(512, len(list(net.state_space())))

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


    def test_update_pin_index_clash(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        with self.assertRaises(ValueError):
          net.update([0,0], index=0, pin=[1])
        with self.assertRaises(ValueError):
          net.update([0,0], index=1, pin=[1])
        with self.assertRaises(ValueError):
          net.update([0,0], index=1, pin=[0,1])


    def test_update_pin(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0])

        net.theta = bnet.WTNetwork.negative_threshold
        xs = [1,1]
        self.assertEqual([1,0], net.update(xs, pin=[0]))

        net.theta = bnet.WTNetwork.positive_threshold
        xs = [0,0]
        self.assertEqual([0,0], net.update(xs, pin=[1]))


    def test_pinning_s_pombe(self):
        from neet.boolean.examples import s_pombe
        self.assertEqual(
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            s_pombe.update([0,0,0,0,1,0,0,0,0], pin=[-1])
        )
        self.assertEqual(
            [0, 0, 1, 1, 0, 0, 1, 0, 0],
            s_pombe.update([0,0,0,0,0,0,0,0,1], pin=[1])
        )
        self.assertEqual(
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            s_pombe.update([0,0,0,0,0,0,0,0,1], pin=range(1,4))
        )
        self.assertEqual(
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
            s_pombe.update([0,0,0,0,0,0,0,0,1], pin=[1,2,3,-1])
        )


    def test_update_values_none(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        xs = [0,0]
        self.assertEqual([0,1], net.update(xs, values=None))
        xs = [0,0]
        self.assertEqual([0,1], net.update(xs, values={}))


    def test_update_invalid_values(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        with self.assertRaises(ValueError):
          net.update([0,0], values={0: 2})
        with self.assertRaises(ValueError):
          net.update([0,0], values={0: -1})


    def test_update_pin_invalid_indicies(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        with self.assertRaises(IndexError):
          net.update([0,0], values={-3: 0})
        with self.assertRaises(IndexError):
          net.update([0,0], values={2: 0})


    def test_update_values_index_clash(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        with self.assertRaises(ValueError):
          net.update([0,0], index=0, values={0: 1})
        with self.assertRaises(ValueError):
          net.update([0,0], index=1, values={1: 0})
        with self.assertRaises(ValueError):
          net.update([0,0], index=1, values={0: 0, 1: 0})


    def test_update_values_pin_clash(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.positive_threshold)
        with self.assertRaises(ValueError):
          net.update([0,0], pin=[0], values={0: 1})
        with self.assertRaises(ValueError):
          net.update([0,0], pin=[1], values={1: 0})
        with self.assertRaises(ValueError):
          net.update([0,0], pin=[1], values={0: 0, 1: 0})
        with self.assertRaises(ValueError):
          net.update([0,0], pin=[1, 0], values={0: 0})


    def test_update_values(self):
        net = bnet.WTNetwork([[1,0],[-1,1]], [0.5,0.0],
          theta=bnet.WTNetwork.negative_threshold)

        xs = [1,1]
        self.assertEqual([1,1], net.update(xs, values={1: 1}))

        net.theta = bnet.WTNetwork.positive_threshold
        xs = [0,0]
        self.assertEqual([1,1], net.update(xs, values={0: 1}))


    def test_values_s_pombe(self):
        from neet.boolean.examples import s_pombe
        self.assertEqual(
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            s_pombe.update([0,0,0,0,1,0,0,0,0], values={1:1,4:1,-1:0})
        )
        self.assertEqual(
            [0, 1, 0, 1, 0, 0, 1, 1, 0],
            s_pombe.update([0,0,0,0,0,0,0,0,1], values={2:0,-2:1})
        )
        self.assertEqual(
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            s_pombe.update([0,0,0,0,0,0,0,0,1], values={1: 0, 2:0, 3:0})
        )

    def test_has_metadata(self):
        net = bnet.WTNetwork([[1]])
        self.assertTrue(hasattr(net,'metadata'))
        self.assertEqual(type(net.metadata),dict)

    def test_neighbors_in_split_threshold(self):

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

        self.assertEqual(net.neighbors_in(2),set([0,1,2,5,8]))

        with self.assertRaises(IndexError):
            self.assertEqual(net.neighbors_in(2.0))

        with self.assertRaises(IndexError):
            self.assertEqual(net.neighbors_in('2'))

    def test_neighbors_out_split_threshold(self):

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

        self.assertEqual(net.neighbors_out(2),set([1,2,5]))

    def test_neighbors_in_positive_threshold(self):

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
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            theta=bnet.WTNetwork.positive_threshold)

        self.assertEqual(net.neighbors_in(2),set([0,1,5,8]))

        with self.assertRaises(IndexError):
            self.assertEqual(net.neighbors_in(2.0))

        with self.assertRaises(IndexError):
            self.assertEqual(net.neighbors_in('2'))

    def test_neighbors_out_negative_threshold(self):

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
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            theta=bnet.WTNetwork.negative_threshold)

        self.assertEqual(net.neighbors_out(2),set([1,5]))
      
    def test_neighbors_both_split_threshold(self):

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

        self.assertEqual(net.neighbors(2),set([0,1,2,5,8]))

    def test_to_networkx_graph_names(self):
        from neet.boolean.examples import s_pombe

        nx_net = s_pombe.to_networkx_graph(labels='names')
        self.assertEqual(set(nx_net),set(s_pombe.names))

    def test_to_networkx_graph_names_fail(self):
        net = bnet.WTNetwork([[1,0],[0,1]])

        with self.assertRaises(ValueError):
            net.to_networkx_graph(labels='names')

    def test_to_networkx_metadata(self):
        from neet.boolean.examples import s_pombe

        nx_net = s_pombe.to_networkx_graph(labels='indices')

        self.assertEqual(nx_net.graph['name'],'s_pombe')
        self.assertEqual(nx_net.graph['name'],s_pombe.metadata['name'])

    # def test_draw(self):
    #     net = bnet.LogicNetwork([((0,), {'0'})])
    #     draw(net,labels='indices')

        


