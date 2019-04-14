import numpy as np
import unittest
from neet.automata import ECA
from neet.boolean.examples import s_pombe
from neet.information import (Architecture, active_information,
                              entropy_rate, transfer_entropy,
                              mutual_information)


class TestInformation(unittest.TestCase):
    """
    Test the information analysis module
    """

    def test_canary(self):
        """
        A canary test to ensure the test suite is working
        """
        self.assertEqual(3, 1 + 2)

    def test_active_info_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            active_information(5, k=3, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            active_information(5, k=3, timesteps=10, local=True)

    def test_active_info_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            active_information(ECA(30), k=3, timesteps=10, local=False)
        active_information(ECA(30), k=3, timesteps=10, size=5, local=False)

    def test_active_info_s_pombe(self):
        """
        ``active_information`` computes the correct values for ``s_pombe``
        """
        known_ai = [0.0, 0.408344, 0.629567, 0.629567, 0.379157, 0.400462,
                    0.670196, 0.670196, 0.391891]
        computed_ai = active_information(s_pombe, k=5, timesteps=20)
        self.assertEqual(9, len(computed_ai))
        for got, expected in zip(computed_ai, known_ai):
            self.assertAlmostEqual(expected, got, places=6)

    def test_local_active_info_s_pombe(self):
        """
        local ``active_information`` averages to the correct values for
        ``s_pombe``
        """
        known_ai = [0.0, 0.408344, 0.629567, 0.629567, 0.379157, 0.400462,
                    0.670196, 0.670196, 0.391891]
        computed_ai = active_information(
            s_pombe, k=5, timesteps=20, local=True)
        self.assertEqual((9, 512, 16), computed_ai.shape)
        for got, expected in zip(computed_ai, known_ai):
            self.assertAlmostEqual(expected, np.mean(got), places=6)

    def test_entropy_rate_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            entropy_rate(5, k=3, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            entropy_rate(5, k=3, timesteps=10, local=True)

    def test_entropy_rate_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            entropy_rate(ECA(30), k=3, timesteps=10, local=False)
        entropy_rate(ECA(30), k=3, timesteps=10, size=5, local=False)

    def test_entropy_rate_s_pombe(self):
        """
        ``entropy_rate`` computes the correct values for ``s_pombe``
        """
        known_er = [0.0, 0.016912, 0.072803, 0.072803, 0.058420, 0.024794,
                    0.032173, 0.032173, 0.089669]
        computed_er = entropy_rate(s_pombe, k=5, timesteps=20)
        self.assertEqual(9, len(computed_er))
        for got, expected in zip(computed_er, known_er):
            self.assertAlmostEqual(expected, got, places=6)

    def test_local_entropy_rate_s_pombe(self):
        """
        local ``entropy_rate`` averages to the correct values for ``s_pombe``
        """
        known_er = [0.0, 0.016912, 0.072803, 0.072803, 0.058420, 0.024794,
                    0.032173, 0.032173, 0.089669]
        computed_er = entropy_rate(s_pombe, k=5, timesteps=20, local=True)
        self.assertEqual((9, 512, 16), computed_er.shape)
        for got, expected in zip(computed_er, known_er):
            self.assertAlmostEqual(expected, np.mean(got), places=6)

    def test_transfer_entropy_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            transfer_entropy(5, k=3, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            transfer_entropy(5, k=3, timesteps=10, local=True)

    def test_transfer_entropy_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            transfer_entropy(ECA(30), k=3, timesteps=10, local=False)
        transfer_entropy(ECA(30), k=3, timesteps=10, size=5, local=False)

    def test_transfer_entropy_s_pombe(self):
        """
        ``transfer_entropy`` computes the correct values for ``s_pombe``
        """
        known_te = np.asarray(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.016912, 0., 0., 0., 0.],
             [0., 0.051370, 0., 0.012225, 0.019947,
                 0.051370, 0.006039, 0.006039, 0.072803],
             [0., 0.051370, 0.012225, 0., 0.019947,
                 0.051370, 0.006039, 0.006039, 0.072803],
             [0., 0.058420, 0.047602, 0.047602, 0.,
                 0.058420, 0.047602, 0.047602, 0.],
             [0., 0., 0.024794, 0.024794, 0., 0., 0.024794, 0.024794, 0.],
             [0., 0.016690, 0.004526, 0.004526, 0.011916,
                 0.016690, 0., 0.002983, 0.032173],
             [0., 0.016690, 0.004526, 0.004526, 0.011916,
                 0.016690, 0.002983, 0., 0.032173],
             [0., 0.060304, 0.048289, 0.048289, 0.089669, 0.060304,
                 0.048927, 0.048927, 0.]])
        computed_te = transfer_entropy(s_pombe, k=5, timesteps=20)
        self.assertEqual(known_te.shape, computed_te.shape)
        for got, expected in zip(computed_te.flatten(), known_te.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

    def test_local_transfer_entropy_s_pombe(self):
        """
        local ``transfer_entropy`` averages to the correct values for
        ``s_pombe``
        """
        known_te = np.asarray(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.016912, 0., 0., 0., 0.],
             [0., 0.051370, 0., 0.012225, 0.019947,
                 0.051370, 0.006039, 0.006039, 0.072803],
             [0., 0.051370, 0.012225, 0., 0.019947,
                 0.051370, 0.006039, 0.006039, 0.072803],
             [0., 0.058420, 0.047602, 0.047602, 0.,
                 0.058420, 0.047602, 0.047602, 0.],
             [0., 0., 0.024794, 0.024794, 0., 0., 0.024794, 0.024794, 0.],
             [0., 0.016690, 0.004526, 0.004526, 0.011916,
                 0.016690, 0., 0.002983, 0.032173],
             [0., 0.016690, 0.004526, 0.004526, 0.011916,
                 0.016690, 0.002983, 0., 0.032173],
             [0., 0.060304, 0.048289, 0.048289, 0.089669, 0.060304,
                 0.048927, 0.048927, 0.]])
        computed_te = transfer_entropy(s_pombe, k=5, timesteps=20, local=True)
        self.assertEqual((9, 9, 512, 16), computed_te.shape)
        for i in range(9):
            for j in range(9):
                self.assertAlmostEqual(
                    known_te[i, j], np.mean(computed_te[i, j]), places=6)

    def test_mutual_information_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            mutual_information(5, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            mutual_information(5, timesteps=10, local=True)

    def test_mutual_information_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            mutual_information(ECA(30), timesteps=10, local=False)
        mutual_information(ECA(30), timesteps=10, size=5, local=False)

    def test_mutual_information_s_pombe(self):
        """
        ``mutual_information`` computes the correct values for ``s_pombe``
        """
        known_mi = np.asarray(
            [[0.162326, 0.013747, 0.004285, 0.004285, 0.013409, 0.015862,
              0.005170, 0.005170, 0.011028],
             [0.013747, 0.566610, 0.007457, 0.007457, 0.006391, 0.327908,
              0.006761, 0.006761, 0.004683],
             [0.004285, 0.007457, 0.838373, 0.475582, 0.211577, 0.004329,
              0.459025, 0.459025, 0.127557],
             [0.004285, 0.007457, 0.475582, 0.838373, 0.211577, 0.004329,
              0.459025, 0.459025, 0.127557],
             [0.013409, 0.006391, 0.211577, 0.211577, 0.574591, 0.007031,
              0.175608, 0.175608, 0.012334],
             [0.015862, 0.327908, 0.004329, 0.004329, 0.007031, 0.519051,
              0.006211, 0.006211, 0.002607],
             [0.005170, 0.006761, 0.459025, 0.459025, 0.175608, 0.006211,
              0.808317, 0.493495, 0.103905],
             [0.005170, 0.006761, 0.459025, 0.459025, 0.175608, 0.006211,
              0.493495, 0.808317, 0.103905],
             [0.011028, 0.004683, 0.127557, 0.127557, 0.012334, 0.002607,
              0.103905, 0.103905, 0.634238]])
        computed_mi = mutual_information(s_pombe, timesteps=20)
        self.assertEqual(known_mi.shape, computed_mi.shape)
        for got, expected in zip(computed_mi.flatten(), known_mi.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

    def test_local_mutual_information_s_pombe(self):
        """
        local ``mutual_information`` averages to the correct values for
        ``s_pombe``
        """
        known_mi = np.asarray(
            [[0.162326, 0.013747, 0.004285, 0.004285, 0.013409, 0.015862,
              0.005170, 0.005170, 0.011028],
             [0.013747, 0.566610, 0.007457, 0.007457, 0.006391, 0.327908,
              0.006761, 0.006761, 0.004683],
             [0.004285, 0.007457, 0.838373, 0.475582, 0.211577, 0.004329,
              0.459025, 0.459025, 0.127557],
             [0.004285, 0.007457, 0.475582, 0.838373, 0.211577, 0.004329,
              0.459025, 0.459025, 0.127557],
             [0.013409, 0.006391, 0.211577, 0.211577, 0.574591, 0.007031,
              0.175608, 0.175608, 0.012334],
             [0.015862, 0.327908, 0.004329, 0.004329, 0.007031, 0.519051,
              0.006211, 0.006211, 0.002607],
             [0.005170, 0.006761, 0.459025, 0.459025, 0.175608, 0.006211,
              0.808317, 0.493495, 0.103905],
             [0.005170, 0.006761, 0.459025, 0.459025, 0.175608, 0.006211,
              0.493495, 0.808317, 0.103905],
             [0.011028, 0.004683, 0.127557, 0.127557, 0.012334, 0.002607,
              0.103905, 0.103905, 0.634238]])
        computed_mi = mutual_information(s_pombe, timesteps=20, local=True)
        self.assertEqual((9, 9, 512, 21), computed_mi.shape)
        for i in range(9):
            for j in range(9):
                self.assertAlmostEqual(
                    known_mi[i, j], np.mean(computed_mi[i, j]), places=6)

    def test_architecture_ai(self):
        """
        The architecture correctly computes the active information
        """
        k, timesteps = 5, 20
        arch = Architecture(s_pombe, k=k, timesteps=timesteps)

        expected_ai = active_information(s_pombe, k=k, timesteps=timesteps)
        got_ai = arch.active_information()
        self.assertEqual(got_ai.shape, expected_ai.shape)
        for got, expected in zip(got_ai, expected_ai):
            self.assertAlmostEqual(expected, got, places=6)

        expected_ai = active_information(
            s_pombe, k=k, timesteps=timesteps, local=True)
        got_ai = arch.active_information(local=True)
        self.assertEqual(got_ai.shape, expected_ai.shape)
        for got, expected in zip(got_ai.flatten(), expected_ai.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

    def test_architecture_er(self):
        """
        The architecture correctly computes the entropy rate
        """
        k, timesteps = 5, 20
        arch = Architecture(s_pombe, k=k, timesteps=timesteps)

        expected_er = entropy_rate(s_pombe, k=k, timesteps=timesteps)
        got_er = arch.entropy_rate()
        self.assertEqual(got_er.shape, expected_er.shape)
        for got, expected in zip(got_er, expected_er):
            self.assertAlmostEqual(expected, got, places=6)

        expected_er = entropy_rate(
            s_pombe, k=k, timesteps=timesteps, local=True)
        got_er = arch.entropy_rate(local=True)
        self.assertEqual(got_er.shape, expected_er.shape)
        for got, expected in zip(got_er.flatten(), expected_er.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

    def test_architecture_te(self):
        """
        The architecture correctly computes the transfer entropy
        """
        k, timesteps = 5, 20
        arch = Architecture(s_pombe, k=k, timesteps=timesteps)

        expected_te = transfer_entropy(s_pombe, k=k, timesteps=timesteps)
        got_te = arch.transfer_entropy()
        self.assertEqual(got_te.shape, expected_te.shape)
        for got, expected in zip(got_te.flatten(), expected_te.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

        expected_te = transfer_entropy(
            s_pombe, k=k, timesteps=timesteps, local=True)
        got_te = arch.transfer_entropy(local=True)
        self.assertEqual(got_te.shape, expected_te.shape)
        for got, expected in zip(got_te.flatten(), expected_te.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

    def test_architecture_mi(self):
        """
        The architecture correctly computes the mutual information
        """
        k, timesteps = 5, 20
        arch = Architecture(s_pombe, k=k, timesteps=timesteps)

        expected_mi = mutual_information(s_pombe, timesteps=timesteps)
        got_mi = arch.mutual_information()
        self.assertEqual(got_mi.shape, expected_mi.shape)
        for got, expected in zip(got_mi.flatten(), expected_mi.flatten()):
            self.assertAlmostEqual(expected, got, places=6)

        expected_mi = mutual_information(
            s_pombe, timesteps=timesteps, local=True)
        got_mi = arch.mutual_information(local=True)
        self.assertEqual(got_mi.shape, expected_mi.shape)
        for got, expected in zip(got_mi.flatten(), expected_mi.flatten()):
            self.assertAlmostEqual(expected, got, places=6)
