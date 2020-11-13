import numpy as np
import unittest
from neet import Information
from neet.boolean.examples import s_pombe, s_cerevisiae


class TestInformation(unittest.TestCase):
    """
    Test the information analysis module
    """

    def test_canary(self):
        """
        A canary test to ensure the test suite is working
        """
        self.assertEqual(3, 1 + 2)

    def test_invalid_network(self):
        """
        Cannot initialize with an invalid network.
        """
        with self.assertRaises(TypeError):
            Information('net', k=5, timesteps=20)

    def test_invalid_history_length(self):
        """
        Cannot initialize with an invalid history length.
        """
        with self.assertRaises(TypeError):
            Information(s_pombe, k=float(5), timesteps=20)

        with self.assertRaises(ValueError):
            Information(s_pombe, k=0, timesteps=20)

        with self.assertRaises(ValueError):
            Information(s_pombe, k=-1, timesteps=20)

    def test_invalid_timeseries_length(self):
        """
        Cannot initialize with an invalid time series length.
        """
        with self.assertRaises(TypeError):
            Information(s_pombe, k=5, timesteps=float(20))

        with self.assertRaises(ValueError):
            Information(s_pombe, k=5, timesteps=0)

        with self.assertRaises(ValueError):
            Information(s_pombe, k=5, timesteps=-1)

    def test_can_initialize(self):
        """
        Can initialize the network properly.
        """
        arch = Information(s_pombe, k=5, timesteps=20)
        self.assertEqual(arch.net, s_pombe)
        self.assertEqual(arch.k, 5)
        self.assertEqual(arch.timesteps, 20)

    def test_ai(self):
        """
        The architecture correctly computes the active information
        """
        k, timesteps = 5, 20
        arch = Information(s_pombe, k=k, timesteps=timesteps)

        expected_ai = np.asarray([0.0, 0.408344, 0.629567, 0.629567, 0.379157,
                                  0.400462, 0.670196, 0.670196, 0.391891])
        got_ai = arch.active_information()
        self.assertEqual(got_ai.shape, expected_ai.shape)
        self.assertTrue(np.allclose(got_ai, expected_ai, atol=1e-6))

        got_ai = arch.active_information(local=True)
        self.assertEqual((9, 512, 16), got_ai.shape)
        self.assertTrue(np.allclose(np.mean(got_ai, axis=(1, 2)), expected_ai, atol=1e-6))

    def test_er(self):
        """
        The architecture correctly computes the entropy rate
        """
        k, timesteps = 5, 20
        arch = Information(s_pombe, k=k, timesteps=timesteps)

        expected_er = np.asarray([0.0, 0.016912, 0.072803, 0.072803, 0.058420,
                                  0.024794, 0.032173, 0.032173, 0.089669])
        got_er = arch.entropy_rate()
        self.assertEqual(got_er.shape, expected_er.shape)
        self.assertTrue(np.allclose(got_er, expected_er, atol=1e-6))

        got_er = arch.entropy_rate(local=True)
        self.assertEqual((9, 512, 16), got_er.shape)
        self.assertTrue(np.allclose(np.mean(got_er, axis=(1, 2)), expected_er, atol=1e-6))

    def test_te(self):
        """
        The architecture correctly computes the transfer entropy
        """
        k, timesteps = 5, 20
        arch = Information(s_pombe, k=k, timesteps=timesteps)

        expected_te = np.asarray(
            [[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
              0.000000, 0.000000, 0.000000],
             [0.000000, 0.000000, 0.051370, 0.051370, 0.058420, 0.000000,
              0.016690, 0.016690, 0.060304],
             [0.000000, 0.000000, 0.000000, 0.012225, 0.047602, 0.024794,
              0.004526, 0.004526, 0.048289],
             [0.000000, 0.000000, 0.012225, 0.000000, 0.047602, 0.024794,
              0.004526, 0.004526, 0.048289],
             [0.000000, 0.016912, 0.019947, 0.019947, 0.000000, 0.000000,
              0.011916, 0.011916, 0.089669],
             [0.000000, 0.000000, 0.051370, 0.051370, 0.058420, 0.000000,
              0.016690, 0.016690, 0.060304],
             [0.000000, 0.000000, 0.006039, 0.006039, 0.047602, 0.024794,
              0.000000, 0.002983, 0.048927],
             [0.000000, 0.000000, 0.006039, 0.006039, 0.047602, 0.024794,
              0.002983, 0.000000, 0.048927],
             [0.000000, 0.000000, 0.072803, 0.072803, 0.000000, 0.000000,
              0.032173, 0.032173, 0.000000]])

        got_te = arch.transfer_entropy()
        self.assertEqual(got_te.shape, expected_te.shape)
        self.assertTrue(np.allclose(got_te, expected_te, atol=1e-6))

        got_te = arch.transfer_entropy(local=True)
        self.assertEqual((9, 9, 512, 16), got_te.shape)
        self.assertTrue(np.allclose(np.mean(got_te, axis=(2, 3)), expected_te, atol=1e-6))

    def test_mi(self):
        """
        The architecture correctly computes the mutual information
        """
        k, timesteps = 5, 20
        arch = Information(s_pombe, k=k, timesteps=timesteps)

        expected_mi = np.asarray(
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
        got_mi = arch.mutual_information()
        self.assertEqual(got_mi.shape, expected_mi.shape)
        self.assertTrue(np.allclose(got_mi, expected_mi, atol=1e-6))

        got_mi = arch.mutual_information(local=True)
        self.assertEqual((9, 9, 512, 21), got_mi.shape)
        self.assertTrue(np.allclose(np.mean(got_mi, axis=(2, 3)), expected_mi, atol=1e-6))

    def test_set_network(self):
        """
        Changing the network resets the internal state.
        """
        arch = Information(s_pombe, k=5, timesteps=20)
        before_ai = arch.active_information()

        arch.net = s_cerevisiae
        after_ai = arch.active_information()

        self.assertNotEqual(after_ai.shape, before_ai.shape)

        with self.assertRaises(TypeError):
            arch.net = 'net'

    def test_set_history_length(self):
        """
        Changing the history length resets the internal state.
        """
        arch = Information(s_pombe, k=1, timesteps=20)
        before_ai = arch.active_information()

        arch.k = 5
        after_ai = arch.active_information()

        self.assertFalse(np.allclose(after_ai, before_ai))

        with self.assertRaises(TypeError):
            arch.k = float(1)

        with self.assertRaises(ValueError):
            arch.k = 0

        with self.assertRaises(ValueError):
            arch.k = -1

    def test_set_timeseries_length(self):
        """
        Changing the number of time steps resets the internal state.
        """
        arch = Information(s_pombe, k=5, timesteps=10)
        before_ai = arch.active_information()

        arch.timesteps = 20
        after_ai = arch.active_information()

        self.assertFalse(np.allclose(after_ai, before_ai))

        with self.assertRaises(TypeError):
            arch.timesteps = float(1)

        with self.assertRaises(ValueError):
            arch.timesteps = 0

        with self.assertRaises(ValueError):
            arch.timesteps = -1
