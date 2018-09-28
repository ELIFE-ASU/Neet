"""
.. currentmodule:: neet.information

.. testsetup:: information

    import numpy as np
    from neet.information import *
    from neet.boolean.examples import s_pombe

Information Architecture
========================

The :mod:`neet.information` provides a collection of functions which compute
various information measures over the dynamics of discrete-state network
models. Each measure computes a time series of a desired length from each
initial state of the network and uses the time series to populate probability
distributions over the state transitions of each node. From there any number
of information or entropy measures may be applied. All of this is bookkeeping
is taken care of by the function; all you have to do is provide a network and
any necessary parameters for the calculation


.. rubric:: Example (Active Information for Fission Yeast)

.. doctest:: information

    >>> active_information(s_pombe, k=5, timesteps=20)
    array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
           0.40046165, 0.67019615, 0.67019615, 0.39189127])

The core information-theoretic computations are supported by the `PyInform
<https://elife-asu.github.io/PyInform>`_ package.


API Documentation
-----------------

The :mod:`neet.information` module is broken into two parts: `Information
Measures`_ and the `Architecture Class`_.

The `Information Measures`_ are a collection of module-level functions which
independently compute a variety of information measures over a network of
interest:

- Active Information Storage (:func:`active_information`)
- Entropy Rate (:func:`entropy_rate`)
- Transfer Entropy (:func:`transfer_entropy`)
- Mutual Information (:func:`mutual_information`).

However, since they do not share any data between them, if you plan to compute
many different measures, it is much more efficient to cache the computed time
series. This is done by the :class:`Architecture` class which provides all of
the same measures, only computes the time series once.

Information Measures
^^^^^^^^^^^^^^^^^^^^
"""
import numpy as np
import pyinform as pi
from neet.synchronous import timeseries


def active_information(net, k, timesteps, size=None, local=False):
    """
    Compute the active information storage for each node in a network.

    .. doctest:: information

        >>> active_information(s_pombe, k=5, timesteps=20)
        array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
               0.40046165, 0.67019615, 0.67019615, 0.39189127])
        >>> lais = active_information(s_pombe, k=5, timesteps=20, local=True)
        >>> lais[1]
        array([[0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                0.13079175],
               [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                0.13079175],
               [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                0.13079175],
               ...,
               [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                0.13079175],
               [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                0.13079175],
               [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                0.13079175]])
        >>> np.mean(lais[1])
        0.4083435963963132

    :param net: a NEET network
    :param k: the history length
    :param timesteps: the number of timesteps to evaluate the network
    :param size: the size of variable-sized network (or ``None``)
    :param local: whether or not to compute the local active information
    :returns: a numpy array of active information values
    """
    series = timeseries(net, timesteps=timesteps, size=size)
    shape = series.shape
    if local:
        active_info = np.empty(
            (shape[0], shape[1], shape[2] - k), dtype=np.float)
        for i in range(shape[0]):
            active_info[i, :, :] = pi.active_info(series[i], k=k, local=local)
    else:
        active_info = np.empty(shape[0], dtype=np.float)
        for i in range(shape[0]):
            active_info[i] = pi.active_info(series[i], k=k, local=local)
    return active_info


def entropy_rate(net, k, timesteps, size=None, local=False):
    """
    Compute the entropy rate for each node in a network.

    .. doctest:: information

        >>> entropy_rate(s_pombe, k=5, timesteps=20)
        array([0.        , 0.01691208, 0.07280268, 0.07280268, 0.05841994,
               0.02479402, 0.03217332, 0.03217332, 0.08966941])
        >>> ler = entropy_rate(s_pombe, k=5, timesteps=20, local=True)
        >>> ler[4]
        array([[0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                0.00507099],
               ...,
               [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                0.00507099]])
        >>> np.mean(ler[4])
        0.0584199434476326

    :param net: a NEET network
    :param k: the history length
    :param timesteps: the number of timesteps to evaluate the network
    :param size: the size of variable-sized network (or ``None``)
    :param local: whether or not to compute the local entropy rate
    :returns: a numpy array of entropy rate values
    """
    series = timeseries(net, timesteps=timesteps, size=size)
    shape = series.shape
    if local:
        rate = np.empty((shape[0], shape[1], shape[2] - k), dtype=np.float)
        for i in range(shape[0]):
            rate[i, :, :] = pi.entropy_rate(series[i], k=k, local=local)
    else:
        rate = np.empty(shape[0], dtype=np.float)
        for i in range(shape[0]):
            rate[i] = pi.entropy_rate(series[i], k=k, local=local)
    return rate


def transfer_entropy(net, k, timesteps, size=None, local=False):
    """
    Compute the transfer entropy matrix for a network.

    .. doctest:: information

        >>> transfer_entropy(s_pombe, k=5, timesteps=20)
        array([[ 0.00000000e+00,  0.00000000e+00, -1.11022302e-16,
                -1.11022302e-16,  0.00000000e+00,  0.00000000e+00,
                -1.11022302e-16,  0.00000000e+00,  0.00000000e+00],
               [-4.44089210e-16, -4.44089210e-16, -4.44089210e-16,
                -4.44089210e-16,  1.69120759e-02, -4.44089210e-16,
                -4.44089210e-16, -4.44089210e-16, -4.44089210e-16],
               [ 4.44089210e-16,  5.13704599e-02,  4.44089210e-16,
                 1.22248438e-02,  1.99473023e-02,  5.13704599e-02,
                 6.03879253e-03,  6.03879253e-03,  7.28026801e-02],
               [ 4.44089210e-16,  5.13704599e-02,  1.22248438e-02,
                 4.44089210e-16,  1.99473023e-02,  5.13704599e-02,
                 6.03879253e-03,  6.03879253e-03,  7.28026801e-02],
               [ 0.00000000e+00,  5.84199434e-02,  4.76020591e-02,
                 4.76020591e-02,  0.00000000e+00,  5.84199434e-02,
                 4.76020591e-02,  4.76020591e-02,  0.00000000e+00],
               [ 2.22044605e-16,  2.22044605e-16,  2.47940243e-02,
                 2.47940243e-02,  2.22044605e-16,  2.22044605e-16,
                 2.47940243e-02,  2.47940243e-02,  2.22044605e-16],
               [-4.44089210e-16,  1.66898258e-02,  4.52634832e-03,
                 4.52634832e-03,  1.19161772e-02,  1.66898258e-02,
                -4.44089210e-16,  2.98276692e-03,  3.21733224e-02],
               [-4.44089210e-16,  1.66898258e-02,  4.52634832e-03,
                 4.52634832e-03,  1.19161772e-02,  1.66898258e-02,
                 2.98276692e-03, -4.44089210e-16,  3.21733224e-02],
               [-4.44089210e-16,  6.03036989e-02,  4.82889077e-02,
                 4.82889077e-02,  8.96694146e-02,  6.03036989e-02,
                 4.89270931e-02,  4.89270931e-02, -4.44089210e-16]])

        >>> lte = transfer_entropy(s_pombe, k=5, timesteps=20, local=True)
        >>> lte[4,3]
        array([[0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                0.00507099],
               ...,
               [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                0.00507099],
               [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                0.00507099]])
        >>> np.mean(lte[4,3])
        0.047602059103704124

    :param net: a NEET network
    :param k: the history length
    :param timesteps: the number of timesteps to evaluate the network
    :param size: the size of variable-sized network (or ``None``)
    :param local: whether or not to compute the local transfer entropy
    :returns: a numpy matrix of transfer entropy values
    """
    series = timeseries(net, timesteps=timesteps, size=size)
    shape = series.shape
    if local:
        trans_entropy = np.empty(
            (shape[0], shape[0], shape[1], shape[2] - k), dtype=np.float)
        for i in range(shape[0]):
            for j in range(shape[0]):
                te = pi.transfer_entropy(series[j], series[i], k=k, local=True)
                trans_entropy[i, j, :, :] = te
    else:
        trans_entropy = np.empty((shape[0], shape[0]), dtype=np.float)
        for i in range(shape[0]):
            for j in range(shape[0]):
                trans_entropy[i, j] = pi.transfer_entropy(series[j], series[i],
                                                          k=k, local=False)
    return trans_entropy


def mutual_information(net, timesteps, size=None, local=False):
    """
    Compute the mutual information matrix for a network.

    .. doctest:: information

        >>> mutual_information(s_pombe, timesteps=20)
        array([[0.16232618, 0.01374672, 0.00428548, 0.00428548, 0.01340937,
                0.01586238, 0.00516987, 0.00516987, 0.01102766],
               [0.01374672, 0.56660996, 0.00745714, 0.00745714, 0.00639113,
                0.32790848, 0.0067609 , 0.0067609 , 0.00468342],
               [0.00428548, 0.00745714, 0.83837294, 0.475582  , 0.21157695,
                0.00432855, 0.4590254 , 0.4590254 , 0.12755745],
               [0.00428548, 0.00745714, 0.475582  , 0.83837294, 0.21157695,
                0.00432855, 0.4590254 , 0.4590254 , 0.12755745],
               [0.01340937, 0.00639113, 0.21157695, 0.21157695, 0.57459066,
                0.00703145, 0.17560769, 0.17560769, 0.01233356],
               [0.01586238, 0.32790848, 0.00432855, 0.00432855, 0.00703145,
                0.51905053, 0.00621124, 0.00621124, 0.00260667],
               [0.00516987, 0.0067609 , 0.4590254 , 0.4590254 , 0.17560769,
                0.00621124, 0.80831657, 0.49349527, 0.10390475],
               [0.00516987, 0.0067609 , 0.4590254 , 0.4590254 , 0.17560769,
                0.00621124, 0.49349527, 0.80831657, 0.10390475],
               [0.01102766, 0.00468342, 0.12755745, 0.12755745, 0.01233356,
                0.00260667, 0.10390475, 0.10390475, 0.63423835]])
        >>> lmi = mutual_information(s_pombe, timesteps=20, local=True)
        >>> lmi[4,3]
        array([[-0.67489772, -0.67489772, -0.67489772, ...,  0.18484073,
                 0.18484073,  0.18484073],
               [-0.67489772, -0.67489772, -0.67489772, ...,  0.18484073,
                 0.18484073,  0.18484073],
               [-0.67489772, -0.67489772, -0.67489772, ...,  0.18484073,
                 0.18484073,  0.18484073],
               ...,
               [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                 0.18484073,  0.18484073],
               [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                 0.18484073,  0.18484073],
               [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                 0.18484073,  0.18484073]])
        >>> np.mean(lmi[4,3])
        0.21157695279993294

    :param net: a NEET network
    :param k: the history length
    :param timesteps: the number of timesteps to evaluate the network
    :param size: the size of variable-sized network (or ``None``)
    :param local: whether or not to compute the local mutual information
    :returns: a numpy matrix of mutual information values
    """
    series = timeseries(net, timesteps=timesteps, size=size)
    shape = series.shape
    if local:
        mutual_info = np.empty(
            (shape[0], shape[0], shape[1], shape[2]), dtype=np.float)
        for i in range(shape[0]):
            for j in range(shape[0]):
                mutual_info[i, j, :, :] = pi.mutual_info(
                    series[j], series[i], local=True)
    else:
        mutual_info = np.empty((shape[0], shape[0]), dtype=np.float)
        for i in range(shape[0]):
            for j in range(shape[0]):
                mutual_info[i, j] = pi.mutual_info(
                    series[j], series[i], local=False)
    return mutual_info


class Architecture(object):
    """
    A class to represent the k-history informational architecture of a network.

    .. note::
        The class:`Architecture` computes the average information measures a
        bit differently than the associated module-level functions.
        Specifically, it first computes the local measures and then averages
        them; this leads different numerical results on account of the subtlties
        of floating-point mathematics. However, they will always be "close". In
        particular

        .. doctest:: information

              >>> arch = Architecture(s_pombe, k=5, timesteps=20)
              >>> function = transfer_entropy(s_pombe, k=5, timesteps=20)
              >>> method = arch.transfer_entropy()
              >>> np.testing.assert_almost_equal(method, function)

    """

    def __init__(self, net, k, timesteps, size=None):
        """
        Initialize the architecture given a network and enough information to
        compute a time series.

        During initialization the following measures are computed and cached:
        * Local and Average Active Information Storage
        * Local and Average Entropy Rate
        * Local and Average Transfer Entropy
        * Local and Average Mutual Information

        .. doctest:: information

            >>> arch = Architecture(s_pombe, k=5, timesteps=20)
            >>> arch.active_information()
            array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
                   0.40046165, 0.67019615, 0.67019615, 0.39189127])

        :param net: a NEET network
        :param k: the history length
        :param timesteps: the number of timesteps to evaluate the network
        :param size: the size of variable-sized network (or ``None``)
        """
        self.__k = k
        self.__series = timeseries(net, timesteps=timesteps, size=size)
        shape = self.__series.shape

        self.__local_active_info = np.empty((shape[0], shape[1], shape[2] - k))
        self.__local_entropy_rate = np.empty(
            (shape[0], shape[1], shape[2] - k))
        self.__local_transfer_entropy = np.empty(
            (shape[0], shape[0], shape[1], shape[2] - k))
        self.__local_mutual_info = np.empty(
            (shape[0], shape[0], shape[1], shape[2]))

        self.__active_info = np.empty(shape[0])
        self.__entropy_rate = np.empty(shape[0])
        self.__transfer_entropy = np.empty((shape[0], shape[0]))
        self.__mutual_info = np.empty((shape[0], shape[0]))

        self.__initialize()

    def __initialize(self):
        """
        Initialize the internal variables storing the computed information
        measures.
        """
        k = self.__k
        series = self.__series
        nodes = self.__series.shape[0]

        local_active_info = self.__local_active_info
        active_info = self.__active_info
        for i in range(nodes):
            local_active_info[i, :, :] = pi.active_info(
                series[i], k=k, local=True)
            active_info[i] = np.mean(local_active_info[i, :, :])

        local_entropy_rate = self.__local_entropy_rate
        ent_rate = self.__entropy_rate
        for i in range(nodes):
            local_entropy_rate[i, :, :] = pi.entropy_rate(
                series[i], k=k, local=True)
            ent_rate[i] = np.mean(local_entropy_rate[i, :, :])

        local_transfer_entropy = self.__local_transfer_entropy
        trans_entropy = self.__transfer_entropy
        for i in range(nodes):
            for j in range(nodes):
                te = pi.transfer_entropy(series[j], series[i], k=k, local=True)
                local_transfer_entropy[i, j, :, :] = te
                trans_entropy[i, j] = np.mean(te)

        local_mutual_info = self.__local_mutual_info
        mutual_info = self.__mutual_info
        for i in range(nodes):
            for j in range(nodes):
                local_mutual_info[i, j, :, :] = pi.mutual_info(
                    series[j], series[i], local=True)
                mutual_info[i, j] = np.mean(local_mutual_info[i, j, :, :])

    def active_information(self, local=False):
        """
        Get the local or average active information

        .. doctest:: information

            >>> arch = Architecture(s_pombe, k=5, timesteps=20)
            >>> arch.active_information()
            array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
                   0.40046165, 0.67019615, 0.67019615, 0.39189127])
            >>> lais = arch.active_information(local=True)
            >>> lais[1]
            array([[0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   ...,
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175]])
            >>> np.mean(lais[1])
            0.4083435963963132

        :param local: whether to return local (True) or global active information
        :type local: bool
        :return: a numpy array containing the (local) active information
        """
        if local:
            return self.__local_active_info
        else:
            return self.__active_info

    def entropy_rate(self, local=False):
        """
        Get the local or average entropy rate

        .. doctest:: information

            >>> arch = Architecture(s_pombe, k=5, timesteps=20)
            >>> arch.entropy_rate()
            array([0.        , 0.01691208, 0.07280268, 0.07280268, 0.05841994,
                   0.02479402, 0.03217332, 0.03217332, 0.08966941])
            >>> ler = arch.entropy_rate(local=True)
            >>> ler[4]
            array([[0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   ...,
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099]])

        :param local: whether to return local (True) or global entropy rate
        :type local: bool
        :return: a numpy array containing the (local) entropy rate
        """
        if local:
            return self.__local_entropy_rate
        else:
            return self.__entropy_rate

    def transfer_entropy(self, local=False):
        """
        Get the local or average transfer entropy

        .. doctest:: information

            >>> arch = Architecture(s_pombe, k=5, timesteps=20)
            >>> arch.transfer_entropy()
            array([[ 0.00000000e+00,  3.37457926e-18, -2.18195687e-18,
                    -2.18195687e-18, -5.39390581e-18,  3.37457926e-18,
                     5.10930274e-18,  5.10930274e-18,  1.80248611e-18],
                   [-3.25260652e-18, -3.25260652e-18, -3.05745013e-17,
                    -3.05745013e-17,  1.69120759e-02, -3.25260652e-18,
                    -3.25260652e-18, -3.25260652e-18, -3.25260652e-18],
                   [-4.20670443e-17,  5.13704599e-02, -4.20670443e-17,
                     1.22248438e-02,  1.99473023e-02,  5.13704599e-02,
                     6.03879253e-03,  6.03879253e-03,  7.28026801e-02],
                   [-4.20670443e-17,  5.13704599e-02,  1.22248438e-02,
                    -4.20670443e-17,  1.99473023e-02,  5.13704599e-02,
                     6.03879253e-03,  6.03879253e-03,  7.28026801e-02],
                   [-1.09531524e-16,  5.84199434e-02,  4.76020591e-02,
                     4.76020591e-02, -1.09531524e-16,  5.84199434e-02,
                     4.76020591e-02,  4.76020591e-02, -1.09531524e-16],
                   [-5.20417043e-18, -5.20417043e-18,  2.47940243e-02,
                     2.47940243e-02, -5.20417043e-18, -5.20417043e-18,
                     2.47940243e-02,  2.47940243e-02, -5.20417043e-18],
                   [ 3.08997619e-17,  1.66898258e-02,  4.52634832e-03,
                     4.52634832e-03,  1.19161772e-02,  1.66898258e-02,
                     3.08997619e-17,  2.98276692e-03,  3.21733224e-02],
                   [ 3.08997619e-17,  1.66898258e-02,  4.52634832e-03,
                     4.52634832e-03,  1.19161772e-02,  1.66898258e-02,
                     2.98276692e-03,  3.08997619e-17,  3.21733224e-02],
                   [ 1.37693676e-17,  6.03036989e-02,  4.82889077e-02,
                     4.82889077e-02,  8.96694146e-02,  6.03036989e-02,
                     4.89270931e-02,  4.89270931e-02,  1.37693676e-17]])

            >>> lte = arch.transfer_entropy(local=True)
            >>> lte[4,3]
            array([[0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   ...,
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099]])

        :param local: whether to return local (True) or global transfer entropy
        :type local: bool
        :return: a numpy array containing the (local) transfer entropy
        """
        if local:
            return self.__local_transfer_entropy
        else:
            return self.__transfer_entropy

    def mutual_information(self, local=False):
        """
        Get the local or average mutual information

        .. doctest:: information

            >>> arch = Architecture(s_pombe, k=5, timesteps=20)
            >>> arch.mutual_information()
            array([[0.16232618, 0.01374672, 0.00428548, 0.00428548, 0.01340937,
                    0.01586238, 0.00516987, 0.00516987, 0.01102766],
                   [0.01374672, 0.56660996, 0.00745714, 0.00745714, 0.00639113,
                    0.32790848, 0.0067609 , 0.0067609 , 0.00468342],
                   [0.00428548, 0.00745714, 0.83837294, 0.475582  , 0.21157695,
                    0.00432855, 0.4590254 , 0.4590254 , 0.12755745],
                   [0.00428548, 0.00745714, 0.475582  , 0.83837294, 0.21157695,
                    0.00432855, 0.4590254 , 0.4590254 , 0.12755745],
                   [0.01340937, 0.00639113, 0.21157695, 0.21157695, 0.57459066,
                    0.00703145, 0.17560769, 0.17560769, 0.01233356],
                   [0.01586238, 0.32790848, 0.00432855, 0.00432855, 0.00703145,
                    0.51905053, 0.00621124, 0.00621124, 0.00260667],
                   [0.00516987, 0.0067609 , 0.4590254 , 0.4590254 , 0.17560769,
                    0.00621124, 0.80831657, 0.49349527, 0.10390475],
                   [0.00516987, 0.0067609 , 0.4590254 , 0.4590254 , 0.17560769,
                    0.00621124, 0.49349527, 0.80831657, 0.10390475],
                   [0.01102766, 0.00468342, 0.12755745, 0.12755745, 0.01233356,
                    0.00260667, 0.10390475, 0.10390475, 0.63423835]])
            >>> lmi = arch.mutual_information(local=True)
            >>> lmi[4,3]
            array([[-0.67489772, -0.67489772, -0.67489772, ...,  0.18484073,
                     0.18484073,  0.18484073],
                   [-0.67489772, -0.67489772, -0.67489772, ...,  0.18484073,
                     0.18484073,  0.18484073],
                   [-0.67489772, -0.67489772, -0.67489772, ...,  0.18484073,
                     0.18484073,  0.18484073],
                   ...,
                   [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                     0.18484073,  0.18484073],
                   [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                     0.18484073,  0.18484073],
                   [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                     0.18484073,  0.18484073]])

        :param local: whether to return local (True) or global mutual information
        :type local: bool
        :return: a numpy array containing the (local) mutual information
        """
        if local:
            return self.__local_mutual_info
        else:
            return self.__mutual_info
