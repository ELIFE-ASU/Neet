"""
.. currentmodule:: neet.information

.. testsetup:: information

    import numpy as np
    from neet.information import *
    from neet.boolean.examples import s_pombe

The :mod:`neet.information` provides the :class:`Information` class to compute
various information measures over the dynamics of discrete-state network
models. 

The core information-theoretic computations are supported by the `PyInform
<https://elife-asu.github.io/PyInform>`_ package.
"""
import numpy as np
import pyinform as pi


class Information(object):
    """
    A class to represent the :math:`k`-history informational architecture of a
    network.

    An Information is initialized with a network, a history length, and time
    series length. A time series of the desired length is computed from each
    initial state of the network, and used populate probability distributions
    over the state transitions of each node. From there any number of
    information or entropy measures may be applied.

    During initialization the following measures are computed and cached:

    .. autosummary::
        :nosignatures:

        active_information
        entropy_rate
        mutual_information
        transfer_entropy

    .. rubric:: Examples

    .. doctest:: information

        >>> arch = Information(s_pombe, k=5, timesteps=20)
        >>> arch.active_information()
        array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
               0.40046165, 0.67019615, 0.67019615, 0.39189127])

    :param net: the network to analyze
    :type net: neet.network.Network
    :param k: the history length
    :type k: int
    :param timesteps: the number of timesteps to evaluate the network
    :type timesteps: int
    """

    def __init__(self, net, k, timesteps):
        self.__k = k
        self.__series = net.timeseries(timesteps=timesteps)
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
        Get the local or average active information.

        .. doctest:: information

            >>> arch = Information(s_pombe, k=5, timesteps=20)
            >>> arch.active_information()
            array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
                   0.40046165, 0.67019615, 0.67019615, 0.39189127])
            >>> lais = arch.active_information(local=True)
            >>> lais[1]
            array([[0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   ...,
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175],
                   [0.13079175, 0.13079175, 0.13079175, ..., 0.13079175, 0.13079175,
                    0.13079175]])
            >>> np.mean(lais[1])
            0.4083435...

        :param local: whether to return local (True) or global active
                      information
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) active
                 information
        """
        if local:
            return self.__local_active_info
        else:
            return self.__active_info

    def entropy_rate(self, local=False):
        """
        Get the local or average entropy rate.

        .. doctest:: information

            >>> arch = Information(s_pombe, k=5, timesteps=20)
            >>> arch.entropy_rate()
            array([0.        , 0.01691208, 0.07280268, 0.07280268, 0.05841994,
                   0.02479402, 0.03217332, 0.03217332, 0.08966941])
            >>> ler = arch.entropy_rate(local=True)
            >>> ler[4]
            array([[0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   ...,
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099]])

        :param local: whether to return local (True) or global entropy rate
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) entropy rate
        """
        if local:
            return self.__local_entropy_rate
        else:
            return self.__entropy_rate

    def transfer_entropy(self, local=False):
        """
        Get the local or average transfer entropy.

        .. doctest:: information

            >>> arch = Information(s_pombe, k=5, timesteps=20)
            >>> arch.transfer_entropy()
            array([[0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.01691208,
                    0.        , 0.        , 0.        , 0.        ],
                   ...,
                   [0.        , 0.01668983, 0.00452635, 0.00452635, 0.01191618,
                    0.01668983, 0.00298277, 0.        , 0.03217332],
                   [0.        , 0.0603037 , 0.04828891, 0.04828891, 0.08966941,
                    0.0603037 , 0.04892709, 0.04892709, 0.        ]])

            >>> lte = arch.transfer_entropy(local=True)
            >>> lte[4,3]
            array([[0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.        , 0.        , ..., 0.00507099, 0.00507099,
                    0.00507099],
                   ...,
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099],
                   [0.        , 0.29604946, 0.00507099, ..., 0.00507099, 0.00507099,
                    0.00507099]])

        :param local: whether to return local (True) or global transfer entropy
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) transfer
                 entropy
        """
        if local:
            return self.__local_transfer_entropy
        else:
            return self.__transfer_entropy

    def mutual_information(self, local=False):
        """
        Get the local or average mutual information.

        .. doctest:: information

            >>> arch = Information(s_pombe, k=5, timesteps=20)
            >>> arch.mutual_information()
            array([[0.16232618, 0.01374672, 0.00428548, 0.00428548, 0.01340937,
                    0.01586238, 0.00516987, 0.00516987, 0.01102766],
                   [0.01374672, 0.56660996, 0.00745714, 0.00745714, 0.00639113,
                    0.32790848, 0.0067609 , 0.0067609 , 0.00468342],
                   ...,
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
                   ...,
                   [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                     0.18484073,  0.18484073],
                   [-2.89794147,  1.7513014 ,  0.18484073, ...,  0.18484073,
                     0.18484073,  0.18484073]])

        :param local: whether to return local (True) or global mutual
                      information
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) mutual
                 information
        """
        if local:
            return self.__local_mutual_info
        else:
            return self.__mutual_info
