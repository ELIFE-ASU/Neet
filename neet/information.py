# -*- coding: utf-8 -*-
"""
.. currentmodule:: neet

.. testsetup:: information

    import numpy as np
    from neet import Information
    from neet.boolean.examples import s_pombe

The :mod:`neet` provides the :class:`Information` class to compute various information measures over
the dynamics of discrete-state network models.

The core information-theoretic computations are supported by the `PyInform
<https://elife-asu.github.io/PyInform>`_ package.
"""
import numpy as np
import pyinform as pi
from .network import Network


class Information(object):
    """
    A class to represent the :math:`k`-history informational architecture of a network.

    An Information is initialized with a network, a history length, and time series length. A time
    series of the desired length is computed from each initial state of the network, and used
    populate probability distributions over the state transitions of each node. From there any
    number of information or entropy measures may be applied.

    The Information class provides three public attributes:

    .. autosummary::
        :nosignatures:

        net
        k
        timesteps

    During following measures can be computed and cached:

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
    :type net: neet.Network
    :param k: the history length
    :type k: int
    :param timesteps: the number of timesteps to evaluate the network
    :type timesteps: int
    """

    def __init__(self, net, k, timesteps):
        if not isinstance(net, Network):
            raise(TypeError('net argument must be a neet.Network'))

        if not isinstance(k, int):
            raise(TypeError('history length must be an integer'))
        elif k < 1:
            raise(ValueError('history length must be at least 1'))

        if not isinstance(timesteps, int):
            raise(TypeError('timesteps must be an integer'))
        elif timesteps < 1:
            raise(ValueError('timesteps must be at least 1'))

        self.__net = net
        self.__k = k
        self.__timesteps = timesteps

        self.__initialize()

    def __initialize(self):
        """
        Initialize the internal state after parameters are reset.
        """
        self.__series = self.__net.timeseries(timesteps=self.__timesteps)

        self.__local_active_info = None
        self.__active_info = None

        self.__local_entropy_rate = None
        self.__entropy_rate = None

        self.__local_transfer_entropy = None
        self.__transfer_entropy = None

        self.__local_mutual_info = None
        self.__mutual_info = None

    @property
    def net(self):
        """
        The network over which to compute the various information measures

        .. Note::

            The cached internal state of the :class:`Information` instances, namely any pre-computed
            time series and information measures, is cleared when the network is changed.

        :type: neet.Network
        """
        return self.__net

    @net.setter
    def net(self, net):
        if not isinstance(net, Network):
            raise(TypeError('net argument must be a neet.Network'))

        self.__net = net
        self.__initialize()

    @property
    def k(self):
        """
        The history length to use to compute the various information measures

        .. Note::

            The cached internal state of the :class:`Information` instances, namely any pre-computed
            time series and information measures, is cleared when the history length is changed.

        :type: int
        """
        return self.__k

    @k.setter
    def k(self, k):
        if not isinstance(k, int):
            raise(TypeError('history length must be an integer'))
        elif k < 1:
            raise(ValueError('history length must be at least 1'))

        self.__k = k
        self.__initialize()

    @property
    def timesteps(self):
        """
        The time series length to use to compute the various information measures

        .. Note::

            The cached internal state of the :class:`Information` instances, namely any pre-computed
            time series and information measures, is cleared when the number of time steps is
            changed.

        :type: int
        """
        return self.__timesteps

    @timesteps.setter
    def timesteps(self, timesteps):
        if not isinstance(timesteps, int):
            raise(TypeError('timesteps must be an integer'))
        elif timesteps < 1:
            raise(ValueError('timesteps must be at least 1'))

        self.__timesteps = timesteps
        self.__initialize()

    def active_information(self, local=False):
        """
        Get the local or average active information.

        Active information (AI) was introduced in [Lizier2012]_ to quantify information storage in
        distributed computation. AI is defined in terms of a temporally local variant

        .. math::

                a_{X,i}(k) = \\log_2 \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)p(x_{i+1})}

        where the probabilites are constructed emperically from an *entire* time series. From this
        local variant, the temporally global active information is defined as

        .. math::

            A_X(k) = \\langle a_{X,i}(k) \\rangle_{i}
                   = \\sum_{x^{(k)}_i,\\, x_{i+1}} p(x^{(k)}_i, x_{i+1}) \\log_2
                        \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)p(x_{i+1})}.

        .. rubric:: Examples

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

        :param local: whether to return local (True) or global active information
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) active information for every node
                 in the network
        """
        if local:
            if self.__local_active_info is None:
                k = self.__k
                series = self.__series
                shape = series.shape
                local_active_info = np.empty((shape[0], shape[1], shape[2] - k))
                for i in range(shape[0]):
                    local_active_info[i, :, :] = pi.active_info(series[i], k=k, local=True)
                self.__local_active_info = local_active_info

            return self.__local_active_info
        else:
            if self.__active_info is None:
                k = self.__k
                series = self.__series
                shape = series.shape
                active_info = np.empty(shape[0])
                for i in range(shape[0]):
                    active_info[i] = pi.active_info(series[i], k=k)
                self.__active_info = active_info
            return self.__active_info

    def entropy_rate(self, local=False):
        """
        Get the local or average entropy rate.

        Entropy rate quantifies the amount of information need to describe a random variable — the
        state of a node in this case — given observations of its :math:`k`-history. In other words,
        it is the entropy of the time series of a node's state conditioned on its
        :math:`k`-history. The time-local entropy rate

        .. math::

            h_{X,i}(k) = \\log_2 \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)}

        can be averaged to obtain the global entropy rate

        .. math::

            H_X(k) = \\langle h_{X,i}(k) \\rangle_{i}
                   = \\sum_{x^{(k)}_i,\\, x_{i+1}} p(x^{(k)}_i, x_{i+1}) \\log_2
                     \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)}.

        .. rubric:: Examples

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
        :return: a :class:`numpy.ndarray` containing the (local) entropy rate for every node in the
                 network
        """
        if local:
            if self.__local_entropy_rate is None:
                k = self.__k
                series = self.__series
                shape = series.shape
                local_entropy_rate = np.empty((shape[0], shape[1], shape[2] - k))
                for i in range(shape[0]):
                    local_entropy_rate[i, :, :] = pi.entropy_rate(series[i], k=k, local=True)
                self.__local_entropy_rate = local_entropy_rate
            return self.__local_entropy_rate
        else:
            if self.__entropy_rate is None:
                k = self.__k
                series = self.__series
                shape = series.shape
                entropy_rate = np.empty(shape[0])
                for i in range(shape[0]):
                    entropy_rate[i] = pi.entropy_rate(series[i], k=k)
                self.__entropy_rate = entropy_rate
            return self.__entropy_rate

    def transfer_entropy(self, local=False):
        """
        Get the local or average transfer entropy.

        Transfer entropy (TE) was introduced by [Schreiber2000]_ to quantify information transfer
        between an information source and destination, in this case a pair of nodes, condition out
        their shared history effects. TE is defined in terms of a time-local variant

        .. math::

            t_{X \\rightarrow Y, i}(k) = \\log_2 \\frac{p(y_{i+1}, x_i~|~y^{(k)}_i)}
                {p(y_{i+1}~|~y^{(k)}_i)p(x_i~|~y^{(k)}_i)}

        Time averaging defines the global transfer entropy

        .. math::

            T_{Y \\rightarrow X}(k) = \\langle t_{X \\rightarrow Y, i}(k) \\rangle_i

        .. rubric:: Examples

        .. doctest:: information

            >>> arch = Information(s_pombe, k=5, timesteps=20)
            >>> arch.transfer_entropy()
            array([[0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        ],
                   [0.        , 0.        , 0.05137046, 0.05137046, 0.05841994,
                    0.        , 0.01668983, 0.01668983, 0.0603037 ],
                   ...,
                   [0.        , 0.        , 0.00603879, 0.00603879, 0.04760206,
                    0.02479402, 0.00298277, 0.        , 0.04892709],
                   [0.        , 0.        , 0.07280268, 0.07280268, 0.        ,
                    0.        , 0.03217332, 0.03217332, 0.        ]])

            >>> lte = arch.transfer_entropy(local=True)
            >>> lte[4,3]
            array([[-1.03562391,  1.77173101,  0.        , ...,  0.        ,
                     0.        ,  0.        ],
                   [-1.03562391,  1.77173101,  0.        , ...,  0.        ,
                     0.        ,  0.        ],
                   [ 1.77173101,  0.        ,  0.        , ...,  0.        ,
                     0.        ,  0.        ],
                   ...,
                   [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                     0.        ,  0.        ],
                   [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                     0.        ,  0.        ],
                   [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                     0.        ,  0.        ]])

        The first and second indices of the resulting arrays are the source and target nodes,
        respectively.

        :param local: whether to return local (True) or global transfer entropy
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) transfer entropy for every pair of
                 nodes in the network
        """
        if local:
            if self.__local_transfer_entropy is None:
                k = self.__k
                series = self.__series
                shape = series.shape
                local_transfer_entropy = np.empty((shape[0], shape[0], shape[1], shape[2] - k))
                for i in range(shape[0]):
                    for j in range(shape[0]):
                        local_transfer_entropy[i, j, :, :] = pi.transfer_entropy(series[i],
                                                                                 series[j], k=k, local=True)
                self.__local_transfer_entropy = local_transfer_entropy
            return self.__local_transfer_entropy
        else:
            if self.__transfer_entropy is None:
                k = self.__k
                series = self.__series
                shape = series.shape
                transfer_entropy = np.empty((shape[0], shape[0]))
                for i in range(shape[0]):
                    for j in range(shape[0]):
                        transfer_entropy[i, j] = pi.transfer_entropy(series[i], series[j], k=k)
                self.__transfer_entropy = transfer_entropy
            return self.__transfer_entropy

    def mutual_information(self, local=False):
        """
        Get the local or average mutual information.

        Mutual information is a measure of the amount of mutual dependence (correlation) between two
        random variables — nodes in this case. The time-local mutual information

        .. math::

            i_{i}(X,Y) = -\\log_2 \\frac{p(x_i, y_i)}{p(x_i)p(y_i)}

        can be time-averaged to define the standard mutual information

        .. math::

            I(X,Y) = -\\sum_{x_i, y_i} p(x_i, y_i) \\log_2 \\frac{p(x_i, y_i)}{p(x_i)p(y_i)}.

        .. rubric:: Examples

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

        :param local: whether to return local (True) or global mutual information
        :type local: bool
        :return: a :class:`numpy.ndarray` containing the (local) mutual information for every pair
                 of nodes in the network
        """
        if local:
            if self.__local_mutual_info is None:
                series = self.__series
                shape = series.shape
                local_mutual_info = np.empty((shape[0], shape[0], shape[1], shape[2]))
                for i in range(shape[0]):
                    for j in range(i, shape[0]):
                        local_mutual_info[i, j, :, :] = pi.mutual_info(series[i], series[j],
                                                                       local=True)
                        local_mutual_info[j, i, :, :] = local_mutual_info[i, j, :, :]
                self.__local_mutual_info = local_mutual_info
            return self.__local_mutual_info
        else:
            if self.__mutual_info is None:
                series = self.__series
                shape = series.shape
                mutual_info = np.empty((shape[0], shape[0]))
                for i in range(shape[0]):
                    for j in range(i, shape[0]):
                        mutual_info[i, j] = pi.mutual_info(series[i], series[j])
                        mutual_info[j, i] = mutual_info[i, j]
                self.__mutual_info = mutual_info
            return self.__mutual_info
