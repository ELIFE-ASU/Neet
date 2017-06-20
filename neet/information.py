# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import pyinform as pi
from neet.landscape import timeseries

def active_information(net, k, timesteps, size=None, local=False):
    """
    Compute the active information storage for each node in a network.

    ::

        >>> active_information(s_pombe, k=5, timesteps=20)
        array([ 0.        ,  0.4083436 ,  0.62956679,  0.62956679,  0.37915718,
                0.40046165,  0.67019615,  0.67019615,  0.39189127])
        >>> lais = active_information(s_pombe, k=5, timesteps=20, local=True)
        >>> lais[1]
        array([[ 0.13079175,  0.13079175,  0.13079175, ...,  0.13079175,
                0.13079175,  0.13079175],
            [ 0.13079175,  0.13079175,  0.13079175, ...,  0.13079175,
                0.13079175,  0.13079175],
            [ 0.13079175,  0.13079175,  0.13079175, ...,  0.13079175,
                0.13079175,  0.13079175],
            ...,
            [ 0.13079175,  0.13079175,  0.13079175, ...,  0.13079175,
                0.13079175,  0.13079175],
            [ 0.13079175,  0.13079175,  0.13079175, ...,  0.13079175,
                0.13079175,  0.13079175],
            [ 0.13079175,  0.13079175,  0.13079175, ...,  0.13079175,
                0.13079175,  0.13079175]])
        >>> np.mean(lais[1])
        0.40834359639631324

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
        active_info = np.empty((shape[0], shape[1], shape[2]-k), dtype=np.float)
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

    ::

        >>> entropy_rate(s_pombe, k=5, timesteps=20)
        array([ 0.        ,  0.01691208,  0.07280268,  0.07280268,  0.05841994,
                0.02479402,  0.03217332,  0.03217332,  0.08966941])
        >>> ler = entropy_rate(s_pombe, k=5, timesteps=20, local=True)
        >>> ler[4]
        array([[ 0.        ,  0.        ,  0.        , ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.        ,  0.        , ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.        ,  0.        , ...,  0.00507099,
                0.00507099,  0.00507099],
            ...,
            [ 0.        ,  0.29604946,  0.00507099, ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.29604946,  0.00507099, ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.29604946,  0.00507099, ...,  0.00507099,
                0.00507099,  0.00507099]])
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
        rate = np.empty((shape[0], shape[1], shape[2]-k), dtype=np.float)
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

    ::

        >>> transfer_entropy(s_pombe, k=5, timesteps=20)
        (9, 9)
        array([[  0.00000000e+00,   0.00000000e+00,  -1.11022302e-16,
                -1.11022302e-16,   0.00000000e+00,   0.00000000e+00,
                -1.11022302e-16,   0.00000000e+00,   0.00000000e+00],
            [  4.44089210e-16,   4.44089210e-16,   0.00000000e+00,
                0.00000000e+00,   1.69120759e-02,   4.44089210e-16,
                0.00000000e+00,   0.00000000e+00,   4.44089210e-16],
            [  4.44089210e-16,   5.13704599e-02,   4.44089210e-16,
                1.22248438e-02,   1.99473023e-02,   5.13704599e-02,
                6.03879253e-03,   6.03879253e-03,   7.28026801e-02],
            [  4.44089210e-16,   5.13704599e-02,   1.22248438e-02,
                4.44089210e-16,   1.99473023e-02,   5.13704599e-02,
                6.03879253e-03,   6.03879253e-03,   7.28026801e-02],
            [  0.00000000e+00,   5.84199434e-02,   4.76020591e-02,
                4.76020591e-02,   0.00000000e+00,   5.84199434e-02,
                4.76020591e-02,   4.76020591e-02,   0.00000000e+00],
            [  2.22044605e-16,   2.22044605e-16,   2.47940243e-02,
                2.47940243e-02,   2.22044605e-16,   2.22044605e-16,
                2.47940243e-02,   2.47940243e-02,   2.22044605e-16],
            [ -4.44089210e-16,   1.66898258e-02,   4.52634832e-03,
                4.52634832e-03,   1.19161772e-02,   1.66898258e-02,
                -4.44089210e-16,   2.98276692e-03,   3.21733224e-02],
            [ -4.44089210e-16,   1.66898258e-02,   4.52634832e-03,
                4.52634832e-03,   1.19161772e-02,   1.66898258e-02,
                2.98276692e-03,  -4.44089210e-16,   3.21733224e-02],
            [ -4.44089210e-16,   6.03036989e-02,   4.82889077e-02,
                4.82889077e-02,   8.96694146e-02,   6.03036989e-02,
                4.89270931e-02,   4.89270931e-02,  -4.44089210e-16]])

        >>> lte = transfer_entropy(s_pombe, k=5, timesteps=20, local=True)
        >>> lte[4,3]
        array([[ 0.        ,  0.        ,  0.        , ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.        ,  0.        , ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.        ,  0.        , ...,  0.00507099,
                0.00507099,  0.00507099],
            ...,
            [ 0.        ,  0.29604946,  0.00507099, ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.29604946,  0.00507099, ...,  0.00507099,
                0.00507099,  0.00507099],
            [ 0.        ,  0.29604946,  0.00507099, ...,  0.00507099,
                0.00507099,  0.00507099]])
        >>> np.mean(lte[4,3])
        0.047602059103704124

    :param net: a NEET network
    :param k: the history length
    :param timesteps: the number of timesteps to evaluate the network
    :param size: the size of variable-sized network (or ``None``)
    :param local: whether or not to compute the local transfer entropy
    :returns: a generator of transfer entropy values
    """
    series = timeseries(net, timesteps=timesteps, size=size)
    shape = series.shape
    if local:
        trans_entropy = np.empty((shape[0], shape[0], shape[1], shape[2]-k), dtype=np.float)
        for i in range(shape[0]):
            for j in range(shape[0]):
                trans_entropy[i, j, :, :] = pi.transfer_entropy(series[j], series[i],
                                                                k=k, local=True)
    else:
        trans_entropy = np.empty((shape[0], shape[0]), dtype=np.float)
        print(trans_entropy.shape)
        for i in range(shape[0]):
            for j in range(shape[0]):
                trans_entropy[i, j] = pi.transfer_entropy(series[j], series[i], k=k, local=False)
    return trans_entropy
