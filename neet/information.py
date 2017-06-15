# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import pyinform as pi
from neet.landscape import timeseries

def active_information(net, k, timesteps, size=None, local=False):
    """
    Compute the active information storage for each node in a network.

    ::

        >>> active_information(s_pombe, k=5, timesteps=20)
        <map object at 0x000001E3DC4096D8>
        >>> list(active_information(s_pombe, k=5, timesteps=20))
        [0.0, 0.4083435963963131, 0.629566787877351, 0.629566787877351,
        0.37915718072043925, 0.4004616479864467, 0.6701961455359506,
        0.6701961455359506, 0.3918912655219793]
        >>> lais = list(active_information(s_pombe, k=5, timesteps=20, local=True))
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
    :returns: a generator of active information values
    """
    series = timeseries(net, timesteps=timesteps, size=size)
    return map(lambda xs: pi.active_info(xs, k=k, local=local), series)
