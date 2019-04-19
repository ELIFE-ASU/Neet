"""
.. currentmodule:: neet.boolean.network

.. testsetup:: boolean_network

    from neet.automata import ECA
    from neet.boolean.network import *
    from neet.statespace import StateSpace

API Documentation
-----------------
"""
from neet.network import Network
from neet.statespace import UniformSpace


class BooleanNetwork(Network):
    def __init__(self, size, names=None, metadata=None):
        super(BooleanNetwork, self).__init__(size, names, metadata)
        self._state_space = UniformSpace(ndim=size, base=2)

    def state_space(self):
        return self._state_space


Network.register(BooleanNetwork)
