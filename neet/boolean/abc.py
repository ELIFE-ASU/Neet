"""
.. currentmodule:: neet.boolean.abc

.. testsetup:: boolean_abc

    from neet.automata import ECA
    from neet.boolean.abc import *
    from neet.statespace import StateSpace

API Documentation
-----------------
"""
from neet.abc import Network
from neet.statespace import StateSpace


class BooleanNetwork(Network):
    def __init__(self, size, names=None, metadata=None):
        super(BooleanNetwork, self).__init__(size, names, metadata)
        self._state_space = StateSpace(self.size, base=2)

    def state_space(self):
        return self._state_space


Network.register(BooleanNetwork)
