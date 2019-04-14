"""
.. currentmodule:: neet.automata

.. testsetup:: automata

    from neet.automata import *
    from neet.synchronous import transitions

Cellular Automata
=================

Cellular automata are a special type of network. They can be envisioned
at a boolean networks wherein every node has exactly 3 incoming edges. The
:mod:`neet.automata` modules provides two submodules, :mod:`neet.automata.eca`
and :mod:`neet.automata.reca`. As a convenience, the classes in each are
exposed in the :mod:`neet.automata`, so you never have to reference the
submodules unless you so choose.

API Documentation
-----------------
"""
from .eca import ECA # noqa
from .reca import RewiredECA # noqa
