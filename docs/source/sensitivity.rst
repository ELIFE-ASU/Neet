.. currentmodule:: neet.boolean

.. testsetup:: introduction

      from neet.boolean.examples import s_pombe

.. _sensitivity:

Sensitivity Analysis
====================

Neet provides an API for computing various measures of sensitivity on Networks via the
:class:`SensitivityMixin`. Sensitivity, in its simplest form, is a measure of how small
perturbations of the network's state change under the dynamics. In the sensitivity parlance, a
network is called, *sub-critical*, *critical*, or *chaotic* if the perturbation tends to shrink,
stay the same, or grow over time.

.. Note::

   As of the v1.0.0 release, only the :mod:`neet.boolean` module provides implementations of the
   sensitivity interface. A subsequent release will generalize this mixin to support a wider range
   of network models.

Boolean Sensitivity
-------------------

The standard definition of `sensitivity
<api/boolean/sensitivity.html#neet.boolean.SensitivityMixin.sensitivity>`__ at a given state of a
Boolean network is defined in terms of the Hamming distance:

.. math::

      D_H(x,y) = \sum_{i} x_i \oplus y_i.

That is, the number of bits differing between two binary states, :math:`x` and :math:`y`. A Hamming
neighbor of a state :math:`x` is a state that differs from it by exactly :math:`1` bit. We can write
:math:`x \oplus e_i` to represent the Hamming neighbor of :math:`x` which differs in the
:math:`i`-th bit. The sensitivity of the state :math:`x` is then defined as

.. math::

      s_f(x) = \frac{1}{N} \sum_{i = 1}^N D_H(f(x), f(x \oplus e_i))

where :math:`f` is the network's update function, and :math:`N` is the number of nodes in the
network.

Neet makes computing sensitivity at a given network state as straightforward as possible:

.. doctest:: introduction

      >>> s_pombe.sensitivity([0, 0, 0, 0, 0, 0, 0, 0 ,0])
      1.5555555555555556

More often than not, though, you'll want to compute the average of the sensitivity over all of the
states of the network. That is

.. math::

      s_f = \frac{1}{2^N} \sum_{x} s_f(x).

In Neet, just ask for it

.. doctest:: introduction

      >>> s_pombe.average_sensitivity()
      0.9513888888888888

For a full range of sensitivity-related features offered by Neet, see the `API References
<api/boolean/sensitivity.html>`_.
