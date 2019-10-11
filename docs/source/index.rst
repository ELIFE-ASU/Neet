Neet: Simulating and analyzing network models
=============================================

**Neet** is a python package designed to provide an easy-to-use API for creating
and evaluating network models. In its current state, **Neet** supports
simulating synchronous Boolean network models, though the API is designed to be
model generic. Future work will implement asynchronous update mechanisms and
more general network types.

If you are interested in using **Neet**, you'll definitely be interested in
checking out the documentation - https://elife-asu.github.io/Neet.

.. image:: https://travis-ci.org/ELIFE-ASU/Neet.svg?branch=master
    :alt: Build Status (Travis CI)
    :target: https://travis-ci.org/ELIFE-ASU/Neet

.. image:: https://ci.appveyor.com/api/projects/status/eyrn6l2wygeglnx5/branch/master?svg=true
    :alt: Build Status (Appveyor)
    :target: https://ci.appveyor.com/project/dglmoore/neet-awnxe/branch/master

.. image:: https://codecov.io/gh/elife-asu/neet/branch/master/graph/badge.svg
    :alt: Code Coverage (Codecov)
    :target: https://codecov.io/gh/elife-asu/neet

Examples
--------

**Neet** provides a hierarchy of network classes with methods designed to make
common tasks as painless as possible. For example, you can read in a collection
of boolean logic equations and immediately probe the dynamics of the network,
and compute values such as the :attr:`neet.LandscapeMixin.attractors` and the
:meth:`neet.boolean.SensitivityMixin.average_sensitivity` of the network

.. doctest::

   >>> from neet.boolean import LogicNetwork
   >>> from neet.boolean.examples import MYELOID_LOGIC_EXPRESSIONS
   >>> net = LogicNetwork.read_logic(MYELOID_LOGIC_EXPRESSIONS)
   >>> net.names
   ['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL', 'C/EBPa', 'PU.1', 'cJun', 'EgrNab', 'Gfi-1']
   >>> net.attractors
   array([array([0]), array([62, 38]), array([46]), array([54]),
          array([1216]), array([1116, 1218]), array([896]), array([960])],
         dtype=object)
   >>> net.average_sensitivity()
   1.0227272727272727
   >>> net.network_graph()
   <networkx.classes.digraph.DiGraph object at 0x...>

See the `examples <https://github.com/ELIFE-ASU/Neet/blob/master/examples>`_
directory of the `GitHub repository <https://github.com/ELIFE-ASU/Neet>`_ for
Jupyter notebooks which demonstrate some of the Neet's features.

Getting Help
------------

**Neet** is developed to help people interested in using and analyzing network
models to get things done quickly and painlessly. Your feedback is
indispensable. Please create an issue if you find a bug, an error in the
documentation, or have a feature you'd like to request. Your contribution will
make **Neet** a better tool for everyone.

If you are interested in contributing to **Neet**, please contact the
developers. We'll get you up and running!

**Neet Source Repository**
    https://github.com/elife-asu/neet

**Neet Issue Tracker**
    https://github.com/elife-asu/neet/issues

Relevant Publications
---------------------

Daniels, B.C., Kim, H., Moore, D.G., Zhou, S., Smith, H.B., Karas, B., Kauffman,
S.A., and Walker, S.I. (2018) "Criticality Distinguishes the Ensemble of
Biological Regulatory Networks" *Phys. Rev.  Lett.* **121** (13), 138102,
`doi:10.1103/PhysRevLett.121.138102 <https://doi.org/10.1103/PhysRevLett.121.138102>`_.

Copyright and Licensing
-----------------------

Copyright © 2017-2019 Bryan C. Daniels, Bradley Karas, Hyunju Kim, Douglas G.
Moore, Harrison Smith, Sara I. Walker, and Siyu Zhou. Free use of this software
is granted under the terms of the MIT License.

See the `LICENSE <https://github.com/elife-asu/neet/blob/master/LICENSE>`_ for
details.

Contents
--------

.. toctree::

   getting-started
   intro
   networks
   statespaces
   landscapes
   information
   sensitivity
   api
   zrefs

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
