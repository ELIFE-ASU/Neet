.. currentmodule:: neet

.. _introduction:

Introduction
============

Neet is a library for simulating and analyzing dynamical network models. It is written entirely in
Python, with minimal external dependencies. It provides a `heirarchy <networks.html>`__ of network
classes and facilities for analyzing the `attractor landscapes <landscapes.html>`__, `informational
structure <information.html>`__ and `sensitivity` of those network models.

Examples
--------

Neet provides a network classes with methods designed to make common tasks as painless as possible.
For example, you can read in a collection of boolean logic equations and immediately probe the
dynamics of the network, and compute values such as the :attr:`LandscapeMixin.attractors` and the
:meth:`boolean.SensitivityMixin.average_sensitivity` of the network

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


Getting Started
---------------

Installation
~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

Neet depends on several packages which will be installed by default when Neet is installed via
`pip`:

* `six <https://pypi.org/project/six/>`_
* `numpy <https://pypi.org/project/numpy/>`_
* `networkx <https://pypi.org/project/networkx/>`_
* `pyinform <https://pypi.org/project/pyinform/>`_
* `deprecated <https://pypi.org/project/Deprecated/>`_

However, network visualization is notoriously problematic, and so we have two optional dependencies
which are only required if you wish to visualize networks using Neet's builtin capabilities:

* `Graphviz <https://graphviz.org/>`_
* `pygraphviz <https://pypi.org/project/pygraphviz/>`_

True to form, these dependencies are a pain. Graphviz, unfortunately, cannot be installed via pip
(see: https://graphviz.gitlab.io/download/ for installation instructions). Once Graphviz has been
installed, you can install `pygraphviz` via `pip`.

Via Pip
^^^^^^^

To install via ``pip``, you can run the following

::

    $ pip install neet

Note that on some systems this will require administrative privileges. If you don't have admin
privileges or would prefer to install Neet for your user only, you do so via the ``--user`` flag:

::

    $ pip install --user neet

From Source
^^^^^^^^^^^

::

    $ git clone https://github.com/elife-asu/neet
    $ cd neet
    $ python setup.py test
    $ pip install .

System Support
~~~~~~~~~~~~~~

So far the python wrapper has been tested under ``python2.7``, ``python3.4`` and ``python3.5``, and
on the following platforms:

.. Note::

   We will continue supporting Python 2.7 until January 1, 2020 when `PEP 373
   <https://www.python.org/dev/peps/pep-0373/#maintenance-releases>`_ states
   that official support for Python 2.7 will end.

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10

