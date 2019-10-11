.. _introduction:

Introduction
============

Neet is a library for simulating and analyzing dynamical network models. It is written entirely in
Python, with minimal external dependencies.

Getting Started
---------------

Installation
~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

**Neet** depends on several packages which will be installed by default when **Neet** is installed
via `pip`:

* `six <https://pypi.org/project/six/>`_
* `numpy <https://pypi.org/project/numpy/>`_
* `networkx <https://pypi.org/project/networkx/>`_
* `pyinform <https://pypi.org/project/pyinform/>`_
* `deprecated <https://pypi.org/project/Deprecated/>`_

However, network visualization is notoriously problematic, and so we have two optional dependencies
which are only required if you wish to visualize networks using **Neet**'s builtin capabilities:

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

Note that on some systems this will require administrative privileges. If you
don't have admin privileges or would prefer to install **Neet** for your user
only, you do so via the ``--user`` flag:

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

So far the python wrapper has been tested under ``python2.7``, ``python3.4`` and
``python3.5``, and on the following platforms:

.. Note::

   We will continue supporting Python 2.7 until January 1, 2020 when `PEP 373
   <https://www.python.org/dev/peps/pep-0373/#maintenance-releases>`_ states
   that official support for Python 2.7 will end.

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10

