# Neet: Simulating and analyzing network models

[![][doc-stable-img]][doc-stable-url] [![][doc-latest-img]][doc-latest-url] [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][doi-img]][doi-url]

[travis-img]: https://travis-ci.org/ELIFE-ASU/Neet.svg?branch=master
[travis-url]: https://travis-ci.org/ELIFE-ASU/Neet

[appveyor-img]: https://ci.appveyor.com/api/projects/status/eyrn6l2wygeglnx5/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/dglmoore/neet-awnxe/branch/master

[codecov-img]: https://codecov.io/gh/elife-asu/neet/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/elife-asu/neet

[doc-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[doc-latest-url]: https://neet.readthedocs.io/en/latest

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://neet.readthedocs.io/en/stable

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.3489785.svg
[doi-url]: https://doi.org/10.5281/zenodo.3489785

**Neet** is a python package designed to provide an easy-to-use API for creating
and evaluating dynamical network models. In its current state, **Neet** supports
simulating synchronous Boolean network models, though the API is designed to be
model generic. Future work will implement asynchronous update mechanisms and
more general network types.

## Examples

**Neet** provides a hierarchy of network classes with methods designed to make common tasks as
painless as possible. For example, you can read in a collection of boolean logic equations and
immediately probe the dynamics of the network, and compute values such as the
[attractor cycles](https://neet.readthedocs.io/en/stable/api/landscape.html#neet.LandscapeMixin.attractors)
and
[average sensitivity](https://neet.readthedocs.io/en/stable/api/boolean/sensitivity.html#neet.boolean.SensitivityMixin.average_sensitivity)
of the network.

```python
>>> from neet.boolean import LogicNetwork
>>> net = LogicNetwork.read_logic('myeloid-logic_expressions.txt')
>>> net.names
['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL', 'C/EBPa', 'PU.1', 'cJun', 'EgrNab', 'Gfi-1']
>>> net.attractors
array([array([0]), array([62, 38]), array([46]), array([54]),
       array([1216]), array([1116, 1218]), array([896]), array([960])],
      dtype=object)
>>> net.average_sensitivity()
1.0227272727272727
>>> net.network_graph()
<networkx.classes.digraph.DiGraph object at 0x7b2ce5508510>
```

See the [examples](examples) directory for Jupyter notebooks which demonstrate some of the Neet's features.

## Installation

### Dependencies

**Neet** depends on several packages which will be installed by default when **Neet** is installed
via `pip`:

* [six](https://pypi.org/project/six/)
* [numpy](https://pypi.org/project/numpy/)
* [networkx](https://pypi.org/project/networkx/)
* [pyinform](https://pypi.org/project/pyinform/)
* [deprecated](https://pypi.org/project/Deprecated/)

However, network visualization is notoriously problematic, and so we have two optional dependencies
which are only required if you wish to visualize networks using **Neet**'s builtin capabilities:

* [Graphviz](https://graphviz.org/)
* [pygraphviz](https://pypi.org/project/pygraphviz/)

True to form, these dependencies are a pain. Graphviz, unfortunately, cannot be installed via pip
(see: https://graphviz.gitlab.io/download/ for installation instructions). Once Graphviz has been
installed, you can install `pygraphviz` via `pip`.

### Via Pip

To install **Neet** via `pip`, you can run the following

```bash
$ pip install neet
```

Note that on some systems this will require administrative privileges. If you
don't have admin privileges or would prefer to install **Neet** for your user
only, you do so via the `--user` flag:

```bash
$ pip install --user neet
```

### From Source
```bash
$ git clone https://github.com/elife-asu/neet
$ cd neet
$ python setup.py test
$ pip install .
```

## Getting Help
**Neet** is developed to help people interested in using and analyzing network
models to get things done quickly and painlessly. Your feedback is
indispensable. Please create an issue if you find a bug, an error in the
documentation, or have a feature you'd like to request. Your contribution will
make **Neet** a better tool for everyone.

If you are interested in contributing to **Neet**, please contact the
developers. We'll get you up and running!

<dl>
  <dt>Neet Source Repository</dt>
  <dd>https://github.com/elife-asu/neet</dd>
  <dt>Neet Issue Tracker</dt>
  <dd>https://github.com/elife-asu/neet/issues</dd>
</dl>

## Relevant Publications

- Daniels, B.C., Kim, H., Moore, D.G., Zhou, S., Smith, H.B., Karas, B.,
  Kauffman, S.A., and Walker, S.I. (2018) "Criticality Distinguishes the
  Ensemble of Biological Regulatory Networks" *Phys. Rev. Lett.* **121** (13),
  138102, doi:[10.1103/PhysRevLett.121.138102](https://doi.org/10.1103/PhysRevLett.121.138102)

## System Support

So far the python wrapper has been tested under `python3.4` and `python3.5`, and on the following
platforms:

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10

> **Note:** As of January 1, 2020, official support for Python 2.X  has ended as per [PEP
373](https://www.python.org/dev/peps/pep-0373/#maintenance-releases). As such, Neet no longer
officially supports 2.X; however, the current version (Neet v1.0) is compatible and all unit tests
pass under Python 2.7.

## Copyright and Licensing
Copyright © 2017-2020 Bryan C. Daniels, Bradley Karas, Hyunju Kim, Douglas G.
Moore, Harrison Smith, Sara I. Walker, and Siyu Zhou. Free use of this software is
granted under the terms of the MIT License.

See the [LICENSE](https://github.com/elife-asu/neet/blob/master/LICENSE) for
details.
