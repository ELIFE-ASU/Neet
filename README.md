# Neet: Simulating and analyzing network models

**Neet** is a python package designed to provide an easy-to-use API for creating
and evaluating network models. In its current state, **Neet** supports
simulating synchronous Boolean network models, though the API is designed to be
model generic. Future work will implement asynchronous update mechanisms and
more general network types.

If you are interested in using **Neet**, you'll definitely be interested in
checking out the documentation - https://elife-asu.github.io/Neet.

[![Build Status (Travis CI)](https://travis-ci.org/ELIFE-ASU/Neet.svg?branch=master)](https://travis-ci.org/ELIFE-ASU/Neet)
[![Build Status (Appveyor)](https://ci.appveyor.com/api/projects/status/eyrn6l2wygeglnx5/branch/master?svg=true)](https://ci.appveyor.com/project/dglmoore/neet-awnxe/branch/master)

## Installation

### Via Pip

To install via `pip`, you can run the following

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
  138102, [doi:10.1103/PhysRevLett.121.138102](https://doi.org/10.1103/PhysRevLett.121.138102)

## System Support

So far the python wrapper has been tested under `python2.7`, `python3.4` and
`python3.5`, and on the following platforms:

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10

## Copyright and Licensing
Copyright © 2017-2018 Bryan C. Daniels, Bradley Karas, Hyunju Kim, Douglas G.
Moore, Harrison Smith, Sara I. Walker, and Siyu Zhou. Free use of this software is
granted under the terms of the MIT License.

See the [LICENSE](https://github.com/elife-asu/neet/blob/master/LICENSE) for
details.
