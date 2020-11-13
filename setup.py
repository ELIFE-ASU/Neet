from setuptools import setup

long_description = """
Neet is a python package designed to provide an easy-to-use API for creating and evaluating network
models. In its current state, Neet supports simulating synchronous Boolean network models, though
the API is designed to be model generic. Future work will implement asynchronous update mechanisms
and more general network types.

.. image:: https://travis-ci.org/ELIFE-ASU/Neet.svg?branch=master
    :alt: Build Status (Travis CI)
    :target: https://travis-ci.org/ELIFE-ASU/Neet

.. image:: https://ci.appveyor.com/api/projects/status/eyrn6l2wygeglnx5/branch/master?svg=true
    :alt: Build Status (Appveyor)
    :target: https://ci.appveyor.com/project/dglmoore/neet-awnxe/branch/master

.. image:: https://codecov.io/gh/elife-asu/neet/branch/master/graph/badge.svg
    :alt: Code Coverage (Codecov)
    :target: https://codecov.io/gh/elife-asu/neet
"""

with open("README.md") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

setup(
    name='neet',
    version='1.0.0',
    description='Simulating and analyzing network models',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    maintainer='Douglas G. Moore',
    maintainer_email='doug@dglmoore.com',
    url='https://github.com/elife-asu/neet',
    license=LICENSE,
    install_requires=['six', 'numpy', 'networkx', 'pyinform', 'deprecated'],
    extra_requires={
        "draw": ['pygraphviz']
    },
    setup_requires=['green'],
    packages=['neet', 'neet.boolean'],
    package_data={'neet.boolean': ['data/*.txt', 'data/*.dat']},
    test_suite='test',
    platforms=['Windows', 'OS X', 'Linux']
)
