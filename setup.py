# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

from setuptools import setup

with open("README.md") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

setup(
    name='neet',
    version='0.0.0',
    description='A brilliant and fundamental contribution to network science',
    long_description=README,
    maintainer='Douglas G. Moore',
    maintainer_email='doug@dglmoore.com',
    url='https://github.com/elife-asu/neet',
    license=LICENSE,
    requires=['numpy', 'networkx', 'pyinform'],
    packages=['neet', 'neet.automata', 'neet.boolean'],
    package_data={'neet.boolean': ['data/*.txt', 'data/*.dat']},
    test_suite='test',
    platforms=['Windows', 'OS X', 'Linux']
)
