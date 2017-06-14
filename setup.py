# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

from setuptools import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name='neet',
    version='0.0.0',
    description='A brilliant and fundamental contribution to network science',
    long_description=readme,
    maintainer='Douglas G. Moore',
    maintainer_email='doug@dglmoore.com',
    url='https://github.com/elife-asu/neet',
    license=license,
    requires=['numpy','networkx'],
    packages=['neet', 'neet.boolean'],
    test_suite='test',
    platforms=['Windows', 'OS X', 'Linux']
)
