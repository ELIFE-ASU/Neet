from setuptools import setup

with open("README.md") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

setup(
    name='neet',
    version='0.1.0',
    description='A brilliant and fundamental contribution to network science',
    long_description=README,
    maintainer='Douglas G. Moore',
    maintainer_email='doug@dglmoore.com',
    url='https://github.com/elife-asu/neet',
    license=LICENSE,
    install_requires=['six', 'numpy', 'networkx', 'pyinform', 'deprecated', 'pygraphviz'],
    setup_requires=['green'],
    packages=['neet', 'neet.boolean'],
    package_data={'neet.boolean': ['data/*.txt', 'data/*.dat']},
    test_suite='test',
    platforms=['Windows', 'OS X', 'Linux']
)
