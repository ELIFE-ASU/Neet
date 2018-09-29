"""
The :mod:`neet.python3` provides type aliases which make it easier to write
code which works under both Python 2 and Python 3.
"""
import sys

if sys.version_info > (3,):
    long = int
    unicode = str
else:
    long = long
    unicode = unicode
