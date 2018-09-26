import sys

if sys.version_info > (3,):
    long = int
    unicode = str
else:
    long = long
    unicode = unicode

