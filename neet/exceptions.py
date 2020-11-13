"""
.. currentmodule:: neet.exceptions

Exceptions are the key mechanism for handling undesirable program state.
Whenever Neet encounters a problem, it raises an exception of some variety.
Whenever possible, we have preferred to use builtin exception classes, e.g.
:class:`ValueError`, :class:`IndexError`, etc... For cases that aren't really
covered by a builtin exception class, we've created subclasses of the standard
library's :class:`Exception` to report those errors.

.. inheritance-diagram:: neet.exceptions
   :parts: 1
"""


class FormatError(Exception):
    """
    An error class to report when a configuration or data file is improperly
    formatted.
    """
    pass
