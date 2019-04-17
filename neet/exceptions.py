"""
.. currentmodule:: neet.exceptions

Exceptions
==========

Exceptions are the key mechanism for handling undesirable program
state. Whenever :mod:`neet` encounters a problem, it raises an exception of
some variety. Whenever possible, we have preferred to use builtin exception
classes, e.g. `ValueError`, `IndexError`, etc... For causes that aren't really
covered by a builtin exception class, we've created subclasses of the standard
library's `Exception` to report those errors.

At the current point, there is only one such exception class,
:class:`FormatError`

API Documentation
-----------------
"""


class FormatError(Exception):
    """
    An error class to report when a configuration or data file is improperly
    formatted.
    """
    pass


class ImplementationError(Exception):
    """
    An error class to report when a subclass fails to implement a required method.
    """
    pass
