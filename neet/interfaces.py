# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

def is_network(thing):
    """
    Determine whether an *object* or *type* meets the interface requirement of
    a network.

    .. rubric:: Example:

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...
        >>> class IsNotNetwork(object):
        ...     pass
        ...
        >>> is_network(IsNetwork())
        True
        >>> is_network(IsNetwork)
        True
        >>> is_network(IsNotNetwork())
        False
        >>> is_network(IsNotNetwork)
        False
        >>> is_network(5)
        False

    :param thing: an object or a type
    :returns: ``True`` if ``thing`` has the minimum interface of a network
    """
    return hasattr(thing, 'update')


def is_fixed_sized(thing):
    """
    Determine whether an *object* or *type* is a network and has a fixed size.

    .. rubric:: Example

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...
        >>> class FixedSized(IsNetwork):
        ...     def size():
        ...         return 5
        ...
        >>> is_fixed_sized(IsNetwork)
        False
        >>> is_fixed_sized(FixedSized)
        True

    :param thing: an object or a type
    :returns: ``True`` if ``thing`` is a network with a size attribute
    :see: :func:`is_network`.
    """
    return is_network(thing) and hasattr(thing, 'size')
