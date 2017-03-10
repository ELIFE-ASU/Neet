# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

def is_network(obj):
    """
    Determine whether an *object* meets the interface requirement of a network.
    
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
        >>> is_network(IsNotNetwork())
        False
        >>> is_network(5)
        False
    
    :param obj: an object
    :returns: ``True`` if ``obj`` is not a type and qualifes as a network
    """
    return not isinstance(obj, type) and hasattr(obj, 'update')

def is_network_type(cls):
    """
    Determine whether a *type* meets the interface requirement of a network.
    
    .. rubric:: Example:
    
    ::
    
        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...
        >>> class IsNotNetwork(object):
        ...     pass
        ...
        >>> is_network_type(IsNetwork)
        True
        >>> is_network_type(IsNotNetwork)
        False
        >>> is_network_type(int)
        False
    
    :param cls: a class
    :returns: ``True`` if ``cls`` is a type and qualifes as a network
    """
    return isinstance(cls, type) and hasattr(cls, 'update')
