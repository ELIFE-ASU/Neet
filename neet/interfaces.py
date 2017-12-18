# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

import networkx as nx

def is_network(thing):
    """
    Determine whether an *object* or *type* meets the interface requirement of
    a network.

    .. rubric:: Example:

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...     def state_space(self):
        ...         return StateSpace(1)
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
    return hasattr(thing, 'update') and hasattr(thing, 'state_space') and hasattr(thing, 'neighbors')


def is_fixed_sized(thing):
    """
    Determine whether an *object* or *type* is a network and has a fixed size.

    .. rubric:: Example

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...     def state_space(self):
        ...         return StateSpace(1)
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


def is_boolean_network(thing):
    """
    Determine whether an *object* is a network with all Boolean states.
    """
    # Boolean networks have a single base equal to 2
    return is_network(thing) and hasattr(thing.state_space(), 'base') and thing.state_space().base == 2

def neighbors(net,index=None,direction='both',**kwargs):
    """
    Return a set of neighbors for a specified node, or a list of sets of
    neighbors for all nodes in the network.

    For ECAs it is possible to call the neighbors of an index which is 
    greater than the size of the network, in the case of networks which have
    fixed boundary conditions.

    The left boundary is at ``index==size+1``
    The right boundary is at ``index==size``

    eg. ``if size(eca)==3 and boundary!=None:``
    The organization of the neighbors list is as follows:
    ``[node_0|node_1|node_2|left_boundary|right_boundary]``
    
    :param index: node index, if neighbors desired for one node only
    :param direction: type of node neighbors to return (can be 'in','out', or 'both')
    :kwarg size: size of ECA, required if network is an ECA
    :returns: a set (if index!=None) or list of sets of neighbors of a node or network or nodes
    :raises ValueError: if ``net.__class__.__name__ == 'ECA' and index >= size and boundary==None``
    :raises ValueError: if ``net.__class__.__name__ == 'ECA' and index >= size+2 and boundary!=None``

    .. rubric:: Basic Use:

    ::

        >>> net = ECA(30)
        >>> net.neighbors(3,index=2,direction='out')
        set([0,1,2])
        >>> net.boundary = (1,1)
        >>> net.neighbors(3,index=2,direction='out')
        set([1,2])

    See `ECA.neighbors()`,`LogicNetwork.neighbors()` or `WTNetwork.neighbors()`
    docstrings for more details and basic use examples.

    """

    if net.__class__.__name__ == 'ECA':

        if 'size' not in kwargs:
            raise AttributeError("A `size` kwarg is required for returning an ECA's neighbors")
        else:
            return net.neighbors(kwargs['size'],index=index,direction=direction)

    else:
        return net.neighbors(index=index,direction=direction)

def to_networkx_graph(net):
    """
    Return networkx graph given neet network.  Requires networkx.
    """
    edges = []
    names = net.names
    for i,jSet in enumerate(net.neighbors(direction='out')):
        for j in jSet:
            edges.append((names[i],names[j]))
    return nx.DiGraph(edges,name=net.metadata.get('name'))

def draw(net,format='pdf',filename=None):
    """
    Output a file with a simple network drawing.  
    Requires networkx and pygraphviz.
    Supported image formats are determined by pygraphviz.
    """
    if filename is None: filename = net.metadata.get('name','network')
    if not filename.endswith('.'+format): filename += '.'+format
    g = to_networkx_graph(net)
    nx.nx_agraph.view_pygraphviz(g,prog='circo',path=filename)


