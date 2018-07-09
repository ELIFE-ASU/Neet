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


def neighbors(net, index, direction='both', **kwargs):
    """
    Return a set of neighbors for a specified node, or a list of sets of
    neighbors for all nodes in the network.

    For ECAs, in the cases of the lattices having fixed boundary conditions,
    the left boundary, being on the left of the leftmost index 0, has an index
    of -1, while the right boundary's index is the size+1. The full state of the
    lattices and the boundaries is equavolent to:
    `[cell0, cell1, ..., cellN, right_boundary, left_boundary]`
    if it is ever presented as a single list in Python.

    :param index: node index, if neighbors desired for one node only
    :param direction: type of node neighbors to return (can be 'in','out', or 'both')
    :kwarg size: size of ECA, required if network is an ECA
    :returns: a set of neighbors of a node

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
    if direction not in ('in', 'out', 'both'):
        raise ValueError('direction must be "in", "out" or "both"')

    neighbor_types = {'in': net.neighbors_in,
                      'out': net.neighbors_out,
                      'both': net.neighbors}

    if net.__class__.__name__ == 'ECA':
        if 'size' not in kwargs:
            raise AttributeError("A `size` kwarg is required for returning an ECA's neighbors")
        else:
            return neighbor_types[direction](index, size=kwargs['size'])

    else:
        return neighbor_types[direction](index)

def to_networkx_graph(net,size=None,labels='indices',**kwargs):
    """
    Return networkx graph given neet network.  Requires networkx.

    :param labels: how node is labeled and thus identified in networkx graph
                   ('names' or 'indices'), only used if network is a LogicNetwork or WTNetwork
    :kwarg size: size of ECA, required if network is an ECA
    :returns : a networkx DiGraph
    """
    if net.__class__.__name__ == 'ECA':
        if size is None:
            raise AttributeError("A `size` kwarg is required for converting an neet ECA to a networkx network")
        else:
            return net.to_networkx_graph(size)

    elif net.__class__.__name__ in ['WTNetwork','LogicNetwork']:
        return net.to_networkx_graph(labels=labels)


# def to_networkx_graph(net):
#     """
#     Return networkx graph given neet network.  Requires networkx.
#     """
#     edges = []
#     names = net.names
#     for i,jSet in enumerate(net.neighbors(direction='out')):
#         for j in jSet:
#             edges.append((names[i],names[j]))
#     return nx.DiGraph(edges,name=net.metadata.get('name'))

# def draw(net,format='pdf',filename=None):
#     """
#     Output a file with a simple network drawing.

#     Requires networkx and pygraphviz.

#     Supported image formats are determined by graphviz.  In particular,
#     pdf support requires 'cairo' and 'pango' to be installed prior to
#     graphviz installation.
#     """
#     if filename is None: filename = net.metadata.get('name','network')
#     if not filename.endswith('.'+format): filename += '.'+format
#     g = to_networkx_graph(net)
#     nx.nx_agraph.view_pygraphviz(g,prog='circo',path=filename)
