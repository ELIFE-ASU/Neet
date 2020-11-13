"""
.. currentmodule:: neet.draw

Utilities for drawing Neet objects and graph representations.
"""
import networkx as nx
import pygraphviz  # noqa
import tempfile
import os


@nx.utils.open_file(5, 'w+b')
def view_pygraphviz(G, edgelabel=None, prog='dot', args='',
                    suffix='', path=None, display_image=True):
    """
    Views the graph G using the specified layout algorithm.

    This is a modified version of ``view_pyagraphviz`` from
    :mod:`networkx.drawing.nx_agraph` to allow display toggle.

    Original copyright::

        Copyright (C) 2004-2019 by
            Aric Hagberg <hagberg@lanl.gov>
            Dan Schult <dschult@colgate.edu>
            Pieter Swart <swart@lanl.gov>
        All rights reserved. BSD license.
        Author: Aric Hagberg (hagberg@lanl.gov)

    :param G: the graph to draw
    :type G: networkx.Graph or networkx.DiGraph
    :param edgelabel: If a string, then it specifes the edge attribute to be
                      displayed on the edge labels. If a callable, then it is
                      called for each edge and it should return the string to
                      be displayed on the edges.  The function signature of
                      `edgelabel` should be edgelabel(data), where `data` is
                      the edge attribute dictionary.
    :type edgelabel: str, callable or None
    :param prog: Name of Graphviz layout program.
    :type prog: str
    :param args: Additional arguments to pass to the Graphviz layout program.
    :type args: str
    :param suffix: If `filename` is None, we save to a temporary file.  The
                   value of `suffix` will appear at the tail end of the
                   temporary filename.
    :type suffix: str
    :param path: The filename used to save the image. If None, save to a
                 temporary file. File formats are the same as those from
                 pygraphviz.agraph.draw.
    :type path: str or None

    :return: the filename of the generated image, and a ``PyGraphviz`` graph instance

    .. Note::
        If this function is called in succession too quickly, sometimes the
        image is not displayed. So you might consider time.sleep(.5) between
        calls if you experience problems.
    """
    if not len(G):
        raise nx.NetworkXException("An empty graph cannot be drawn.")

    # If we are providing default values for graphviz, these must be set
    # before any nodes or edges are added to the PyGraphviz graph object.
    # The reason for this is that default values only affect incoming objects.
    # If you change the default values after the objects have been added,
    # then they inherit no value and are set only if explicitly set.

    # to_agraph() uses these values.
    attrs = ['edge', 'node', 'graph']
    for attr in attrs:
        if attr not in G.graph:
            G.graph[attr] = {}

    # These are the default values.
    edge_attrs = {'fontsize': '10'}
    node_attrs = {'style': 'filled',
                  'fillcolor': '#0000FF40',
                  'height': '0.75',
                  'width': '0.75',
                  'shape': 'circle'}
    graph_attrs = {}

    def update_attrs(which, attrs):
        # Update graph attributes. Return list of those which were added.
        added = []
        for k, v in attrs.items():
            if k not in G.graph[which]:
                G.graph[which][k] = v
                added.append(k)

    def clean_attrs(which, added):
        # Remove added attributes
        for attr in added:
            del G.graph[which][attr]
        if not G.graph[which]:
            del G.graph[which]

    # Update all default values
    update_attrs('edge', edge_attrs)
    update_attrs('node', node_attrs)
    update_attrs('graph', graph_attrs)

    # Convert to agraph, so we inherit default values
    A = nx.nx_agraph.to_agraph(G)

    # Remove the default values we added to the original graph.
    clean_attrs('edge', edge_attrs)
    clean_attrs('node', node_attrs)
    clean_attrs('graph', graph_attrs)

    # If the user passed in an edgelabel, we update the labels for all edges.
    if edgelabel is not None:
        if not hasattr(edgelabel, '__call__'):
            def func(data):
                return ''.join(["  ", str(data[edgelabel]), "  "])
        else:
            func = edgelabel

        # update all the edge labels
        if G.is_multigraph():
            for u, v, key, data in G.edges(keys=True, data=True):
                # PyGraphviz doesn't convert the key to a string. See #339
                edge = A.get_edge(u, v, str(key))
                edge.attr['label'] = str(func(data))
        else:
            for u, v, data in G.edges(data=True):
                edge = A.get_edge(u, v)
                edge.attr['label'] = str(func(data))

    if path is None:
        ext = 'png'
        if suffix:
            suffix = '_%s.%s' % (suffix, ext)
        else:
            suffix = '.%s' % (ext,)
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    else:
        # Assume the decorator worked and it is a file-object.
        pass

    display_pygraphviz(A, path=path, prog=prog, args=args, display_image=display_image)

    return path.name, A


def display_pygraphviz(graph, path, format=None, prog=None, args='', display_image=True):
    """Internal function to display a graph in OS dependent manner.

    Modified from networkx.drawing.nx_agraph functions to allow display toggle.
    Original copyright of display_pygraphviz:
        Copyright (C) 2004-2019 by
        Aric Hagberg <hagberg@lanl.gov>
        Dan Schult <dschult@colgate.edu>
        Pieter Swart <swart@lanl.gov>
        All rights reserved.
        BSD license.
        Author: Aric Hagberg (hagberg@lanl.gov)

    Parameters
    ----------
    graph : PyGraphviz graph
        A PyGraphviz AGraph instance.
    path :  file object
        An already opened file object that will be closed.
    format : str, None
        An attempt is made to guess the output format based on the extension
        of the filename. If that fails, the value of `format` is used.
    prog : string
        Name of Graphviz layout program.
    args : str
        Additional arguments to pass to the Graphviz layout program.

    Notes
    -----
    If this function is called in succession too quickly, sometimes the
    image is not displayed. So you might consider time.sleep(.5) between
    calls if you experience problems.

    """

    if format is None:
        filename = path.name
        format = os.path.splitext(filename)[1].lower()[1:]
    if not format:
        # Let the draw() function use its default
        format = None

    # Save to a file and display in the default viewer.
    # We must close the file before viewing it.
    graph.draw(path, format, prog, args)
    path.close()
    if display_image:
        nx.utils.default_opener(filename)
