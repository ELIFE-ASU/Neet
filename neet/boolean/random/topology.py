import networkx as nx
import numpy as np

from .randomizer import AbstractRandomizer
from .constraints import TopologicalConstraint, GenericTopological, ConstraintError


class TopologyRandomizer(AbstractRandomizer):
    """
    An abstract base class for all randomizers which implement topological
    randomization.
    """
    @property
    def constraints(self):
        return super().constraints

    @constraints.setter
    def constraints(self, constraints):
        """
        Set the randomizer's constraints.

        :param constraints: the new constraints
        :type constraints: a seq of AbstractConstraint instances
        :raises TypeError: if any of the contraints are not an AbstractConstraint
        """
        if constraints is None:
            constraints = []
        elif not isinstance(constraints, list):
            constraints = list(constraints)

        for i, constraint in enumerate(constraints):
            if isinstance(constraint, TopologicalConstraint):
                pass
            elif callable(constraint):
                constraints[i] = GenericTopological(constraint)
            else:
                raise TypeError('constraints must be callable or type TopologicalConstraint')

        AbstractRandomizer.constraints.__set__(self, constraints)  # type: ignore

    def add_constraint(self, constraint):
        """
        Append a constraint to the randomizer's list of constraints.

        :param constraint: the new constraint
        :type constraint: TopologicalConstraint
        :raises TypeError: if the constraint is not an TopologicalConstraint
        """
        if isinstance(constraint, TopologicalConstraint):
            pass
        elif callable(constraint):
            constraint = GenericTopological(constraint)
        else:
            raise TypeError('constraints must be callable or type TopologicalConstraint')

        super().add_constraint(constraint)


class FixedTopology(TopologyRandomizer):
    @property
    def constraints(self):
        return super().constraints

    @constraints.setter
    def constraints(self, constraints):
        """
        Set the randomizer's constraints.

        :param constraints: the new constraints
        :type constraints: a seq of AbstractConstraint instances
        :raises TypeError: if any of the contraints are not an AbstractConstraint
        """
        if constraints is None:
            constraints = []
        elif not isinstance(constraints, list):
            constraints = list(constraints)

        for i, constraint in enumerate(constraints):
            if isinstance(constraint, TopologicalConstraint):
                pass
            elif callable(constraint):
                constraints[i] = GenericTopological(constraint)
            else:
                raise TypeError('constraints must be callable or type TopologicalConstraint')
            if not constraints[i].satisfies(self.graph):
                msg = 'the provided network is inconsistent with the provided constraints'
                raise ConstraintError(msg)

        TopologyRandomizer.constraints.__set__(self, constraints)  # type: ignore

    def add_constraint(self, constraint):
        """
        Append a constraint to the randomizer's list of constraints.

        :param constraint: the new constraint
        :type constraint: TopologicalConstraint
        :raises TypeError: if the constraint is not an TopologicalConstraint
        """
        if isinstance(constraint, TopologicalConstraint):
            pass
        elif callable(constraint):
            constraint = GenericTopological(constraint)
        else:
            raise TypeError('constraints must be callable or type TopologicalConstraint')

        if not constraint.satisfies(self.graph):
            msg = 'the provided network is inconsistent with the provided constraints'
            raise ConstraintError(msg)

        super().add_constraint(constraint)

    def random(self):
        """
        Create a random network variant. Because we check the constraints
        against the randomizer's graph when they are added, and we are just
        returning the graph, we can be certain that this will always succeed.
        That is, this method **will not** raise a ``ConstraintError``.

        :returns: a random network or graph
        """
        return self._randomize()

    def _randomize(self):
        """
        Return a graph that is isomorphic to the desired graph.

        :returns: networkx.DiGraph
        """
        return self.graph


class MeanDegree(TopologyRandomizer):
    """
    Generate a topology with the same mean degree as the initial network. This
    amounts to randomly constructing a graph with the same number of edges as
    the original graph.

    :returns: networkx.DiGraph
    """
    def _randomize(self):
        n = len(self.graph)
        edgeindices = np.random.choice(n * n, self.graph.size(), replace=False)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(map(lambda i: divmod(i, n), edgeindices))
        return G


class InDegree(TopologyRandomizer):
    """
    Generate a topology with the same in-degree distribution as the initial
    network. This amounts iterating over all nodes and selecting :math:`k`
    nodes from which to draw an edge, where :math:`k` is the in-degree of the
    node in the original graph.

    :returns: networkx.DiGraph
    """
    def _randomize(self):
        n = len(self.graph)
        edges = []
        for j in range(n):
            for i in np.random.choice(n, self.graph.in_degree(j), replace=False):
                edges.append((i, j))

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        return G


class OutDegree(TopologyRandomizer):
    """
    Generate a topology with the same out-degree distribution as the initial
    network. This amounts iterating over all nodes and selecting :math:`k`
    nodes to which to draw an edge, where :math:`k` is the in-degree of the
    node in the original graph.

    :returns: networkx.DiGraph
    """
    def _randomize(self):
        n = len(self.graph)
        edges = []
        for i in range(n):
            for j in np.random.choice(n, self.graph.out_degree(i), replace=False):
                edges.append((i, j))

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        return G
