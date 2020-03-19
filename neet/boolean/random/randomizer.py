import neet
import networkx as nx
from abc import ABCMeta, abstractmethod
from .constraints import AbstractConstraint, ConstraintError


class AbstractRandomizer(object, metaclass=ABCMeta):
    def __init__(self, network, constraints=None, timeout=1000, **kwargs):
        """
        An abstract interface for all randomizers based on randomly modifying a
        base network or graph. Rejection testing is used to enforce
        user-specified constraints. Networks/graphs will be repeatedly randomized
        until a instance satisfying all constraints is found. If a
        ```timeout``` is provided, the rejection testing will stop after that
        many attempts and raise ``ConstraintError`` if no valid network was found.
        If ``timeout <= 0``, then the rejection testing will never time out.

        :param network: a base network or graph
        :type network: neet.Network or networkx.DiGraph
        :param constraints: constraints used for rejection testing
        :type constraints: a sequence of AbstractConstraint instances
        :param timeout: the number of attempts before rejection testing times
                        out. If less than 1, the rejection testing will never
                        time out.
        """
        if isinstance(network, neet.Network):
            self.network = network
        elif isinstance(network, nx.DiGraph):
            self.__network = None
            self.__graph = network
        else:
            raise TypeError('network must be a neet.Network or a networkx.DiGraph')

        self.timeout = timeout
        self.constraints = constraints

    @property
    def network(self):
        """
        Get the randomizer's network

        :returns: neet.Network or None
        """
        return self.__network

    @network.setter
    def network(self, network):
        """
        Set the randomizer's network and replace the graph with the network's
        graph.

        :param network: the new network
        :type network: neet.Network
        :raises TypeError: if the argument is not a neet.Network
        """
        if not isinstance(network, neet.Network):
            raise TypeError('network must be an instance of neet.Network')
        self.__network = network
        self.__graph = self.__network.network_graph()

    @property
    def graph(self):
        """
        Get the randomizer's graph

        :returns: networkx.DiGraph
        """
        return self.__graph

    @graph.setter
    def graph(self, graph):
        """
        Set the randomizer's graph and replace the network with ``None``.

        :param graph: the new graph
        :type graph: networkx.DiGraph
        :raises TypeError: if the argument is not a networkx.DiGraph
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError('graph must be an instance of networkx.DiGraph')
        self.__network = None
        self.__graph = graph

    @property
    def constraints(self):
        """
        Get the randomizer's constraints.

        :returns: a list of AbstractConstraint instances
        """
        return self.__constraints

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
            if not isinstance(constraint, AbstractConstraint):
                raise TypeError('constraints must be instances of AbstractConstraint')

        self.__constraints = constraints

    def add_constraint(self, constraint):
        """
        Append a constraint to the randomizer's list of constraints.

        :param constraint: the new constraint
        :type constraint: AbstractConstraint
        :raises TypeError: if the constraint is not an AbstractConstraint
        """
        if not isinstance(constraint, AbstractConstraint):
            raise TypeError('constraints must be instances of AbstractConstraint')
        self.__constraints.append(constraint)

    def _check_constraints(self, net):
        """
        Check a network or graph against the randomizer's constraints.

        :param net: the network or directed graph
        :type net: neet.Network or networkx.DiGraph
        :returns: ``True`` if the network/graph satisfies all constraints
        """
        for constraint in self.constraints:
            if not constraint.satisfies(net):
                return False
        return True

    def __iter__(self):
        """
        Generate an infinite sequence of random networks or graphs.
        """
        while True:
            yield self.random()

    def random(self):
        """
        Create a random network variant.

        :returns: a random network or graph
        :raises ConstraintError: if a constraint could not be satisfied before
                                 the randomizer's timeout.
        """
        loop = 0
        while self.timeout <= 0 or loop < self.timeout:
            net = self._randomize()
            if self._check_constraints(net):
                return net
            loop += 1
        raise ConstraintError('failed to generate a network that statisfies all constraints')

    @abstractmethod
    def _randomize(self):
        """
        Create and *unconstrained* network variant.

        :returns: a random network or graph
        """
        pass
