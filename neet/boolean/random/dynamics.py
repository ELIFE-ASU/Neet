import neet
import numpy as np

from abc import abstractmethod
from .randomizer import AbstractRandomizer
from .topology import TopologyRandomizer, FixedTopology, InDegree
from .constraints import DynamicalConstraint, TopologicalConstraint, GenericDynamical, \
    NodeConstraint, GenericNodeConstraint, ConstraintError
from inspect import isclass


class NetworkRandomizer(AbstractRandomizer):
    def __init__(self, network, trand=None, constraints=None, timeout=1000, **kwargs):
        """
        An abstract base class for all randomizers which implement dynamical
        randomization.

        :param network: a base network or graph
        :type network: neet.Network or networkx.DiGraph
        :param trand: how to randomize the topology (default: FixedTopology)
        :type trand: instance or subclass of TopologyRandomizer, or None
        :param constraints: constraints used for rejection testing
        :type constraints: a sequence of AbstractConstraint instances
        :param timeout: the number of attempts before rejection testing times
                        out. If less than 1, the rejection testing will never
                        time out.
        """
        if trand is None:
            trand = FixedTopology(network, timeout=timeout, **kwargs)
        elif isclass(trand) and issubclass(trand, TopologyRandomizer):
            trand = trand(network, timeout=timeout, **kwargs)
        elif isinstance(trand, TopologyRandomizer):
            pass
        else:
            raise TypeError('trand must be an instance or subclass of TopologyRandomizer')
        self.trand = trand
        super().__init__(network, constraints, timeout, **kwargs)

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

        tconstraints, nconstraints, dconstraints = [], [], []

        for i, constraint in enumerate(constraints):
            if isinstance(constraint, NodeConstraint):
                nconstraints.append(constraint)
            elif isinstance(constraint, DynamicalConstraint):
                dconstraints.append(constraint)
            elif isinstance(constraint, TopologicalConstraint):
                tconstraints.append(constraint)
            elif callable(constraint):
                dconstraints.append(GenericDynamical(constraint))
            else:
                msg = 'constraints must be callable, a DynamicalConstraint or TopologicalConstraint'
                raise TypeError(msg)

        self.trand.constraints = tconstraints
        self.node_constraints = nconstraints
        AbstractRandomizer.constraints.__set__(self, dconstraints)  # type: ignore

    @property
    def node_constraints(self):
        return self.__node_constraints

    @node_constraints.setter
    def node_constraints(self, constraints):
        if constraints is None:
            constraints = []
        elif not isinstance(constraints, list):
            constraints = list(constraints)

        for i, constraint in enumerate(constraints):
            if isinstance(constraint, NodeConstraint):
                pass
            elif callable(constraint):
                constraint[i] = GenericNodeConstraint(constraint)
            else:
                msg = 'constraints must be callable, a DynamicalConstraint or TopologicalConstraint'
                raise TypeError(msg)

        self.__node_constraints = constraints

    def add_constraint(self, constraint):
        """
        Append a constraint to the randomizer's list of constraints.

        :param constraint: the new constraint
        :type constraint: AbstractConstraint
        :raises TypeError: if the constraint is not an AbstractConstraint
        """
        if isinstance(constraint, NodeConstraint):
            self.__node_constraints.append(constraint)
        elif isinstance(constraint, DynamicalConstraint):
            super().add_constraint(constraint)
        elif callable(constraint):
            super().add_constraint(GenericDynamical(constraint))
        elif isinstance(constraint, TopologicalConstraint):
            self.trand.add_constraint(constraint)
        else:
            msg = 'constraints must be callable, a DynamicalConstraint or TopologicalConstraint'
            raise TypeError(msg)

    def random(self):
        topology = self.trand.random()

        loop = 0
        while self.timeout <= 0 or loop < self.timeout:
            net = self._randomize(topology)
            if self._check_constraints(net):
                return net
            loop += 1
        raise ConstraintError('failed to generate a network that statisfies all constraints')

    def _randomize(self, topology):
        table = []
        for node in sorted(topology.nodes):
            predecessors = tuple(topology.predecessors(node))
            params = self._function_class_parameters(topology, node)
            table.append((predecessors, self._random_function(**params)))
        return neet.boolean.LogicNetwork(table)

    def _check_node_constraints(self, f):
        for constraint in self.node_constraints:
            if not constraint.satisfies(f):
                return False
        return True

    def _random_function(self, k, p, **kwargs):
        volume = 2**k
        integer, decimal = divmod(p * volume, 1)

        loop = 0
        while self.timeout <= 0 or loop < self.timeout:
            num_states = int(integer + np.random.choice(2, p=[1 - decimal, decimal]))
            indices = np.random.choice(volume, num_states, replace=False)
            f = set('{0:0{1}b}'.format(index, k) for index in indices)
            if self._check_node_constraints(f):
                return f
        raise ConstraintError('failed to generate a function that statisfies all constraints')

    @abstractmethod
    def _function_class_parameters(self, topology, node, **kwargs):
        return {'topology': topology, 'node': node, 'k': topology.in_degree(node)}


class UniformBias(NetworkRandomizer):
    def __init__(self, network, p=0.5, **kwargs):
        """
        Generate random Boolean networks with the same bias on each non-external
        node.
        """
        super().__init__(network, **kwargs)
        self.p = p

    def _function_class_parameters(self, topology, node, **kwargs):
        params = super()._function_class_parameters(topology, node)
        params.update({'p': self.p})
        return params


class MeanBias(UniformBias):
    def __init__(self, network, **kwargs):
        """
        Generate random Boolean networks with the same mean bias (on average)
        as the original network.
        """
        if not isinstance(network, neet.boolean.LogicNetwork):
            raise NotImplementedError()
        super().__init__(network, self._mean_bias(network), **kwargs)

    def _mean_bias(self, network):
        """
        Get the mean bias of a network
        """
        return np.mean([float(len(row[1]) / 2**len(row[0])) for row in network.table])


class LocalBias(NetworkRandomizer):
    def __init__(self, network, trand=None, **kwargs):
        """
        Generate networks with the same bias on each node. This scheme can only
        be applied in conjunction with the ``FixedTopology`` and ``InDegree`
        topological randomizers.
        """
        if not isinstance(network, neet.boolean.LogicNetwork):
            raise NotImplementedError(type(network))
        elif trand is not None:
            if isclass(trand) and not issubclass(trand, (FixedTopology, InDegree)):
                raise NotImplementedError(trand)
            elif not isclass(trand) and not isinstance(trand, (FixedTopology, InDegree)):
                raise NotImplementedError(type(trand))

        super().__init__(network, trand, **kwargs)
        self.local_bias = [float(len(row[1]) / 2**len(row[0])) for row in network.table]

    def _function_class_parameters(self, topology, node, **kwargs):
        params = super()._function_class_parameters(topology, node)
        params.update({'p': self.local_bias[node]})
        return params


class FixCanalizingMixin(NetworkRandomizer):
    def _randomize(self, topology):
        table = []
        if self.network is None:  # type: ignore
            raise NotImplementedError('Randomizer is based on a graph, cannot infer canalization')
        canalizing = self.network.canalizing_nodes()
        for node in sorted(topology.nodes):
            predecessors = tuple(topology.predecessors(node))
            params = self._function_class_parameters(topology, node)
            if node in canalizing:
                table.append((predecessors, self._random_canalizing_function(**params)))
            else:
                table.append((predecessors, self._random_function(**params)))
        return neet.boolean.LogicNetwork(table)

    def _random_canalizing_function(self, k, p, **kwargs):
        integer, decimal = divmod(2**k * p, 1)
        num_states = int(integer + np.random.choice(2, p=[1 - decimal, decimal]))

        canalizing_input = np.random.choice(k)
        canalizing_value = np.random.choice(2)
        if num_states > 2**(k - 1):
            canalized_value = 1
        elif num_states < 2**(k - 1):
            canalized_value = 0
        else:
            canalized_value = np.random.choice(2)

        fixed_states = self._all_states_with_one_node_fixed(k, canalizing_input, canalizing_value)
        other_states = np.lib.arraysetops.setxor1d(np.arange(2**k), fixed_states, assume_unique=True)

        if canalized_value == 1:
            state_idxs = np.random.choice(other_states, num_states - len(fixed_states), replace=False)
            state_idxs = np.concatenate((state_idxs, np.array(fixed_states)))
        elif canalized_value == 0:
            state_idxs = np.random.choice(other_states, num_states, replace=False)

        return set('{0:0{1}b}'.format(idx, k) for idx in state_idxs)

    def _all_states_with_one_node_fixed(self, k, fixed_index, fixed_value):
        return [idx for idx in range(2**k)
                if '{0:0{1}b}'.format(idx, k)[fixed_index] == str(fixed_value)]
