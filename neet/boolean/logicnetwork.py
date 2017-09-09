# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import re
from neet.statespace import StateSpace


class LogicNetwork(object):
    """
    The LogicNetwork class represents boolean networks whose update rules
    follow logic relations among nodes. Each node state is expressed as ``0``
    or ``1``.
    """

    def __init__(self, table, names=None):
        """
        Construct a network from a logic truth table.

        A truth table stores a list of of tuples, one for each node in order.
        The tuple with the form of `(A, {C1, C2, ...})` at index `i` contains
        the activation conditions for the node of index `i`. `A` is a tuple
        marking the indices of the nodes which influence the state of node `i`
        via logic relations. `{C1, C2, ...}` being a set, each element is the
        collective binary state of these influencing nodes that would activate
        node `i`, setting it `1`. Any other collective states of nodes `A` not
        in the set are assumed to deactivate node `i`, setting it `0`. `C1`,
        `C2`, etc. are sequences (`tuple` or `str`) of binary digits, each
        being the binary state of corresponding node in `A`.

        :param table: the logic table
        :param names: names of nodes, default None

        .. rubric:: Examples

        ::
            >>> net = LogicNetwork([((0,), {'0'})])
            >>> net.size
            1
            >>> net.table
            [((0,), {'0'})]

        ::

            >>> net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
            >>> net.size
            2
            >>> net.table
            [((1,), {'0', '1'}), ((0,), {'1'})]

        ::

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), ((0, 1), '10', [1, 1]),
                                    ((0, 1), {'11'})], ['A', 'B'])
            >>> net.size
            3
            >>> net.names
            ['A', 'B']
            >>> net.table
            [((1, 2), {'01', '10'}), ((0, 2), {'01', '10', '11'}), ((0, 1), {'11'})]

        """
        if not isinstance(table, (list, tuple)):
            raise TypeError("table must be a list or tuple")

        self.size = len(table)

        if names:
            if not isinstance(names, (list, tuple)):
                raise TypeError("names must be a list or tuple")
            elif len(names) != self.size:
                raise ValueError("number of names must match network size")
            else:
                self.names = list(names)

        self._state_space = StateSpace(self.size, base=2)
        self._encoded_table = []

        for row in table:
            # Validate mask.
            if not (isinstance(row, (list, tuple)) and len(row) == 2):
                raise ValueError("Invalid table format")
            # Encode the mask.
            mask_code = 0
            for idx in row[0]:
                if idx >= self.size:
                    raise IndexError("mask index out of range")
                mask_code += 2 ** idx  # Low order, low index.
            # Validate truth table of the sub net.
            if not isinstance(row[1], (list, tuple, set)):
                raise ValueError("Invalid table format")
            # Encode each condition of truth table.
            encoded_sub_table = set()
            for condition in row[1]:
                encoded_condition = 0
                for idx, state in zip(row[0], condition):
                    encoded_condition += 2 ** idx if int(state) else 0
                encoded_sub_table.add(encoded_condition)
            self._encoded_table.append((mask_code, encoded_sub_table))
        # Store positive truth table for human reader.
        self.table = []
        for row in table:
            conditions = set()
            for condition in row[1]:
                conditions.add(''.join([str(int(s)) for s in condition]))
            self.table.append((row[0], conditions))

    def state_space(self):
        return self._state_space

    def _unsafe_update(self, net_state, index=None, pin=None, values=None):
        """
        Update node states according to the truth table. Core update function.

        If `index` is provided, update only node at `index`. If `index` is not
        provided, update all ndoes. The input `net_state` is not modified.

        :param net_state: a sequence of binary node states
        :type net_state: sequence
        :param index: the index to update (or None)
        :type index: int or None
        :param pin: the indices to pin (or None)
        :type pin: sequence
        :returns: the updated states

        .. rubric:: Basic Use:

        ::

            >>> net = LogicNetwork([((0,), {'0'})])
            >>> net._unsafe_update([0], 0)
            [1]
            >>> net._unsafe_update([1])
            [0]

        ::

            >>> net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
            >>> net._unsafe_update([1, 0], 0))
            [1, 0]
            >>> net._unsafe_update([1, 0], 1))
            [1, 1]
            >>> net._unsafe_update([0, 0])
            [1, 0]

        ::

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), {(0, 1), '10', (1, 1)}),
                                    ((0, 1), {'11'})])
            >>> net.size
            3
            >>> net._unsafe_update([0, 1, 0])
            [1, 0, 0]
            >>> net._unsafe_update([0, 0, 1])
            [1, 1, 0]
            >>> net._unsafe_update([0, 0, 1], 1)
            [0, 1, 1]
            >>> net._unsafe_update([0, 0, 1], pin=[1])
            [1, 0, 0]
            >>> net._unsafe_update([0, 0, 1], pin=[0, 1])
            [0, 0, 0]
            >>> net._unsafe_update([0, 0, 1], values={0: 0})
            [0, 1, 0]
            >>> net._unsafe_update([0, 0, 1], pin=[1], values={0: 0})
            [0, 0, 0]
        """
        encoded_state = self.state_space().encode(net_state)

        if index is None:
            indices = range(self.size)
        else:
            indices = [index]

        if pin is None:
            pin = []

        for idx in indices:
            if idx in pin:
                continue
            mask, condition = self._encoded_table[idx]
            sub_net_state = mask & encoded_state
            net_state[idx] = 1 if sub_net_state in condition else 0

        if values:
            for k, v in values.items():
                net_state[k] = v

        return net_state

    def update(self, net_state, index=None, pin=None, values=None):
        """
        Update node states according to the truth table.

        If `index` is provided, update only node at `index`. If `index` is not
        provided, update all ndoes. `pin` provides the indices of which the
        nodes' states are forced to remain unchanged. Update is inplace.

        :param net_state: a sequence of binary node states
        :type net_state: sequence
        :param index: the index to update (or None)
        :type index: int or None
        :param pin: the indices to pin (or None)
        :type pin: sequence
        :returns: the updated states

        .. rubric:: Basic Use:

        ::

            >>> net = LogicNetwork([((0,), {'0'})])
            >>> net.update([0], 0)
            [1]
            >>> net.update([1])
            [0]
            >>>

        ::

            >>> net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
            >>> net.update([1, 0], 0))
            [1, 0]
            >>> net.update([1, 0], 1))
            [1, 1]
            >>> net.update([0, 0])
            [1, 0]

        ::

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), {(0, 1), '10', (1, 1)}),
                                    ((0, 1), {'11'})])
            >>> net.size
            3
            >>> net.update([0, 0, 1], 1)
            [0, 1, 1]
            >>> net.update([0, 1, 0])
            [1, 0, 0]
            >>> net.update([0, 0, 1])
            [1, 1, 0]
            >>> net.update([0, 0, 1], pin=[1])
            [1, 0, 0]
            >>> net.update([0, 0, 1], pin=[0, 1])
            [0, 0, 0]
            >>> net.update([0, 0, 1], values={0: 0})
            [0, 1, 0]
            >>> net.update([0, 0, 1], pin=[1], values={0: 0})
            [0, 0, 0]
        """
        if net_state not in self.state_space():
            raise ValueError(
                "the provided state is not in the network's state space")

        if values and any([v not in (0, 1) for v in values.values()]):
            raise ValueError("invalid state in values argument")

        if pin and values and any([k in pin for k in values]):
            raise ValueError("cannot set a value for a pinned state")

        return self._unsafe_update(net_state, index, pin, values)

    @classmethod
    def read_table(cls, table_file):
        """
        Read a network from a truth table file.

        A logic table file starts with a table title which contains names of
        all nodes. It is a line marked by `##` at the begining with node names
        seperated by commas or spaces. This line is required. For artificial
        network without node names, arbitrary names will be put in place, e.g.:

        `## A B C D`

        Following are the sub-tables of logic conditions for every node. Each
        sub-table nominates a node and its logically connected nodes in par-
        enthesis as a comment line:

        `# A (B C)`

        The rest of the sub-table are states of those nodes in parenthesis
        (B, C) that would activate the state of A. States that would deactive A
        should not be included in the sub-table.

        A complete logic table with 3 nodes A, B, C would look like this:

        '''
        # A B C
        # A (B C)
        1 0
        1 1
        # B (A)
        1
        # C (B C A)
        1 0 1
        0 1 0
        0 1 1
        '''

        Custom comments can be added above the table title, but not below.

        :param table_file: a truth table file
        :returns: a :class:LogicNetwork

        .. rubric:: Examples:

        ::

            >>> net = LogicNetwork.read('myeloid-truth_table.txt')
            >>> net.size
            11
            >>> net.names
            ['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL', 'C/EBPa',
             'PU.1', 'cJun', 'EgrNab', 'Gfi-1']
        """
        names_format = re.compile(r'^\s*##[^#]+$')
        node_title_format = re.compile(
            r'^\s*#\s*(\S+)\s*\((\s*(\S+\s*)+)\)\s*$')

        with open(table_file, 'r') as f:
            lines = f.read().splitlines()
            # Search for node names.
            i = 0
            names = []
            while not names:
                try:
                    if names_format.match(lines[i]):
                        names = re.split(r'\s*,\s*|\s+', lines[i].strip())[1:]
                    i += 1
                except IndexError:
                    raise FormatError("node names not found in file")

            table = [()] * len(names)
            # Create condition tables for each node.
            for line in lines[i:]:
                node_title = node_title_format.match(line)
                if node_title:
                    node_name = node_title.group(1)
                    # Read specifications for node.
                    if node_name not in names:
                        raise FormatError(
                            "'{}' not in node names".format(node_name))
                    node_index = names.index(node_name)
                    sub_net_nodes = re.split(
                        r'\s*,\s*|\s+', node_title.group(2).strip())
                    table[node_index] = (
                        tuple([names.index(node) for node in sub_net_nodes]), set())
                elif re.match(r'^\s*#.*$', line):
                    # Skip a comment.
                    continue
                else:
                    # Read activation conditions for node.
                    try:
                        if line.strip():
                            condition = re.split(r'\s*,\s*|\s+', line.strip())
                        else:
                            # Skip an empty line.
                            continue

                        if len(condition) != len(table[node_index][0]):
                            raise FormatError(
                                "number of states and nodes must match")
                        for state in condition:
                            if state not in ('0', '1'):
                                raise FormatError("node state must be binary")
                        table[node_index][1].add(''.join(condition))

                    except NameError:  # node_index not defined
                        raise FormatError(
                            "node must be specified before logic conditions")

        # If no truth table is provided for a node, that node is considered
        # an "external" node, i.e, its state stays on or off by itself.
        for i, sub_table in enumerate(table):
            if not sub_table:  # Empty truth table.
                table[i] = ((i), {'1'})
        return cls(table, names)

    @classmethod
    def read_logic(cls, logic_file, external_nodes_file=None):
        """
        Read a network from a file of logic equations.

        A logic equations has the form of `A = B AND ( C OR D )`, each term
        being separated from parantheses and logic operators with at least a
        space. The optional `external_nodes_file` takes a file that contains
        nodes in a column whose states do not depend on any nodes. These are
        considered "external" nodes. Equivalently, such a node would have a
        logic equation `A = A`, for its state stays on or off unless being set
        externally, but now the node had to be excluded from `external_nodes_file`
        to avoid duplication and confusion.

        :param logic_file: a .txt file of logic equations
        :param external_nodes_file: a .txt file of external nodes
        :returns: a :class:LogicNetwork

        .. rubric:: Basic Use:

        ::

            >>> net = LogicNetwork.read_logic('myeloid-logic_expressions.txt')
            >>> net.size
            11
            >>> net.names
            ['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL', 'C/EBPa',
             'PU.1', 'cJun', 'EgrNab', 'Gfi-1']
        """
        names = []
        expressions = []
        with open(logic_file) as eq_file:
            for eq in eq_file:
                name, expr = eq.split('=')
                names.append(name.strip())
                expressions.append(expr.strip())

        if external_nodes_file:
            with open(external_nodes_file) as extra_file:
                extras = [name.strip() for name in extra_file]
            names += extras

        ops = {'AND', 'OR', 'NOT'}

        table = []
        for expr in expressions:
            sub_nodes = []
            conditions = set()

            expr_split = expr.split()
            for i, item in enumerate(expr_split):
                if item not in ops and item not in '()':
                    if item not in names:
                        raise ValueError("unknown component '{}'".format(item))
                    if item not in sub_nodes:
                        expr_split[i] = '{' + str(len(sub_nodes)) + '}'
                        sub_nodes.append(item)
                    else:
                        expr_split[i] = '{' + str(sub_nodes.index(item)) + '}'
                else:
                    expr_split[i] = item.lower()
            logic_expr = ' '.join(expr_split)

            indices = tuple([names.index(node) for node in sub_nodes])

            for dec_state in range(2**len(sub_nodes)):
                bin_state = '{0:0{1}b}'.format(dec_state, len(sub_nodes))
                if eval(logic_expr.format(*bin_state)):
                    conditions.add(bin_state)

            table.append((indices, conditions))

        # Add empty logic tables for external components.
        if external_nodes_file:
            for i in range(len(extras)):
                table.append((((len(names) - len(extras) + i),), set('1')))

        return cls(table, names)


class FormatError(Exception):
    """Exception for errors in logic table's format"""
    pass
