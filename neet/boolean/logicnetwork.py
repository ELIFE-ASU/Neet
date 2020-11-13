"""
.. currentmodule:: neet.boolean

.. testsetup:: logicnetwork

    from neet.boolean import LogicNetwork
    from neet.boolean.examples import MYELOID_LOGIC_EXPRESSIONS, MYELOID_TRUTH_TABLE
"""
import re
from neet.python import long
from neet.exceptions import FormatError
from .network import BooleanNetwork


class LogicNetwork(BooleanNetwork):
    """
    LogicNetwork represents a network of logic functions. This type of Boolean
    network model is common in biological modeling.

    .. inheritance-diagram:: LogicNetwork
        :parts: 1

    In addition to methods inherited from :class:`neet.boolean.BooleanNetwork`,
    LogicNetwork exposes the following attributes

    +---------------+----------------------------+
    | :attr:`table` | The network's truth table. |
    +---------------+----------------------------+

    and methods:

    .. autosummary::
        :nosignatures:

        is_dependent
        reduce_table
        read_table
        read_logic

    At a minimum, LogicNetworks accept a truth table at initialization.  A
    truth table stores a list of tuples, one for each node in order. A tuple of
    the form ``(A, {C1, C2, ...})`` at index ``i`` provides the activation
    conditions for the node of index ``i``. ``A`` is a tuple marking the
    indices of the nodes which influence the state of node ``i`` via logic
    relations. ``{C1, C2, ...}`` is a set, each element of which is the
    collection of binary states of these influencing nodes that would activate
    node ``i``, setting it to ``1``. Any other collection of states of nodes in
    ``A`` are assumed to deactivate node ``i``, setting it to ``0``.

    ``C1``, ``C2``, etc. are sequences (``tuple`` or ``str``) of binary digits,
    each being the binary state of corresponding node in ``A``.

    The following network has a single node, which is only activates when it is
    in the ``0`` state. That is, it alternates between ``0`` and ``1``.

    .. doctest:: logicnetwork

        >>> net = LogicNetwork([((0,), {'0'})])
        >>> net.size
        1
        >>> net.table
        [((0,), {'0'})]

    A more complicated network, with three nodes. Here, node ``0`` activates in
    the next state whenever node ``1`` is deactivated; node ``1`` activates
    based on the state of nodes ``1`` and ``2``; and node ``2`` activates based
    on its own state.

    .. doctest:: logicnetwork

        >>> net = LogicNetwork([((1,), {'0'}), ((1,2), {'10', '11'}), ((2,), {'1'})])
        >>> net.size
        3
        >>> net.table == [((1,), {'0'}), ((1, 2), {'10', '11'}), ((2,), {'1'})]
        True

    Notice that node ``1`` will fall into the activated state regardless of
    what node ``2`` is doing. In other words, the edge :math:`2 \\rightarrow
    1`` is not a real edge. The table can be reduced to remove such an "fake"
    edge using the ``reduced`` argument:

    .. doctest:: logicnetwork

        >>> net = LogicNetwork([((1,), {'0'}), ((1,2), {'10', '11'}), ((2,), {'1'})])
        >>> net.table == [((1,), {'0'}), ((1, 2), {'10', '11'}), ((2,), {'1'})]
        True
        >>> net = LogicNetwork([((1,), {'0'}), ((1,2), {'10', '11'}), ((2,), {'1'})], reduced=True)
        >>> net.table == [((1,), {'0'}), ((1,), {'1'}), ((2,), {'1'})]
        True

    :param table: the logic table
    :type table: list, tuple
    :param reduced: reduce the table
    :type reduced: bool
    :param names: an iterable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
    :raises TypeError: if the rows of the table are neither ``list`` nor ``tuple``
    :raises IndexError: if a node depends another which doesn't have a row in the table
    :raises TypeError: if the truth conditions are neither ``list``, ``tuple`` nor ``set``.
    """

    def __init__(self, table, reduced=False, names=None, metadata=None):
        if not isinstance(table, (list, tuple)):
            raise TypeError("table must be a list or tuple")

        super(LogicNetwork, self).__init__(size=len(table), names=names, metadata=metadata)

        # Store positive truth table for human reader.
        self.table = []
        for row in table:
            # Validate incoming indices.
            if not (isinstance(row, (list, tuple)) and len(row) == 2):
                raise TypeError("Invalid table format")
            for idx in row[0]:
                if idx >= self.size:
                    raise IndexError("mask index out of range")
            # Validate truth table of the sub net.
            if not isinstance(row[1], (list, tuple, set)):
                raise TypeError("Invalid table format")
            conditions = set()
            for condition in row[1]:
                conditions.add(''.join([str(long(s)) for s in condition]))
            self.table.append((row[0], conditions))

        if reduced:
            self.reduce_table()

        # Encode truth table for faster computation.
        self._encode_table()

    def _encode_table(self):
        self._encoded_table = []
        for indices, conditions in self.table:
            # Encode the mask.
            mask_code = long(0)
            for idx in indices:
                mask_code += 2 ** long(idx)  # Low order, low index.
            # Encode each condition of truth table.
            encoded_sub_table = set()
            for condition in conditions:
                encoded_condition = long(0)
                for idx, state in zip(indices, condition):
                    encoded_condition += 2 ** long(idx) if int(state) else 0
                encoded_sub_table.add(encoded_condition)
            self._encoded_table.append((mask_code, encoded_sub_table))

    def is_dependent(self, target, source):
        """
        Is the ``target`` node dependent on the state of ``source``?

        .. doctest:: logicnetwork

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
            ... ((0, 2), {'01', '10', '11'}),
            ... ((0, 1), {'11'})])
            >>> net.is_dependent(0, 0)
            False
            >>> net.is_dependent(0, 2)
            True

        :param target: index of the target node
        :type target: int
        :param source: index of the source node
        :type source: int
        :return: whether the target node is dependent on the source
        """
        sub_table = self.table[target]
        if source not in sub_table[0]:  # No explicit dependency.
            return False

        # Determine implicit dependency.
        i = sub_table[0].index(source)
        counter = {}
        for state in sub_table[1]:
            # State excluding source.
            state_sans_source = state[:i] + state[i + 1:]
            if long(state[i]) == 1:
                counter[state_sans_source] = counter.get(
                    state_sans_source, 0) + 1
            else:
                counter[state_sans_source] = counter.get(
                    state_sans_source, 0) - 1

        if any(counter.values()):  # States uneven.
            return True
        return False

    def reduce_table(self):
        """
        Reduce truth table by removing input nodes which have no logic
        influence from the truth table of each node.

        .. note::

            This function introduces the identity function for all nodes which
            have no inputs. This ensure that every node has a well-defined
            logical function. The example below demonstrates this with node
            ``1``.

        .. doctest:: logicnetwork

            >>> net = LogicNetwork([((0,1), {'00', '10'}), ((0,), {'0', '1'})])
            >>> net.table == [((0,1), {'00', '10'}), ((0,), {'0', '1'})]
            True
            >>> net.reduce_table()
            >>> net.table == [((1,), {'0'}), ((1,), {'0', '1'})]
            True

        """
        reduced_table = []
        for node, (sources, conditions) in enumerate(self.table):
            reduced_sources = []
            reduced_indices = []
            for idx, source in enumerate(sources):
                if self.is_dependent(node, source):
                    reduced_sources.append(source)
                    reduced_indices.append(idx)

            if reduced_sources:  # Node state is influenced by other nodes.
                reduced_conditions = set()
                for condition in conditions:
                    reduced_condition = ''.join([str(condition[idx])
                                                 for idx in reduced_indices])
                    reduced_conditions.add(reduced_condition)
            else:
                # Node state is not influenced by other nodes including itself.
                reduced_sources = (node, )
                if not conditions:
                    # If original conditions is empty, node is never activated.
                    reduced_conditions = set()
                else:
                    # Node is always activated no matter its previous state.
                    reduced_conditions = {'0', '1'}

            reduced_table.append((tuple(reduced_sources), reduced_conditions))

        self.table = reduced_table

        self._encode_table()

    def _unsafe_update(self, net_state, index=None, pin=None, values=None):
        encoded_state = self._unsafe_encode(net_state)

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

    @classmethod
    def read_table(cls, table_path, reduced=False, metadata=None):
        """
        Read a network from a truth table file.

        A logic table file starts with a table title which contains names of
        all nodes. It is a line marked by ``##`` at the begining with node
        names seperated by commas or spaces. This line is required. For
        artificial network without node names, arbitrary names must be put in
        place, e.g.:

        ::

            ## A B C D

        Following are the sub-tables of logic conditions for every node. Each
        sub-table nominates a node and its logically connected nodes in par-
        enthesis as a comment line:

        ::

            # A (B C)

        The rest of the sub-table are states of those nodes in parenthesis
        ``(B, C)`` that would activate the state of A. States that would
        deactivate ``A`` should not be included in the sub-table.

        A complete logic table with 3 nodes A, B, C would look like this:

        ::

            ## A B C
            # A (B C)
            1 0
            1 1
            # B (A)
            1
            # C (B C A)
            1 0 1
            0 1 0
            0 1 1

        Custom comments can be added above or below the table title (as long as
        they are preceeded with more or less than two ``#`` (e.g. ``#`` or
        ``###`` but not ``##``)).

        .. rubric:: Examples:

        .. testcode:: logicnetwork

            print(open(MYELOID_TRUTH_TABLE, 'r').read())

        .. testoutput:: logicnetwork

            ## GATA-2, GATA-1, FOG-1, EKLF, Fli-1, SCL, C/EBPa, PU.1, cJun, EgrNab, Gfi-1
            # GATA-2 (GATA-2, GATA-1, FOG-1, PU.1)
            1 1 0 0
            1 0 1 0
            1 0 0 0
            # GATA-1 (GATA-1, GATA-2, Fli-1, PU.1)
            1 0 0 0
            0 1 0 0
            0 0 1 0
            1 1 0 0
            1 0 1 0
            0 1 1 0
            1 1 1 0
            # FOG-1 (GATA-1)
            1
            ...

        .. doctest:: logicnetwork

            >>> net = LogicNetwork.read_table(MYELOID_TRUTH_TABLE)
            >>> net.size
            11
            >>> net.names
            ['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL', 'C/EBPa', 'PU.1', 'cJun', 'EgrNab', 'Gfi-1']
            >>> net.table ==  [((0, 1, 2, 7), {'1000', '1010', '1100'}),
            ... ((1, 0, 4, 7), {'0010', '0100', '0110', '1000', '1010', '1100', '1110'}),
            ... ((1,), {'1'}),
            ... ((1, 4), {'10'}),
            ... ((1, 3), {'10'}),
            ... ((1, 7), {'10'}),
            ... ((6, 1, 2, 5), {'1000', '1001', '1010', '1011', '1100', '1101', '1110'}),
            ... ((6, 7, 1, 0), {'0100', '1000', '1100'}),
            ... ((7, 10), {'10'}),
            ... ((7, 8, 10), {'110'}),
            ... ((6, 9), {'10'})]
            True

        :param table_path: a path to a table table file
        :type table_path: str
        :param reduced: reduce the table
        :type reduced: bool
        :param names: an iterable object of the names of the nodes in the network
        :type names: seq
        :param metadata: metadata dictionary for the network
        :type metadata: dict
        :return: a :class:`LogicNetwork`
        """
        names_format = re.compile(r'^\s*##[^#]+$')
        node_title_format = re.compile(
            r'^\s*#\s*(\S+)\s*\((\s*(\S+\s*)+)\)\s*$')

        with open(table_path, 'r') as f:
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
                        raise FormatError("'{}' not in node names".format(node_name))
                    node_index = names.index(node_name)
                    sub_net_nodes = re.split(
                        r'\s*,\s*|\s+', node_title.group(2).strip())

                    in_nodes = tuple(map(names.index, sub_net_nodes))
                    table[node_index] = (in_nodes, set())
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
                table[i] = ((i,), {'1'})

        return cls(table, reduced=reduced, names=names, metadata=metadata)

    @classmethod
    def read_logic(cls, logic_path, external_nodes_path=None, reduced=False, metadata=None):
        """
        Read a network from a file of logic equations.

        A logic equations has the form of ``A = B AND ( C OR D )``, each term
        being separated from parantheses and logic operators with at least a
        space. The optional ``external_nodes_path`` takes a file that contains
        nodes in a column whose states do not depend on any nodes. These are
        considered "external" nodes. Equivalently, such a node would have a
        logic equation ``A = A``, for its state stays on or off unless being
        set externally.

        .. rubric:: Examples

        .. testcode:: logicnetwork

            print(open(MYELOID_LOGIC_EXPRESSIONS, 'r').read())

        .. testoutput:: logicnetwork

            GATA-2 = GATA-2 AND NOT ( GATA-1 AND FOG-1 ) AND NOT PU.1
            GATA-1 = ( GATA-1 OR GATA-2 OR Fli-1 ) AND NOT PU.1
            FOG-1 = GATA-1
            EKLF = GATA-1 AND NOT Fli-1
            Fli-1 = GATA-1 AND NOT EKLF
            SCL = GATA-1 AND NOT PU.1
            C/EBPa = C/EBPa AND NOT ( GATA-1 AND FOG-1 AND SCL )
            PU.1 = ( C/EBPa OR PU.1 ) AND NOT ( GATA-1 OR GATA-2 )
            cJun = PU.1 AND NOT Gfi-1
            EgrNab = ( PU.1 AND cJun ) AND NOT Gfi-1
            Gfi-1 = C/EBPa AND NOT EgrNab

        .. doctest:: logicnetwork

            >>> net = LogicNetwork.read_logic(MYELOID_LOGIC_EXPRESSIONS)
            >>> net.size
            11
            >>> net.names
            ['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL', 'C/EBPa', 'PU.1', 'cJun', 'EgrNab', 'Gfi-1']
            >>> net.table ==  [((0, 1, 2, 7), {'1000', '1010', '1100'}),
            ... ((1, 0, 4, 7), {'0010', '0100', '0110', '1000', '1010', '1100', '1110'}),
            ... ((1,), {'1'}),
            ... ((1, 4), {'10'}),
            ... ((1, 3), {'10'}),
            ... ((1, 7), {'10'}),
            ... ((6, 1, 2, 5), {'1000', '1001', '1010', '1011', '1100', '1101', '1110'}),
            ... ((6, 7, 1, 0), {'0100', '1000', '1100'}),
            ... ((7, 10), {'10'}),
            ... ((7, 8, 10), {'110'}),
            ... ((6, 9), {'10'})]
            True

        :param logic_path: path to a file of logial expressions
        :type logic_path: str
        :param external_nodes_path: a path to a file of external nodes
        :type external_nodes_path: str
        :param reduced: reduce the table
        :type reduced: bool
        :param names: an iterable object of the names of the nodes in the network
        :type names: seq
        :param metadata: metadata dictionary for the network
        :type metadata: dict
        :return: a :class:`LogicNetwork`
        """
        names = []
        expressions = []
        with open(logic_path) as eq_file:
            for eq in eq_file:
                name, expr = eq.split('=')
                names.append(name.strip())
                expressions.append(expr.strip())

        if external_nodes_path:
            with open(external_nodes_path) as extra_file:
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
                        raise FormatError("unknown component '{}'".format(item))
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
        if external_nodes_path:
            for i in range(len(extras)):
                table.append((((len(names) - len(extras) + i),), set('1')))

        return cls(table, reduced=reduced, names=names, metadata=metadata)

    def neighbors_in(self, index, *args, **kwargs):
        return set(self.table[index][0])

    def neighbors_out(self, index, *args, **kwargs):
        outgoing_neighbors = set()
        for i, incoming_neighbors in enumerate([row[0] for row in self.table]):
            if index in incoming_neighbors:
                outgoing_neighbors.add(i)

        return outgoing_neighbors


BooleanNetwork.register(LogicNetwork)
