import random
from neet.statespace import StateSpace
from .logicnetwork import LogicNetwork


def random_logic(logic_net, p=0.5, fixed_connections=True):
    """
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    if fixed_connections:
        return random_logic_fixed_connections(logic_net, p)
    else:
        return random_logic_free_connections(logic_net, p)


def random_logic_fixed_connections(logic_net, p=0.5):
    """
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    new_table = []
    for row in logic_net.table:
        indices = row[0]

        conditions = set()
        for state in StateSpace(len(indices)):
            if random.random() > p:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)


def random_logic_free_connections(logic_net, p=0.5):
    """
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    new_table = []
    for i in range(logic_net.size):
        n_indices = random.randint(1, logic_net.size)
        indices = tuple(sorted(random.sample(
            range(logic_net.size), k=n_indices)))

        conditions = set()
        for state in StateSpace(n_indices):
            if random.random() > p:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)
