import neet.statespace
from neet.interfaces import Network, BooleanNetwork


class MockObject(object):
    """
    A Generic Object
    """
    pass


class MockNetwork(Network):
    """
    A mock network
    """
    def __init__(self, size):
        """
        mock init simply calls super's __init__
        """
        super(MockNetwork, self).__init__(size)

    def update(self, lattice):
        """
        mock update method
        """
        pass

    def state_space(self):
        """
        mock state space method
        """
        return neet.statespace.StateSpace(self.size)

    def neighbors_in(self, *args, **kwargs):
        """
        mock neighbors method
        """
        pass

    def neighbors_out(self, *args, **kwargs):
        """
        mock neighbors method
        """
        pass

    def to_networkx_graph(self, **kwargs):
        """
        mock method to create networkx graph
        """
        return super(MockNetwork, self).to_networkx_graph(**kwargs)


Network.register(MockNetwork)


class MockBooleanNetwork(BooleanNetwork):
    """
    A mock Boolean network
    """
    def __init__(self, size):
        """
        mock init simply calls super's __init__
        """
        super(MockBooleanNetwork, self).__init__(size)

    def update(self, lattice):
        """
        mock update method
        """
        pass

    def neighbors_in(self, *args, **kwargs):
        """
        mock neighbors method
        """
        pass

    def neighbors_out(self, *args, **kwargs):
        """
        mock neighbors method
        """
        pass

    def to_networkx_graph(self, **kwargs):
        """
        mock method to create networkx graph
        """
        return super(MockNetwork, self).to_networkx_graph(**kwargs)


BooleanNetwork.register(MockBooleanNetwork)
