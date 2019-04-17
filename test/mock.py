import neet.statespace
from neet.interfaces import Network


class MockObject(object):
    """
    A Generic Object
    """
    pass


class MockNetwork(Network):
    """
    A mock, variable sized network
    """

    def update(self, lattice):
        """
        mock update method
        """
        pass

    def state_space(self, size):
        """
        mock state space method
        """
        return neet.statespace.StateSpace(size)

    def neighbors(self):
        """
        mock neighbors method
        """
        pass

    @property
    def size(self):
        return 0


Network.register(MockNetwork)


class MockFixedSizedNetwork(Network):
    """
    A mock fixed-sized network
    """

    def update(self, lattice):
        """
        mock update method
        """
        pass

    @property
    def size(self):
        """
        mock size property
        """
        pass

    def state_space(self):
        """
        mock state space method
        """
        return neet.statespace.StateSpace(1)

    def neighbors(self):
        """
        mock neighbors method
        """
        pass


Network.register(MockFixedSizedNetwork)
