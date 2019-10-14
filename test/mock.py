from neet import Network, UniformNetwork
from neet.boolean import BooleanNetwork


class MockObject(object):
    """
    A Generic Object
    """
    pass


class MockNetwork(Network):
    """
    A mock network
    """

    def __init__(self, shape, names=None, metadata=None):
        """
        mock init simply calls super's __init__
        """
        super(MockNetwork, self).__init__(shape, names=names, metadata=metadata)

    def _unsafe_update(self, lattice, index=None, pin=None, value=None):
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


Network.register(MockNetwork)


class MockUniformNetwork(UniformNetwork):
    """
    A mock Uniform network
    """

    def __init__(self, size, base, names=None, metadata=None):
        """
        mock init simply class super's __init__
        """
        super(MockUniformNetwork, self).__init__(size, base, names=None, metadata=None)

    def _unsafe_update(self, lattice, index=None, pin=None, value=None):
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


MockUniformNetwork.register(MockUniformNetwork)


class MockBooleanNetwork(BooleanNetwork):
    """
    A mock Boolean network
    """

    def __init__(self, size):
        """
        mock init simply calls super's __init__
        """
        super(MockBooleanNetwork, self).__init__(size)

    def _unsafe_update(self, lattice, index=None, pin=None, value=None):
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


BooleanNetwork.register(MockBooleanNetwork)
