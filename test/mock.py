# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import neet.statespace

class MockObject(object):
    """
    A Generic Object
    """
    pass

class MockNetwork(object):
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

class MockFixedSizedNetwork(object):
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
