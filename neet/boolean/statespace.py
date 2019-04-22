from neet.python import long
from neet.statespace import UniformSpace
import copy


class BooleanSpace(UniformSpace):
    def __init__(self, ndim):
        super(BooleanSpace, self).__init__(ndim, base=2)

    def __iter__(self):
        ndim = self.ndim
        state = [0] * ndim
        yield state[:]
        i = 0
        while i != ndim:
            if state[i] == 0:
                state[i] = 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def __contains__(self, state):
        try:
            if len(state) != self.ndim:
                return False

            for x in state:
                if x != 0 and x != 1:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)
        for x in state:
            encoded += place * long(x)
            place <<= 1
        return encoded

    def decode(self, encoded):
        ndim = self.ndim
        state = [0] * ndim
        for i in range(ndim):
            state[i] = encoded & 1
            encoded >>= 1
        return state

    def subspace(self, indices, state=None):
        ndim = self.ndim

        if state is not None and state not in self:
            raise ValueError('provided state is not in the state space')
        elif state is None:
            state = [0] * ndim

        indices = list(set(indices))
        indices.sort()
        nindices = len(indices)

        if nindices == 0:
            yield copy.copy(state)
        elif indices[0] < 0 or indices[-1] >= ndim:
            raise IndexError('index out of range')
        elif nindices == ndim:
            for state in self:
                yield state
        else:

            initial = copy.copy(state)

            yield copy.copy(state)
            i = 0
            while i != nindices:
                if state[indices[i]] == initial[indices[i]]:
                    state[indices[i]] ^= 1
                    for j in range(i):
                        state[indices[j]] = initial[indices[j]]
                    i = 0
                    yield copy.copy(state)
                else:
                    i += 1

    def hamming_neighbors(self, state):
        if state not in self:
            raise ValueError('state is not in state space')
        neighbors = [None] * self.ndim
        for i in range(self.ndim):
            neighbors[i] = copy.copy(state)
            neighbors[i][i] ^= 1
        return neighbors

    def distance(self, a, b):
        if a not in self:
            raise ValueError('first state is not in state space')
        if b not in self:
            raise ValueError('second state is not in state space')
        out = 0
        for i in range(self.ndim):
            out += a[i] ^ b[i]
        return out
