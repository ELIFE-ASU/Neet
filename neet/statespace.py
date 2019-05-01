from .python import long


class StateSpace(object):
    def __init__(self, shape):
        if isinstance(shape, list):
            if len(shape) == 0:
                raise ValueError("shape cannot be empty")
            else:
                self._volume = 1
                for base in shape:
                    if not isinstance(base, int):
                        raise TypeError("shape must be a list of ints")
                    elif base < 1:
                        raise ValueError("shape may only contain positive elements")
                    self._volume *= base
                self._size = len(shape)
                self._shape = shape[:]
        else:
            raise TypeError("shape must be a list")

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def volume(self):
        return self._volume

    def __iter__(self):
        size, shape = self.size, self.shape
        state = [0] * size
        yield state[:]
        i = 0
        while i != size:
            base = shape[i]
            if state[i] + 1 < base:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def __contains__(self, states):
        try:
            if len(states) != self.size:
                return False

            for state, base in zip(states, self.shape):
                if state < 0 or state >= base:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)

        for (x, b) in zip(state, self.shape):
            encoded += place * long(x)
            place *= b

        return encoded

    def encode(self, state):
        if state not in self:
            raise ValueError("state is not in state space")

        return self._unsafe_encode(state)

    def decode(self, encoded):
        size = self.size
        state = [0] * size
        for (i, base) in enumerate(self.shape):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state


class UniformSpace(StateSpace):
    def __init__(self, size, base):
        super(UniformSpace, self).__init__([base] * size)
        self._base = base

    @property
    def base(self):
        return self._base

    def __iter__(self):
        size, base = self.size, self.base
        state = [0] * size
        yield state[:]
        i = 0
        while i != size:
            if state[i] + 1 < base:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def __contains__(self, state):
        try:
            if len(state) != self.size:
                return False

            base = self.base
            for x in state:
                if x < 0 or x >= base:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)

        base = self.base
        for x in state:
            encoded += place * long(x)
            place *= base

        return encoded

    def decode(self, encoded):
        size, base = self.size, self.base
        state = [0] * size
        for i in range(size):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state
