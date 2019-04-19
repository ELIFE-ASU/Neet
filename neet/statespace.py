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
                self._ndim = len(shape)
                self._shape = shape[:]
        else:
            raise TypeError("shape must be a list")

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def volume(self):
        return self._volume

    def __iter__(self):
        ndim, shape = self.ndim, self.shape
        state = [0] * ndim
        yield state[:]
        i = 0
        while i != ndim:
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
            if len(states) != self.ndim:
                return False

            for state, base in zip(states, self.shape):
                if state < 0 or state >= base:
                    return False
            return True
        except TypeError:
            return False
        except IndexError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)

        for (x, b) in zip(state, self.shape):
            encoded += place * x
            place *= b

        return long(encoded)

    def encode(self, state):
        if state not in self:
            raise ValueError("state is not in state space")

        return self._unsafe_encode(state)

    def decode(self, encoded):
        ndim = self.ndim
        state = [0] * ndim
        for (i, base) in enumerate(self.shape):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state


class UniformSpace(StateSpace):
    def __init__(self, ndim, base):
        super(UniformSpace, self).__init__([base] * ndim)
        self._base = base

    @property
    def base(self):
        return self._base

    def __iter__(self):
        ndim, base = self.ndim, self.base
        state = [0] * ndim
        yield state[:]
        i = 0
        while i != ndim:
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
            if len(state) != self.ndim:
                return False

            base = self.base
            for x in state:
                if x < 0 or x >= base:
                    return False
            return True
        except TypeError:
            return False
        except IndexError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)

        base = self.base
        for x in state:
            encoded += place * long(x)
            place *= base

        return encoded

    def decode(self, encoded):
        ndim, base = self.ndim, self.base
        state = [0] * ndim
        for i in range(ndim):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state


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
                state[i] += 1
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
        except IndexError:
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
