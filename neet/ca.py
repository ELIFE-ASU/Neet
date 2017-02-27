# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

class ECA(object):
    def __init__(self, code):
        if 255 < code or code < 0:
            raise(ValueError("invalid ECA code"))
        self.code = code

    @classmethod
    def __check_arguments(self, lattice, n):
        if n < 1:
            raise(ValueError("cannot update lattice fewer than once"))

        if len(lattice) < 3:
            raise(ValueError("lattice is too short"))

        for x in lattice:
            if x != 0 and x != 1:
                msg = "invalid value {} in lattice".format(x)
                raise(ValueError(msg))

    def __unsafe_update(self, lattice, n):
        for m in range(n):
            a = lattice[0]
            d = 2 * lattice[-1] + lattice[0]
            for i in range(1,len(lattice)):
                d = 7 & (2 * d + lattice[i])
                lattice[i-1] = 1 & (self.code >> d)
            d = 7 & (2 * d + a)
            lattice[-1] = 1 & (self.code >> d)

    def update(self, lattice, n=1):
        ECA.__check_arguments(lattice, n)
        self.__unsafe_update(lattice, n)

    def step(self, lattice, n=1):
        ECA.__check_arguments(lattice, n)
        l = lattice[:]
        self.__unsafe_update(l, n)
        return l
