import numpy as np

inf = float("inf")


class DubinsNode:

    def __init__(self, loc, hash_coeffs, parent=None):
        self.loc = loc
        self.parent = parent
        self.hash_coeffs = hash_coeffs

    def __str__(self):
        return "<" + str(self.loc[0]) + ", " + str(self.loc[1]) + ", " + str(self.loc[2]) + ", " + str(self.loc[3]) + ", " + str(
            self.loc[4]) + ">"

    def __repr__(self):
        return str(self)

    def __eq__(self, othr):
        return isinstance(othr, type(self)) and np.array_equal(self.loc, othr.loc)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.loc.T.dot(self.hash_coeffs)