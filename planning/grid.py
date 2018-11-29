import numpy as np
from math import ceil, floor

margin = 4
half_margin = int(margin / 2)


class Grid:
    def __init__(self, min_val, max_val, resolution):

        # set up grid
        self.max_val = max_val.reshape((-1, 1)) 
        self.n_dim = self.max_val.shape[0]
        self.min_val = min_val.reshape((self.n_dim, 1))
        self.resolution = resolution.reshape((self.n_dim, 1))
        self.n = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            self.n[i] = ceil(ceil((self.max_val[i] - self.min_val[i]) / self.resolution[i])) + margin
        self.n = self.n 
        self.grid = np.ones(map(tuple, self.n.T)[0]) * 50.0

        self.last_ind = None
        self.last_x = None

    def get(self, x):

        ind = self.to_index(x)
        try:
            val = self.grid[ind]
        except IndexError:
            val = float("inf") # out of bounds => inf
        return val

    def to_index(self, x):
        if self.last_x is not None and np.array_equal(self.last_x, x):
            return self.last_ind

        ind = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            ind[i] =  floor(((x[i] - self.min_val[i]) / self.resolution[i])) + half_margin  
        ind = map(tuple, ind.T)[0]
        self.last_x = x
        self.last_ind = ind
        return ind

    def set(self, x, val):
        try:
            self.grid[self.to_index(x)] = val
            return
        except IndexError:
            return # do nothing for values out of bounds
