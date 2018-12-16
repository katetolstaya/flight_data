import numpy as np
from math import ceil, floor

margin = 4
half_margin = int(margin / 2)


class Grid:
    def __init__(self, config, min_val, max_val):

        # set up grid
        self.config = config
        self.max_val = max_val[0:4].reshape((-1, 1))
        self.n_dim = self.max_val.shape[0]
        self.min_val = min_val[0:4].reshape((self.n_dim, 1))

        xy_res = float(config['grid_res_xy'])
        z_res = float(config['grid_res_z'])
        theta_res = float(config['grid_res_theta'])

        self.resolution = np.array([xy_res, xy_res, z_res, theta_res]).flatten()

        self.n = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            self.n[i] = ceil(ceil((self.max_val[i] - self.min_val[i]) / self.resolution[i])) + margin
        self.n = self.n 
        self.grid = np.zeros(map(tuple, self.n.T)[0]) #* 50.0

        self.last_ind = None
        self.last_x = None

        self.lookup_res_xy = float(config['dind_res_xy'])
        self.lookup_res_z = float(config['dind_res_z'])
        self.lookup_res_theta = float(config['dind_res_theta'])
        self.lookup_res = np.array(
            [self.lookup_res_xy, self.lookup_res_xy, self.lookup_res_z, self.lookup_res_theta])
        self.lookup_res = self.lookup_res.flatten() / self.resolution.flatten()

    def get(self, x):

        if x.dtype == int:
            ind = self.ind_to_index(x)
        else:
            ind = self.loc_to_index(x)

        try:
            val = self.grid[ind]
        except IndexError:
            val = float("inf") # out of bounds => inf
        return val

    def ind_to_index(self, x):
        arr = np.multiply(x[0:4], self.lookup_res).astype(int) + half_margin
        return tuple(arr)

    def loc_to_index(self, x):
        if self.last_x is not None and np.array_equal(self.last_x, x):
            return self.last_ind

        ind = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            ind[i] = floor(((x[i] - self.min_val[i]) / self.resolution[i])) + half_margin
        ind = map(tuple, ind.T)[0]
        self.last_x = x
        self.last_ind = ind
        return ind

    def set(self, x, val):
        try:
            self.grid[self.loc_to_index(x)] = val
            return
        except IndexError:
            return # do nothing for values out of bounds
