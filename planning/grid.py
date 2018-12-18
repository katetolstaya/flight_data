import numpy as np
from math import ceil, floor
import pickle
# from sklearn.gaussian_process.kernels import RBF


class Grid:
    def __init__(self, config, min_val, max_val):

        # set up grid
        self.config = config
        self.max_val = max_val[0:4].reshape((-1, 1))
        self.n_dim = self.max_val.shape[0]
        self.min_val = min_val[0:4].reshape((self.n_dim, 1))

        self.margin = 4
        self.half_margin = int(self.margin / 2)

        self.sigma = float(config['grid_sigma'])
        xy_res = float(config['grid_res_xy'])
        z_res = float(config['grid_res_z'])
        theta_res = float(config['grid_res_theta'])
        self.resolution = np.array([xy_res, xy_res, z_res, theta_res]).flatten()

        self.lookup_res_xy = float(config['dind_res_xy'])
        self.lookup_res_z = float(config['dind_res_z'])
        self.lookup_res_theta = float(config['dind_res_theta'])
        self.lookup_res = np.array(
            [self.lookup_res_xy, self.lookup_res_xy, self.lookup_res_z, self.lookup_res_theta])
        self.lookup_res = self.lookup_res.flatten() / self.resolution.flatten()

        self.fname = config['grid_filename']
        self.fname = 'model/' + self.fname + '.pkl'

        self.n = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            self.n[i] = ceil(ceil((self.max_val[i] - self.min_val[i]) / self.resolution[i])) + self.margin
        self.grid = np.zeros(map(tuple, self.n.T)[0])

        # want to use a kernel- update more than one cell, with some variance
        # vec = np.arange(-2, 3).astype(float)
        # X, Y, Z, T = np.meshgrid(vec * xy_res, vec * xy_res, vec * z_res, vec * theta_res)
        # X = X.reshape((-1,1))
        # Y = Y.reshape((-1,1))
        # Z = Z.reshape((-1,1))
        # T = T.reshape((-1,1))
        #
        # coords = np.stack((X, Y, Z, T), axis=1).reshape((-1,4))
        # rbf = RBF(length_scale=self.sigma)
        #
        # self.coord_kernels = rbf(coords, np.zeros((1, 4))).reshape((5,5,5,5))

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
        arr = np.multiply(x[0:4], self.lookup_res).astype(int) + self.half_margin
        return tuple(arr)

    def loc_to_index(self, x):
        ind = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            ind[i] = floor(((x[i] - self.min_val[i]) / self.resolution[i])) + self.half_margin
        ind = map(tuple, ind.T)[0]
        return ind

    def set(self, x, val):
        try:
            self.grid[self.loc_to_index(x)] = val
            return
        except IndexError:
            return # do nothing for values out of bounds

    def load_grid(self, fname=None):
        if fname is None:
            fname = self.fname
        pickle.load(open(fname, 'rb'))

    def save_grid(self, fname=None):
        if fname is None:
            fname = self.fname
        pickle.dump(self.grid, open(fname, 'wb'))

    # def update(self, x, u):
    #     try:
    #         x_ind = self.loc_to_index(x)
    #         temp = self.grid[x_ind[0]-2:x_ind[0] + 3, x_ind[1]-2:x_ind[1] + 3, x_ind[2]-2:x_ind[2] + 3, x_ind[3]-2:x_ind[3] + 3]
    #         self.grid[x_ind[0]-2:x_ind[0] + 3, x_ind[1]-2:x_ind[1]+3, x_ind[2]-2:x_ind[2]+3, x_ind[3]-2:x_ind[3]+3] = temp + self.coord_kernels * u
    #         return
    #     except IndexError:
    #         return # do nothing for values out of bounds
    #     except ValueError:
    #         return
