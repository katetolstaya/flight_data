import numpy as np
from math import ceil, floor
from planning.dubins_util import neg_pi_to_pi
import pickle
# from sklearn.gaussian_process.kernels import RBF
from scipy.sparse import csc_matrix

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
        self.fname = 'model/' + config['grid_filename'] + '.pkl'
        self.n = np.zeros((self.n_dim, 1), dtype=int)
        for i in range(0, self.n_dim):
            self.n[i] = ceil(ceil((self.max_val[i] - self.min_val[i]) / self.resolution[i])) + self.margin

        self.grid = {}

    def get(self, x):

        if x.dtype == int:
            ind = self.ind_to_index(x)  # convert from planning indices to grid indices
        else:
            ind = self.loc_to_index(x)  # convert from locations to grid indices

        try:

            if ind not in self.grid:
                self.grid[ind] = 0
                val = 0
            else:
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
            return  # do nothing for values out of bounds

    def load_grid(self, fname=None):
        if fname is None:
            fname = self.fname
        #self.grid = np.load(fname)
        self.grid = pickle.load(open(fname, "rb"))

    def save_grid(self, fname=None):
        if fname is None:
            fname = self.fname
        #np.save(fname, self.grid)
        pickle.dump(self.grid, open(fname, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def update(self, path, coeff):
        n_path_points = path.shape[0]
        for i in range(0, n_path_points):
            noise = np.random.normal(0, 0.25, size=(4,))
            noise[2] = 0.1 * noise[2]
            noise[3] = 0.1 * noise[3]
            temp = path[i, 0:4]  + noise
            temp[3] = neg_pi_to_pi(temp[3])
            self.set(temp, self.get(temp) + coeff * 1.0 / n_path_points)
            # grid.update(temp, coeff * 1.0 / M)

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
