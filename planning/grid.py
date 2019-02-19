import numpy as np
from math import ceil, floor
from planning.dubins_util import neg_pi_to_pi
import pickle


class Grid:
    def __init__(self, config, min_val, max_val):

        # allow checking for overflow or underflow
        np.seterr(over='raise')
        np.seterr(under='raise')

        # set up grid
        self.config = config
        self.max_val = max_val[0:4].reshape((-1, 1))
        self.n_dim = self.max_val.shape[0]
        self.min_val = min_val[0:4].reshape((self.n_dim, 1))

        self.margin = 4
        self.half_margin = int(self.margin / 2)

        self.default_val = float(config['grid_weight'])

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

        noise_var = 2.0
        self.noise_res = self.lookup_res.reshape((1, -1)) * noise_var
        self.noise_mean = np.array([0.0, 0.0, 0.0, 0.0]).reshape((1, -1))

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

        val = self.grid.get(ind)

        if val is None:
            val = self.default_val
        # if val>100:
        #     print(val)
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
        self.grid = pickle.load(open(fname, "rb"))

    def save_grid(self, fname=None):
        if fname is None:
            fname = self.fname
        pickle.dump(self.grid, open(fname, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def gradient_step(self, path, step_size):  # path in world coordinates

        n_path_points = path.shape[0]
        noise = np.random.normal(self.noise_mean, self.noise_res, size=(n_path_points, 4))
        for i in range(0, n_path_points):
            loc_noise = path[i, 0:4] + noise[i, :]
            loc_noise[3] = neg_pi_to_pi(loc_noise[3])
            try:
                old_val = self.get(loc_noise)
                if old_val > 0:
                    new_val = max(old_val + step_size, 0)
                    self.set(loc_noise, new_val)
            except FloatingPointError:  # don't update if overflow or underflow
                pass

    # def gradient_step(self, path, step_size):  # path in world coordinates,
    #
    #     n_path_points = path.shape[0]
    #     step_size_n = step_size / n_path_points
    #     noise = np.random.normal(self.noise_mean, self.noise_res, size=(n_path_points, 4))
    #     for i in range(0, n_path_points):
    #         loc_noise = path[i, 0:4] + noise[i, :]
    #         loc_noise[3] = neg_pi_to_pi(loc_noise[3])
    #         try:
    #             old_val = self.get(loc_noise)
    #             #new_val = old_val * np.exp(step_size_n * max(1.0, old_val)) # true gradient
    #             new_val = old_val * np.exp(step_size_n)  # fixed step
    #             #print(new_val)
    #             self.set(loc_noise, new_val)
    #         except FloatingPointError:
    #             # don't update if overflow or underflow
    #             pass
