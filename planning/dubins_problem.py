import numpy as np
import math
from math import sqrt
from planning.dubins_util import dubins_path
inf = float("inf")

class DubinsProblem:

    def __init__(self, config, coord_min, coord_max):

        # TODO shorten primitive length to be short enough for interpolation needed for cost computations

        self.config = config
        self.coord_min = coord_min
        self.coord_max = coord_max

        self.v = float(config['velocity'])
        self.delta_theta = float(config['delta_theta'])
        self.curvature = self.delta_theta / self.v
        self.delta_z = float(config['delta_z'])

        self.ps_theta = np.array([0.0, -1.0, 1.0]) * self.delta_theta
        self.ps_z = np.array([0.0, -1.0, 1.0]) * self.delta_z
        self.num_ps_theta = len(self.ps_theta)
        self.num_ps_z = len(self.ps_z)

        self.dt = float(config['dt'])
        self.ddt = float(config['ddt'])

        self.lookup_res_xyz = float(config['dind_res_xyz'])
        self.lookup_res_theta = float(config['dind_res_theta'])
        self.lookup_res_time = float(config['dind_res_time'])
        self.lookup_res = np.array(
            [self.lookup_res_xyz, self.lookup_res_xyz, self.lookup_res_xyz, self.lookup_res_theta, self.lookup_res_time])

        # TODO make a single array of all the primitives.
        # TODO this is more complicated actually due to dz.
        self.primitive_cost = self.v * self.dt

        # generate from data
        self.lookup_num_thetas = int(2 * math.pi / self.lookup_res_theta) + 1
        self.lookup_num_x = int((coord_max[0] - coord_min[0]) / self.lookup_res_xyz) + 1
        self.lookup_num_y = int((coord_max[1] - coord_min[1]) / self.lookup_res_xyz) + 1
        self.lookup_num_z = int((coord_max[2] - coord_min[2]) / self.lookup_res_xyz) + 1

        self.hash_a = self.lookup_num_x * self.lookup_num_y * self.lookup_num_z * self.num_ps_theta
        self.hash_b = self.lookup_num_y * self.lookup_num_z * self.num_ps_theta
        self.hash_c = self.lookup_num_z * self.num_ps_theta
        self.hash_d = self.num_ps_theta
        self.hash_coeffs = np.array([self.hash_a, self.hash_b, self.hash_c, self.hash_d, 1])

        # make lookup tables here for the end of each primitive
        # z depends only on the z primitive
        self.lookup_delta_z = (self.ps_z / self.lookup_res_xyz).astype(int)

        # t depends only on the time resolution
        self.delta_time = int(self.dt / self.lookup_res_time)

        # x,y will depend on theta primitive and current theta
        self.lookup_delta_x = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)
        self.lookup_delta_y = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)

        # theta also depends on both because modulo 2pi
        self.lookup_theta = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)

        for i in range(0, self.num_ps_theta):
            for j in range(0, self.lookup_num_thetas):
                theta = j * self.lookup_res_theta + self.coord_min[3]
                dx = dy = 0
                for t in range(0, int(self.dt / self.ddt)):
                    dx = dx + self.ddt * self.v * math.cos(theta)
                    dy = dy + self.ddt * self.v * math.sin(theta)
                    theta = np.mod(theta + self.ddt * self.ps_theta[i], 2 * np.pi)

                self.lookup_delta_x[i, j] = int(dx / self.lookup_res_xyz)
                self.lookup_delta_y[i, j] = int(dy / self.lookup_res_xyz)
                self.lookup_theta[i, j] = int(theta / self.lookup_res_theta)

    def to_ind(self, loc):
        return ((loc - self.coord_min) / self.lookup_res).astype(int)

    def to_loc(self, ind):
        return ind * self.lookup_res + self.coord_min

    def get_neighbors(self, ind):
        neighbors = []
        for dzi in range(0, self.num_ps_z):
            for dti in range(0, self.num_ps_theta):
                neigh = np.array((5,))
                neigh[0] = ind[0] + self.lookup_delta_x[dti, ind[3]]
                neigh[1] = ind[1] + self.lookup_delta_y[dti, ind[3]]
                neigh[2] = ind[2] + self.lookup_delta_z[dzi]
                neigh[3] = self.lookup_theta[dti, ind[3]]
                neigh[4] = ind[4] + self.delta_time
                neighbors.append(neigh)
        return neighbors

    def hash_ind(self, ind):
        return ind.T.dot(self.hash_coeffs)

    def heuristic(self, ind_start, ind_end):
        return self.dubins_distance(ind_start, ind_end)

    # TODO - implement dubins distance in primitive coordinates (easy if x,y,z resolutions are same)
    def dubins_distance(self, start, goal):

        bc = self.curvature  # largest curvature is always more efficient
        bcost, bt, bp, bq, bmode = dubins_path(self.x, self.y, self.theta, goal.x, goal.y, goal.theta, bc)

        turn_speed = bc * self.v
        tpq = [bt, bp, bq]

        dt = np.zeros((3, 1))

        for i in range(0, 3):
            if bmode[i] == "L":
                dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "R":
                dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "S":
                dt[i] = tpq[i] / bc / self.v

        delta_time = np.sum(dt)

        delta_z = abs(self.z - goal.z)

        while delta_z / delta_time > max(self.z_prims):
            delta_time = delta_time + 2 * math.pi / turn_speed

        return sqrt((delta_time * self.v) ** 2 + (delta_z) ** 2)  # * 2 # L2

    # TODO interpolate the full path in physical space using splines
    def interpolate_waypoints(self, waypoints):
        return None















