import numpy as np
import math
from math import sqrt
from planning.dubins_util import dubins_path
from dubins_node import DubinsNode
inf = float("inf")

class DubinsProblem:

    def __init__(self, config, coord_min, coord_max):

        # TODO shorten primitive length to be short enough for interpolation needed for cost computations

        self.config = config
        self.coord_min = coord_min
        self.coord_max = coord_max

        self.dt = float(config['dt'])
        self.ddt = float(config['ddt'])

        self.v_xy = float(config['velocity_x'])
        self.v_theta = float(config['velocity_theta'])
        self.v_z = float(config['velocity_z'])
        self.curvature = self.v_theta / self.v_xy

        self.ps_theta = np.array([0.0, -1.0, 1.0]) * self.v_theta
        self.ps_z = np.array([0.0, -1.0, 1.0]) * self.v_z

        self.max_ps_z = max(self.ps_z)
        self.num_ps_theta = 3 #len(self.ps_theta)
        self.num_ps_z = 3 #len(self.ps_z)

        self.lookup_res_xy = float(config['dind_res_xy'])
        self.lookup_res_z = float(config['dind_res_z'])
        self.lookup_res_theta = float(config['dind_res_theta'])
        self.lookup_res_time = float(config['dind_res_time'])
        self.lookup_res = np.array(
            [self.lookup_res_xy, self.lookup_res_xy, self.lookup_res_z, self.lookup_res_theta, self.lookup_res_time])

        self.goal_res_xy = float(config['goal_res_xy'])
        self.goal_res_z = float(config['goal_res_z'])
        self.goal_res_theta = float(config['goal_res_theta'])
        self.goal_res_time = float(config['goal_res_time'])
        self.goal_res = np.array(
            [self.goal_res_xy, self.goal_res_xy, self.goal_res_z, self.goal_res_theta, self.goal_res_time])
        self.goal_res = self.goal_res / self.lookup_res

        # generate from data
        self.lookup_num_thetas = int(2 * math.pi / self.lookup_res_theta) + 1
        self.lookup_num_x = int((coord_max[0] - coord_min[0]) / self.lookup_res_xy) + 1
        self.lookup_num_y = int((coord_max[1] - coord_min[1]) / self.lookup_res_xy) + 1
        self.lookup_num_z = int((coord_max[2] - coord_min[2]) / self.lookup_res_z) + 1

        self.hash_a = self.lookup_num_x * self.lookup_num_y * self.lookup_num_z * self.num_ps_theta
        self.hash_b = self.lookup_num_y * self.lookup_num_z * self.num_ps_theta
        self.hash_c = self.lookup_num_z * self.num_ps_theta
        self.hash_d = self.num_ps_theta
        self.hash_coeffs = np.array([self.hash_a, self.hash_b, self.hash_c, self.hash_d, 1])

        # make lookup tables here for the end of each primitive
        # z depends only on the z primitive
        self.lookup_delta_z = (self.dt * self.ps_z / self.lookup_res_z).astype(int)

        self.lookup_prim_cost = np.sqrt(np.power(self.ps_z * self.dt, 2) + (self.v_xy * self.dt)**2)
        self.lookup_prim_cost = np.sqrt(np.power(self.ps_z * self.dt, 2) + (self.v_xy * self.dt)**2)

        # t depends only on the time resolution
        self.delta_time = int(self.dt / self.lookup_res_time)

        # x,y will depend on theta primitive and current theta
        self.lookup_delta_x = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)
        self.lookup_delta_y = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)

        # theta also depends on both because modulo 2pi
        self.lookup_theta = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)

        for i in range(0, self.num_ps_theta):
            for j in range(0, self.lookup_num_thetas):
                theta = float(j) * self.lookup_res_theta #+ self.coord_min[3]
                dx = 0.0
                dy = 0.0
                for t in range(0, int(self.dt / self.ddt)):
                    dx = dx + self.ddt * self.v_xy * math.cos(theta)
                    dy = dy + self.ddt * self.v_xy * math.sin(theta)
                    theta = self.theta_2pi(theta + self.ddt * self.ps_theta[i])

                self.lookup_delta_x[i, j] = int(dx / self.lookup_res_xy)
                self.lookup_delta_y[i, j] = int(dy / self.lookup_res_xy)
                self.lookup_theta[i, j] = int(theta / self.lookup_res_theta)

        self.bc = self.curvature * self.lookup_res_xy  # convert curvature from world to indices - should be right
        self.turn_speed = self.curvature * self.v_xy  # turn speed dtheta / dt is constant
        self.scaled_v = self.bc * self.v_xy / self.lookup_res_xy # TODO check this!!
        self.scaled_vz = self.max_ps_z / self.lookup_res_z

    def to_ind(self, loc):
        loc[3] = self.theta_2pi(loc[3])
        return ((loc - self.coord_min) / self.lookup_res).astype(int)

    def to_loc(self, ind):
        return ind.astype(float) * self.lookup_res + self.coord_min

    def to_angle(self, ind):
        return float(ind) * self.lookup_res[3] + self.coord_min[3]

    def get_neighbors(self, parent_node):
        neighbors = []
        parent = parent_node.loc
        for dzi in range(0, self.num_ps_z):
            for dti in range(0, self.num_ps_theta):
                neigh_loc = np.zeros((5,), dtype=int)
                neigh_loc[0] = parent[0] + self.lookup_delta_x[dti, parent[3]]
                neigh_loc[1] = parent[1] + self.lookup_delta_y[dti, parent[3]]
                neigh_loc[2] = parent[2] + self.lookup_delta_z[dzi]
                neigh_loc[3] = self.lookup_theta[dti, parent[3]]
                neigh_loc[4] = parent[4] #+ self.delta_time
                if neigh_loc.min() >= 0:  # in bounds
                    neighbors.append((self.new_node(neigh_loc, parent_node), self.lookup_prim_cost[dzi]))
        return neighbors

    def new_node(self, loc, parent_node=None):
        if loc.dtype != int:
            loc = self.to_ind(loc)
        return DubinsNode(loc, self.hash_coeffs, parent_node)

    def heuristic(self, start_node, end_node):
        return self.dubins_distance(start_node.loc, end_node.loc)

    def dubins_distance(self, si, gi):
        _, bt, bp, bq, bmode = dubins_path(si[0], si[1], self.to_angle(si[3]), gi[0], gi[1], self.to_angle(gi[3]), self.bc)
        tpq = [bt, bp, bq]
        delta_time = 0

        for i in range(0, 3):
            if bmode[i] == "L":
                delta_time = delta_time + tpq[i] / self.turn_speed  # turn speed const
            elif bmode[i] == "R":
                delta_time = delta_time + tpq[i] / self.turn_speed  # turn speed is const
            elif bmode[i] == "S":
                delta_time = delta_time + tpq[i] / self.scaled_v

        delta_z = abs(si[2] - gi[2])
        while delta_z / delta_time > self.scaled_vz:
            delta_time = delta_time + 2 * math.pi / self.turn_speed

        dist = sqrt((delta_time * self.v_xy) ** 2 + (delta_z * self.lookup_res_z) ** 2)
        return dist

    # TODO interpolate the full path in physical space using splines
    def interpolate_waypoints(self, waypoints):
        return None

    # TODO what about if we don't care about goal time?
    def at_goal_position(self, start, goal):
        return np.all(np.less(np.abs(start.loc-goal.loc), self.goal_res))

    def theta_2pi(self, theta):
        return (theta + 2 * np.pi) % (2 * np.pi)

    def reconstruct_path(self, n):
        path = np.zeros((0, 5))
        while n.parent is not None:
            path = np.concatenate((path, self.to_loc(n.loc).reshape(1, -1)), axis=0)
            n = n.parent
        return np.flip(path, 0)














