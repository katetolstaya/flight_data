import numpy as np
import math
from math import sqrt
from planning.dubins_util import dubins_path, neg_pi_to_pi
from planning.dubins_node import DubinsNode
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spline.Greville3D import Bspline4

inf = float("inf")


class DubinsProblem:
    def __init__(self, config, coord_min, coord_max):

        self.config = config
        self.coord_min = coord_min
        self.coord_max = coord_max

        self.dt = float(config['dt'])
        self.ddt = float(config['ddt'])

        self.v_xy = float(config['velocity_x'])
        self.v_theta = float(config['velocity_theta'])
        self.v_z = float(config['velocity_z'])
        self.curvature = self.v_theta / self.v_xy

        self.ps_theta = np.array([0.0, -1.0, 1.0, -0.5, 0.5]) * self.v_theta
        self.ps_z = np.array([0.0, -1.0, 1.0, -0.5, 0.5]) * self.v_z

        self.max_ps_z = max(self.ps_z)
        self.num_ps_theta = len(self.ps_theta)
        self.num_ps_z = len(self.ps_z)

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

        self.lookup_prim_cost = np.sqrt(np.power(self.ps_z * self.dt, 2) + (self.v_xy * self.dt) ** 2)

        # t depends only on the time resolution
        self.delta_time = int(self.dt / self.lookup_res_time)

        # x,y will depend on theta primitive and current theta
        self.lookup_delta_x = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)
        self.lookup_delta_y = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)

        # theta also depends on both because modulo 2pi
        self.lookup_theta = np.zeros((self.num_ps_theta, self.lookup_num_thetas), dtype=int)

        for i in range(0, self.num_ps_theta):
            for j in range(0, self.lookup_num_thetas):
                theta = self.to_angle(j)
                dx = 0.0
                dy = 0.0
                for t in range(0, int(self.dt / self.ddt)):
                    dx = dx + self.ddt * self.v_xy * math.cos(theta)
                    dy = dy + self.ddt * self.v_xy * math.sin(theta)
                    theta = neg_pi_to_pi(theta + self.ddt * self.ps_theta[i])

                self.lookup_delta_x[i, j] = int(dx / self.lookup_res_xy)
                self.lookup_delta_y[i, j] = int(dy / self.lookup_res_xy)
                self.lookup_theta[i, j] = int((theta - self.coord_min[3]) / self.lookup_res_theta)

        self.bc = self.curvature * self.lookup_res_xy * 1.1  # convert curvature from world to indices - should be right
        self.recip_bc = 1.0 / self.bc
        # self.turn_speed = self.curvature * self.v_xy  # turn speed dtheta / dt is constant
        self.scaled_vxy = self.v_xy / self.lookup_res_xy
        self.scaled_vz = self.max_ps_z / self.lookup_res_z

        self.recip_turn_speed = 1.0 / self.v_theta
        self.recip_scaled_vxy = 1.0 / self.scaled_vxy

        self.full_turn_time = 2 * math.pi / self.v_theta

    def to_ind(self, loc):
        loc[3] = neg_pi_to_pi(loc[3])
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
                neigh_loc[4] = parent[4] + self.delta_time
                if np.all(neigh_loc >= 0):  # in bounds
                    neighbors.append((self.new_node(neigh_loc, parent_node), self.lookup_prim_cost[dzi]))
        return neighbors

    def new_node(self, loc, parent_node=None):
        if loc.dtype != int:
            loc = self.to_ind(loc)
        return DubinsNode(loc, self.hash_coeffs, parent_node)

    def heuristic(self, start_node, end_node):
        # if np.all(np.less(np.abs(start_node.loc[0:3]-end_node.loc[0:3]), 3*self.goal_res[0:3])):
        #     min_cost = inf
        #     for n in range(0, 10):
        #         new_end = end_node.loc + np.random.uniform(low=-self.goal_res, high=self.goal_res)
        #         min_cost = min(min_cost, self.dubins_distance(start_node.loc, new_end))
        #
        #         #min_cost = min(min_cost, self.dubins_distance(start_node.loc, end_node.loc, self.bc * (1+(1.0*np.random.random()))))
        #     return min_cost
        # else:
        return self.dubins_distance(start_node.loc, end_node.loc)

    def dubins_distance(self, si, gi, bc=None):

        delta_time_sg = gi[4] - si[4]
        if delta_time_sg < 0:
            return inf

        if bc is None:
            bc = self.bc

        _, bt, bp, bq, bmode = dubins_path(si[0], si[1], self.to_angle(si[3]), gi[0], gi[1], self.to_angle(gi[3]), bc)
        tpq = [bt, bp, bq]
        delta_time = 0

        for i in range(0, 3):
            if bmode[i] == 'L':
                delta_time = delta_time + tpq[i] * self.recip_turn_speed  # turn speed const
            elif bmode[i] == 'R':
                delta_time = delta_time + tpq[i] * self.recip_turn_speed  # turn speed is const
            elif bmode[i] == 'S':
                delta_time = delta_time + tpq[i] * self.recip_bc * self.recip_scaled_vxy

        delta_z = abs(si[2] - gi[2])
        while delta_z > self.scaled_vz * delta_time:
            delta_time = delta_time + self.full_turn_time

        delta_dist = delta_time * self.v_xy
        delta_z = delta_z * self.lookup_res_z
        dist = sqrt(delta_dist * delta_dist + delta_z * delta_z)

        if delta_time_sg < delta_time:  # not enough time to reach the goal
            return inf

        return dist # * 0.5 + delta_time_sg * self.v_xy * 0.5

    # TODO constrain airplane arrival time, will need to tune velocities
    def at_goal_position(self, start, goal):
        return np.all(np.less(np.abs(start.loc[0:4] - goal.loc[0:4]), self.goal_res[0:4]))
        #return np.all(np.less(np.abs(start.loc - goal.loc), self.goal_res))

    def path_to_ind(self, path):
        ind_path = np.zeros(path.shape)

        for i in range(path.shape[0]):
            ind_path[i, :] = self.to_ind(path[i,:])

        return ind_path

    def ind_to_path(self, ind_path):
        path = np.zeros(ind_path.shape)

        for i in range(path.shape[0]):
            path[i, :] = self.to_loc(ind_path[i,:])

        return path

    def reconstruct_path(self, n):
        path = np.zeros((0, 5))
        while n.parent is not None:
            path = np.concatenate((path, self.to_loc(n.loc).reshape(1, -1)), axis=0)
            n = n.parent
        return np.flip(path, 0)

    def reconstruct_path_ind(self, n):
        path = np.zeros((0, 5))
        while n.parent is not None:
            path = np.concatenate((path, n.loc.reshape(1, -1)), axis=0)
            n = n.parent
        return np.flip(path, 0)


    def initialize_plot(self, start, goal):
        plt.ion()
        # self.fig = plt.figure()
        # self.ax = self.fig.gca(projection='3d')

        self.fig, self.ax = plt.subplots()

        start_loc = self.to_loc(start.loc)
        goal_loc = self.to_loc(goal.loc)

        self.ax.plot([goal_loc[0]], [goal_loc[1]], 'rx')

        self.x_plot = [start_loc[0]]
        self.y_plot = [start_loc[1]]
        self.line, = self.ax.plot(self.x_plot, self.y_plot, 'go')

    def update_plot(self, s):
        loc = self.to_loc(s.loc)
        self.x_plot.append(loc[0])
        self.y_plot.append(loc[1])

        self.line.set_data(self.x_plot, self.y_plot)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def resample_path(path, start, goal, n_ts=400):

        # s = 0.1
        #
        # ts = np.linspace( start=path[0, 4], stop=path[-1, 4], num=n_ts)
        #
        # s_x = UnivariateSpline(path[:, 4], path[:, 0], s=s)
        # s_y = UnivariateSpline(path[:, 4], path[:, 1], s=s)
        # s_z = UnivariateSpline(path[:, 4], path[:, 2], s=s)
        #
        # # interpolate new x,y,z,bearing coordinates
        # xs = s_x(ts).reshape(-1, 1)
        # ys = s_y(ts).reshape(-1, 1)
        # zs = s_z(ts).reshape(-1, 1)
        # bs = np.arctan2(ys[1:] - ys[:-1], xs[1:] - xs[:-1])
        # bs = np.append(bs, [bs[-1]], axis=0)
        # ts = ts.reshape(-1, 1)
        #
        # smoothed_path = np.stack((xs, ys, zs, bs, ts), axis=1).reshape(-1, 5)
        # return smoothed_path

        n_path_pts = path.shape[0]

        # Build knot vector tk
        maxtk = n_path_pts - 4
        s = (1, 4)
        tk = np.zeros(s)
        tkmiddle = np.arange(maxtk + 1)
        tkend = maxtk * np.ones(s)
        tk = np.append(tk, tkmiddle)
        tk = np.append(tk, tkend)

        ts = np.linspace(start=path[0, 4], stop=path[-1, 4], num=n_ts)
        smoothed_path, B4, tau = Bspline4(path[:,0:3], n_ts, tk, maxtk)  # interpolate in XYZ

        xs = smoothed_path[:, 0].reshape(-1, 1)
        ys = smoothed_path[:, 1].reshape(-1, 1)
        zs = smoothed_path[:, 2].reshape(-1, 1)
        bs = np.arctan2(ys[1:] - ys[:-1], xs[1:] - xs[:-1])
        bs = np.append(bs, [bs[-1]], axis=0)
        ts = ts.reshape(-1, 1)

        smoothed_path = np.stack((xs, ys, zs, bs, ts), axis=1).reshape(-1, 5)
        return smoothed_path

    @staticmethod
    def resample_path_dt(path, s, dt):

        start = np.ceil(path[0, 4]/dt)
        stop = np.floor(path[-1, 4]/dt)
        ts = np.arange(start=start, stop=stop) * dt

        s_x = UnivariateSpline(path[:, 4], path[:, 0], s=s)
        s_y = UnivariateSpline(path[:, 4], path[:, 1], s=s)
        s_z = UnivariateSpline(path[:, 4], path[:, 2], s=s)

        # interpolate new x,y,z,bearing coordinates
        xs = s_x(ts).reshape(-1, 1)
        ys = s_y(ts).reshape(-1, 1)
        zs = s_z(ts).reshape(-1, 1)
        bs = np.arctan2(ys[1:] - ys[:-1], xs[1:] - xs[:-1])
        bs = np.append(bs, [bs[-1]], axis=0)
        ts = ts.reshape(-1, 1)

        smoothed_path = np.stack((xs, ys, zs, bs, ts), axis=1).reshape(-1, 5)
        return smoothed_path
