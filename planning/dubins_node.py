import numpy as np
from planning.dubins_util import dubins_path
import math
from math import sqrt
import configparser
import sys

inf = float("inf")


class DubinsNode:

    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']

    # TODO move these params to config
    v = float(config['velocity'])
    delta_theta = float(config['delta_theta'])
    delta_z = float(config['delta_z'])

    theta_prims = np.array([0.0, -1.0, 1.0]) * delta_theta
    z_prims = np.array([0.0, -1.0, 1.0]) * delta_z
    ntheta_prims = len(theta_prims)
    nz_prims = len(z_prims)

    dt = float(config['dt'])
    ddt = float(config['ddt'])

    dind_res_x = float(config['dind_res_x'])
    dind_res_y = float(config['dind_res_y'])
    dind_res_z = float(config['dind_res_z'])
    dind_res_theta = float(config['dind_res_theta'])
    dind_res_time = float(config['dind_res_time'])
    dind_res = np.array(
        [dind_res_x, dind_res_y, dind_res_z, dind_res_theta, dind_res_time])

    # generate from data
    dind_min_x = 0.0
    dind_min_y = 0.0
    dind_min_z = 0.0
    dind_min_theta = 0.0
    dind_min_time = 0.0
    dind_min = np.array([dind_min_x, dind_min_y, dind_min_z, dind_min_theta, dind_min_time])
    dind_num_theta = int(2 * math.pi / dind_res_theta) + 1

    # make lookup tables here for the end of each primitive
    # z depends only on the z primitive
    dind_z = z_prims / dind_res_z

    # t depends only on the time resolution
    dind_time = dt / dind_res_time

    # x,y will depend on theta primitive and current theta
    dind_x = np.zeros((len(theta_prims), dind_num_theta))
    dind_y = np.zeros((len(theta_prims), dind_num_theta))

    # theta also depends on both because modulo 2pi
    ind_theta = np.zeros((len(theta_prims), dind_num_theta))


    for i in range(0, len(theta_prims)):
        for j in range(0, dind_num_theta):
            theta = j * dind_res_theta + dind_min_theta
            dx = dy = 0
            for t in range(0, int(dt / ddt)):
                dx = dx + ddt * v * math.cos(theta)
                dy = dy + ddt * v * math.sin(theta)
                theta = np.mod(theta + ddt * theta_prims[i], 2 * np.pi)

            dind_x[i, j] = int(dx / dind_res[0])
            dind_y[i, j] = int(dy / dind_res[1])
            ind_theta[i, j] = int(theta / dind_res[3])

    # TODO also make tables for trajectory interpolation

    # TODO precompute the map of costs ( given airport location, and the location of the other airplanes)
    # TODO store only node indices in Dubins nodes, and only convert to dense trajectories in physical space
    # TODO shorten primitive length to be short enough for interpolatio n needed for cost computations
    # TODO a function to interpolate the full path in physical space


    def to_node(self, x, y, z, theta, time):
        ind = ((np.array([x, y, z, theta, time]) - dind_min) / dind_res).astype(int)
        return DubinsNode(ind[0], ind[1], ind[2], ind[3], ind[4])


    def to_loc(self, node):
        loc = np.array([node.x, node.y, node.z, node.theta, node.time]) * dind_res + dind_min
        return loc[0], loc[1], loc[2], loc[3], loc[4]


    def precompute_cost(self):
        # TODO already have the cost grid, just have to resample more finely

        return 0

    @staticmethod
    def get_neighbors(node):
        neighbors = []
        for dzi in range(0, nz_prims):
            for dti in range(0, ntheta_prims):
                new_x = node.x + dind_x[dti, node.theta]
                new_y = node.y + dind_y[dti, node.theta]
                new_theta = ind_theta[dti, node.theta]
                new_z = node.z + dind_z[dzi]
                new_time = node.time + dt
                neighbors.append(DubinsNode(new_x, new_y, new_z, new_theta, new_time))

        return neighbors




    ###############################################3

    v = 0.1
    delta_theta = 0.125 * math.pi / 100.0 * 5.0 * 1.5
    theta_prims = [0, -1.0 * delta_theta, 1.0 * delta_theta]
    z_prims = np.array([0.0, -1.0, 1.0]) / 1000.0 * 6.0
    dt = 30.0  # 25.0
    ddt = 0.5
    dt_theta = dt
    curvatures = [delta_theta / v]

    dist_tol = 0.05
    theta_tol = 0.05  # 0.0 * np.pi


    # TODO make lookup tables here for the end of each primitive
    # TODO move these params to config
    dind_res_x = 1.0
    dind_res_y = 1.0
    dind_res_z = 1.0
    dind_res_theta = 1.0
    dind_res_time = 0.1

    dind_num_theta = int(2 * math.pi / dind_res_theta)

    # x,y will depend on theta primitive and current theta
    dind_x = np.zeros((len(theta_prims), dind_num_theta))
    dind_y = np.zeros((len(theta_prims), dind_num_theta))

    # theta also depends on both because modulo 2pi
    dind_theta = np.zeros((len(theta_prims), dind_num_theta))

    # z depends only on the z primitive
    dind_z = z_prims / dind_res_z

    # t depends only on the time resolution
    dind_time = dt_theta / dind_res_time

    # TODO also make tables for trajectory interpolation

    # TODO precompute the map of costs ( given airport location, and the location of the other airplanes)
    # TODO store only node indices in Dubins nodes, and only convert to dense trajectories in physical space
    # TODO shorten primitive length to be short enough for interpolation needed for cost computations
    # TODO a function to interpolate the full path in physical space

    ##############################################

    def __init__(self, x=None, y=None, z=None, theta=None, time=None, parent=None, dz=None, dtheta=None, dt=None):

        # need to define these:
        # space/time location of node
        self.x, self.y, self.z, self.theta, self.time = x, y, z, theta, time

        # or these: (or both)
        self.parent, self.dz, self.dtheta = parent, dz, dtheta  # these are arrays
        self.delta_t = dt

    def __str__(self):
        return "<" + str(self.x) + ", " + str(self.y) + "," + str(self.z) + "," + str(self.theta) + "," + str(
            self.time) + ">"

    def __repr__(self):
        return str(self)

    def __eq__(self, othr):
        return (isinstance(othr, type(self)) and (self.x, self.y, self.z, self.theta, self.time) == (
            othr.x, othr.y, othr.z, othr.theta, othr.time))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.x) ^ hash(self.y) ^ hash(self.z) ^ hash(self.theta) ^ hash(self.time) ^ hash(
            (self.x, self.y, self.z, self.theta, self.time))

    ##############################################

    def cost_to_parent(self, obj):
        cost = obj.integrate_path_cost(self.interpolate())
        return cost

    def at_goal_position(self, goal):
        if abs(self.x - goal.x) > DubinsNode.dist_tol:
            return False
        elif abs(self.y - goal.y) > DubinsNode.dist_tol:
            return False
        elif abs(self.z - goal.z) > DubinsNode.dist_tol:
            return False
        elif self.theta_distance(goal) > DubinsNode.theta_tol:
            return False
        elif self.euclid_distance(goal) > DubinsNode.dist_tol:
            return False
        else:
            return True

    def heuristic(self, goal, n_goal=False):
        if n_goal or self.at_goal_position(goal):
            return 0
        else:
            return (1.0) * self.dubins_distance(goal)  # return self.distance(end)

    def distance(self, other):
        other = other[1]
        return self.time_distance(other) + self.euclid_distance(other) + self.theta_distance(other)

    def euclid_distance(self, other):
        dist = sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)  # L2
        return dist

        # difference in angles

    def theta_distance(self, target):
        x = target.theta
        y = self.theta
        return abs(math.pi - abs(abs(x - y) - math.pi))  # abs(math.atan2(math.sin(x - y), math.cos(x - y))) #

    def time_distance(self, other):
        return abs(self.time - other.time)

    # Dubins distance used for heuristic
    def dubins_distance(self, goal):

        bc = max(DubinsNode.curvatures)  # largest curvature is always more efficient
        bcost, bt, bp, bq, bmode = dubins_path(self.x, self.y, self.theta, goal.x, goal.y, goal.theta, bc)

        turn_speed = bc * DubinsNode.v
        tpq = [bt, bp, bq]

        dt = np.zeros((3, 1))

        for i in range(0, 3):
            if bmode[i] == "L":
                dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "R":
                dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "S":
                dt[i] = tpq[i] / bc / DubinsNode.v

        delta_time = np.sum(dt)

        delta_z = abs(self.z - goal.z)

        while delta_z / delta_time > max(DubinsNode.z_prims):
            delta_time = delta_time + 2 * math.pi / turn_speed

        return sqrt((delta_time * DubinsNode.v) ** 2 + (delta_z) ** 2)  # * 2 # L2

    ############################################

    def get_neighbors(self, goal=None):
        neighbors = []
        dt = DubinsNode.dt
        dt_theta = DubinsNode.dt_theta
        for dtheta in DubinsNode.theta_prims:
            for dz in DubinsNode.z_prims:

                if dtheta != 0:
                    n = DubinsNode(None, None, None, None, None, self, [dz], [dtheta], [dt_theta])
                else:
                    n = DubinsNode(None, None, None, None, None, self, [dz], [dtheta], [dt])
                neighbors.append(n)

                # n.interpolate()
                # if n.in_bounds():
                #    neighbors.append(n)  #- TODO in cost/objective class

        if goal is not None:
            goal_neighbor = self.path_to_goal(goal)  # try to make a dubins path to goal
            if goal_neighbor is not None: neighbors.append(goal_neighbor)

        return neighbors

    # TODO interpolate for given ddt, but return for requested dt
    # TODO endpoints will be in the path twice??
    def interpolate(self):
        if self.parent is not None:
            if self.parent.x is None:
                self.parent.interpolate()

            N = int(math.ceil(np.sum(self.delta_t) / DubinsNode.ddt))
            cum_time = np.cumsum(self.delta_t) + self.parent.time

            path = np.zeros((N, 5))
            path[0, :] = np.array(
                [self.parent.x, self.parent.y, self.parent.z, self.parent.theta, self.parent.time]).reshape((1, 5))

            j = 0
            for i in range(1, N):
                path[i, 0] = path[i - 1, 0] + DubinsNode.ddt * DubinsNode.v * math.cos(path[i - 1, 3])
                path[i, 1] = path[i - 1, 1] + DubinsNode.ddt * DubinsNode.v * math.sin(path[i - 1, 3])
                path[i, 2] = path[i - 1, 2] + DubinsNode.ddt * self.dz[j]
                path[i, 3] = path[i - 1, 3] + DubinsNode.ddt * self.dtheta[j]
                path[i, 4] = path[i - 1, 4] + DubinsNode.ddt

                path[i, 3] = (path[i, 3] + 2 * math.pi) % (2 * math.pi)
                j = np.searchsorted(cum_time, path[i, 4])

            if self.x is None:  # lazy evaluation of end point
                self.x, self.y, self.z = path[N - 1, 0], path[N - 1, 1], path[N - 1, 2]
                self.theta, self.time = path[N - 1, 3], path[N - 1, 4]
            return path[0::5, :]
        else:
            return None

    def path_to_goal(self, goal):

        bcost, bt, bp, bq, bmode, bc = inf, inf, inf, inf, None, inf

        # TODO create list of nodes with different paths for cost, collision checking
        for c in DubinsNode.curvatures:
            cost, t, p, q, mode = dubins_path(self.x, self.y, self.theta, goal.x, goal.y, goal.theta, c)
            if cost < bcost:
                bcost, bt, bp, bq, bmode, bc = cost, t, p, q, mode, c

        turn_speed = bc * DubinsNode.v
        tpq = [bt, bp, bq]

        # generate dz, dtheta -> Node object
        dtheta = np.zeros((3, 1))
        dt = np.zeros((3, 1))

        for i in range(0, 3):
            if bmode[i] == "L":
                dtheta[i] = turn_speed
                dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "R":
                dtheta[i] = turn_speed * -1.0
                dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "S":
                dtheta[i] = 0.0
                dt[i] = tpq[i] / bc / DubinsNode.v

        delta_time = np.sum(dt)
        delta_z = goal.z - self.z

        if delta_time == 0.0:
            return None

        dz = np.ones((3, 1)) * delta_z / delta_time

        if np.abs(delta_z) / delta_time > max(DubinsNode.z_prims):
            return None

        return DubinsNode(goal.x, goal.y, goal.z, goal.theta, self.time + delta_time, self, dz, dtheta, dt)

        ############################################


def reconstruct_path(n):
    path = np.zeros((0, 5))
    while n.parent is not None:
        new_path = np.flip(n.interpolate(), 0)
        path = np.concatenate((path, new_path), axis=0)
        n = n.parent
    return np.flip(path, 0)


def plot_path(path):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'o')
    plt.show()
