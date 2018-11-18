import numpy as np
from planning.dubins_util import dubins_path
import math
from math import sqrt  

goal_dist_tol = 0.05
goal_theta_tol = 0.05 #0.0 * np.pi

dist_delta = 0.1 # 10 meters
theta_delta = 0.3 # in radians
t_delta = 25.0 # seconds

inf = float("inf")

class DubinsNode:

    v = 0.1 #0.002 #2.0/100.0
    delta_theta = 0.125 * math.pi / 100.0 * 5.0 #* 1.5 #* 1.5 #* 5.0 #/ 2.0  # * pi
    #thetas = [0, delta_theta, 2.0 * delta_theta, -1.0 * delta_theta, -2.0 * delta_theta]
    thetas = [0,  -1.0 * delta_theta, 1.0 * delta_theta]
    #zs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 1000.0 * 6.0
    zs = np.array([ 0.0, -1.0, 1.0]) / 1000.0 * 6.0
    dt = 30.0 #60.0 #60.0 #120.0 #100.0 #120.0 #100.0 #50.0 #1.0 * 100.0 / 2.0  #* 2.0 #4.0 #/ 2.0 #0.5
    ddt = 2.0 #dt / 30.0 #3.0 #5.0 / 2.0/2.0 #0.05 * 100.0 / 5.0 #/ 2.0

    dt_theta = dt #/ 2.0
    curvatures = [delta_theta / v]

    ##############################################

    def __init__(self, x=None, y=None, z=None, theta=None, time=None, parent=None, dz=None, dtheta=None, dt=None):
        # space/time location of node

        # need to define these:
        self.x, self.y, self.z, self.theta, self.time = x, y, z, theta, time

        # or these: (or both)
        self.parent, self.dz, self.dtheta = parent, dz, dtheta # these are arrays
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
        if abs(self.x - goal.x) >= goal_dist_tol or abs(self.y - goal.y) >= goal_dist_tol or abs(self.z - goal.z) >= goal_dist_tol:
            return False
        return self.theta_distance(goal) <= goal_theta_tol and self.euclid_distance(goal) <= goal_dist_tol
       

    def heuristic(self, goal, n_goal=False):
        if n_goal or self.at_goal_position(goal):
            return 0
        else:
            return (1.0) * self.dubins_distance(goal)         #return self.distance(end)


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
        return abs(math.pi - abs(abs(x - y) - math.pi)) # abs(math.atan2(math.sin(x - y), math.cos(x - y))) #

    def time_distance(self, other):
        return abs(self.time - other.time)

    # Dubins distance used for heuristic
    def dubins_distance(self, goal):

        bc = max(DubinsNode.curvatures) # largest curvature is always more efficient
        bcost, bt, bp, bq, bmode = dubins_path(self.x, self.y, self.theta, goal.x, goal.y, goal.theta, bc)

        turn_speed = bc * DubinsNode.v
        tpq = [bt, bp, bq]

        dt = np.zeros((3,1))

        for i in range(0,3):
            if bmode[i] == "L":   dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "R": dt[i] = tpq[i] / turn_speed
            elif bmode[i] == "S": dt[i] = tpq[i] / bc / DubinsNode.v

        delta_time = np.sum(dt)

        delta_z = abs(self.z - goal.z)

        while delta_z / delta_time > max(DubinsNode.zs):
            delta_time = delta_time + 2 * math.pi / turn_speed

        return sqrt((delta_time * DubinsNode.v) ** 2 + (delta_z) ** 2) #* 2 # L2


    ############################################

    def get_neighbors(self, goal=None):
        neighbors = []
        dt = DubinsNode.dt
        dt_theta = DubinsNode.dt_theta
        for dtheta in DubinsNode.thetas:
            for dz in DubinsNode.zs:

                if dtheta != 0:
                    n = DubinsNode(None, None, None, None, None, self, [dz], [dtheta], [dt_theta])
                else:
                    n = DubinsNode(None, None, None, None, None, self, [dz], [dtheta], [dt])
                neighbors.append(n)

                # n.interpolate()
                # if n.in_bounds():
                #    neighbors.append(n)  #- TODO in cost/objective class

        if goal is not None:
            goal_neighbor = self.path_to_goal(goal) # try to make a dubins path to goal
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
            path[0,:] = np.array([self.parent.x, self.parent.y, self.parent.z, self.parent.theta, self.parent.time]).reshape((1,5))

            j = 0
            for i in range(1, N):
                path[i, 0] =  path[i-1, 0] + DubinsNode.ddt * DubinsNode.v * math.cos(path[i-1, 3])
                path[i, 1] =  path[i-1, 1] + DubinsNode.ddt * DubinsNode.v * math.sin(path[i-1, 3])
                path[i, 2] =  path[i-1, 2] + DubinsNode.ddt * self.dz[j]
                path[i, 3] = path[i-1, 3] + DubinsNode.ddt * self.dtheta[j]
                path[i, 4] = path[i-1, 4] + DubinsNode.ddt

                path[i, 3] = (path[i, 3] + 2 * math.pi) % (2 * math.pi)
                j = np.searchsorted(cum_time, path[i, 4])

            if self.x is None: # lazy evaluation of end point
                self.x, self.y, self.z = path[N-1, 0], path[N-1, 1], path[N-1, 2]
                self.theta, self.time = path[N-1, 3], path[N-1, 4]
            return path
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
        dtheta = np.zeros((3,1))
        dt = np.zeros((3,1))

        for i in range(0,3):
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

        # while abs(delta_z) / delta_time > max(Node.zs):
        #     delta_time = delta_time + 2 * math.pi / turn_speed
        if delta_time == 0.0:
            return None

        dz = np.ones((3, 1)) * delta_z / delta_time

        if np.abs(delta_z) / delta_time > max(DubinsNode.zs):
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
    ax.plot(path[:,0], path[:,1], path[:,2], 'o')
    plt.show()



