import numpy as np
from planning.dubins_node import DubinsNode
from math import exp

inf = float("inf")

from dubins_objective import sigmoid, llu


class DubinsMultiAirplaneObjective:
    def __init__(self, config, grid=None):
        # self.others = others # a tuple of arrays for every other plane
        self.grid = grid
        self.cost_type = config['grid_cost_type']
        self.w = float(config['grid_weight'])  # 0.01 #20.0 #0.5 # the expected cost for the cost is 1.5x the heuristic

        sx = 20
        sy = 20
        sz = 20
        self.obstacle_lims = np.array([sx, sy, sz])
        self.obstacle_cost = 100.0

        self.obstacle_paths = {}

        if self.cost_type == 'sigmoid':
            self.cost_func = sigmoid
        elif self.cost_type == 'exp':
            self.cost_func = exp
        elif self.cost_type == 'llu':
            self.cost_func = llu
        else:
            raise NotImplementedError

    def get_cost(self, ind):
        if isinstance(ind, DubinsNode):
            ind = ind.loc
        grid_cost = self.w * self.cost_func(self.grid.get(ind))
        return grid_cost + self.obstacle_costs(ind)

    def integrate_path_cost(self, path): # TODO
        cost = 0
        for i in range(1, np.size(path, 0)):
            # integrate grid cost
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])

            cost_mult = 1.0
            if self.grid is not None:
                cost_mult = cost_mult + self.get_cost(path[i, :])
            # TODO obstacle computation here

            cost = cost + cost_mult * euclid_dist

            if cost is inf:
                return inf

        return cost

    def add_obstacle(self, obstacle_path):

        for i in range(obstacle_path.shape[0]):
            time = obstacle_path[i, 4]
            grid_loc = self.grid.loc_to_index(obstacle_path[i])[0:3]
            if time not in self.obstacle_paths:
                self.obstacle_paths[time] = np.zeros((0, 3))
            self.obstacle_paths[time] = np.vstack((self.obstacle_paths[time], grid_loc))

    def clear_obstacles(self):
        self.obstacle_paths = {}

    def get_obstacle_cost(self, diff):

        if np.all(diff < self.obstacle_lims):
            return self.obstacle_cost
        else:
            return 0  # do nothing for values out of bounds

    def obstacle_costs(self, ind):
        obstacles = self.obstacle_paths.get(ind[4])
        if obstacles is not None:
            cost_sum = 0
            for i in range(obstacles.shape[0]):
                diff = np.abs(obstacles[i, :] - ind[0:3])
                cost_sum = cost_sum + self.get_obstacle_cost(diff)
            return cost_sum
        else:
            return 0

    def get_obstacle_distances(self, path):

        distances = []
        for j in range(0, path.shape[0]):
            ind = path[j, :]
            obstacles = self.obstacle_paths.get(ind[4])
            if obstacles is not None:
                for i in range(obstacles.shape[0]):
                    diff = np.abs(obstacles[i, :] - ind[0:3])
                    distances.append(diff)
        return distances

