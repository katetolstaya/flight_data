import numpy as np
from planning.dubins_node import DubinsNode
from math import exp

inf = float("inf")


class DubinsObjective:
    def __init__(self, config, grid=None):
        # self.others = others # a tuple of arrays for every other plane
        self.grid = grid
        self.cost_type = config['grid_cost_type']
        self.w = float(config['grid_weight'])  # 0.01 #20.0 #0.5 # the expected cost for the cost is 1.5x the heuristic

    def get_cost(self, ind):
        if isinstance(ind, DubinsNode):
            ind = ind.loc

        if self.cost_type == 'sigmoid':
            return self.w * sigmoid(self.grid.get(ind))
        elif self.cost_type == 'exp':
            return self.w * exp(self.grid.get(ind))
        elif self.cost_type == 'llu':
            return self.w * llu(self.grid.get(ind))
        else:
            raise NotImplementedError

    def integrate_path_cost(self, path):
        cost = 0
        for i in range(1, np.size(path, 0)):

            # below ground is out of bounds
            if path[i, 2] < -0.5 or cost == inf:
                return inf

            # integrate grid cost
            if self.grid is not None:
                euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
                cost = cost + (1.0 + self.get_cost(path[i, :])) * euclid_dist
        return cost


def sigmoid(x):
    return 2.0 * exp(x) / (exp(x) + 1.0)


def llu(x):
    if x < 0:
        return exp(x)
    else:
        return 1 + x
