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

        # if self.cost_type == 'sigmoid':
        #     self.cost_func = sigmoid
        # elif self.cost_type == 'exp':
        #     self.cost_func = exp
        # elif self.cost_type == 'llu':
        #     self.cost_func = llu
        # else:
        #     raise NotImplementedError

    def get_cost(self, ind):
        if isinstance(ind, DubinsNode):
            ind = ind.loc
        return self.grid.get(ind)  #self.w * self.cost_func(self.grid.get(ind))

    def integrate_path_cost(self, path):
        cost = 0
        for i in range(1, np.size(path, 0)):
            # integrate grid cost
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
            if self.grid is not None:
                cost = cost + (1.0 + self.get_cost(path[i, :])) * euclid_dist
            else:
                cost = cost + euclid_dist

            if cost is inf:
                return inf

        return cost


# def sigmoid(x):
#     return 2.0 / (exp(-1.0 * x) + 1.0)
#
#
# def llu(x):
#     if x < 0:
#         return exp(x)
#     else:
#         return 1 + x
