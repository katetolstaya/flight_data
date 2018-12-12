import numpy as np
from planning.dubins_node import DubinsNode
from math import exp

inf = float("inf")

class DubinsObjective:

    def __init__(self, grid=None):
        # self.others = others # a tuple of arrays for every other plane
        self.grid = grid
        self.cost_type = sigmoid
        self.dN = 10 # don't check all points for efficiency

        self.w = 1.0 #0.01 #20.0 #0.5 # the expected cost for the cost is 1.5x the heuristic

        self.v = 0.1
        self.N = 10

    def integrate_path_cost(self, path):

        last_node = DubinsNode(path[0, 0], path[0, 1], path[0, 2], path[0, 3], path[0, 4])
        cost = 0
        N = 100
        dN = 1 #20 #max(int(np.size(path, 0)/N), 1)
        for i in range(1, np.size(path, 0), dN): 
            
            # below ground is out of bounds
            if path[i, 2] < -0.5 or cost == inf:
                return inf

            node = DubinsNode(path[i, 0], path[i, 1], path[i, 2], path[i, 3], path[i, 4])

            # integrate grid cost
            if self.grid is not None:
                cost = cost + self.w  * self.v * llu(self.grid.get(path[i, :])) * (node.time - last_node.time)

            # integrate path length
            cost = cost + last_node.euclid_distance(node) 
            last_node = node

        return cost

def sigmoid(x):
    return 2.0 * exp(x) / (exp(x) + 1.0)

def llu(x):
    if x < 0:
        return exp(x)
    else:
        return 1 + x
