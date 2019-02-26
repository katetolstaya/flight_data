import numpy as np
from planning.dubins_node import DubinsNode

inf = float("inf")

class DubinsObjective:
    def __init__(self, config, grid=None):
        self.grid = grid
        self.sxy = float(config['obstacle_init_xy'])
        self.sz = float(config['obstacle_init_z'])
        self.obstacle_lims = np.array([self.sxy, self.sxy, self.sz])
        self.obstacle_cost = float(config['obstacle_cost']) # 1.0 #0.0000001 #1.0 #1000.0

        self.step_xy = float(config['obstacle_step_xy'])
        self.step_z = float(config['obstacle_step_z'])
        self.obstacle_step = np.array([self.step_xy, self.step_xy, self.step_z]).flatten()
        self.clip = float(config['obstacle_clip'])
        self.obstacle_paths = {}

    def get_cost(self, ind):
        if isinstance(ind, DubinsNode):
            ind = ind.loc

        cost = 1.0
        if self.grid is not None:
            cost = cost + self.grid.get(ind)
        if self.obstacle_paths:
            cost = cost + self.get_obstacles_cost(ind)
        return cost

    def integrate_path_cost(self, path, path_ind=None):  # TODO
        cost = 0
        for i in range(1, np.size(path, 0)):

            # Ja and Jo cost
            cost_mult = 1.0

            if self.obstacle_paths:
                cost_mult = cost_mult + self.get_obstacles_cost(path_ind[i, :])

            if self.grid is not None:
                cost_mult = cost_mult + self.get_cost(path[i, :])

            # Approximate the line integral using lengths of straight segments
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
            cost = cost + cost_mult * euclid_dist

            if cost is inf:
                return inf
        return cost

    def add_obstacle(self, obstacle_path):
        for i in range(obstacle_path.shape[0]):
            time = int(obstacle_path[i, 4])
            ind = obstacle_path[i,0:3]
            if time not in self.obstacle_paths:
                self.obstacle_paths[time] = np.zeros((0, 3))
            self.obstacle_paths[time] = np.vstack((self.obstacle_paths[time], ind))

    def clear_obstacles(self):
        self.obstacle_paths = {}

    def get_obstacle_threshold(self, diff):
        return self.obstacle_cost * np.product(np.maximum(self.obstacle_lims - diff, 0))

    def get_obstacles_cost(self, ind):
        obstacles = self.obstacle_paths.get(int(ind[4]))
        if obstacles is not None:
            cost_sum = 0
            for i in range(obstacles.shape[0]):
                diff = np.abs(obstacles[i, :] - ind[0:3])
                cost_sum = cost_sum + self.get_obstacle_threshold(diff)
            return cost_sum
        else:
            return 0

    def compute_gradient(self, path):
        grad_sum = np.zeros((3, ))
        for j in range(0, path.shape[0]):
            ind = path[j, :]
            obstacles = self.obstacle_paths.get(int(ind[4]))
            if obstacles is not None:
                for i in range(obstacles.shape[0]):

                    diff = np.maximum(self.obstacle_lims - np.abs(obstacles[i, :] - ind[0:3]), 0)
                    if np.product(diff) > 0:
                        prod_grad = self.obstacle_cost * np.array([diff[1]*diff[2], diff[0]* diff[2], diff[0] * diff[1]])
                        grad_sum = grad_sum + prod_grad.flatten()

        return grad_sum

    def update_obstacle_lims(self, path_expert, path_planner):
        grad_expert = self.compute_gradient(path_expert)
        if path_planner is not None:
            grad_planner = self.compute_gradient(path_planner)
            delta = grad_planner - grad_expert
        else:
            delta = -1.0 * grad_expert

        self.obstacle_lims = self.obstacle_lims + self.obstacle_step * np.clip(delta, -self.clip, self.clip)
        self.obstacle_lims = np.maximum(self.obstacle_lims, 0)


    # def __init__(self, config, grid=None):
    #     # self.others = others # a tuple of arrays for every other plane
    #     self.grid = grid
    #     self.cost_type = config['grid_cost_type']
    #     self.w = float(config['grid_weight'])  # 0.01 #20.0 #0.5 # the expected cost for the cost is 1.5x the heuristic
    #
    # def get_cost(self, ind):
    #     if isinstance(ind, DubinsNode):
    #         ind = ind.loc
    #     return self.grid.get(ind)
    #
    # def integrate_path_cost(self, path):
    #     cost = 0
    #     for i in range(1, np.size(path, 0)):
    #         # integrate grid cost
    #         euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
    #         if self.grid is not None:
    #             cost = cost + (1.0 + self.get_cost(path[i, :])) * euclid_dist
    #         else:
    #             cost = cost + euclid_dist
    #
    #         if cost is inf:
    #             return inf
    #
    #     return cost


