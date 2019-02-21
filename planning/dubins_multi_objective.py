import numpy as np
from planning.dubins_node import DubinsNode
inf = float("inf")


class DubinsMultiAirplaneObjective:
    def __init__(self, config, grid=None):
        # TODO move to .cfg
        self.grid = grid
        self.sxy = 7000
        self.sz = 300
        self.obstacle_lims = np.array([self.sxy, self.sxy, self.sz])
        self.obstacle_cost = 0.01  # 1.0 #0.0000001 #1.0 #1000.0
        self.obstacle_step = np.array([0.1, 0.1, 0.01]).flatten()
        self.clip = 500.0
        self.obstacle_paths = {}
        #[5400. 5400.  140.]

    def get_cost(self, ind):
        if isinstance(ind, DubinsNode):
            ind = ind.loc
        return self.grid.get(ind) + self.get_obstacles_cost(ind)

    def integrate_path_cost(self, path, path_ind):  # TODO
        cost = 0
        for i in range(1, np.size(path, 0)):
            # integrate grid cost
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
            cost_mult = 1.0 + self.get_obstacles_cost(path_ind[i, :])
            if self.grid is not None:
                cost_mult = cost_mult + self.get_cost(path_ind[i, :])
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

    def get_obstacle_threshold(self, diff):
        return self.obstacle_cost * np.product(np.maximum(self.obstacle_lims - diff, 0))

    def get_obstacles_cost(self, ind):
        obstacles = self.obstacle_paths.get(ind[4])
        if obstacles is not None:
            cost_sum = 0
            for i in range(obstacles.shape[0]):
                diff = np.abs(obstacles[i, :] - ind[0:3])
                cost_sum = cost_sum + self.get_obstacle_threshold(diff)
            return cost_sum
        else:
            return 0

    def ind_path_to_costs(self, path):
        cost_sum = 0
        for j in range(0, path.shape[0]):
            cost_sum = cost_sum + self.get_obstacles_cost(path[j, :])
        return cost_sum

    def update_obstacle_lims(self, path_expert, path_planner, step):
        costs_expert = self.ind_path_to_costs(path_expert)
        if path_planner is not None:
            costs_planner = self.ind_path_to_costs(path_planner)
            delta = costs_planner - costs_expert
        else:
            delta = -1.0 * costs_expert

        self.obstacle_lims = self.obstacle_lims + self.obstacle_step * np.clip(delta, -self.clip, self.clip)
        self.obstacle_lims = np.maximum(self.obstacle_lims, 0)
