import numpy as np
from planning.dubins_node import DubinsNode
from math import exp

# from dubins_objective import sigmoid, llu

inf = float("inf")


class DubinsMultiAirplaneObjective:
    def __init__(self, config, grid=None):
        # self.others = others # a tuple of arrays for every other plane
        self.grid = grid
        self.cost_type = config['grid_cost_type']
        self.w = float(config['grid_weight'])  # 0.01 #20.0 #0.5 # the expected cost for the cost is 1.5x the heuristic

        # sx = 100.0
        # sy = 100.0
        # sz = 100.0

        self.sxy = 50
        self.sz = 50
        self.obstacle_lims = np.array([self.sxy, self.sxy, self.sz])
        self.obstacle_cost = 100.0
        self.obstacle_step = 0.001

        self.obstacle_grid = np.zeros((self.sxy, self.sxy, self.sz))
        self.lookup_res_xy = float(config['dind_res_xy'])
        self.lookup_res_z = float(config['dind_res_z'])
        self.lookup_res_theta = float(config['dind_res_theta'])
        self.lookup_res = np.array(
            [self.lookup_res_xy, self.lookup_res_xy, self.lookup_res_z, self.lookup_res_theta])

        for i in range(sx):
            dx = i * self.lookup_res_xy
            for j in range(sy):
                dy = j * self.lookup_res_xy
                for k in range(sz):
                    dz = k * self.lookup_res_z

                    self.obstacle_grid[i, j, k] = np.exp(-np.linalg.norm(np.array([dx, dy, dz]))) * 100

        self.obstacle_paths = {}

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
        return self.grid.get(ind) + self.get_obstacles_cost(ind)

    # def get_cost(self, ind):
    #     if isinstance(ind, DubinsNode):
    #         ind = ind.loc
    #     grid_cost = self.w * self.cost_func(self.grid.get(ind))
    #     return grid_cost + self.get_obstacles_cost(ind)

    def integrate_path_cost(self, path, path_ind):  # TODO
        cost = 0
        dt = 20
        for i in range(1, np.size(path, 0)):
            # integrate grid cost
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
            cost_mult = 1.0 + self.get_obstacles_cost(path_ind[i, :])
            if self.grid is not None:
                cost_mult = cost_mult + self.get_cost(path_ind[i, :])
            cost = cost + cost_mult * euclid_dist
            if cost is inf:
                return inf

        # for i in range(1, np.size(path, 0)):
        #     # integrate grid cost
        #     euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
        #     cost_mult = 1.0  #+ self.get_obstacles_cost(path_ind[i, :])
        #     if self.grid is not None:
        #         cost_mult = cost_mult + self.get_cost(path[i, :])
        #     cost = cost + cost_mult * euclid_dist
        #     if cost is inf:
        #         return inf
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
        if np.all(diff < self.obstacle_lims):
            # return self.obstacle_cost
            ind = map(tuple, diff.T)[0]
            return self.obstacle_grid[ind]
        else:
            return 0  # do nothing for values out of bounds

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

    def get_path_obstacle_distances(self, path):
        distances = np.zeros((0, 3))
        for j in range(0, path.shape[0]):
            ind = path[j, :]
            obstacles = self.obstacle_paths.get(ind[4])
            if obstacles is not None:
                for i in range(obstacles.shape[0]):
                    diff = np.abs(obstacles[i, :] - ind[0:3])
                    distances = np.vstack((distances, diff))
        return distances

    def update_obstacle_lims(self, path_expert, path_planner):

        dist_expert = self.get_path_obstacle_distances(path_expert)
        dist_planner = self.get_path_obstacle_distances(path_planner)

        if dist_expert.shape[0] > 0 and dist_planner.shape[0] > 0:
            delta = np.mean(dist_expert, axis=0) - np.mean(dist_planner, axis=0)  # diff by feature
            delta_exp = (np.exp(delta[0]) * np.zeros((self.sxy, 1, 1))) * (np.exp(delta[1]) * np.zeros(
                (1, self.sxy, 1))) * (np.exp(delta[0]) * np.zeros((1, 1, self.sz)))
            self.obstacle_grid = self.obstacle_grid * delta_exp

            # self.obstacle_lims = self.obstacle_lims + self.obstacle_step * delta.flatten()
