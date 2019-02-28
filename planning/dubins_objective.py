import numpy as np
from planning.dubins_node import DubinsNode

inf = float("inf")


class DubinsObjective:

    def __init__(self, config, grid=None):
        """

        :param config:
        :type config:
        :param grid:
        :type grid:
        """

        self.obstacle_cost = float(config['obstacle_cost'])
        self.obstacle_lims = np.array([float(config['obstacle_init_xy']), float(config['obstacle_init_z'])])
        self.obstacle_step = np.array([float(config['obstacle_step_xy']), float(config['obstacle_step_z'])])
        self.clip_grad = float(config['obstacle_clip'])
        self.grid = grid
        self.obstacle_paths = {}

    def get_cost(self, ind):
        """

        :param ind:
        :type ind:
        :return:
        :rtype:
        """
        if isinstance(ind, DubinsNode):
            ind = ind.loc

        cost = 1.0
        if self.grid is not None:
            cost = cost + self.grid.get(ind)
        if self.obstacle_paths:
            cost = cost + self.get_obstacles_cost(ind)
        return cost

    def integrate_path_cost(self, path, path_ind=None):  # TODO
        """

        :param path:
        :type path:
        :param path_ind:
        :type path_ind:
        :return:
        :rtype:
        """
        path_cost = 0
        for i in range(1, np.size(path, 0)):

            # Ja and Jo cost
            state_cost = 1.0  # path length

            if self.obstacle_paths:
                state_cost = state_cost + self.get_obstacles_cost(path_ind[i, :])

            if self.grid is not None:
                state_cost = state_cost + self.get_cost(path[i, :])

            # Approximate the line integral using lengths of straight segments
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
            path_cost = path_cost + state_cost * euclid_dist

            if path_cost is inf:
                return inf
        return path_cost

    def add_obstacle(self, obstacle_path):
        """

        :param obstacle_path:
        :type obstacle_path:
        """
        for i in range(obstacle_path.shape[0]):
            time = int(obstacle_path[i, 4])
            ind = obstacle_path[i, 0:3]
            if time not in self.obstacle_paths:
                self.obstacle_paths[time] = np.zeros((0, 3))
            self.obstacle_paths[time] = np.vstack((self.obstacle_paths[time], ind))

    def clear_obstacles(self):
        """

        """
        self.obstacle_paths = {}

    def get_obstacle_threshold(self, diff):
        """

        :param diff:
        :type diff:
        :return:
        :rtype:
        """
        temp = self.obstacle_lims - np.array([np.linalg.norm([diff[0], diff[1]]), diff[2]])
        return self.obstacle_cost * np.product(np.maximum(temp, 0))

    def get_obstacles_cost(self, ind):
        """

        :param ind:
        :type ind:
        :return:
        :rtype:
        """
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
        """

        :param path:
        :type path:
        :return:
        :rtype:
        """
        grad_sum = np.zeros((2,))
        for j in range(0, path.shape[0]):
            ind = path[j, :]
            obstacles = self.obstacle_paths.get(int(ind[4]))
            if obstacles is not None:
                for i in range(obstacles.shape[0]):

                    diff = np.abs(obstacles[i, :] - ind[0:3])
                    diff_xy = max(self.obstacle_lims[0] - np.linalg.norm([diff[0], diff[1]]), 0)
                    diff_z = max(self.obstacle_lims[1] - diff[2], 0)

                    # the derivative has the terms swapped
                    if diff_xy * diff_z > 0:
                        grad_sum = grad_sum + self.obstacle_cost * np.array([diff_z, diff_xy])

        return grad_sum

    @staticmethod
    def update_obstacle_lims(obj_expert, path_expert, obj, path_planner):
        """

        :param obj_expert:
        :type obj_expert:
        :param path_expert:
        :type path_expert:
        :param obj:
        :type obj:
        :param path_planner:
        :type path_planner:
        """
        grad_expert = obj_expert.compute_gradient(path_expert)
        if path_planner is not None:
            grad_planner = obj.compute_gradient(path_planner)
            delta = grad_planner - grad_expert
        else:
            delta = -1.0 * grad_expert

        obj.obstacle_lims = obj.obstacle_lims + obj.obstacle_step * np.clip(delta, -obj.clip_grad, obj.clip_grad)
        obj.obstacle_lims = np.maximum(obj.obstacle_lims, 0)

        obj_expert.obstacle_lims = obj.obstacle_lims
