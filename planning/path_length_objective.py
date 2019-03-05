import numpy as np

inf = float("inf")


class PathLengthObjective:

    def __init__(self, config, grid=None):
        """

        :param config:
        :type config:
        :param grid:
        :type grid:
        """

    def get_cost(self, ind):
        """

        :param ind:
        :type ind:
        :return:
        :rtype:
        """
        return 1.0

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

            # Approximate the line integral using lengths of straight segments
            euclid_dist = np.linalg.norm(path[i - 1, 0:3] - path[i, 0:3])
            path_cost = path_cost + state_cost * euclid_dist

            if path_cost is inf:
                return inf
        return path_cost
