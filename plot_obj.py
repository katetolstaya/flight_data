import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import configparser
import random
from process import get_min_max_all
from train import load_flight_data, make_planner
from plot_utils import plot_planner_expert
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective

from planning.dubins_problem import DubinsProblem

def main():
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    to = float(config['timeout'])
    seed = int(config['random_seed'])
    planner = make_planner(config['planner_type'])
    n_samples = int(config['num_samples'])

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    print('Loading cost...')
    # set up cost grid
    # n_iters = int(config['num_iterations'])
    # n_samples = int(config['num_samples'])
    grid = Grid(config, xyzbea_min, xyzbea_max)
    grid.load_grid(fname="model/grid19.pkl")

    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    max_x = 500
    max_y = 500

    offset_x = -100 #200
    offset_y = -600
    cost_min = np.ones((max_x, max_y)) * 100
    count = np.ones((max_x, max_y))
    cost_sum = np.ones((max_x, max_y)) * 100

    for k in grid.grid.keys():
        x = k[0] + offset_x
        y = k[1] + offset_y

        if x > 0 and y > 0 and x < max_x and y < max_y:
            cost_sum[x, y] = cost_sum[x, y] + grid.grid[k]
            cost_min[x, y] = min(cost_min[x, y], grid.grid[k])

            count[x,y] = count[x,y] + 1

    avg_cost = cost_sum / count
    cost_min = cost_min.T

    flight_summaries.pop(0)
    for flight in flight_summaries:
        if flight.get_path_len() < 4:
            continue
        start, goal = flight.get_start_goal()
        node = planner(problem, start, goal, obj).plan(to)
        if node is not None:
            # planner_path = problem.reconstruct_path(node)
            planner_path = problem.reconstruct_path_ind(node)

            planner_path_ind = np.zeros((planner_path.shape[0], 2))
            for t in range(planner_path.shape[0]):
                temp = grid.ind_to_index(planner_path[t,:])
                planner_path_ind[t,:] = temp[0:2]

            fig, ax = plt.subplots()
            ax.imshow(cost_min, extent=[0, max_x, 0, max_y], interpolation='spline16', alpha=0.5, origin='lower')
            ax.plot(planner_path_ind[:,0]+offset_x, planner_path_ind[:,1]+offset_y, linewidth=5, color='firebrick')
            plt.show()

        else:
            print("timeout")



    # # pick best orientation in each location (x,y,z)
    # # plot only those location that are > 0 (good)
    # # convert orientation to quiver direction, u, v, w
    #
    # orientation = np.argmin(grid, axis=3)  # min cost is best
    #
    # # print(orientation.nonzero())
    #
    # good = np.min(grid, axis=3) < -30
    #
    # x, y, z = good.nonzero()
    #
    # orientation = orientation[x, y, z]
    # # print(orientation)
    #
    # # print(orientation)
    # half_margin = 2
    #
    # x = (x - half_margin) * obj.grid.resolution[0] + obj.grid.min_val[0]
    # y = (y - half_margin) * obj.grid.resolution[1] + obj.grid.min_val[1]
    # z = (z - half_margin) * obj.grid.resolution[2] + obj.grid.min_val[2]
    # orientation = (orientation - half_margin) * obj.grid.resolution[3] + obj.grid.min_val[3]
    #
    # # print()
    # # print(orientation)
    #
    # u = np.cos(orientation)
    # v = np.sin(orientation)
    # w = np.zeros(np.shape(orientation))
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # ax.quiver(x, y, z, u, v, w, color='g', length=1.0)
    # plt.show()

    # print(min(grid[abs(grid)>0]))


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # n = 100

    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    #     ax.scatter(xs, ys, zs, c=c, marker=m)

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # plt.show()


if __name__ == "__main__":
    main()
