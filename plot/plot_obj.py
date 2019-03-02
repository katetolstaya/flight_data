import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import configparser
import random
import sys
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective
from planning.dubins_problem import DubinsProblem
from data_utils import load_flight_data, make_planner, load_lims, get_multi_airplane_segments, time_sync_flight_data


def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'cfg/params.cfg'

    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    to = float(config['timeout'])
    seed = int(config['random_seed'])
    planner = make_planner(config['planner_type'])

    if seed >= 0:
        random.seed(seed)

    # get plane data
    fnames = ['flights20160112']  # , 'flights20160112', 'flights20160113']
    flight_summaries = load_flight_data(config, fnames)

    print('Loading cost...')
    folder = "model/"
    fname = "grid19"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    max_x = 300
    max_y = 300

    offset_x = -250
    offset_y = -750

    offset = np.array([offset_x, offset_y, 0, 0, 0]).reshape((1, 5))
    cost_min = np.ones((max_x, max_y)) * 100

    for k in grid.grid.keys():
        x = k[0] + offset_x
        y = k[1] + offset_y

        if x > 0 and y > 0 and x < max_x and y < max_y:
            cost_min[x, y] = min(cost_min[x, y], grid.grid[k])

    cost_min = cost_min.T

    plt.ion()

    flight_summaries.pop(0)
    for flight in flight_summaries:
        if flight.get_num_waypoints() < 4:
            continue
        start, goal = flight.get_start_goal()
        node = planner(problem, start, goal, obj).plan(to)
        if node is not None:
            # planner_path = problem.reconstruct_path(node)
            planner_path = problem.reconstruct_path_ind(node)

            planner_path_ind = np.zeros((planner_path.shape[0], 2))
            for t in range(planner_path.shape[0]):
                temp = grid.ind_to_index(planner_path[t, :])
                planner_path_ind[t, :] = temp[0:2]
            planner_path_ind = planner_path_ind + np.array([offset_x, offset_y]).reshape((1, 2))

            fig, ax = plt.subplots()
            ax.imshow(cost_min, extent=[0, max_x, 0, max_y], cmap='Blues', interpolation='spline16', alpha=0.5,
                      origin='lower')

            line1, = ax.plot(planner_path_ind[0, 0], planner_path_ind[0, 1], linewidth=5, color='forestgreen')
            for t in range(planner_path.shape[0] - 1):
                line1.set_xdata(planner_path_ind[0:t + 1, 0])
                line1.set_ydata(planner_path_ind[0:t + 1, 1])
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.5)

        else:
            print("timeout")


if __name__ == "__main__":
    main()
