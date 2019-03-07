import random
from process import get_min_max_all
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective
import configparser
from planning.dubins_problem import DubinsProblem
import numpy as np
import sys

from data_utils import log_fname, load_flight_data, make_planner, save_lims, log

import matplotlib.pyplot as plt

def main():
    # read in parameters

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'cfg/params.cfg'

    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    planner = make_planner(config['planner_type'])

    to = float(config['timeout'])
    n_iters = int(config['num_iterations'])
    n_samples = int(config['num_samples'])
    seed = int(config['random_seed'])

    log_file_name = 'logs/' + config['grid_filename'] + "_log.txt"
    folder = 'model/'
    fname = config['grid_filename']

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data(config)
    random.shuffle(flight_summaries)

    # # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    save_lims(xyzbea_min, xyzbea_max, folder, fname)

    # print('Initializing...')
    # set up cost grid
    grid = Grid(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    dt = 1.0
    ind = 0

    min_ind = int(config['min_ind'])
    step_ind = int(config['step_ind'])
    save_ind = int(config['save_ind'])

    log('T\tPlanner\tExpert\tDiff')

    plt.ion()
    fig, ax = plt.subplots(facecolor='white')

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    time_text = plt.text(20.0, 20.0, '', fontsize=18)
    for i in range(0, n_iters):

        for flight in flight_summaries:
            # can't interpolate paths with len < 4
            if flight.get_num_waypoints() < 4:
                continue

            start, goal = flight.get_start_goal()
            expert_path = flight.to_path()
            expert_dense_path = DubinsProblem.resample_path_dt(expert_path, s=0.1, dt=dt)
            expert_cost = obj.integrate_path_cost(expert_dense_path)

            if expert_cost > 100000:
                continue

            # try planning
            expert_only = True


            if expert_only:
                # log(str(ind) + '\t' + '0\t' + str(expert_cost) + '\t' + str(np.inf))
                grid.gradient_step(expert_dense_path, -10.0)

            ind = ind + 1


            if ind % 100 == 0:
                max_x = 250
                max_y = 250

                offset_x = -250
                offset_y = -750

                cost_min = np.ones((max_x, max_y)) * 100

                for k in grid.grid.keys():
                    x = k[0] + offset_x
                    y = k[1] + offset_y

                    if x > 0 and y > 0 and x < max_x and y < max_y:
                        cost_min[x, y] = min(cost_min[x, y], grid.grid[k])

                cost_min = cost_min.T

                ax.set_xlim([0, max_x])
                ax.set_ylim([0, max_y])

                ax.imshow(-1.0 * cost_min, extent=[0, max_x, 0, max_y], cmap='Greens', interpolation='spline16',
                          origin='lower')
                time_text.set_text("$i=${0:d}".format(int(ind)))
                fig.canvas.draw()
                fig.canvas.flush_events()

    obj.grid.save_grid()


if __name__ == "__main__":
    main()
