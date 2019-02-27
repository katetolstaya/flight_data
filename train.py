import random
from process import get_min_max_all
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective
import configparser
from planning.dubins_problem import DubinsProblem
import numpy as np
import sys

from data_utils import log_fname, load_flight_data, make_planner, save_lims, log

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

    log_file_name = 'logs/' + config['grid_filename']+"_log.txt"
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

    #print('Initializing...')
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
    for i in range(0, n_iters):

        for flight in flight_summaries:
            # can't interpolate paths with len < 4
            if flight.get_path_len() < 4:
                continue

            start, goal = flight.get_start_goal()
            expert_path = flight.to_path()
            expert_dense_path = DubinsProblem.resample_path_dt(expert_path, s=0.1, dt=dt)
            expert_cost = obj.integrate_path_cost(expert_dense_path)

            if expert_cost > 100000:
                continue

            # try planning
            expert_only = False
            if ind > min_ind or ind % step_ind == 0:

                node = planner(problem, start, goal, obj).plan(to)
                if node is not None and (i < min_ind or i % step_ind == 0):
                    planner_path = problem.reconstruct_path(node)
                    planner_dense_path = DubinsProblem.resample_path_dt(planner_path, s=0.1, dt=dt)

                    # compute cost
                    planner_cost = obj.integrate_path_cost(planner_dense_path)
                    path_diff = problem.compute_avg_path_diff(expert_dense_path, planner_dense_path)

                    log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_diff))
                    # print(planner_cost - expert_cost)
                    grid.gradient_step(planner_dense_path, 10.0)
                    grid.gradient_step(expert_dense_path, -10.0)
                else:
                    expert_only = True
            else:
                expert_only = True

            if expert_only:
                log(str(ind) + '\t' + '0\t' + str(expert_cost) + '\t' + str(np.inf))
                grid.gradient_step(expert_dense_path, -10.0)

            ind = ind + 1
            # if ind % save_ind == 0:
            #     #print('Saving grid...')
            #     obj.grid.save_grid()

    obj.grid.save_grid()


if __name__ == "__main__":
    main()
