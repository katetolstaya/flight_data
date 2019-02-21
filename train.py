import random
from process import get_min_max_all
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective
import configparser
from planning.dubins_problem import DubinsProblem
import numpy as np

from data_utils import log, load_flight_data, make_planner, save_lims

def main():
    # read in parameters
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    planner = make_planner(config['planner_type'])

    to = float(config['timeout'])
    n_iters = int(config['num_iterations'])
    n_samples = int(config['num_samples'])
    seed = int(config['random_seed'])

    log_file = open(config['grid_filename']+"_log.txt", "wb")
    folder = 'model/'
    fname = config['grid_filename']

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    # # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    save_lims(xyzbea_min, xyzbea_max, folder, fname)

    log('Initializing...')
    # set up cost grid
    grid = Grid(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    log('T\tPlanner\tExpert\tDiff', log_file)
    #initialize cost with one pass through the data
    dt = 1.0
    ind = 0
    # if flight_summaries is not None:
    #
    #     for flight in flight_summaries:
    #         # can't interpolate paths with len < 4
    #         if flight.get_path_len() < 5:
    #             continue
    #
    #         path = flight.to_path()
    #         dense_path = DubinsProblem.resample_path_dt(path, s=0.1, dt=dt)
    #         expert_cost = obj.integrate_path_cost(dense_path)
    #
    #         if ind % 10 == 0:
    #             start, goal = flight.get_start_goal()
    #             # try planning
    #             node = planner(problem, start, goal, obj).plan(to)
    #             if node is not None:
    #                 planner_path = problem.reconstruct_path(node)
    #                 planner_dense_path = DubinsProblem.resample_path_dt(planner_path, s=0.1, dt=dt)
    #
    #                 planner_cost = obj.integrate_path_cost(planner_dense_path)
    #                 path_diff = problem.compute_avg_path_diff(dense_path, planner_dense_path)
    #                 log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_diff), log_file)
    #             else:
    #                 log(str(ind) + '\t' + '0\t' + str(expert_cost) + '\t' + str(np.inf), log_file)
    #
    #         for i in range(0, n_iters):
    #             grid.gradient_step(dense_path, -10.0)  # TODO
    #             ind = ind + 1
    #
    # log('Saving grid...')
    # grid.save_grid()

    min_ind = 3000
    step_ind = 50
    save_ind = 250

    log('Planning...')
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

                    log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_diff), log_file)
                    # print(planner_cost - expert_cost)
                    grid.gradient_step(planner_dense_path, 10.0)
                    grid.gradient_step(expert_dense_path, -10.0)
                else:
                    expert_only = True
            else:
                expert_only = True

            if expert_only:
                log(str(ind) + '\t' + '0\t' + str(expert_cost) + '\t' + str(np.inf), log_file)
                grid.gradient_step(expert_dense_path, -10.0)

            ind = ind + 1
            if ind % save_ind == 0:
                log('Saving grid...')
                obj.grid.save_grid()

    obj.grid.save_grid()


if __name__ == "__main__":
    main()
