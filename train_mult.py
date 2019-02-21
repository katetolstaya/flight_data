import random
import configparser
from planning.dubins_problem import DubinsProblem
from planning.dubins_multi_objective import DubinsMultiAirplaneObjective
from planning.grid import Grid
from data_utils import load_flight_data, make_planner, load_lims, get_multi_airplane_segments, time_sync_flight_data, \
    log
import sys
inf = float("inf")


def main():
    # read in parameters
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'cfg/params.cfg'

    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']

    to = float(config['timeout'])
    n_iters = int(config['num_iterations'])
    seed = int(config['random_seed'])
    log_file = open('logs/' + config['grid_filename'] + "_mult_log.txt", "wb")

    if seed >= 0:
        random.seed(seed)

    print('Processing trajectories...')
    fnames = ['flights20160112']  # , 'flights20160112', 'flights20160113']
    flight_summaries = load_flight_data(fnames)
    lists = get_multi_airplane_segments(flight_summaries)
    random.shuffle(lists)

    folder = "model/"
    fname = "grid19"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)
    obj = DubinsMultiAirplaneObjective(config, grid)
    obj_expert = DubinsMultiAirplaneObjective(config, grid)
    planner = make_planner(config['planner_type'])
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    n_updates = 0

    # start training
    ind = 0
    for n in range(n_iters):
        random.shuffle(lists)
        for l in lists:
            paths = time_sync_flight_data(l, problem)

            print('Planning for ' + str(len(l)) + ' airplanes...')
            obj.clear_obstacles()
            obj_expert.clear_obstacles()
            timeout = False

            for expert_path in paths:

                # expert path computation
                expert_path_ind = problem.path_to_ind(expert_path)
                expert_dense_path = DubinsProblem.resample_path_dt(expert_path, s=0.1, dt=1.0)
                expert_cost = obj_expert.integrate_path_cost(expert_path, expert_path_ind)

                # default values
                planner_cost = 0
                path_diff = inf
                planner_path_ind = None

                if not timeout:

                    # plan trajectory
                    node = planner(problem, expert_path[0, :], expert_path[-1, :], obj).plan(to)

                    if node is not None:

                        # planner path computation
                        planner_path_ind = problem.reconstruct_path_ind(node)
                        planner_path = problem.ind_to_path(planner_path_ind)
                        planner_dense_path = DubinsProblem.resample_path_dt(planner_path, s=0.1, dt=1.0)
                        planner_cost = obj.integrate_path_cost(planner_path, planner_path_ind)
                        path_diff = problem.compute_avg_path_diff(expert_dense_path, planner_dense_path)

                    else:
                        timeout = True

                # gradient step
                obj.update_obstacle_lims(expert_path_ind, planner_path_ind, 1.0)
                obj_expert.obstacle_lims = obj.obstacle_lims

                log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_diff) + '\t' + str(
                    obj.obstacle_lims), log_file)

                obj_expert.add_obstacle(expert_path_ind)
                if not timeout:
                    obj.add_obstacle(planner_path_ind)

                ind = ind + 1



if __name__ == "__main__":
    main()
