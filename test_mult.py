import configparser
import random
from process import get_min_max_all
from train import load_flight_data, make_planner
from plot_utils import plot_planner_expert
from planning.grid import Grid
from planning.dubins_problem import DubinsProblem
import matplotlib.pyplot as plt
from train_mult import get_multi_airplane_segments, time_sync_flight_data
from planning.dubins_multi_objective import DubinsMultiAirplaneObjective

def main():
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    to = float(config['timeout'])
    seed = int(config['random_seed'])
    planner = make_planner(config['planner_type'])

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    lists = get_multi_airplane_segments(flight_summaries)

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    print('Loading cost...')
    # set up cost grid
    # n_iters = int(config['num_iterations'])
    # n_samples = int(config['num_samples'])

    grid = Grid(config, xyzbea_min, xyzbea_max)
    grid.load_grid()
    obj = DubinsMultiAirplaneObjective(config, grid)
    obj_expert = DubinsMultiAirplaneObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    for l in lists:

        print('Planning for ' + str(len(l)) + ' airplanes...')
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()
        obj_expert.clear_obstacles()

        for expert_path in paths:

            expert_path_ind = problem.path_to_ind(expert_path)
            # obj.update_obstacle_lims(expert_path_ind, expert_path_ind)

            node = planner(problem, expert_path[0, :], expert_path[-1, :], obj).plan(to)

            if node is not None:

                planner_path_ind = problem.reconstruct_path_ind(node)
                planner_path = problem.ind_to_path(planner_path_ind)
                expert_path_ind = problem.path_to_ind(expert_path)

                expert_cost = obj.integrate_path_cost(expert_path, expert_path_ind)
                planner_cost = obj.integrate_path_cost(planner_path, planner_path_ind)

                # path_diff = problem.compute_avg_path_diff(expert_path, planner_path)
                # print(path_diff)

                print(planner_cost - expert_cost)

                # print('time')
                # print(planner_path[-1,4] - expert_path[-1,4])

                planner_spline = problem.resample_path(planner_path)
                expert_spline = problem.resample_path(expert_path)
                plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline)

                obj.add_obstacle(planner_path_ind)
                obj_expert.add_obstacle(expert_path_ind)

            else:
                print('Timeout')
                break

if __name__ == "__main__":
    main()
