import configparser
import random
import sys
from plot_utils import plot_planner_expert
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective

from planning.dubins_problem import DubinsProblem
from data_utils import load_flight_data, make_planner, load_lims


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
    n_samples = int(config['num_samples'])

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data(config)
    random.shuffle(flight_summaries)

    # set up cost grid

    print('Loading cost...')
    folder = "model/"
    fname = "grid19"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Planning...')
    for flight in flight_summaries:
        if flight.get_path_len() < 4:
            continue
        start, goal = flight.get_start_goal()
        node = planner(problem, start, goal, obj).plan(to)
        if node is not None:
            planner_path = problem.reconstruct_path(node)

            expert_path = flight.to_path()
            planner_spline = problem.resample_path(planner_path, n_samples)
            expert_spline = problem.resample_path(expert_path, n_samples)
            expert_cost = obj.integrate_path_cost(expert_path)
            planner_cost = obj.integrate_path_cost(planner_path)
            print(str(planner_cost - expert_cost))

            plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline)

        else:
            print('Timeout')

if __name__ == "__main__":
    main()
