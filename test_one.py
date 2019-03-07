import configparser
import random
import sys
from matplotlib import rc

from planning.grid import Grid
from planning.dubins_objective import DubinsObjective
from planning.dubins_problem import DubinsProblem
from utils.data_utils import load_flight_data, make_planner, load_lims, log
from utils.plot_utils import plot_planner_expert

rc('text', usetex=True)
font = {'family': 'serif', 'weight': 'bold', 'size': 14}
rc('font', **font)


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
    folder = "models/"
    fname = "grid22"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Planning...')
    ind = 0
    for flight in flight_summaries:

        if flight.get_num_waypoints() < 4:
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

            path_min_diff = problem.compute_avg_min_diff(planner_path, expert_spline)

            log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_min_diff))

            plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline)
            ind = ind + 1
        # else:
        #     print('Timeout')


if __name__ == "__main__":
    main()
