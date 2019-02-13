import configparser
import random
from process import get_min_max_all
from train import load_flight_data, make_planner
from train import init_obj_prob
from plot_utils import plot_planner_expert


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

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    obj, problem = init_obj_prob(config, xyzbea_min, xyzbea_max, flight_summaries)
    # obj.grid.load_grid()

    print('Planning...')
    for flight in flight_summaries:
        if flight.get_path_len() < 4:
            continue
        start, goal = flight.get_start_goal()
        node = planner(problem, start, goal, obj).plan(to)
        if node is not None:
            planner_path = problem.reconstruct_path(node)

            expert_path = flight.to_path()
            planner_spline = problem.resample_path(planner_path, start, goal)
            expert_spline = problem.resample_path(expert_path, start, goal)
            plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline)
        else:
            print('Timeout')

if __name__ == "__main__":
    main()
