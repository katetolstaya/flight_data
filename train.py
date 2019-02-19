import pickle
import random
from parameters import Parameters
from process import get_flights, get_min_max_all
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective
from planning.arastar import ARAStar
from planning.astar import AStar
import configparser
from planning.dubins_problem import DubinsProblem
from plot_utils import plot_planner_expert
import numpy as np

def log(s, f=None):
    print(s)
    if f is not None:
        f.write(s)
        f.write('\n')
        f.flush()


def load_flight_data():
    params = Parameters()
    fnames = ['flights20160111', 'flights20160112', 'flights20160113']
    flight_summaries = []
    for fname in fnames:
        flights = pickle.load(open('data/' + fname + '.pkl', 'rb'))
        _, summaries = get_flights(flights,
                                   params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)
    return flight_summaries


def make_planner(planner_type):
    if planner_type == 'AStar':
        planner = AStar
    elif planner_type == 'ARAStar':
        planner = ARAStar
    else:
        raise NotImplementedError
    return planner


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

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    # # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)

    log('Initializing...')
    # set up cost grid
    grid = Grid(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    log('T\tPlanner\tExpert\tDiff', log_file)
    #initialize cost with one pass through the data
    dt = 1.0
    ind = 0
    if flight_summaries is not None:

        for flight in flight_summaries:
            # can't interpolate paths with len < 4
            if flight.get_path_len() < 5:
                continue

            path = flight.to_path()
            dense_path = DubinsProblem.resample_path_dt(path, s=0.1, dt=dt)
            expert_cost = obj.integrate_path_cost(dense_path)

            if ind % 200 == 0:
                start, goal = flight.get_start_goal()
                # try planning
                node = planner(problem, start, goal, obj).plan(to)
                if node is not None:
                    planner_path = problem.reconstruct_path(node)
                    planner_dense_path = DubinsProblem.resample_path_dt(planner_path, s=0.1, dt=dt)

                    planner_cost = obj.integrate_path_cost(planner_path)
                    path_diff = problem.compute_avg_path_diff(dense_path, planner_dense_path)
                    log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_diff), log_file)
                else:
                    log(str(ind) + '\t' + '0\t' + str(expert_cost) + '\t' + str(np.inf), log_file)

            for i in range(0, n_iters):
                grid.gradient_step(dense_path, -10.0) # TODO
                ind = ind + 1

    log('Saving grid...')
    grid.save_grid()

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

            # try planning
            node = planner(problem, start, goal, obj).plan(to)
            if node is not None:
                planner_path = problem.reconstruct_path(node)
                planner_dense_path = DubinsProblem.resample_path_dt(planner_path, s=0.1, dt=dt)

                # compute cost
                planner_cost = obj.integrate_path_cost(planner_path)
                path_diff = problem.compute_avg_path_diff(expert_dense_path, planner_dense_path)

                log(str(ind) + '\t' + str(planner_cost) + '\t' + str(expert_cost) + '\t' + str(path_diff), log_file)
                # print(planner_cost - expert_cost)
                grid.gradient_step(expert_dense_path, -10.0)
                grid.gradient_step(planner_dense_path, 10.0)
            else:
                log(str(ind) + '\t' + '0\t' + str(expert_cost) + '\t' + str(np.inf), log_file)
                grid.gradient_step(expert_dense_path, -10.0)
            ind = ind + 1
            if ind % 50 == 0:
                log('Saving grid...')
                obj.grid.save_grid()

    obj.grid.save_grid()


if __name__ == "__main__":
    main()
