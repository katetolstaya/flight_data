import numpy as np
import pickle, random
from parameters import Parameters
from process import get_flights, get_min_max_all
from planning.grid import Grid
from planning.objective import DubinsObjective
from planning.arastar import ARAStar
from planning.astar import AStar
import configparser
from planning.dubins_problem import DubinsProblem
from planning.dubins_util import neg_pi_to_pi


def update_grid(grid, path, coeff):
    n_path_points = path.shape[0]
    for i in range(0, n_path_points):
        noise = np.random.normal(0, 0.5, size=(4,))
        noise[2] = 0.1 * noise[2]
        noise[3] = 0.1 * noise[3]
        temp = path[i, 0:4] + noise
        temp[3] = neg_pi_to_pi(temp[3])
        grid.set(temp, grid.get(temp) + coeff * 1.0 / n_path_points)
        # grid.update(temp, coeff * 1.0 / M)


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
    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    grid = Grid(config, xyzbea_min, xyzbea_max)

    # initialize cost with one pass through the data
    for flight in flight_summaries:
        # can't interpolate paths with len < 4
        if flight.get_path_len() < 4:
            continue

        path = flight.to_path()
        dense_path = DubinsProblem.resample_path(path, 3, n_samples)
        for n in range(0, n_iters):
            update_grid(grid, dense_path, -1000.0)
            #update_grid(grid, path, -1000.0)

    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    obj.grid.save_grid()

    print('Planning...')
    ind = 0
    for i in range(0, n_iters):

        for flight in flight_summaries:
            # can't interpolate paths with len < 4
            if flight.get_path_len() < 4:
                continue

            start, goal = flight.get_start_goal()
            node = planner(problem, start, goal, obj).plan(to)

            if node is not None:
                planner_path = problem.reconstruct_path(node)
                expert_path = flight.to_path()
                # print(obj.integrate_path_cost(expert_path) - obj.integrate_path_cost(planner_path))
                #
                # update_grid(grid, expert_path, -100.0)
                # update_grid(grid, planner_path, 100.0)

                expert_dense_path = DubinsProblem.resample_path(expert_path, 3, n_samples)
                planner_dense_path = DubinsProblem.resample_path(planner_path, 3, n_samples)

                print(obj.integrate_path_cost(expert_dense_path) - obj.integrate_path_cost(planner_dense_path))

                update_grid(grid, expert_dense_path, -100.0)
                update_grid(grid, planner_dense_path, 100.0)

                ind = ind + 1
                if ind % 30 == 0:
                    obj.grid.save_grid()
                break
            else:
                print('Timeout')

    obj.grid.save_grid()


if __name__ == "__main__":
    main()
