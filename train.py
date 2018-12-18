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
from planning.dubins_util import zero_to_2pi


def update_grid(grid, path, coeff):
    M = path.shape[0]
    for i in range(0, M):
        noise = np.random.normal(0, 0.5, size=(4,))
        noise[2] = 0.1 * noise[2]
        noise[3] = 0.1 * noise[3]
        temp = path[i, 0:4] + noise
        temp[3] = zero_to_2pi(temp[3])
        # grid.update(temp, coeff * 1.0 / M)
        grid.set(temp, grid.get(temp) + coeff * 1.0 / M)


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


def main():
    # read in parameters
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']

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
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)
    grid = Grid(config, xyzbea_min, xyzbea_max)

    # initialize cost with one pass through the data
    for n in range(0, n_iters):
        for flight in flight_summaries:

            # can't interpolate paths with len < 4
            if flight.get_path_len() < 4:
                continue

            path = flight.to_path()
            dense_path = DubinsProblem.resample_path(path, 3, n_samples)
            update_grid(grid, dense_path, -1000.0)

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
            node = ARAStar(problem, start, goal, obj).plan(to)

            if node is not None:
                planner_path = problem.reconstruct_path(node)
                expert_path = flight.to_path()
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
