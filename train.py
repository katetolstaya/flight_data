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


def save_grid(obj):
    pickle.dump(obj, open('model/grid.pkl', 'wb'))


def interp_expert(flight, N):
    path = np.concatenate((flight.loc_xyzbea, flight.time.reshape((-1,1))), axis=1)
    return path


def flight_to_path(flight):
    path = np.concatenate((flight.loc_xyzbea, flight.time.reshape((-1,1))), axis=1)
    return path


def update_grid(grid, path, coeff):
    M = path.shape[0]
    for i in range(0, M):

        noise = np.random.normal(0, 0.5, size=(4, ))
        noise[2] = 0.1 * noise[2]
        noise[3] = 0.1 * noise[3]
        temp = path[i, 0:4] + noise
        temp[3] = zero_to_2pi(temp[3])
        #grid.update(temp, coeff * 1.0 / M)
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

    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']

    to = float(config['timeout'])
    n_iters = int(config['num_iterations'])

    flight_summaries = load_flight_data()

    # set up grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)

    xyzbea_min[4] = 0.0
    xyzbea_max[4] = 1.0


    grid = Grid(config, xyzbea_min, xyzbea_max)

    # initialize cost with one pass through the data
    for n in range(0, n_iters):
        for flight in flight_summaries:
            path = flight_to_path(flight)
            update_grid(grid, path, -1000.0)
    obj = DubinsObjective(config, grid)
    save_grid(obj.grid.grid)

    random.shuffle(flight_summaries)
    # random.seed(0)

    ind = 0

    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Planning...')

    for i in range(0, n_iters):

        for flight in flight_summaries:
            xyzb = flight.loc_xyzbea
            # flight.time stores all the time info!
            # TODO use this to constrain airplane arrival time

            start = np.array([xyzb[0, 0], xyzb[0, 1], xyzb[0, 2], xyzb[0, 3], 1.0]).flatten()
            goal = np.array([xyzb[-1, 0], xyzb[-1, 1], xyzb[-1, 2], xyzb[-1, 3], 1.0]).flatten()
            node = ARAStar(problem, start, goal, obj).plan(to)

            if node is not None:
                planner_path = problem.reconstruct_path(node)
                planner_path = planner_path[0::5, :]
                expert_path = flight_to_path(flight)
                print(obj.integrate_path_cost(expert_path) - obj.integrate_path_cost(planner_path))

                update_grid(grid, planner_path, 100.0)
                update_grid(grid, expert_path, -100.0)
                ind = ind + 1
                if ind % 30 == 0:
                    save_grid(obj.grid.grid)
                break
            else:
                print('Timeout')

            
    save_grid(obj.grid.grid)


if __name__ == "__main__":
    main()
