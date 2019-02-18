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



def load_flight_data():
    params = Parameters()
    fnames = ['flights20160111', 'flights20160112', 'flights20160113']
    #fnames = ['flights20160111']
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

    # # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)

    print('Initializing...')
    # set up cost grid
    n_iters = int(config['num_iterations'])
    n_samples = int(config['num_samples'])
    grid = Grid(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    #initialize cost with one pass through the data
    if flight_summaries is not None:

        for flight in flight_summaries:
            # can't interpolate paths with len < 4
            if flight.get_path_len() < 5:
                continue

            path = flight.to_path()
            dense_path = DubinsProblem.resample_path(path, n_samples)
            print(-1.0 * obj.integrate_path_cost(path))
            for i in range(0, n_iters):
                grid.gradient_step(dense_path, -10000.0)

    print('Saving grid...')
    grid.save_grid()

    print('Planning...')
    ind = 0
    for i in range(0, n_iters):

        for flight in flight_summaries:
            # can't interpolate paths with len < 4
            if flight.get_path_len() < 4:
                continue

            start, goal = flight.get_start_goal()
            expert_path = flight.to_path()
            expert_dense_path = DubinsProblem.resample_path(expert_path, n_samples)
            expert_cost = obj.integrate_path_cost(expert_dense_path)

            # try planning
            node = planner(problem, start, goal, obj).plan(to)
            if node is not None:
                planner_path = problem.reconstruct_path(node)
                planner_dense_path = DubinsProblem.resample_path(planner_path, n_samples)
                planner_cost = obj.integrate_path_cost(planner_dense_path)
                print(planner_cost - expert_cost)
                grid.gradient_step(expert_dense_path, -100.0)
                grid.gradient_step(planner_dense_path, 100.0)
            else:
                print('Timeout')
                print(-1.0 * expert_cost)
                grid.gradient_step(expert_dense_path, -100.0)
            ind = ind + 1
            if ind % 50 == 0:
                print('Saving grid...')
                obj.grid.save_grid()

    obj.grid.save_grid()


if __name__ == "__main__":
    main()
