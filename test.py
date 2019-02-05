import numpy as np
import pickle, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from process import get_min_max_all
from train import load_flight_data
from planning.arastar import ARAStar
from planning.astar import AStar
from planning.dubins_problem import DubinsProblem
import configparser
from planning.grid import Grid
from planning.objective import DubinsObjective


def plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['g', 'r']

    for i, arr in enumerate([expert_path, planner_path]):
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        orientation = arr[:, 3]
        u = np.cos(orientation)
        v = np.sin(orientation)
        w = np.zeros(np.shape(orientation))
        ax.quiver(x, y, z, u, v, w, color=colors[i], length=1.0)

    x = planner_spline[:, 0].flatten()
    y = planner_spline[:, 1].flatten()
    z = planner_spline[:, 2].flatten()
    ax.plot(xs=x, ys=y, zs=z)

    x = expert_spline[:, 0].flatten()
    y = expert_spline[:, 1].flatten()
    z = expert_spline[:, 2].flatten()
    ax.plot(xs=x, ys=y, zs=z)

    plt.show()


def main():
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    to = float(config['timeout'])
    seed = int(config['random_seed'])
    planner_type = config['planner_type']
    if planner_type == 'AStar':
        planner = AStar
    elif planner_type == 'ARAStar':
        planner = ARAStar
    else:
        raise NotImplementedError

    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    grid = Grid(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    obj.grid.load_grid()

    # initialize planner problem
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Planning...')
    for flight in flight_summaries:

        # can't interpolate paths with len < 4
        if flight.get_path_len() < 4:
            continue

        start, goal = flight.get_start_goal()

        expert_path = flight.to_path()
        expert_spline = problem.resample_path(expert_path, 3)

        print(expert_path[:,3])
        print(expert_spline[:,3])

        # node = planner(problem, start, goal, obj).plan(to)
        # if node is not None:
        #     planner_path = problem.reconstruct_path(node)
        #
        #     planner_spline = DubinsProblem.resample_path(planner_path, 2)
        #
        #     plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline)
        # else:
        #     print('Timeout')


if __name__ == "__main__":
    main()
