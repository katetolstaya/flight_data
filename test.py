import numpy as np
import pickle, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from process import get_min_max_all
from train import flight_to_path, load_flight_data
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

    flight_summaries = load_flight_data()
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    xyzbea_min[4] = 0.0
    xyzbea_max[4] = 1.0

    grid = Grid(config, xyzbea_min, xyzbea_max)
    obj = DubinsObjective(config, grid)
    obj.grid.grid = pickle.load(open('model/grid.pkl', 'rb'))
    # obj = None
    random.seed(5)
    random.shuffle(flight_summaries)

    to = float(config['timeout'])
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Planning...')
    for flight in flight_summaries:
        time = flight.time
        xyzb = flight.loc_xyzbea
        start = np.array([xyzb[0, 0], xyzb[0, 1], xyzb[0, 2], xyzb[0, 3], flight.time[0]]).flatten()
        goal = np.array([xyzb[-1, 0], xyzb[-1, 1], xyzb[-1, 2], xyzb[-1, 3], flight.time[-1]]).flatten()
        node = ARAStar(problem, start, goal, obj).plan(to)
        if node is not None:
            planner_path = problem.reconstruct_path(node)
            expert_path = flight_to_path(flight)
            planner_spline = problem.smoothing_spline(planner_path, 2)
            expert_spline = problem.smoothing_spline(expert_path, 3)
            plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline)
        else:
            print('Timeout')


if __name__ == "__main__":
    main()
