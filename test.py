import numpy as np
import pickle, random, math, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from process import get_min_time, timestamp, min_dist_to_airport, get_flights, get_min_max
from train import interp_expert, load_flight_data, interp_expert
from planning.dubins_node import reconstruct_path
# from planning.objective import DubinsObjective
# from planning.grid import Grid
# from planning.arastar import ARAStar
from planning.astar import AStar
from planning.dubins_problem import DubinsProblem
import configparser

# def plot_planner_expert(planner_path, expert_path):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot(expert_path[:,0], expert_path[:,1], expert_path[:,2], '.')
#     ax.plot(planner_path[:,0], planner_path[:,1], planner_path[:,2], '.')
    


def plot_planner_expert(planner_path, expert_path):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colors = ['g', 'r']

    for i, arr in enumerate([expert_path, planner_path]):
        x = arr[:,0]
        y = arr[:,1]
        z = arr[:,2]
        orientation = arr[:,3]    
        u = np.cos(orientation)
        v = np.sin(orientation)
        w = np.zeros(np.shape(orientation))
        ax.quiver(x, y, z, u, v, w, color=colors[i], length=1.0)

    plt.show()

def main():

    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']

    flight_summaries = load_flight_data()
    xyzbea_min, xyzbea_max = get_min_max(flight_summaries)
    xyzbea_min = np.append(xyzbea_max, [0.0])
    xyzbea_max = np.append(xyzbea_max, [1.0])
    obj = pickle.load(open('model/objective_exp.pkl', 'rb'))
    random.seed(0)
    random.shuffle(flight_summaries)

    ind = 0 
    n_iters = 10
    to =   600.0
    N = 100
    print('Planning...')
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    for flight in flight_summaries:
        xyzb = flight.loc_xyzbea

        start = np.array([xyzb[0,0], xyzb[0,1], xyzb[0,2], xyzb[0,3], 0]).flatten()
        goal = np.array([xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], xyzb[-1,3] , 0]).flatten()


        node = AStar(problem, start, goal, obj).plan(to)

        if node is not None:
            planner_path = reconstruct_path(node)
            planner_path = planner_path[0::5,:] #interp_path(planner_path, N)
            expert_path = interp_expert(flight, N)
            print(obj.integrate_path_cost(expert_path) - obj.integrate_path_cost(planner_path))
            plot_planner_expert(planner_path, expert_path)
        else:
            print('None')

if __name__ == "__main__":
    main()
