import numpy as np
import pickle, random, math, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport, get_flights, get_min_max
from planning.dubins_node import DubinsNode, reconstruct_path, plot_path
from planning.objective import DubinsObjective
from planning.grid import Grid
from planning.arastar import ARAStar
from planning.astar import AStar
from inverse_rl import interp_expert, load_flight_data, interp_path, interp_expert, reconstruct_path

def plot_planner_expert(planner_path, expert_path):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(expert_path[:,0], expert_path[:,1], expert_path[:,2], '.')
    ax.plot(planner_path[:,0], planner_path[:,1], planner_path[:,2], '.')
    plt.show()

def main():

    flight_summaries = load_flight_data()
    obj = pickle.load(open('model/objective.pkl', 'rb'))
    random.seed(0)
    random.shuffle(flight_summaries)

    ind = 0 
    n_iters = 10
    to = 120.0
    N = 100
    print('Planning...')

    for flight in flight_summaries:
        xyzb = flight.loc_xyzbea

        start = DubinsNode(xyzb[0,0], xyzb[0,1], xyzb[0,2], xyzb[0,3], 0)
        goal = DubinsNode(xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], xyzb[-1,3] , 0)
        node = ARAStar(start, goal, obj).plan(to)

        if node is not None:
            planner_path = reconstruct_path(node)
            planner_path = interp_path(planner_path, N)
            expert_path = interp_expert(flight, N)
            print(obj.integrate_path_cost(expert_path) - obj.integrate_path_cost(planner_path))
            plot_planner_expert(planner_path, expert_path)
        else:
            print('None')

if __name__ == "__main__":
    main()
