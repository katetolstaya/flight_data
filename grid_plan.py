import sys
sys.path.insert(0, "../dubins_planning")

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math

from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport, get_flights, get_min_max

from dubins_node import Node
from astar import astar, reconstruct_path, plot_path
from objective import Objective
from grid import Grid
from inverse_rl import interp_expert


def plot_planner_expert(planner_path, expert_path):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(expert_path[:,0], expert_path[:,1], expert_path[:,2], '.')
    ax.plot(planner_path[:,0], planner_path[:,1], planner_path[:,2], '.')
    plt.show()

def main():

    # load data
    params = Parameters()
    fnames = ['flights20160111','flights20160112','flights20160113']
    flight_summaries = []

    for fname in fnames:
        flights = pickle.load(open('data/' + fname+ '.pkl', 'rb'))
        _, summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)

    # make or load grid
    if True:
        obj = pickle.load(open('model/objective.pkl', 'rb'))
    else:
        xyzbea_min, xyzbea_max = get_min_max(flight_summaries)
        resolution = (xyzbea_max - xyzbea_min)/20.0
        resolution[0,3] = resolution[0,3] * 4.0

        # initialize grid using expert trajectories
        grid = Grid(xyzbea_min, xyzbea_max, resolution)
        for flight in flight_summaries:
            for i in range(0, flight.loc_xyzbea.shape[0]):
                val = grid.get(flight.loc_xyzbea[i,:])
                grid.set(flight.loc_xyzbea[i, :], val-1.0)

        obj = Objective(grid)

    random.seed(3)
    random.shuffle(flight_summaries)

    for flight in flight_summaries:
        xyzb = flight.loc_xyzbea
        expert_path = interp_expert(flight, 100)

        start = Node(xyzb[0,0], xyzb[0,1], xyzb[0,2], xyzb[0,3], 0)
        goal = Node(xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], xyzb[-1,3], 0)

        node = astar(start, goal, obj, 60.0)

        #break
        if node is not None:
            planner_path = reconstruct_path(node)
            print(obj.integrate_path_cost(expert_path) - obj.integrate_path_cost(planner_path))
            plot_planner_expert(planner_path, expert_path)

if __name__ == "__main__":
    main()