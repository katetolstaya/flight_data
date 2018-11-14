import sys
sys.path.insert(0, "../dubins_planning")

import numpy as np
import matplotlib.pyplot as plt
import pickle, random, math

from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport, get_flights, get_min_max

from dubins_node import Node
from astar import astar, reconstruct_path, plot_path
from objective import Objective
from grid import Grid

def save_objective(obj):
    pickle.dump(obj, open('model/objective.pkl','wb') )


def interp_expert(flight, N):
    path = np.concatenate((flight.loc_xyzbea, flight.time.reshape((-1,1))), axis=1)

    min_t = np.min(path[:,4])
    max_t = np.max(path[:,4])

    new_path = np.zeros((N,5))
    new_path[:,4] = np.linspace(min_t, max_t, N) # time vector

    for i in range(0,4):
        if i == 3:
            period = 2*math.pi
        else:
            period = None
        new_path[:,i] = np.interp(new_path[:,4], path[:,4], path[:,i], period=period)
    return new_path

def update_grid(grid, path, coeff):
    M = path.shape[0]
    for i in range(0, M):
        grid.set(path[i, 0:4], grid.get(path[i,0:4]) + coeff * 1.0/M)

def main():

    params = Parameters()

    # load flight data
    fnames = ['flights20160111','flights20160112','flights20160113']
    flight_summaries = []
    for fname in fnames:
        flights = pickle.load(open('data/' + fname+ '.pkl', 'rb'))
        _, summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)


    # set up grid
    xyzbea_min, xyzbea_max = get_min_max(flight_summaries)
    resolution = np.array([2.0, 2.0, 0.2, 0.314]) #/2.0  #(xyzbea_max - xyzbea_min)/20.0
    grid = Grid(xyzbea_min, xyzbea_max, resolution)
    
    # initialize cost with one pass through the data
    N = 100
    for flight in flight_summaries:
        path = interp_expert(flight, N)
        update_grid(grid, path, -5.0)

    objective = Objective(grid)
    save_objective(objective)

    random.shuffle(flight_summaries)
    #random.seed(0)

    ind = 0 
    n_iters = 10
    astar_timeout = 30.0

    for iter in range(0, n_iters):

        for flight in flight_summaries:
            xyzb = flight.loc_xyzbea

            start = Node(xyzb[0,0], xyzb[0,1], xyzb[0,2], xyzb[0,3], 0)
            goal = Node(xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], xyzb[-1,3] , 0)

            node = astar(start, goal, objective, astar_timeout)

            if node is not None:

                planner_path = reconstruct_path(node)
                expert_path = interp_expert(flight, N)
                print(objective.integrate_path_cost(expert_path) - objective.integrate_path_cost(planner_path))

                update_grid(grid, planner_path, 1.0)
                update_grid(grid, expert_path, -1.0)

                ind = ind + 1
                if ind % 30 == 0 :
                    save_objective(objective)

if __name__ == "__main__":
    main()
