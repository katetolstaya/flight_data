import sys
sys.path.insert(0, "../dubins_planning")

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport, get_flights, get_min_max

from dubins_node import Node
from astar import astar, reconstruct_path, plot_path
from objective import Objective
from grid import Grid
import pickle

def main():

    params = Parameters()
    fnames = ['flights20160111','flights20160112','flights20160113']
    flight_summaries = []

    for fname in fnames:
        flights = pickle.load(open('data/' + fname+ '.pkl', 'rb'))
        _, summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)
    
    xyzbea_min, xyzbea_max = get_min_max(flight_summaries)

    resolution = np.array([2.0, 2.0, 0.2, 0.314])  #(xyzbea_max - xyzbea_min)/20.0
    print(resolution)

    #resolution[0,3] = resolution[0,3] 

    # set grid to number of visits by all trajectories
    grid = Grid(xyzbea_min, xyzbea_max, resolution)
    # for flight in flight_summaries:
    #     for i in range(0, flight.loc_xyzbea.shape[0]):
    #         val = grid.get(flight.loc_xyzbea[i,:])
    #         #print(val)
    #         grid.set(flight.loc_xyzbea[i, :], val-1.0)

    n_iters = 10
    astar_timeout = 30.0

    objective = Objective(grid)
    #random.seed(0)
    random.shuffle(flight_summaries)

    ind = 0 

    for iter in range(0, n_iters):

        for flight in flight_summaries:
            xyzb = flight.loc_xyzbea

            # start = Node(xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], (np.pi - xyzb[-1,3]) % (2.0 * np.pi), 0)
            # goal = Node(xyzb[0,0], xyzb[0,1], xyzb[0,2], (np.pi - xyzb[0,3]) % (2.0 * np.pi), 0)
            goal = Node(xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], xyzb[-1,3] , 0)
            start = Node(xyzb[0,0], xyzb[0,1], xyzb[0,2], xyzb[0,3], 0)

            node = astar(start, goal, objective, astar_timeout)

            if node is not None:
                path2 = reconstruct_path(node)

                # loss
                #old_grid = np.copy(grid.grid)
                path = np.concatenate((xyzb, flight.time.reshape((-1,1))), axis=1)
                print(objective.integrate_path_cost(path) - objective.integrate_path_cost(path2) )
                
                # update

                # expert traj
                N = xyzb.shape[0]
                for i in range(0, N):
                    val = grid.get(xyzb[i,:])
                    grid.set(xyzb[i, :], val - 1.0/N)

                # planner traj
                M = path2.shape[0]
                for i in range(0, M):
                    val = grid.get(path2[i,0:4])
                    grid.set(path2[i, 0:4], val + 1.0/M)

                #print(np.sum((old_grid - grid.grid)**2))

                ind = ind + 1

                if ind % 10 == 0 :
                    pickle.dump(objective, open('model/objective.pkl','wb') )

if __name__ == "__main__":
    main()
