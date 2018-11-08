import sys
sys.path.insert(0, "../dubins_planning")

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from process import get_flights, get_colors, get_min_max
from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport

from dubins_node import Node
from astar import astar, reconstruct_path, plot_path
from objective import Objective

def main():

    params = Parameters()
    fnames = ['flights20160111','flights20160112','flights20160113']
    flight_summaries = []

    for fname in fnames:
        flights = pickle.load(open('data/' + fname+ '.pkl', 'rb'))
        _, summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)
    
    xyzbea_min, xyzbea_max = get_min_max(flight_summaries)

    resolution = (xyzbea_max - xyzbea_min)/20.0

    resolution[0,3] = resolution[0,3] * 5

    # set grid to number of visits by all trajectories
    grid = Objective.Grid(xyzbea_min, xyzbea_max, resolution)

    for flight in flight_summaries:
        for i in range(0, flight.loc_xyzbea.shape[0]):
            val = grid.get(flight.loc_xyzbea[i,:])
            #print(val)
            grid.set(flight.loc_xyzbea[i, :], val+1)



    objective = Objective(grid)
    random.shuffle(flight_summaries)

    for flight in flight_summaries:
        xyzb = flight.loc_xyzbea
        path = np.concatenate((xyzb, flight.time.reshape((-1,1))), axis=1)

        print('............................')
        #print(flight.time[-1] - flight.time[0])
        print(objective.integrate_path_cost(path))
        plot_path(path)


        goal = Node(xyzb[-1,0], xyzb[-1,1], xyzb[-1,2], xyzb[-1,3], 0)
        start = Node(xyzb[0,0], xyzb[0,1], xyzb[0,2], xyzb[0,3], 0)

        node = astar(start, goal, objective)
        path = reconstruct_path(node)
        #print(path[-1,4] - path[0,4])
        print(objective.integrate_path_cost(path))
        plot_path(path)

if __name__ == "__main__":
    main()