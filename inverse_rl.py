import numpy as np
import pickle, random, math, sys

#sys.path.insert(0, "./util")

from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport, get_flights, get_min_max
from planning.dubins_node import DubinsNode, reconstruct_path, plot_path
from planning.grid import Grid
from planning.objective import DubinsObjective
from planning.arastar import ARAStar
from planning.astar import AStar

def save_objective(obj):
    pickle.dump(obj, open('model/objective.pkl','wb') )

def interp_expert(flight, N):
    path = np.concatenate((flight.loc_xyzbea, flight.time.reshape((-1,1))), axis=1)
    return interp_path(path, N)

def interp_path(path, N):
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

def load_flight_data():
    params = Parameters()
    fnames = ['flights20160111','flights20160112','flights20160113']
    flight_summaries = []
    for fname in fnames:
        flights = pickle.load(open('data/' + fname+ '.pkl', 'rb'))
        _, summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)
    return flight_summaries

def main():

    flight_summaries = load_flight_data()
    # set up grid
    xyzbea_min, xyzbea_max = get_min_max(flight_summaries)
    resolution = np.array([4.0, 4.0, 0.4, 0.4]) #/ 1.2 #/2.0  #(xyzbea_max - xyzbea_min)/20.0
    grid = Grid(xyzbea_min, xyzbea_max, resolution)
    
    # initialize cost with one pass through the data
    N = 500
    for flight in flight_summaries:
        path = interp_expert(flight, N)
        update_grid(grid, path, -10.0)

    obj = DubinsObjective(grid)
    save_objective(obj)

    random.shuffle(flight_summaries)
    #random.seed(0)

    ind = 0 
    n_iters = 10
    to = 120.0

    print('Planning...')

    for iter in range(0, n_iters):

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

                update_grid(grid, planner_path, 100.0)
                update_grid(grid, expert_path, -100.0)

                ind = ind + 1
                if ind % 30 == 0 :
                    save_objective(obj)
            else:
                print('None')

if __name__ == "__main__":
    main()
