import numpy as np
import matplotlib.pyplot as plt
import pickle

from process import get_flights, get_colors
from segment import make_segments, get_actual_costs, get_costs_from_segments
from parameters import Parameters


def main():

    params = Parameters()
    flights = pickle.load(open(params.fname + '.pkl', 'rb'))
    flights_arr, flight_summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw


    actual_costs, interp_paths = get_actual_costs(flight_summaries, params)
    segments = make_segments(flights_arr, params)
    my_costs = get_costs_from_segments(segments)

    print(actual_costs)
    print(my_costs)
    print [x / actual_costs[3] for x in actual_costs]
    print [x / my_costs[3] for x in my_costs]

    # plot results
    colors = get_colors(flights_arr)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for segment in segments:
        segment.plot(ax, colors)
        #print(segment.interval, segment.refs)

    for flight in flight_summaries:
        loc_xyzbea = flight.loc_xyzbea
        ax.plot(loc_xyzbea[:, 0], loc_xyzbea[:, 1], loc_xyzbea[:, 2], colors[flight.ref], marker='o', lw=0)
        #print (flight.time[0], flight.time[flight.T-1], flight.ref)

    ax.legend()
    plt.show()





if __name__ == "__main__":
    main()