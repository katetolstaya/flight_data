import numpy as np
import matplotlib.pyplot as plt
import pickle

from process import get_flights, get_colors
from segment import make_segments, get_actual_costs, get_costs_from_segments
from parameters import Parameters
from process import get_min_time, timestamp, min_dist_to_airport


def main():

    params = Parameters()
    flights = pickle.load(open(params.fname + '.pkl', 'rb'))

    flights_arr, flight_summaries = get_flights(flights, params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
    # actual_costs, actual_segments = get_actual_costs(flight_summaries, params)

    # # plot results
    # colors = get_colors(flights_arr)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # for segment in actual_segments:
    #     segment.plot(ax, colors)
    #     #print(segment.interval, segment.refs)
    #
    # for flight in flight_summaries:
    #     loc_xyzbea = flight.loc_xyzbea
    #     ax.plot(loc_xyzbea[:, 0], loc_xyzbea[:, 1], loc_xyzbea[:, 2], colors[flight.ref], marker='o')
    #     #print (flight.time[0], flight.time[flight.T-1], flight.ref)
    #
    # ax.legend()
    # plt.show()

    params.min_time = get_min_time(flights)
    params.start_t = params.min_time + params.center_t - params.range_t
    params.end_t = params.min_time + params.center_t + params.range_t

    airport = params.airport
    lat0 = params.lat0
    lon0 = params.lon0
    alt0 = params.alt0

    for i, k in enumerate(sorted(flights.keys())):
        flight = flights[k]

        # time within range
        in_range = np.array([timestamp(t) < params.end_t and timestamp(t) > params.start_t for t in flight.time])

        # alt within range
        in_range = np.logical_and(in_range, np.array([af < params.alt_lim for af in flight.altitude]))

        if (flight.departure != airport and flight.departure != airport): #np.min(np.abs(flight.latitude)) == 0.0 or
            continue
        if min_dist_to_airport(params.start_t, params.end_t, flight, lat0, lon0, alt0) > params.dist_lim:
            continue  # far from airport
        elif np.sum(in_range) < 2:  # fewer than 2 points in path
            continue
        else:

            time = np.array([timestamp(t) - params.start_t for t in flight.time])[in_range]

            lat = np.array(flight.latitude)[in_range]
            lon = np.array(flight.longitude)[in_range]
            alt = np.array(flight.altitude)[in_range]
            next_lat = np.array(flight.next_latitude)[in_range]
            next_lon = np.array(flight.next_longtiude)[in_range]

            plt.plot(time, alt, marker='o', lw=0, ms=4)

    #         plt.plot(lon, lat, marker='o', lw=0, ms=2)
    #
    # plt.ylim(46, 49)
    # plt.xlim(-126,-119)

    # plt.ylim(46, 49)
    #plt.xlim(42000,44000)

    # plt.ylabel('Latitude')
    # plt.xlabel('Longitude')

    plt.xlabel('Time')
    plt.ylabel('Altitude')

    plt.show()


if __name__ == "__main__":
    main()