import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from process import get_flights, get_colors
from segment import make_segments, get_actual_costs, get_costs_from_segments
from parameters import Parameters


def main():
    failures = 0
    success = 0

    total_costs = np.zeros((4,))
    total_actual_costs = np.zeros((4,))

    params = Parameters()
    flights = pickle.load(open(params.fname + '.pkl', 'rb'))


    params.center_t = 26000 #86000

    flights_arr, flight_summaries = get_flights(flights,
                                                params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw

    print(flights_arr)

    for flight in flight_summaries:
        print(flight.loc_xyzbea)


    if not len(flights_arr) == 0:

        print('interpolating actual')
        actual_costs, segments_actual = get_actual_costs(flight_summaries, params)
        print(actual_costs)


        if segments_actual is not None:

            params.switch_params("learn")
            print('planning traj')
            segments = make_segments(flights_arr, params)

            if segments is not None:
                my_costs = get_costs_from_segments(segments)

                total_costs = np.add(total_costs, np.array(my_costs))
                total_actual_costs = np.add(total_actual_costs, np.array(actual_costs))

                print my_costs
                print actual_costs


                # plot results
                colors = get_colors(flights_arr)
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                for plan in segments:
                    plan.plot(ax, colors)

                for plan in segments_actual:
                    plan.plot(ax, colors)

                for flight in flight_summaries:
                    loc_xyzbea = flight.loc_xyzbea
                    ax.plot(loc_xyzbea[:, 0], loc_xyzbea[:, 1], loc_xyzbea[:, 2], colors[flight.ref], marker='o', lw=0)

                ax.legend()
                plt.show()
            else:
                print('planning failed')

        else:
            print('interpolation failed')








if __name__ == "__main__":
    main()