import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import math
from parse_data import Flight
from sklearn.neighbors import NearestNeighbors
import numpy as np
from geo import *
from process import haversine, dist_time, dist_euclid, bearing, print_xyz, min_dist_to_airport, interp_loc
import sys

# SEATAC coordinates
lat0 = 47.4435903
lon0 = -122.2960726
alt0 = 109.37  # meters

SCALE = 100.0

##########################
def main():

    #'flights7.pkl'
    fname = 'flights20160113'


    flights = pickle.load(open(fname+'.pkl', 'rb'))
    locs = np.zeros((0,6))
    start_locs = np.zeros((0, 6))
    end_locs = np.zeros((0, 6))
    # min_dist = 10000000000.0
    # min_bar = []

    dists_neigh = np.zeros((0,))
    dists_end = np.zeros((0,))

    pl = 20

    center_t = 1452754900.0 + 1100.0
    center_t = 1452750000.0 + 1900.0

    center_t = 1452754900.0 + 1100.0
    center_t = 1452748000.0 + 2200.0
    range_t =  200.0

    start_t = center_t - range_t
    end_t = center_t + range_t

    alt_lim = 10000
    dist_lim = 10000

    path_length = 0

    #sys.stdout = None

##########################
    for i, k in enumerate(sorted(flights.keys())):

        flight = flights[k]
        ref = flight.ref

        # time within range
        len_path = np.array([t.timestamp() < end_t and t.timestamp() > start_t for t in flight.time])
        # alt within range
        len_path = np.logical_and(len_path, np.array([a < alt_lim for a in flight.altitude]))

        if np.min(np.abs(flight.latitude)) == 0.0 or (flight.departure != "KSEA" and flight.departure != 'KSEA'): # not enough data or parsing issue
            continue
        if min_dist_to_airport(start_t, end_t, flight, lat0, lon0, alt0) > dist_lim:
            continue
        elif not len_path.any():
            continue
        else:
            #print(flight.time)
            tim_temp = np.array([t.timestamp() for t in flight.time])
            tim = tim_temp[len_path]
            lat = np.array(flight.latitude)[len_path]
            lon = np.array(flight.longitude)[len_path]
            alt = np.array(flight.altitude)[len_path]

            end_loc = interp_loc(end_t, flight)
            start_loc = interp_loc(start_t, flight)

            end_locs = np.vstack([end_locs, end_loc])
            start_locs = np.vstack([start_locs, start_loc])

        locs_new = np.zeros((0, 6)) # make a list of new locations

##########################
        # compute path length
        locarr = []
        loclast = []
        bea = 0
        #print(flight.ref)
        for j in range(0, len(tim)):
            #loclast = locarr

            if j < len(tim)-1:
                bea = bearing(lat[j], lon[j], lat[j+1], lon[j+1])
            #print_xyz(np.array([ref, lat[j], lon[j], alt[j], tim[j], bea]).reshape((1,-1)))

            locarr = np.array([ref, lat[j], lon[j], alt[j], tim[j], bea])
            locs_new = np.vstack([locs_new, locarr.reshape((1,-1))])

            if locarr.size >0:
                print_xyz(locarr, lat0, lon0, alt0, SCALE)

            # if len(loclast) > 0: # add up path lengths
            #     path_length = path_length + dist_euclid(locarr[0],loclast[0])

        # dists_end = np.append(dists_end, [dist_euclid(loc, end_loc) for loc in locs_new])
        locs = np.vstack([locs, locs_new])

##########################

    print('Starts')
    for l in range(0, np.shape(start_locs)[0]):
        print_xyz(start_locs[l], lat0, lon0, alt0,  SCALE)
    print('Ends')
    for l in range(0, np.shape(end_locs)[0]):
        print_xyz(end_locs[l], lat0, lon0, alt0,  SCALE)

if __name__== "__main__":
    main()

    #
    # def main():
    #     #'flights7.pkl'
    #     fname = 'flights20160112'
    #
    #     flights = pickle.load(open(fname+'.pkl', 'rb'))
    #     locs = np.zeros((0,5))
    #     # min_dist = 10000000000.0
    #     # min_bar = []
    #
    #     dists_neigh = np.zeros((0,))
    #     dists_end = np.zeros((0,))
    #
    #     pl = 10
    #
    #     for i, k in enumerate(sorted(flights.keys())):
    #
    #         flight = flights[k]
    #         ref = flight.ref
    #
    #         if len(flight.time) < pl or np.min(np.abs(flight.latitude)) == 0.0: # not enough data or parsing issue
    #             continue
    #         elif flight.departure == 'KSEA' and flight.altitude[0] < 1000: # taking off
    #             tim = flight.time[0:pl]
    #             lat = flight.latitude[0:pl]
    #             lon = flight.longitude[0:pl]
    #             alt = flight.altitude[0:pl]
    #             end_loc = np.array([ref, lat[0], lon[0], alt[0], timestamp(tim[0])]) #.reshape((1, -1))
    #         elif flight.arrival == 'KSEA' and flight.altitude[-1] < 1000:  # landing
    #             tim = flight.time[-pl:]
    #             lat = flight.latitude[-pl:]
    #             lon = flight.longitude[-pl:]
    #             alt = flight.altitude[-pl:]
    #             #print(flight.altitude[-1])
    #             end_loc = np.array([ref, lat[-1], lon[-1], alt[-1], timestamp(tim[-1])]) #.reshape((1, -1))
    #             end_loc = np.array([ref, lat[-1], lon[-1], alt[-1], timestamp(tim[-1])]) #.reshape((1, -1))
    #         else:
    #             continue
    #
    #         locs_new = np.zeros((0, 5)) # make a list of new locations
    #
    #         for j in range(0,pl):
    #             locarr = np.array([ref, lat[j], lon[j], alt[j], timestamp(tim[j])]).reshape((1,-1))
    #             #print(locarr)
    #             locs_new = np.vstack([locs_new, locarr])
    #
    #         dists_end_new = [dist_euclid(loc, end_loc) for loc in locs_new]
    #         dists_end = np.append(dists_end, dists_end_new)
    #         locs = np.vstack([locs, locs_new])
    #
    #         #if (locs.size > 0): # compute nearest neighbor
    #         # new_min = np.min(dists_neigh_new)
    #         #dists_neigh = np.append(dists_neigh, dists_neigh_new)
    #         #print(np.min(distances))
    #
    #         # if new_min <= 1000 and new_min > 0:
    #         #     min_bar.append(new_min)
    #
    #         #plt.plot(lat, lon)
    #         plt.plot(tim, lat)
    #     plt.show()
    #
    #     # max_dim = 20000
    #     # xmax = max_dim
    #     # ymax = max_dim
    #     # xmin = 0
    #     # ymin = 0
    #     #
    #     # print('Fitting neighbors')
    #     # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=dist_time).fit(locs)
    #     # dists_neigh, indices = nbrs.kneighbors(locs)
    #     # print('Got nearest')
    #     #
    #     # distances = np.hstack([dists_end.reshape((-1, 1)), dists_neigh.reshape((-1, 1))])
    #     # hist_dist = distances[np.isfinite(distances).all(axis=1)]
    #     #
    #     # print('saving')
    #     # pickle.dump(hist_dist, open(fname + '_dists'+'.pkl', 'wb'))
    #
    #     # H, xedges, yedges = np.histogram2d(hist_dist[:, 0], hist_dist[:, 1], bins = 25, range=[[xmin, xmax], [ymin, ymax]])
    #     #
    #     # fig = plt.figure(figsize=(7, 3))
    #     # fig.tight_layout()
    #     # ax = fig.add_subplot(132)
    #     # X, Y = np.meshgrid(xedges, yedges)
    #     # ax.pcolormesh(X, Y, H)
    #     #
    #     # plt.xlabel('Distance to Goal')
    #     # plt.ylabel('Distance to Neighbors')
    #     #
    #     #
    #     # plt.show()
    #
    #
    #         #plt.plot(lat, lon)
    #         #plt.plot(tim,alt)
    #     #print(min_dist)
    #     #plt.show()
    #
    #     # plt.hist(min_bar, bins=100)
    #     # plt.show()
    #
    # if __name__== "__main__":
    #   main()