import datetime
import numpy as np
from geo import *


class FlightSummary(object):

    def __init__(self, flight, in_range, config):

        lat0 = float(config['lat0'])
        lon0 = float(config['lon0'])
        alt0 = float(config['alt0'])
        scale = float(config['scale'])

        self.ref = flight.ref
        self.time = np.array([timestamp(t) for t in flight.time])[in_range]
        self.T = len(self.time)

        inds = np.argsort(self.time)
        self.time = self.time[inds]

        lat = np.array(flight.latitude)[in_range][inds]
        lon = np.array(flight.longitude)[in_range][inds]
        alt = np.array(flight.altitude)[in_range][inds]
        self.loc_xyzbea = np.zeros((self.T, 4))

        for k in range(0, self.T):
            # bea = bearing(lat[k], lon[k], next_lat[k], next_lon[k])

            if k == self.T - 1:  # or k == self.T - 2:  # last bearing would be wrong otherwise
                bea = self.loc_xyzbea[k - 1, 3]
            else:
                # bea = bearing(lat[k], lon[k], next_lat[k], next_lon[k])
                bea = bearing(lat[k], lon[k], lat[k + 1], lon[k + 1])

            loc_lla = np.array([flight.ref, lat[k], lon[k], alt[k], self.time[k], bea])
            self.loc_xyzbea[k, :] = np.array(get_xyzbea(loc_lla, lat0, lon0, alt0, scale))

    def to_path(self):
        path = np.concatenate((self.loc_xyzbea, self.time.reshape((-1, 1))), axis=1)
        return path

    def get_start_goal(self):
        xyzb = self.loc_xyzbea
        start = np.array([xyzb[0, 0], xyzb[0, 1], xyzb[0, 2], xyzb[0, 3], self.time[0]]).flatten()
        goal = np.array([xyzb[-1, 0], xyzb[-1, 1], xyzb[-1, 2], xyzb[-1, 3], self.time[-1]]).flatten()
        return start, goal

    def get_num_waypoints(self):
        return self.loc_xyzbea.shape[0]

    def overlap(self, other):
        return max(0, min(self.time[-1], other.time[-1]) - max(self.time[0], other.time[0]))


def timestamp(tztime):
    return (tztime - datetime.datetime(1970, 1, 1)).total_seconds()


def get_xyzbea(loc, lat0, lon0, alt0, SCALE):
    lat = loc[1]
    lon = loc[2]
    alt = loc[3]
    return np.append(np.array(geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)) / SCALE, loc[5])


def min_dist_to_airport(start_t, end_t, flight, lat0, lon0, alt0):
    len_path = np.array([timestamp(t) < end_t and timestamp(t) > start_t for t in flight.time])

    lat = np.array(flight.latitude)[len_path]
    lon = np.array(flight.longitude)[len_path]
    alt = np.array(flight.altitude)[len_path]

    min_dist = np.Inf

    for j in range(0, len(lat)):
        dist = dist_euclid(np.array([0, lat[j], lon[j], alt[j]]), np.array([0, lat0, lon0, alt0]))
        if dist < min_dist:
            min_dist = dist

    return min_dist


def bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    brng = math.atan2(y, x)  # + math.pi/2.0 #+ math.pi/2 # bearing in radians! #TODO
    return brng


def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) * math.sin(dphi / 2) + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def dist_euclid(x, y):
    lat1 = x[1]
    lat2 = y[1]
    lon1 = x[2]
    lon2 = y[2]
    alt1 = x[3]
    alt2 = y[3]
    dist = math.sqrt(haversine(lat1, lon1, lat2, lon2) ** 2 + (alt1 - alt2) ** 2)
    return dist


def get_min_time(flights):
    min_time = np.Inf
    for i, k in enumerate(sorted(flights.keys())):
        min_time = np.minimum(np.min(np.array([timestamp(t) for t in flights[k].time])), min_time)
    return min_time


def get_flights(flights, config):
    td = int(config['time_delta'])
    min_time = get_min_time(flights)
    center_t = int(config['center_t'])
    range_t = int(config['range_t'])

    start_t = min_time + center_t - range_t
    end_t = min_time + center_t + range_t

    airport = config['airport']
    lat0 = float(config['lat0'])
    lon0 = float(config['lon0'])
    alt0 = float(config['alt0'])
    alt_lim = float(config['alt_lim'])
    dist_lim = float(config['dist_lim'])

    flight_summaries = []

    for i, k in enumerate(sorted(flights.keys())):
        flight = flights[k]

        if np.min(np.abs(flight.latitude)) == 0.0 or (flight.arrival != airport):  # and flight.arrival != airport):
            continue

        # timestamp in range between start_t and end_t
        in_range = np.array([timestamp(t) < end_t and timestamp(t) > start_t for t in flight.time])

        # trim the beginning or end of trajectory to discard actual landing
        time_delta = datetime.timedelta(0, td)
        if flight.departure == airport:
            in_range = np.logical_and(in_range, np.array([t < flight.time[0] + time_delta for t in flight.time]))
        if flight.arrival == airport:
            in_range = np.logical_and(in_range, np.array([t > flight.time[-1] - time_delta for t in flight.time]))

        # altitude within range
        in_range = np.logical_and(in_range, np.array([af < alt_lim for af in flight.altitude]))

        # fewer than 3 points in path
        if np.sum(in_range) < 3:
            continue

        # too far from the aiport
        if min_dist_to_airport(start_t, end_t, flight, lat0, lon0, alt0) > dist_lim:
            continue

        flight_summary = FlightSummary(flight, in_range, config)
        flight_summaries.append(flight_summary)

    return flight_summaries


def get_min_max_all(flight_summaries):
    xyzbea_min = np.Inf * np.ones((1, 4))
    xyzbea_max = -1.0 * np.Inf * np.ones((1, 4))

    min_time = np.Inf
    max_time = 0
    # get range of xyzbea and time
    for flight in flight_summaries:
        xyzbea_max = np.maximum(xyzbea_max, np.amax(flight.loc_xyzbea, axis=0))
        xyzbea_min = np.minimum(xyzbea_min, np.amin(flight.loc_xyzbea, axis=0))

        min_time = np.minimum(min_time, np.min(flight.time))
        max_time = np.minimum(max_time, np.max(flight.time))

    # set angle range
    xyzbea_max[0, 3] = 1.0 * math.pi
    xyzbea_min[0, 3] = -1.0 * math.pi

    # set time range
    xyzbea_min = np.append(xyzbea_min, [min_time])
    xyzbea_max = np.append(xyzbea_max, [max_time])

    return xyzbea_min, xyzbea_max
