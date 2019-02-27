import pickle
from process import get_flights
from planning.arastar import ARAStar
from planning.astar import AStar


def log(s, f=None):
    print(s)
    if f is not None:
        f.write(s)
        f.write('\n')
        f.flush()


def log_fname(s, fname, append=True):
    print(s)

    if append:
        f = open(fname, "a")
    else:
        f = open(fname, "wb")

    f.write(s)
    f.write('\n')
    f.flush()
    f.close()


def load_flight_data(config, fnames=None):
    if fnames is None:
        fnames = ['flights20160111', 'flights20160112',
                  'flights20160113']  # , 'flights0501', 'flights0502', 'flights0503']

    flight_summaries = []
    for fname in fnames:
        flights = pickle.load(open('data/' + fname + '.pkl', 'rb'))
        summaries = get_flights(flights, config)
        # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
        flight_summaries.extend(summaries)
    return flight_summaries


def make_planner(planner_type):
    if planner_type == 'AStar':
        planner = AStar
    elif planner_type == 'ARAStar':
        planner = ARAStar
    else:
        raise NotImplementedError
    return planner


def save_lims(xyzbea_min, xyzbea_max, folder, fname):
    pickle.dump(xyzbea_max, open(folder + 'max_' + fname + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(xyzbea_min, open(folder + 'min_' + fname + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_lims(folder, fname):
    xyzbea_min = pickle.load(open(folder + 'min_' + fname + ".pkl", "rb"))
    xyzbea_max = pickle.load(open(folder + 'max_' + fname + ".pkl", "rb"))
    return xyzbea_min, xyzbea_max


def get_multi_airplane_segments(flight_summaries):
    overlap_length = 200
    lists = []
    for s in flight_summaries:
        added = False
        for l in lists:
            for s2 in l:
                if s.overlap(s2) > overlap_length:
                    added = True
                    l.append(s)
                    break
            if added:
                break

        if not added:
            lists.append([s])

    # remove non-overlapping trajectories
    # and sort each list of trajectories in order of airplane's start time
    lists = [sorted(l, key=lambda x: x.time[0]) for l in lists if len(l) >= 2]
    return lists


def time_sync_flight_data(flights, problem):
    paths = []
    s = 0.0
    for flight in flights:
        paths.append(problem.resample_path_dt(flight.to_path(), s, problem.dt))
    return paths
