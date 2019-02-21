import pickle
from parameters import Parameters
from process import get_flights
from planning.arastar import ARAStar
from planning.astar import AStar

def log(s, f=None):
    print(s)
    if f is not None:
        f.write(s)
        f.write('\n')
        f.flush()


def load_flight_data():
    params = Parameters()
    fnames = ['flights20160111', 'flights20160112', 'flights20160113']
        #, 'flights0501', 'flights0502', 'flights0503']
    flight_summaries = []
    for fname in fnames:
        flights = pickle.load(open('data/' + fname + '.pkl', 'rb'))
        _, summaries = get_flights(flights,
                                   params)  # read in flight data: id, start time, starting X,Y,Z,yaw, ending X,Y,Z,yaw
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