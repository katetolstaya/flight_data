import planner
import intervals as I
import numpy as np
from itertools import compress


class Segment:
    def __init__(self, refs, params, starts, ends, start_time, end_time=None, plan=None):
        self.refs = refs
        self.starts = starts
        self.ends = ends

        if self.starts is not None:
            self.N = np.shape(starts)[0]
        self.params = params
        self.velocity = params.velocity

        self.plan = plan
        self.start_time = start_time
        self.end_time = end_time
        if end_time is not None:
            self.interval = I.closedopen(start_time, end_time)

    def run_planner(self):
        self.plan = planner.Planner(self.starts, self.ends, self.params)  # run the planner

        if self.plan.path is not None:
            lengths = self.plan.get_lengths()
            max_length = np.max(lengths)
            min_length = np.min(lengths)

            end_time = max_length / self.velocity + self.start_time

            if self.end_time is not None:
                tol = 0.01
                if np.abs(self.end_time - end_time) > tol or np.abs(max_length - min_length) > tol:
                    raise NotImplementedError  # TODO readjust next intervals to account for this
            else:
                self.end_time = end_time
                self.interval = I.closedopen(self.start_time, self.end_time)

    def overlaps(self, other_interval):
        return self.interval.overlaps(other_interval.interval)

    def intersection(self, other_interval):
        return self.interval.intersection(other_interval.interval)

    def atomic_interval(self):
        return self.interval.to_atomic()

    def interpolate(self, curr_time):
        state = self.plan.interpolate_time(
            (curr_time - self.atomic_interval().lower) / (self.atomic_interval().upper - self.atomic_interval().lower))
        return np.reshape(state, (-1, 4))

    def plot(self, ax, colors=None):
        if colors is not None:
            self.plan.plot(ax, [colors[x] for x in self.refs])
        else:
            self.plan.plot(ax, None)

def make_segments(flights_arr, params):
    params.switch_params("learn")
    flights_arr = flights_arr[flights_arr[:, 1].argsort()]  # sort flights by start time

    N = np.shape(flights_arr)[0]
    segments = []

    candidate_segments = []

    for i in range(0, N):
        my_id = flights_arr[i, 0]
        takeoff_time = flights_arr[i, 1]
        starts = np.reshape(flights_arr[i, 2:6], (1, -1))
        ends = np.reshape(flights_arr[i, 6:], (1, -1))

        new_segment = Segment([my_id], params, starts, ends, takeoff_time)
        new_segment.run_planner()
        if new_segment.plan.path is None:
            return None

        # all non-intersecting candidate segments are now final
        if len(candidate_segments) > 0:
            intersect_bool = [new_segment.overlaps(segment) for segment in candidate_segments]
            segments.extend(list(compress(candidate_segments, [not x for x in intersect_bool])))
            candidate_segments = list(compress(candidate_segments, intersect_bool))

        # add in new segment and make new segments valid
        if len(candidate_segments) == 0:
            candidate_segments.append(new_segment)
        else:  # only intersecting candidates left

            # process intersections
            new_candidate_segments = []
            for segment in candidate_segments:
                interval = new_segment.intersection(segment).to_atomic()  # intersection of atomic intervals is atomic

                if not interval.is_empty():
                    curr_starts = np.vstack([segment.interpolate(interval.lower),
                                             new_segment.interpolate(interval.lower)])
                    curr_ends = np.vstack(
                        [segment.interpolate(interval.upper), new_segment.interpolate(interval.upper)])
                    new_candidate_segments.append(
                        Segment(segment.refs + [my_id], params, curr_starts, curr_ends, interval.lower, interval.upper))

            # process any remaining parts of new segment
            interval_left = new_segment.interval
            for segment in candidate_segments:
                interval_left = interval_left - segment.interval

            for interval in list(interval_left):
                curr_starts = new_segment.interpolate(interval.lower)
                curr_ends = new_segment.interpolate(interval.upper)
                new_candidate_segments.append(
                    Segment([my_id], params, curr_starts, curr_ends, interval.lower, interval.upper))

            # process non-intersecting parts of old segments
            for segment in candidate_segments:
                for interval in list(segment.interval - new_segment.interval):
                    curr_starts = segment.interpolate(interval.lower)
                    curr_ends = segment.interpolate(interval.upper)
                    new_candidate_segments.append(
                        Segment(segment.refs, params, curr_starts, curr_ends, interval.lower, interval.upper))

            # sort segments by intervals, should be disjoint
            new_candidate_segments.sort(key=lambda z: z.atomic_interval())

            candidate_segments = []
            for segment in new_candidate_segments:
                segment.run_planner()
                if segment.plan.path is None:
                    return None
                candidate_segments.append(segment)

    segments.extend(candidate_segments)
    return segments


def get_actual_costs(flight_summaries, params):
    params.switch_params("interp")
    times = []
    for flight in flight_summaries:
        times.extend(flight.time)

    times = list(set(times))

    times.sort()

    segments = np.zeros((len(times), len(flight_summaries), 4))
    intersects = np.zeros((len(times), len(flight_summaries)), dtype=bool)

    costs = np.zeros((4,))

    refs = dict()

    for k, flight in enumerate(flight_summaries):
        refs[k] = flight.ref
        for j in range(1, len(flight.time)):
            start_t = flight.time[j - 1]
            end_t = flight.time[j]
            start_loc = np.reshape(flight.loc_xyzbea[j - 1, :], (-1, 4))
            end_loc = np.reshape(flight.loc_xyzbea[j, :], (-1, 4))

            my_plan = planner.Planner(start_loc, end_loc, params)
            if my_plan.path is None:
                return None, None

            for i in range(0, len(times)):
                if times[i] >= start_t and times[i] <= end_t:
                    t = (times[i] - start_t) / (end_t - start_t)
                    state = my_plan.interpolate_time(t)
                    segments[i, k, :] = state
                    intersects[i, k] = True

    final_segments = []

    for i in range(1, len(times)):
        inter_temp = np.logical_and(intersects[i-1, :], intersects[i, :])

        if inter_temp.any():
            starts = np.reshape(segments[i-1, inter_temp, :], (-1, 4))
            ends = np.reshape(segments[i, inter_temp, :], (-1, 4))
            my_plan = planner.Planner(starts, ends, params)
            if my_plan.path is None:
                return None, None

            curr_refs = [refs[j] for j in np.where(inter_temp)[0]]
            seg = Segment(curr_refs, params, None, None, None, None, my_plan)
            final_segments.append(seg)

            costs[0] = costs[0] + my_plan.get_length_obj().value()
            costs[1] = costs[1] + my_plan.get_clear_obj().value()
            costs[2] = costs[2] + my_plan.get_work_obj().value()
            costs[3] = costs[3] + my_plan.get_balanced_obj().value()

    return costs, final_segments

def get_costs_from_segments(segments):
    costs = np.zeros((4,))
    for segment in segments:
        costs[0] = costs[0] + segment.plan.get_length_obj().value()
        costs[1] = costs[1] + segment.plan.get_clear_obj().value()
        costs[2] = costs[2] + segment.plan.get_work_obj().value()
        costs[3] = costs[3] + segment.plan.get_balanced_obj().value()
    return costs