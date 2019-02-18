import random
from process import get_min_max_all
import configparser
from planning.dubins_problem import DubinsProblem
from train import load_flight_data, make_planner
from planning.dubins_multi_objective import DubinsMultiAirplaneObjective
from planning.grid import Grid
from plot_utils import add_path_to_plot
import matplotlib.pyplot as plt
from plot_utils import plot_planner_expert



def get_multi_airplane_segments(flight_summaries):
    overlap_length = 800
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
    s = 0.1
    for flight in flights:
        paths.append(problem.resample_path_dt(flight.to_path(), s, problem.dt))
    return paths


def main():
    # read in parameters
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']

    to = float(config['timeout'])
    n_iters = int(config['num_iterations'])
    n_samples = int(config['num_samples'])
    seed = int(config['random_seed'])

    if seed >= 0:
        random.seed(seed)

    print('Processing trajectories...')
    flight_summaries = load_flight_data()
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    lists = get_multi_airplane_segments(flight_summaries)
    random.shuffle(lists)

    print('Initializing the grid...')
    # initialize grid with one pass through the data
    grid = Grid(config, xyzbea_min, xyzbea_max)
    for flight in flight_summaries:
        if flight.get_path_len() < 5:  # can't interpolate paths with len < 4
            continue
        path = flight.to_path()
        start, goal = flight.get_start_goal()
        # use a dense interpolation with 200 pts
        dense_path = DubinsProblem.resample_path(path, n_samples)
        # grid.update(dense_path, -1000.0 * n_iters)
        # grid.gradient_step(dense_path, -1000.0 * n_iters)
        grid.gradient_step(dense_path, -1000.0)
    grid.save_grid()

    # set up planner and objectives
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)
    obj = DubinsMultiAirplaneObjective(config, grid)
    obj_expert = DubinsMultiAirplaneObjective(config, grid)
    planner = make_planner(config['planner_type'])

    n_updates = 0

    # start training
    for l in lists:

        print('Planning for ' + str(len(l)) + ' airplanes...')
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()
        obj_expert.clear_obstacles()

        for expert_path in paths:

            # plan trajectory
            node = planner(problem, expert_path[0, :], expert_path[-1, :], obj).plan(to)

            if node is not None:

                # get path in space from A* result
                planner_path_ind = problem.reconstruct_path_ind(node)
                planner_path = problem.ind_to_path(planner_path_ind)
                expert_path_ind = problem.path_to_ind(expert_path)

                expert_dense_path = DubinsProblem.resample_path_dt(expert_path, s=0.1, dt=1.0)
                planner_dense_path = DubinsProblem.resample_path_dt(planner_path, s=0.1, dt=1.0)

                # expert_dense_path = DubinsProblem.resample_path(expert_path, n_samples)
                # planner_dense_path = DubinsProblem.resample_path(planner_path, n_samples)


                # compute cost
                expert_cost = obj_expert.integrate_path_cost(expert_path, expert_path_ind)
                planner_cost = obj.integrate_path_cost(planner_path, planner_path_ind)
                print(planner_cost - expert_cost)

                path_diff = problem.compute_avg_path_diff(expert_dense_path, planner_dense_path)
                print(path_diff)

                ################################

                # gradient step
                # grid.update(expert_dense_path, -100.0)
                # grid.update(planner_dense_path, 100.0)

                grid.gradient_step(expert_dense_path, -1.0)
                grid.gradient_step(planner_dense_path, 1.0)

                # TODO
                # obj.update_obstacle_lims(expert_path_ind, planner_path_ind)
                # obj_expert.obstacle_lims = obj.obstacle_lims

                ###############################

                # add this plane's trajectory to obstacles for the next plane
                obj.add_obstacle(planner_path_ind)
                obj_expert.add_obstacle(expert_path_ind)
                n_updates = n_updates + 1


            else:
                print('Timeout')
                break


            if n_updates > 0 and n_updates % 10 == 0:
                obj.grid.save_grid()




if __name__ == "__main__":
    main()
