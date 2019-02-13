import random
import numpy as np
from process import get_min_max_all
import configparser
from planning.dubins_problem import DubinsProblem
from train import load_flight_data, make_planner
from planning.dubins_multi_objective import DubinsMultiAirplaneObjective
from planning.dubins_objective import DubinsObjective
from planning.grid import Grid
from plot_utils import add_path_to_plot
import matplotlib.pyplot as plt
from plot_utils import plot_planner_expert

def time_sync_flight_data(flights, problem):
    paths = []
    s = 0
    for flight in flights:
        #start, goal = flight.get_start_goal()
        #new_path = {"start": start, "goal": goal, "path": problem.resample_path_dt(flight.to_path(), s, problem.dt)}
        paths.append(problem.resample_path_dt(flight.to_path(), s, problem.dt))
    return paths


def main():
    # read in parameters
    config_file = 'params.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    planner = make_planner(config['planner_type'])

    to = float(config['timeout'])
    n_iters = int(config['num_iterations'])
    n_samples = int(config['num_samples'])
    seed = int(config['random_seed'])
    if seed >= 0:
        random.seed(seed)

    # get plane data
    flight_summaries = load_flight_data()

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)

    print('Processing trajectories...')
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

    # remove non-overlapping trajectories, and sort each list in order of start time
    lists = [sorted(l, key=lambda x: x.time[0]) for l in lists if len(l) >= 2]

    # set up cost grid
    # grid = Grid(config, xyzbea_min, xyzbea_max)
    #grid.load_grid()

    print('Initializing the grid...')
    # initialize cost with one pass through the data
    # set up cost grid
    grid = Grid(config, xyzbea_min, xyzbea_max)
    # initialize cost with one pass through the data

    #print(grid.grid.shape)

    for flight in flight_summaries:
        # can't interpolate paths with len < 4
        if flight.get_path_len() < 5:
            continue

        path = flight.to_path()
        start, goal = flight.get_start_goal()
        dense_path = DubinsProblem.resample_path(path, start, goal, n_samples)
        grid.update(dense_path, -1000.0 * n_iters)

    obj = DubinsMultiAirplaneObjective(config, grid)
    obj_expert = DubinsMultiAirplaneObjective(config, grid)

    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    obj.grid.save_grid()

    fig = None
    ax = None

    print('Planning...')

    random.shuffle(lists)

    lists.pop(0)
    lists.pop(0)

    for l in lists:

        print(len(l))
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()
        obj_expert.clear_obstacles()

        if fig is not None:
            plt.close(fig)
            fig = None
            ax = None


        expert_distances = []
        planner_distances = []

        for expert_path in paths:

            start = expert_path[0, :]
            goal = expert_path[-1, :]

            #expert_path = path_dict["path"]
            # start = #path_dict["start"]
            # goal = #path_dict["goal"]
            #print(start)
            # print(goal)
            node = planner(problem, start, goal, obj).plan(to)

            if node is not None:

                planner_path_ind = problem.reconstruct_path_ind(node)
                planner_path = problem.ind_to_path(planner_path_ind)


                expert_path_ind = problem.path_to_ind(expert_path)

                planner_distances.extend(obj.get_obstacle_distances(planner_path_ind))
                expert_distances.extend(obj_expert.get_obstacle_distances(expert_path_ind))

                for d in expert_distances:
                    print(d)
                print("")
                for d in planner_distances:
                    print(d)
                # TODO use these for gradient update on objective

                obj.grid.update(expert_path, -100.0)
                obj.grid.update(planner_path, 100.0)



                # fig, ax = add_path_to_plot(planner_path, fig, ax)
                # expert_dense_path = DubinsProblem.resample_path(expert_path, start, goal)
                # planner_dense_path = DubinsProblem.resample_path(planner_path, start, goal)
                # print(obj.integrate_path_cost(expert_dense_path) - obj.integrate_path_cost(planner_dense_path))
                # plot_planner_expert(planner_path, expert_path, planner_dense_path, expert_dense_path)

                obj.add_obstacle(planner_path_ind)
                obj_expert.add_obstacle(expert_path_ind)

            else:
                print('Timeout')
                break

        obj.grid.save_grid()

if __name__ == "__main__":
    main()
