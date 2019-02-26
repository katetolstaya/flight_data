import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import configparser
import random
import sys
from planning.grid import Grid
from planning.dubins_problem import DubinsProblem
from planning.dubins_objective import DubinsObjective
from data_utils import load_flight_data, make_planner, load_lims, get_multi_airplane_segments, time_sync_flight_data


def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'cfg/params.cfg'

    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['plan1']
    to = float(config['timeout'])
    seed = int(config['random_seed'])
    planner = make_planner(config['planner_type'])
    n_samples = int(config['num_samples'])

    if seed >= 0:
        random.seed(seed)

    # get plane data
    fnames = ['flights20160112'] #, 'flights20160112', 'flights20160113']
    flight_summaries = load_flight_data(fnames)

    print('Loading cost...')
    folder = "model/"
    fname = "grid19"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)
    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    max_x = 300
    max_y = 300

    offset_x = -250
    offset_y = -750

    offset = np.array([offset_x, offset_y, 0, 0, 0]).reshape((1, 5))
    cost_min = np.ones((max_x, max_y)) * 100

    for k in grid.grid.keys():
        x = k[0] + offset_x
        y = k[1] + offset_y

        if x > 0 and y > 0 and x < max_x and y < max_y:
            cost_min[x, y] = min(cost_min[x, y], grid.grid[k])

    cost_min = cost_min.T


    colors = ['orangered', 'gold', 'dodgerblue', 'orchid']
    #colors = ['#78C74D', "#BDAE33", "#BC5A33", "#A0373F"]
    #colors = ['forestgreen', 'firebrick', 'purple', 'darkblue', 'darkorange']
    plt.ion()

    obj = DubinsObjective(config, grid)
    #obj_expert = DubinsMultiAirplaneObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Syncing trajectories...')

    lists = get_multi_airplane_segments(flight_summaries)

    lists.pop(0)
    lists.pop(0)
    lists.pop(0)
    lists.pop(0)
    lists.pop(0)
    lists.pop(0)
    lists.pop(0)

    for l in lists:

        learner_trajs = []

        print('Planning for ' + str(len(l)) + ' airplanes...')
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()

        for expert_path in paths:

            if len(learner_trajs) >= 4:
                break

            print('Planning #' + str(len(learner_trajs)+1))

            # expert_path_ind = problem.path_to_ind(expert_path)
            # planner_path_grid = np.zeros((expert_path_ind.shape[0], 5))
            # for t in range(expert_path_ind.shape[0]):
            #     ind = np.append(np.asarray(grid.ind_to_index(expert_path_ind[t, :])), expert_path_ind[t, 4])
            #     planner_path_grid[t, :] = ind
            # planner_path_grid = planner_path_grid + offset
            # learner_trajs.append(planner_path_grid)

            node = planner(problem, expert_path[0, :], expert_path[-1, :], obj).plan(to)

            if node is not None:
                planner_path = problem.reconstruct_path(node)
                planner_path_ind = problem.reconstruct_path_ind(node)
                obj.add_obstacle(planner_path_ind)

                planner_path_grid = np.zeros((planner_path_ind.shape[0], 5))
                for t in range(planner_path_ind.shape[0]):
                    ind = np.append(np.asarray(grid.ind_to_index(planner_path_ind[t, :])), planner_path[t, 4])
                    planner_path_grid[t, :] = ind
                planner_path_grid = planner_path_grid + offset
                learner_trajs.append(planner_path_grid)

            else:
                print('Timeout')
                break

        n_learners = len(learner_trajs)
        if n_learners > 1:

            start_time = learner_trajs[0][0,4]
            dt = learner_trajs[0][1,4] - learner_trajs[0][0,4]
            end_time = learner_trajs[-1][-1,4]

            lines = []
            markers = []
            fig, ax = plt.subplots()
            ax.imshow(-1.0 * cost_min, extent=[0, max_x, 0, max_y], cmap='Greens', interpolation='spline16',  origin='lower', alpha=0.5)

            for i in range(len(learner_trajs)):
                line, = ax.plot([0,1], [0,1], linewidth=4, color=colors[i])
                lines.append(line)

            for i in range(len(learner_trajs)):
                marker, = ax.plot([0,1], [0,1], linewidth=0, marker='o', markersize=10, color=colors[i])
                markers.append(marker)

            time_text = plt.text(-2.0, 5, '', fontsize=18)

            lines.reverse()
            markers.reverse()

            inds = np.zeros((n_learners,),dtype=np.int)

            for t in np.arange(start_time, end_time, dt):
                print(t)
                #time_text.set_text(" {0:.2f} s".format(t-start_time))
                for i in reversed(range(n_learners)):
                    if learner_trajs[i].shape[0] > inds[i] and t <= learner_trajs[i][inds[i], 4]:
                        lines[i].set_xdata(learner_trajs[i][0:inds[i]+1, 0])
                        lines[i].set_ydata(learner_trajs[i][0:inds[i]+1, 1])
                        markers[i].set_xdata(learner_trajs[i][inds[i], 0])
                        markers[i].set_ydata(learner_trajs[i][inds[i], 1])
                        inds[i] = inds[i] + 1

                    if learner_trajs[i][-1, 4] < t:
                        markers[i].set_marker("None")
                        markers[i].set_marker("None")


                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(1.0)


if __name__ == "__main__":
    main()
