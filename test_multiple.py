import numpy as np
import matplotlib.pyplot as plt
import time
import configparser
import random
import sys
from planning.grid import Grid
from planning.dubins_problem import DubinsProblem
from planning.dubins_objective import DubinsObjective
from utils.data_utils import load_flight_data, make_planner, load_lims, get_multi_airplane_segments, \
    time_sync_flight_data
from matplotlib import rc

rc('text', usetex=True)
font = {'family': 'serif', 'weight': 'bold', 'size': 14}
rc('font', **font)


def main():
    plot_expert = False

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

    if seed >= 0:
        random.seed(seed)

    # get plane data
    fnames = ['flights20160112']  # 'flights20160112', 'flights20160113']
    flight_summaries = load_flight_data(config, fnames)

    print('Loading cost...')
    folder = "models/"
    fname = "grid19"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)

    max_x = 250
    max_y = 250

    offset_x = -250
    offset_y = -750

    radius = float(config['obstacle_init_xy']) * grid.lookup_res[0]

    offset = np.array([offset_x, offset_y, 0, 0, 0]).reshape((1, 5))
    cost_min = np.ones((max_x, max_y)) * 100

    for k in grid.grid.keys():
        x = k[0] + offset_x
        y = k[1] + offset_y

        if 0 < x < max_x and 0 < y < max_y:
            cost_min[x, y] = min(cost_min[x, y], grid.grid[k])

    cost_min = cost_min.T

    colors = ['orangered', 'dodgerblue', 'gold', 'orchid', 'cyan']
    plt.ion()

    obj = DubinsObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Syncing trajectories...')

    lists = get_multi_airplane_segments(flight_summaries)
    lists = [lists[12], lists[16], lists[27], lists[32], lists[33]]

    for l in lists:

        print('Planning for ' + str(len(l)) + ' airplanes...')
        learner_trajs = []
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()

        for expert_path in paths:

            if len(learner_trajs) >= 5:
                break

            print('Planning #' + str(len(learner_trajs) + 1))

            if plot_expert:

                expert_path_ind = problem.path_to_ind(expert_path)
                print(obj.integrate_path_cost(expert_path, expert_path_ind))
                obj.add_obstacle(expert_path_ind)
                planner_path_grid = np.zeros((expert_path_ind.shape[0], 5))
                for t in range(expert_path_ind.shape[0]):
                    ind = np.append(np.asarray(grid.ind_to_index(expert_path_ind[t, :])), expert_path[t, 4])
                    planner_path_grid[t, :] = ind
                planner_path_grid = planner_path_grid + offset
                learner_trajs.append(planner_path_grid)

            else:
                node = planner(problem, expert_path[0, :], expert_path[-1, :], obj).plan(to)

                if node is not None:
                    planner_path = problem.reconstruct_path(node)
                    planner_path_ind = problem.reconstruct_path_ind(node)

                    print("Cost margin: " + str(obj.integrate_path_cost(planner_path, planner_path_ind)))
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

        # plot results
        n_learners = len(learner_trajs)
        if n_learners > 1:

            start_time = learner_trajs[0][0, 4]
            dt = learner_trajs[0][1, 4] - learner_trajs[0][0, 4]
            end_time = learner_trajs[-1][-1, 4]

            lines = []
            markers = []
            circles = []
            fig, ax = plt.subplots(facecolor='white')

            ax.set_xlim([0, max_x])
            ax.set_ylim([0, max_y])

            ax.imshow(-1.0 * cost_min, extent=[0, max_x, 0, max_y], cmap='Greens', interpolation='spline16',
                      origin='lower', alpha=0.5)

            time_text = plt.text(20.0, 20.0, '', fontsize=18)

            plt.axis('off')

            for i in range(len(learner_trajs)):
                line, = ax.plot([-100, -99], [-100, -99], linewidth=4, color=colors[i])
                lines.append(line)

            for i in range(len(learner_trajs)):
                marker, = ax.plot([-100, -99], [-100, -99], linewidth=0, marker='o', markersize=10, color=colors[i])
                markers.append(marker)
                circle = plt.Circle((-100, -100), radius, edgecolor=colors[i], fill=False, linewidth=2)
                circles.append(circle)
                ax.add_patch(circle)

            inds = np.zeros((n_learners,), dtype=np.int)

            for t in np.arange(start_time, end_time, dt):
                for i in reversed(range(n_learners)):

                    # show next time step
                    while learner_trajs[i].shape[0] > inds[i] and learner_trajs[i][inds[i], 4] < t:
                        lines[i].set_xdata(learner_trajs[i][0:inds[i] + 1, 0])
                        lines[i].set_ydata(learner_trajs[i][0:inds[i] + 1, 1])
                        markers[i].set_xdata(learner_trajs[i][inds[i], 0])
                        markers[i].set_ydata(learner_trajs[i][inds[i], 1])
                        circles[i].center = learner_trajs[i][inds[i], 0], learner_trajs[i][inds[i], 1]
                        inds[i] = inds[i] + 1

                    # traj done
                    if learner_trajs[i].shape[0] <= inds[i] or learner_trajs[i][-1, 4] < t:
                        markers[i].set_marker("None")
                        circles[i].set_radius(0.0)

                time_text.set_text("{0:d} s".format(int(t - start_time)))

                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(1.0)


if __name__ == "__main__":
    main()
