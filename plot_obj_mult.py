import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import configparser
import random
from planning.grid import Grid
from planning.dubins_problem import DubinsProblem
from train_mult import get_multi_airplane_segments, time_sync_flight_data
from planning.dubins_multi_objective import DubinsMultiAirplaneObjective
from data_utils import load_flight_data, make_planner, load_lims

def main():
    config_file = 'params.cfg'
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
    flight_summaries = load_flight_data()
    random.shuffle(flight_summaries)

    print('Loading cost...')
    folder = "model/"
    fname = "grid19"
    xyzbea_min, xyzbea_max = load_lims(folder, fname)
    grid = Grid(config, xyzbea_min, xyzbea_max, fname=fname)
    obj = DubinsMultiAirplaneObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    max_x = 400
    max_y = 400

    offset_x = -200
    offset_y = -700

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

    obj = DubinsMultiAirplaneObjective(config, grid)
    #obj_expert = DubinsMultiAirplaneObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Syncing trajectories...')

    lists = get_multi_airplane_segments(flight_summaries)

    lists.pop(0)
    lists.pop(0)


    for l in lists:

        learner_trajs = []

        print('Planning for ' + str(len(l)) + ' airplanes...')
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()

        #for expert_path in l:
        for expert_path in paths:
            #expert_path = expert_path.to_path()

            if len(learner_trajs) >= 4:
                break

            print('Planning #' + str(len(learner_trajs)+1))

            expert_path_ind = problem.path_to_ind(expert_path)

            print(expert_path_ind)
            planner_path_grid = np.zeros((expert_path_ind.shape[0], 5))
            for t in range(expert_path_ind.shape[0]):
                ind = np.append(np.asarray(grid.ind_to_index(expert_path_ind[t, :])), expert_path_ind[t, 4])
                planner_path_grid[t, :] = ind
            planner_path_grid = planner_path_grid + offset
            learner_trajs.append(planner_path_grid)
            #
            # node = planner(problem, expert_path[0, :], expert_path[-1, :], obj).plan(to)
            #
            # if node is not None:
            #
            #     planner_path = problem.reconstruct_path(node)
            #     planner_path_ind = problem.reconstruct_path_ind(node)
            #     obj.add_obstacle(planner_path_ind)
            #
            #     planner_path_grid = np.zeros((planner_path_ind.shape[0], 5))
            #     for t in range(planner_path_ind.shape[0]):
            #         ind = np.append(np.asarray(grid.ind_to_index(planner_path_ind[t, :])), planner_path[t, 4])
            #         planner_path_grid[t, :] = ind
            #     planner_path_grid = planner_path_grid + offset
            #     learner_trajs.append(planner_path_grid)
            #
            # else:
            #     print('Timeout')
            #     break

        n_learners = len(learner_trajs)
        if n_learners > 1:

            start_time = learner_trajs[0][0,4]
            dt = learner_trajs[0][1,4] - learner_trajs[0][0,4]
            end_time = learner_trajs[-1][-1,4]

            lines = []
            fig, ax = plt.subplots()
            ax.imshow(-1.0 * cost_min, extent=[0, max_x, 0, max_y], cmap='Greens', interpolation='spline16', alpha=0.5, origin='lower')

            for i in range(len(learner_trajs)):
                line, = ax.plot([0,1], [0,1], linewidth=3, color=colors[i])
                lines.append(line)

            lines.reverse()

            inds = np.zeros((n_learners,),dtype=np.int)

            for t in np.arange(start_time, end_time, dt):
                for i in reversed(range(n_learners)):
                    if learner_trajs[i].shape[0] > inds[i] and t <= learner_trajs[i][inds[i], 4]:
                        lines[i].set_xdata(learner_trajs[i][0:inds[i], 0])
                        lines[i].set_ydata(learner_trajs[i][0:inds[i], 1])
                        inds[i] = inds[i] + 1
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(1.0)


if __name__ == "__main__":
    main()
