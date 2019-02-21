import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import configparser
import random
from process import get_min_max_all
from train import load_flight_data, make_planner
from planning.grid import Grid
from planning.dubins_objective import DubinsObjective

from planning.dubins_problem import DubinsProblem
from train_mult import get_multi_airplane_segments, time_sync_flight_data
from planning.dubins_multi_objective import DubinsMultiAirplaneObjective

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

    # set up cost grid
    xyzbea_min, xyzbea_max = get_min_max_all(flight_summaries)
    print('Loading cost...')
    # set up cost grid
    # n_iters = int(config['num_iterations'])
    # n_samples = int(config['num_samples'])
    grid = Grid(config, xyzbea_min, xyzbea_max)
    grid.load_grid(fname="model/grid19.pkl")

    max_x = 300
    max_y = 300

    offset_x = -250 #200
    offset_y = -750
    offset = np.array([offset_x, offset_y, 0, 0, 0]).reshape((1, 5))
    cost_min = np.ones((max_x, max_y)) * 100
    count = np.ones((max_x, max_y))
    cost_sum = np.ones((max_x, max_y)) * 100

    for k in grid.grid.keys():
        x = k[0] + offset_x
        y = k[1] + offset_y

        if x > 0 and y > 0 and x < max_x and y < max_y:
            cost_sum[x, y] = cost_sum[x, y] + grid.grid[k]
            cost_min[x, y] = min(cost_min[x, y], grid.grid[k])

            count[x,y] = count[x,y] + 1

    avg_cost = cost_sum / count
    cost_min = cost_min.T


    colors = ['forestgreen', 'brickred', 'purple', 'darkblue', 'darkorange']
    plt.ion()
    obj = DubinsMultiAirplaneObjective(config, grid)
    #obj_expert = DubinsMultiAirplaneObjective(config, grid)
    problem = DubinsProblem(config, xyzbea_min, xyzbea_max)

    print('Syncing trajectories...')

    lists = get_multi_airplane_segments(flight_summaries)

    lists.pop(0)
    for l in lists:

        learner_trajs = []

        print('Planning for ' + str(len(l)) + ' airplanes...')
        paths = time_sync_flight_data(l, problem)
        obj.clear_obstacles()

        for expert_path in paths:

            if len(learner_trajs) >= 5:
                break

            print('Planning #' + str(len(learner_trajs)))

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
            print(dt)

            lines = []
            fig, ax = plt.subplots()
            ax.imshow(cost_min, extent=[0, max_x, 0, max_y], interpolation='spline16', alpha=0.5, origin='lower')

            for i in range(len(learner_trajs)):
                line, = ax.plot([0,1], [0,1], linewidth=5, color=colors[i])
                lines.append(line)

            inds = np.zeros((n_learners,),dtype=np.int)

            for t in np.arange(start_time, end_time, dt):
                for i in range(n_learners):
                    if learner_trajs[i].shape[0] > inds[i] and t <= learner_trajs[i][inds[i], 4]:
                        lines[i].set_xdata(learner_trajs[i][0:inds[i], 0])
                        lines[i].set_ydata(learner_trajs[i][0:inds[i], 1])
                        inds[i] = inds[i] + 1
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.5)


if __name__ == "__main__":
    main()
