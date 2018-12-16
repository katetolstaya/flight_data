import numpy as np
import pickle, random, math, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from train import interp_expert, load_flight_data, interp_path, interp_expert, reconstruct_path
from planning.dubins_node3 import DubinsNode, reconstruct_path, plot_path
from planning.objective import DubinsObjective
from planning.grid import Grid
from planning.arastar import ARAStar
from planning.astar import AStar


def main():
    flight_summaries = load_flight_data()
    obj = pickle.load(open('model/objective_exp2.pkl', 'rb'))
    random.seed(0)
    random.shuffle(flight_summaries)

    grid = obj.grid.grid

    # pick best orientation in each location (x,y,z)
    # plot only those location that are > 0 (good)
    # convert orientation to quiver direction, u, v, w

    orientation = np.argmin(grid, axis=3)  # min cost is best

    # print(orientation.nonzero())

    good = np.min(grid, axis=3) < -30

    x, y, z = good.nonzero()

    orientation = orientation[x, y, z]
    # print(orientation)

    # print(orientation)
    half_margin = 2

    x = (x - half_margin) * obj.grid.resolution[0] + obj.grid.min_val[0]
    y = (y - half_margin) * obj.grid.resolution[1] + obj.grid.min_val[1]
    z = (z - half_margin) * obj.grid.resolution[2] + obj.grid.min_val[2]
    orientation = (orientation - half_margin) * obj.grid.resolution[3] + obj.grid.min_val[3]

    # print()
    # print(orientation)

    u = np.cos(orientation)
    v = np.sin(orientation)
    w = np.zeros(np.shape(orientation))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(x, y, z, u, v, w, color='g', length=1.0)
    plt.show()

    # print(min(grid[abs(grid)>0]))


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # n = 100

    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    #     ax.scatter(xs, ys, zs, c=c, marker=m)

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # plt.show()


if __name__ == "__main__":
    main()
