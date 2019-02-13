import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def add_path_to_plot(path, fig, ax):
    if fig == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2]
    orientation = path[:, 3]
    u = np.cos(orientation)
    v = np.sin(orientation)
    w = np.zeros(np.shape(orientation))
    ax.quiver(x, y, z, u, v, w, color='r', length=1.0)

    plt.draw()
    plt.pause(0.001)
    return fig, ax

def plot_planner_expert(planner_path, expert_path, planner_spline, expert_spline):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['g', 'r']

    for i, arr in enumerate([expert_path, planner_path]):
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        orientation = arr[:, 3]
        u = np.cos(orientation)
        v = np.sin(orientation)
        w = np.zeros(np.shape(orientation))
        ax.quiver(x, y, z, u, v, w, color=colors[i], length=1.0)

    x = planner_spline[:, 0].flatten()
    y = planner_spline[:, 1].flatten()
    z = planner_spline[:, 2].flatten()
    ax.plot(xs=x, ys=y, zs=z)

    x = expert_spline[:, 0].flatten()
    y = expert_spline[:, 1].flatten()
    z = expert_spline[:, 2].flatten()
    ax.plot(xs=x, ys=y, zs=z)

    plt.show()

def plot_obj(grid):

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

    x = (x - half_margin) * grid.resolution[0] + grid.min_val[0]
    y = (y - half_margin) * grid.resolution[1] + grid.min_val[1]
    z = (z - half_margin) * grid.resolution[2] + grid.min_val[2]
    orientation = (orientation - half_margin) * grid.resolution[3] + grid.min_val[3]

    # print()
    # print(orientation)

    u = np.cos(orientation)
    v = np.sin(orientation)
    w = np.zeros(np.shape(orientation))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(x, y, z, u, v, w, color='g', length=1.0)
    plt.show()