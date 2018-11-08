import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# OMPL
ompl_app_root = "/home/t-ektol/omplapp-1.3.2-Source/"
sys.path.insert(0, join(ompl_app_root, 'ompl/py-bindings'))  # import compiled ompl
from ompl import base as ob
from ompl import geometric as og
from ompl import app as oa
from ompl import util as ou

# OMPL meshes
ompl_resources_dir = join(ompl_app_root, 'resources/3D')
cube_path = join(ompl_resources_dir, "cube.obj")


class Planner:
    def __init__(self, starts, ends, params):

        # Set the OMPL log level to WARN
        # ou.setLogLevel(ou.LOG_WARN)

        self.N = np.shape(starts)[0]

        # initialize and setup Multi-Airplane planner
        self.setup = oa.MultiDubinsAirplanePlanning(self.N)
        self.si = self.setup.getSpaceInformation()
        self.state_space = self.setup.getStateSpace()

        # meshes, bounds, collision checking
        self.setup.setRobotMesh(cube_path)
        for i in range(1, self.N):
            self.setup.addRobotMesh(join(ompl_resources_dir, cube_path))

        # defining good bounds is extremely important!!
        bounds = ob.RealVectorBounds(3)
        min_bound = -2 * np.abs(np.minimum(np.amin(starts), np.amin(ends)))
        max_bound = 2 * np.abs(np.maximum(np.amax(starts), np.amax(ends)))
        bounds.setLow(min_bound)
        bounds.setHigh(max_bound)
        bounds.low[2] = 0  # ground plane

        rr = float(params.rise_rate) / params.turning_radius
        for i in range(0, self.N):
            self.state_space.getSubspace(i).setBounds(bounds)
            self.state_space.getSubspace(i).setTurningRadius(params.turning_radius)
            self.state_space.getSubspace(i).setRiseRate(rr)
        self.si.setStateValidityCheckingResolution(0.005)

        # set planner and objectives
        self.setup.setPlanner(og.BITstar(self.si))
        self.length_obj = oa.MultiDubinsAirplanePlanning.getPathLengthObjective(self.si)
        self.clear_obj = oa.MultiDubinsAirplanePlanning.getClearanceObjective(self.si)
        self.work_obj = oa.MultiDubinsAirplanePlanning.getAirplaneWorkObjective(self.si, self.N)


        # doesn't work! Sets weights to 1.0
        # self.balanced_obj = oa.MultiDubinsAirplanePlanning.getBalancedObjective(self.si, self.length_obj,
        #                                                                         self.clear_obj, self.work_obj,
        #                                                                         params.w1, params.w2, params.w3)

        self.balanced_obj = ob.MultiOptimizationObjective(self.si)
        self.balanced_obj.addObjective(self.length_obj, params.w1)
        self.balanced_obj.addObjective(self.clear_obj, params.w2)
        self.balanced_obj.addObjective(self.work_obj, params.w3)

        self.setup.setOptimizationObjective(self.balanced_obj)

        # set start and goal states
        start = ob.State(self.setup.getSpaceInformation())
        goal = ob.State(self.setup.getSpaceInformation())
        for i in range(0, self.N):
            starti = start()[i]
            goali = goal()[i]

            starti.setX(starts[i][0])
            starti.setY(starts[i][1])
            starti.setZ(starts[i][2])
            starti.setYaw(starts[i][3])

            goali.setX(ends[i][0])
            goali.setY(ends[i][1])
            goali.setZ(ends[i][2])  # z coordinate is flipped?
            goali.setYaw(ends[i][3])
        self.setup.setStartAndGoalStates(start, goal)
        self.setup.setup()

        # try to solve the problem
        self.path = None
        if self.setup.solve(params.time_limit):
            # simplify & print the solution
            self.setup.simplifySolution()
            self.path = self.setup.getSolutionPath()

    def get_path(self):
        return self.path

    def get_length_obj(self):
        return self.path.cost(self.length_obj)

    def get_clear_obj(self):
        return self.path.cost(self.clear_obj)

    def get_work_obj(self):
        return self.path.cost(self.work_obj)

    def get_balanced_obj(self):
        return self.path.cost(self.balanced_obj)

    def get_lengths(self):
        lengths = np.zeros((self.N,))
        for i in range(0, self.N):
            for t in range(1, self.path.getStateCount()):
                state1 = self.path.getState(t - 1)[i]
                state2 = self.path.getState(t)[i]
                lengths[i] = lengths[i] + self.state_space.getSubspace(i).distance(state1, state2)
        return lengths

    def to_array(self):
        n_states = self.path.getStateCount()
        states = self.path.getStates()
        path_array = np.zeros((n_states, 4 * self.N))
        for i in range(0, self.N):
            for t in range(0, n_states):
                path_array[t, 4 * i:(4 * i + 4)] = [states[t][i].getX(), states[t][i].getY(), states[t][i].getZ(),
                                                    states[t][i].getYaw()]
        return path_array

    def interpolate(self, num_pts):
        self.path.interpolate(num_pts)

    def interpolate_time(self, t):
        n_states = self.path.getStateCount()
        total_length = np.sum(self.get_lengths())
        temp_state = None
        length_so_far = 0
        for i in range(1, n_states):
            state1 = self.path.getState(i - 1)
            state2 = self.path.getState(i)
            length = self.state_space.distance(state1, state2)

            if (length_so_far + length) / total_length > t:
                new_t = (t * total_length - length_so_far) / length
                temp_state = self.state_space.allocState()
                self.state_space.interpolate(state1, state2, new_t, temp_state)
                break
            else:
                length_so_far = length_so_far + length

        if temp_state is None:
            temp_state = self.path.getState(n_states - 1)

        states = np.zeros((self.N, 4))
        for i in range(0, self.N):
            states[i, :] = [temp_state[i].getX(), temp_state[i].getY(), temp_state[i].getZ(), temp_state[i].getYaw()]
        return states

    def plot(self, ax=None, color_list=None):
        self.path.interpolate(100)
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        data = self.to_array()

        for i in range(0, self.N):
            x = data[:, 0 + 4 * i]
            y = data[:, 1 + 4 * i]
            z = data[:, 2 + 4 * i]
            if color_list is None:
                ax.plot(x, y, z)
            else:
                ax.plot(x, y, z, color_list[i])

        if ax is None:
            ax.legend()
            plt.show()

