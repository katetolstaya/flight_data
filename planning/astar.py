from planning.priority_queue import PriorityQueue
import time
from planning.dubins_node import DubinsNode

inf = float("inf")
import matplotlib.pyplot as plt

class AStar:

    def __init__(self, problem, start, goal, obj):
        self.prob = problem
        self.start = problem.new_node(start)
        self.goal = problem.new_node(goal)
        self.obj = obj
        self.cost = {} # map for cost estimates
        self.cost[self.start] = 0
        self.open_set = PriorityQueue() # set of open nodes
        self.open_set.put(self.start, 0.0)

    def plan(self, to=30):

        timeout = time.time() + to  # seconds till timeout

        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #
        # start_loc = self.prob.to_loc(self.start.loc)
        # goal_loc = self.prob.to_loc(self.goal.loc)
        # x = [start_loc[0],goal_loc[0] ]
        # y = [start_loc[1],goal_loc[1] ]
        # ax.plot()
        # line, = ax.plot(x, y, 'ro')

        while not self.open_set.empty():
            if time.time() > timeout: 
                print('Timeout')
                return None

            # pop off the next node
            s = self.open_set.get()

            # loc = self.prob.to_loc(s.loc)
            # x.append(loc[0])
            # y.append(loc[1])
            #
            # line.set_data(x, y)
            # ax.relim()
            # ax.autoscale_view()
            # fig.canvas.draw()
            # fig.canvas.flush_events()


            if self.prob.at_goal_position(s, self.goal):  # return if goal
                return s

            for (n, c) in self.prob.get_neighbors(s):
                print(c)
                n_cost = c + self.cost[self.prob.hash(s)]
                hash_n = self.prob.hash(n)
                if n_cost < inf and (hash_n not in self.cost or n_cost < self.cost[hash_n]):
                    self.cost[hash_n] = n_cost
                    self.open_set.put(n, (n_cost + self.prob.heuristic(n, self.goal)))


        print('Empty list')
        return None
