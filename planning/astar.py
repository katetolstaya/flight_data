from planning.priority_queue import PriorityQueue
import time

inf = float("inf")


class AStar:

    def __init__(self, problem, start, goal, obj):
        self.prob = problem
        self.start = self.prob.to_ind(start)
        self.goal = self.prob.to_ind(goal)
        self.obj = obj

        self.cost = {} # map for cost estimates
        self.cost[self.prob.hash(self.start)] = 0
        self.open_set = PriorityQueue() # set of open nodes
        self.open_set.put(self.start, 0.0)

    def plan(self, to=30):

        timeout = time.time() + to  # seconds till timeout

        while not self.open_set.empty():

            if time.time() > timeout: 
                print('Timeout')
                return None

            # pop off the next node
            s = self.open_set.get()
            if self.prob.at_goal_position(s, self.goal):  # return if goal
                return s

            for n in self.prob.get_neighbors(s):
                n_cost = self.prob.primitive_cost + self.cost[self.prob.hash(s)]
                hash_n = self.prob.hash(n)

                if n_cost < inf and (hash_n not in self.cost or n_cost < self.cost[hash_n]):
                    self.cost[hash_n] = n_cost
                    self.open_set.put(n, (n_cost + self.prob.heuristic(n, self.goal)))

        print('Empty list')
        return None
