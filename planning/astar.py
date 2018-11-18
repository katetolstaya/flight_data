from planning.priority_queue import PriorityQueue
import time

inf = float("inf")

class AStar:

    def __init__(self, start, goal, obj):
        self.start = start
        self.goal = goal
        self.obj = obj

        self.cost = {} # map for cost estimates
        self.cost[start] = 0
        self.open_set = PriorityQueue() # set of open nodes
        self.open_set.put(start, 0.0)


    def plan(self, to=30):

        timeout = time.time() + to  # seconds till timeout

        while not self.open_set.empty():

            if time.time() > timeout: 
                print('Timeout')
                return None

            # pop off the next node
            s = self.open_set.get()
            if s.at_goal_position(self.goal): # return if goal
                return s

            for n in s.get_neighbors():
                n_cost = n.cost_to_parent(self.obj) + self.cost[s]

                if n_cost < inf and (n not in self.cost or n_cost < self.cost[n]):
                    self.cost[n] = n_cost
                    self.open_set.put(n, (n_cost + n.heuristic(self.goal)))

        print('Empty list')
        return None
