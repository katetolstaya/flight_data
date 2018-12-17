from planning.priority_queue import PriorityQueue
import time

inf = float("inf")

class AStar:

    def __init__(self, problem, start, goal, obj):
        self.prob = problem
        self.start = problem.new_node(start)
        self.goal = problem.new_node(goal)
        self.obj = obj
        self.cost = {}  # map for cost estimates
        self.cost[self.start] = 0
        self.open_set = PriorityQueue()  # set of open nodes
        self.open_set.put(self.start, 0.0)

        self.animate_plot = True

        if self.animate_plot:
            self.prob.initialize_plot(self.start, self.goal)

    def plan(self, to=30):

        timeout = time.time() + to  # seconds till timeout
        while not self.open_set.empty():
            if time.time() > timeout:
                return None

            # pop off the next node
            s = self.open_set.get()

            if self.animate_plot:
                self.prob.update_plot(s)

            if self.prob.at_goal_position(s, self.goal):  # return if goal
                return s

            for (n, c) in self.prob.get_neighbors(s):
                if self.obj is not None:
                    c = c * (1.0 + self.obj.get_cost(n))
                n_cost = c + self.cost[s]
                if n_cost < inf and (n not in self.cost or n_cost < self.cost[n]):
                    self.cost[n] = n_cost
                    self.open_set.put(n, (n_cost + self.prob.heuristic(n, self.goal)))
        return None
