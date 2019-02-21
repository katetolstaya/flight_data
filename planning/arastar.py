from planning.priority_queue import PriorityQueue
import time

inf = float("inf")


class ARAStar:
    def __init__(self, problem, start, goal, obj=None):
        self.prob = problem
        self.start = problem.new_node(start)
        self.goal = problem.new_node(goal)
        self.obj = obj

        self.open_set = PriorityQueue()  # set of open nodes
        self.closed_set = set()
        self.incons_set = set()
        self.g = {}  # cache cost
        self.h = {}  # cache heuristic

        self.open_set.put(self.start, 0.0)
        self.g[self.start] = 0
        self.h[self.start] = self.prob.heuristic(self.start, self.goal)

        self.g[self.goal] = inf
        self.h[self.goal] = 0

        self.eps = 5.0
        self.eps_ = self.eps
        self.mult_eps = 0.8

        self.goal_node = None  # result

        self.animate_plot = False

        if self.animate_plot:
            self.prob.initialize_plot(self.start, self.goal)

    def plan(self, to=30.0):
        timeout = time.time() + to  # seconds till timeout
        while self.eps_ > 1.0:
            self.eps = self.eps * self.mult_eps
            if time.time() > timeout or self.improve_path(timeout) == 1:
                break
            min_val = self.update_sets()
            self.eps_ = min(self.eps, self.g[self.goal_node] / min_val)
            # print(self.eps_)
        return self.goal_node

    def update_sets(self):
        # update priorities for s in open to fval(s)
        min_val = inf
        for i in range(0, len(self.open_set.elements)):  # probably inefficient
            (_, n) = self.open_set.elements[i]
            self.open_set.elements[i] = (self.f_val(n), n)
            min_val = min(min_val, self.g[n] + self.h[n])

        # move states from incons into open
        temp_set = PriorityQueue()
        for n in self.incons_set:  # probably inefficient
            temp_set.put(n, self.f_val(n))
            min_val = min(min_val, self.g[n] + self.h[n])
        self.open_set.extend(temp_set)
        self.open_set.heapify()

        self.closed_set = set()
        self.incons_set = set()
        return min_val

    def improve_path(self, timeout):

        while not self.open_set.empty() and (
                self.goal_node is None or self.f_val(self.goal_node) > self.open_set.peek()[0]):

            if time.time() > timeout:
                return 1

            s = self.open_set.get()
            self.closed_set.add(s)

            if self.animate_plot:
                self.prob.update_plot(s)

            for (n, c) in self.prob.get_neighbors(s):  # check each neighbor
                if self.obj is not None:
                    c = c * (1.0 + self.obj.get_cost(n))
                n_cost = c + self.g[s]

                if n_cost < inf and (n not in self.g or n_cost < self.g[n]):

                    n_goal = self.prob.at_goal_position(n, self.goal)
                    if n_goal:
                        self.goal_node = n

                    self.g[n] = n_cost
                    self.h[n] = self.prob.heuristic(n, self.goal)

                    if n not in self.closed_set:
                        self.open_set.put(n, self.f_val(n))
                    else:
                        self.incons_set.add(n)

        return 0

    def f_val(self, s):
        return self.g[s] + self.eps * self.h[s]
