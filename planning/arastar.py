from planning.priority_queue import PriorityQueue
import time

inf = float("inf")

class ARAStar:

    def __init__(self, problem, start, goal, obj):
        self.prob = problem
        self.start = problem.new_node(start)
        self.goal = problem.new_node(goal)
        self.obj = obj

        self.open_set = PriorityQueue() # set of open nodes
        self.closed_set = set()
        self.incons_set = set()
        self.g = {} # cache cost
        self.h = {} # cache heuristic


        self.open_set.put(self.start, 0.0)
        self.g[self.start] = 0
        self.h[self.start] = self.prob.heuristic(self.start, self.goal)

        self.g[self.goal] = inf
        self.h[self.goal] = 0

        self.eps = 3.0
        self.eps_ = self.eps
        self.mult_eps = 0.9

        self.goal_node = None # result


    def plan(self, to=30.0):

        timeout = time.time() + to  # seconds till timeout
        start_time = time.time()

        while self.eps_ > 1.0:
            self.eps = self.eps * self.mult_eps
            if time.time() > timeout or self.__improve_path(timeout) == 1:
                break
            min_val = self.__update_sets()
            self.eps_ = min(self.eps, self.g[self.goal_node_hash] / min_val)
            #print(self.eps_)
        return self.goal_node

    def __update_sets(self):

        # update priorities for s in open to fval(s)
        min_val = inf
        for i in range(0, len(self.open_set.elements)): # probably inefficient
            (_, n) = self.open_set.elements[i]
            self.open_set.elements[i] = (self.__f_val(n), n)
            min_val = min(min_val, self.g[n] + self.h[n])

        # move states from incons into open
        temp_set = PriorityQueue()
        for n in self.incons_set: # probably inefficient
            temp_set.put(n, self.__f_val(n))
            min_val = min(min_val, self.g[n] + self.h[n])
        self.open_set.extend(temp_set)

        self.closed_set = set()
        self.incons_set = set()
        return min_val

    def __improve_path(self, timeout):

        while self.goal_node is None or self.__f_val(self.goal_node) > self.open_set.peek()[0]:

            if time.time() > timeout: 
                return 1

            s = self.open_set.get()
            self.closed_set.add(s)

            for (n, c) in self.prob.get_neighbors(s):  # check each neighbor
                n_cost = c * (1.0 + self.obj.get_cost(n)) + self.g[s]

                if n_cost < inf and (n not in self.g or n_cost < self.g[n]):

                    n_goal = self.prob.at_goal_position(n, self.goal)
                    if n_goal:
                        self.goal_node = n
                        self.goal_node_hash = n

                    self.g[n] = n_cost
                    self.h[n] = self.prob.heuristic(n, self.goal)

                    if n not in self.closed_set:
                        self.open_set.put(n, self.__f_val(n))
                    else:
                        self.incons_set.add(n)

        return 0

    def __f_val(self, s):
        return self.g[s] + self.eps * self.h[s]