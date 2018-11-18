import heapq

# from https://www.redblobgames.com/pathfinding/a-star/implementation.html

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self, i = None):
        return heapq.heappop(self.elements)[1]

    def peek(self):
    	return heapq.nsmallest(1, self.elements)[0]

    def heapify(self):
        heapq.heapify(self.elements)

    def extend(self, other):
        self.elements = self.elements + other.elements
        self.heapify()
