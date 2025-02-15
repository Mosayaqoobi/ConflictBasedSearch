import copy
import heapq

class State:
    """
    Class to represent a state on grid-based pathfinding problems. The class contains two static variables:
    map_width and map_height containing the width and height of the map. Although these variables are properties
    of the map and not of the state, they are used to compute the hash value of the state, which is used
    in the CLOSED list. 

    Each state has the values of x, y, g, h, and cost. The cost is used as the criterion for sorting the nodes
    in the OPEN list. 
    """
    map_width = 0
    map_height = 0
    
    def __init__(self, x, y):
        """
        Constructor - requires the values of x and y of the state. All the other variables are
        initialized with the value of 0.
        """
        self._x = x
        self._y = y
        self._g = 0
        self._cost = 0
        self._parent = None
        
    def __repr__(self):
        """
        This method is invoked when we call a print instruction with a state. It will print [x, y],
        where x and y are the coordinates of the state on the map. 
        """
        state_str = "[" + str(self._x) + ", " + str(self._y) + "]"
        return state_str
    
    def __lt__(self, other):
        """
        Less-than operator; used to sort the nodes in the OPEN list
        """
        return self._cost < other._cost
    
    def __hash__(self):
        """
        Given a state (x, y), this method returns the value of x * map_width + y. This is a perfect 
        hash function for the problem (i.e., no two states will have the same hash value). This function
        is used to implement the CLOSED list of the algorithms. 
        """
        return hash((self._y * State.map_width + self._x, self._g))
    
    def __eq__(self, other):
        """
        Method that is invoked if we use the operator == for states. It returns True if self and other
        represent the same state; it returns False otherwise. 
        """
        return self._x == other._x and self._y == other._y and self._g == other._g

    def is_goal(self, goal):
        return self._x == goal._x and self._y == goal._y

    def get_x(self):
        """
        Returns the x coordinate of the state
        """
        return self._x
    
    def get_y(self):
        """
        Returns the y coordinate of the state
        """
        return self._y
    
    def get_g(self):
        """
        Returns the g-value of the state
        """
        return self._g
        
    def set_g(self, g):
        """
        Sets the g-value of the state
        """
        self._g = g

    def get_cost(self):
        """
        Returns the cost of a state; the cost is determined by the search algorithm
        """
        return self._cost
    
    def set_cost(self, cost):
        """
        Sets the cost of the state; the cost is determined by the search algorithm 
        """
        self._cost = cost

    def set_parent(self, parent):
        """
        Defines the parent of a node in the A* search
        """
        self._parent = parent

    def get_parent(self):
        """
        Returns the parent of a node in the A* search
        """
        return self._parent

    def get_heuristic(self, target_state):
        """
        Returns the Manhattan distance heuristic between the state and the target state.
        """
        dist_x = abs(self.get_x() - target_state.get_x())
        dist_y = abs(self.get_y() - target_state.get_y())

        return dist_x + dist_y
    

class CBSState:
        
    def __init__(self, map, starts, goals):
        """
        Constructor of the CBS state. Initializes cost, constraints, maps, start and goal locations, 
        number of agents, and the solution paths.
        """
        self._cost = 0
        self._constraints = {}
        self._map = map
        self._starts = starts
        self._goals = goals
        self._k = len(starts)

        # One dictionary of constraints for each agent
        for i in range(0, self._k):
            self._constraints[i] = {}

        self._paths = {}

    def compute_cost(self):
        """
        Computes the cost of a CBS state. Assumes the sum of the cost of the paths as the objective function.
        """
        astar = AStar(self._map)
        total_cost = 0  # initialize the total cost
        updated_paths = {}

        for i, (start, goal) in enumerate(zip(self._starts, self._goals)):  # iterate over agents
            path_cost, path = astar.search(start, goal, self._constraints[i])
            
            if path is None: 
                self._cost = float('inf')  # mark as infeasible
                return self._cost, {}  
            
            total_cost += path_cost 
            updated_paths[i] = path  # store path for agent i

        self._cost = total_cost  # update the state's total cost
        self._paths = updated_paths  
        return self._cost, self._paths 

    
    def is_solution(self):
        """
        Verifies whether a CBS state is a solution. If it isn't, it returns False and a tuple with 
        the conflicting state and time step; returns True, None otherwise. 
        """
        max_steps = max(len(path) for path in self._paths.values())  # find longest path
        occupied_pos = {}  # track agent positions at each step
        
        for agent, path in self._paths.items():
            for time in range(max_steps):
                if time < len(path):
                    position = path[time]
                    
                    if (time, position) in occupied_pos:  # check for conflict
                        return False, (position, time, occupied_pos[(time, position)], agent)
                    
                    occupied_pos[(time, position)] = agent  # occupied position
        
        return True, (None, None, None, None)

    def successors(self):
        """
        Generates the two children of a CBS state that doesn't represent a solution.
        """
        has_conflict, (conflict_state, conflict_time, agent_a, agent_b) = self.is_solution()

        if has_conflict:  # return empty tuple if no conflict is found
            return []

        node_a = CBSState(self._map, self._starts, self._goals)  # create first child
        node_a._constraints = copy.deepcopy(self._constraints)  # copy constraints
        node_a.set_constraint(conflict_state, conflict_time, agent_a)  # add constraint for agent a

        node_b = CBSState(self._map, self._starts, self._goals)  # create second child
        node_b._constraints = copy.deepcopy(self._constraints)  # copy constraints
        node_b.set_constraint(conflict_state, conflict_time, agent_b)  # add constraint for agent b

        return node_a, node_b  # return the two children

    def set_constraint(self, conflict_state, conflict_time, agent):
        """
        Sets a constraint for agent in conflict_state and conflict_time
        """
        if (conflict_state.get_x(), conflict_state.get_y()) not in self._constraints[agent]:
            self._constraints[agent][(conflict_state.get_x(), conflict_state.get_y())] = set()
        
        self._constraints[agent][(conflict_state.get_x(), conflict_state.get_y())].add(conflict_time)

    def __lt__(self, other):
        """
        Less-than operator; used to sort the nodes in the OPEN list
        """
        return self._cost < other._cost
    
    def get_cost(self):
        """
        Returns the cost of a state
        """
        return self._cost
    
    def set_cost(self, cost):
        """
        Sets the cost of the state
        """
        self._cost = cost

class CBS():
    def search(self, start):
        """
        Performs CBS search for the problem defined in start.
        """
        state = start
        total_cost, _ = state.compute_cost()  # get initial cost

        if total_cost == float('inf'):  # check if the initial state is unsolvable
            return None, total_cost
    
        open_list = [(total_cost, state)]

        while open_list:
            _, state = heapq.heappop(open_list)
            
            if state.is_solution()[0]:  # check if we found a valid solution
                return state._paths, state.get_cost()
            
            for next_state in state.successors():
                next_cost, _ = next_state.compute_cost()  # compute cost for the new state
                
                if next_cost < float('inf'):  # check if valid
                    heapq.heappush(open_list, (next_cost, next_state))

        return None, float('inf')  #if queue empty, return None
        
class AStar():

    def __init__(self, gridded_map):
        """
        Constructor of A*. Creates the datastructures OPEN and CLOSED.
        """
        self.map = gridded_map
        self.OPEN = []
        self.CLOSED = {}
    
    def compute_cost(self, state):
        """
        Computes the f-value of nodes in the A* search
        """
        state.set_cost(state.get_g() + state.get_heuristic(self.goal))

    def _recover_path(self, node):
        """
        Recovers the solution path A* finds. 
        """
        path = []
        while node.get_parent() is not None:
            path.append(node)
            node = node.get_parent()
        path.append(node)
        return path[::-1]

    def search(self, start, goal, constraints=None):
        """
        A* Algorithm: receives a start state and a goal state as input. It returns the
        cost of a path between start and goal and the number of nodes expanded.

        If a solution isn't found, it returns -1 for the cost.
        """
        self.start = start
        self.goal = goal

        self.compute_cost(self.start)
        
        self.OPEN.clear()
        self.CLOSED.clear()
        
        heapq.heappush(self.OPEN, self.start)
        self.CLOSED[start.__hash__()] = self.start
        while len(self.OPEN) > 0:
            node = heapq.heappop(self.OPEN)

            if node.is_goal(self.goal):
                return node.get_g(), self._recover_path(node)

            children = self.map.successors(node, constraints)

            for child in children:
                hash_value = child.__hash__()
                self.compute_cost(child)
                child.set_parent(node)
                
                if hash_value not in self.CLOSED:
                    heapq.heappush(self.OPEN, child)
                    self.CLOSED[hash_value] = child

                if hash_value in self.CLOSED and self.CLOSED[hash_value].get_g() > child.get_g():
                    heapq.heappush(self.OPEN, child)
                    self.CLOSED[hash_value] = child
        return -1, None
    