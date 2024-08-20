from multiprocessing import parent_process
from pydoc_data.topics import topics
from typing import Literal, List, Tuple, TypeAlias, Annotated
import heapq


import numpy as np

from search.env import find_agent

State: TypeAlias = Tuple[int, int, str]

#===============================================================================
# 7.1 FORMULATION
#===============================================================================

def state_func(grid: np.ndarray) -> State:
    """Return a state based on the grid (observation).

    Number mapping:
    -  0: dirt (passable)
    -  1: wall (not passable)
    -  2x: agent is facing up (north)
    -  3x: agent is facing right (east)
    -  4x: agent is facing down (south)
    -  5x: agent is facing left (west)
    -  6: goal
    -  7: mud (passable, but cost more)
    -  8: grass (passable, but cost more)

    State is a tuple of
    - x (int)
    - y (int)
    - facing ('N', 'E', 'S', or 'W')
    """
    # TODO
    pass


# TODO
ACTIONS: List[str] = []

def transition(state: State, action: str, grid: np.ndarray) -> State:
    """Return a new state."""
      # TODO
    action_map = {'E':(1,0),'S':(0,1),'W':(-1,0),'N':(-1,0),}
    dir = action_map.get(action)
    if grid[state.y + dir[1]][state.x + dir[0]] != 1:
        return node(state.x + dir[0],state.y + dir[1],'-',None)
    else:
        return None


def is_goal(state: State, grid: np.ndarray) -> bool:
    """Return whether the state is a goal state."""
    # TODO
    return grid[state.y][state.x] == 6


def cost(state: State, actoin: str, grid: np.ndarray) -> float:
    """Return a cost of an action on the state."""
    # TODO
    # Place the following lines with your own implementation
    # State (x,y,f)
    if type(state) is node:
        if(grid[state.y][state.x] > 19 and grid[state.y][state.x] < 60):
            return 0
        return grid[state.y][state.x]
    else:
        if(grid[state[1]][state[0]] > 19 and grid[state[1]][state[0]] < 60):
            return 0
        return grid[state[1]][state[0]]



#===============================================================================
# 7.2 SEARCH
#===============================================================================


def heuristic(state: State, goal_state: State) -> float:
    """Return the heuristic value of the state."""
    # TODO

    #  state (current cell position)
    #  goal_state (goal cell psoition)

    # 2 method

    dx =  state.x - goal_state.x
    dy = state.y - goal_state.y

    # 1. manhattan distance

    # return abs(dx) + abs(dy)


    # 2. euclidian distance (most reliable to gaurantee shortest path)

    return int((dx * dx + dy * dy)**(0.5))

    #  3. custom

    # return (dx * dx + dy * dy)/5



def graph_search(
        grid: np.ndarray,
        strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    # This function is very long since this function is required to return multiple output which are <action description> <plan selected from explored cell> <list of explored cell>
    # which is contrary to what i would implement. I would only return action
    # and because i did not use cell as a class but a tuble instead, this make it very long due to technical implementation
    # moreover, i shoulve have coded each search algorithm as a different helper function instead of having all of them in the same function definition

    # TODO
    # Replace the lines below with your own implementation

    actions = []
    plans = []
    explored = []

    # goal
    goal = (len(grid) - 2,len(grid[0]) - 2)


    # find agent
    ax,ay = find_agent(grid)
    ax = int(ax)
    ay = int(ay)
    dir_str_map = ['N','E','S','W']
    dir_map = [(1,0),(0,1),(-1,0),(0,-1)]

    # use for backtracking from goal to start cell (dfs)
    dict_assign_index = 0
    parent_dict = {}
    # use for backtracking from goal to start cell (dfs)

    



    """
    '^', '>', 'v', '<'
     2   3     4    5
    agent positions are 0-indexed so plus one to make it look like 1 - indexed
    
    Symbol mapping:
    -  0: ' ', empty (passable)
    -  1: '#', wall (not passable)
    -  2x: '^', agent is facing up (north)
    -  3x: '>', agent is facing right (east)
    -  4x: 'v', agent is facing down (south)
    -  5x: '<', agent is facing left (west)
    -  6: 'G', goal
    -  7: '~', mud (passable, but cost more)
    -  8: '.', grass (passable, but cost more)
    """
    if strategy == 'DFS':  ############################################################################################
        

        return depth_first_search(grid)
        

    elif strategy == 'BFS': ############################################################################################

        return breadth_first_search(grid)
        
    elif strategy == 'A*' or strategy == 'UCS' or strategy == 'GS':  ############################################################################################
        
        return cost_search(grid,strategy)


class node:
    def __init__(self,x,y,direction,parent) -> None:
        self.x = x
        self.y = y
        self.direction = direction
        self.parent = parent
        self.cost_sum = np.inf # use for cost search

    def assign_cost_sum(self,sum):
        self.cost_sum = sum

    def __eq__(self, __o: object) -> bool: # for comparison 
        return True

    def xy_tuble(self):
        return (int(self.x),int(self.y))
    
    def state_tuble(self):
        return (int(self.x),int(self.y),self.direction)


def depth_first_search( ##########################################
        grid: np.ndarray
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""


    facing_map = ['N','E','S','W'] # to determine facing based on value 20 30 40 50

    action_space = ['W','N','E','S'] # action space
    action_value = [0.4,0.3,0.2,0.1] # action value offset


    # declare initial state
    (ax,ay) = find_agent(grid)
    init_node = node(ax,ay,facing_map[grid[ay][ax]//10 - 2],None)
    
    tovisit = []
    explored_set = set() # for cheking if the node ahs been visited or not in o(1)

    explored = [] # keep the order of exploration
    actions = [] 
    plans = []

    action_space = ['W','N','E','S'] # action space
    action_str = ['Move West','Move North','Move East','Move South','Stop']
    action_value = [0.4,0.3,0.2,0.1] # action value offset

    # layout -> (depth,node) # depth is a value that determine the priority

    tovisit.append((0,init_node))

    while True:
        heapq.heapify(tovisit)
        current_node = heapq.heappop(tovisit)
        explored.append((current_node[1].state_tuble()))
        explored_set.add(current_node[1].xy_tuble())


        if is_goal(current_node[1],grid):
            # backtracking to form path on plans list
            backtrack_node = current_node[1]
            child_node = None
            plans.append(backtrack_node.state_tuble())
            actions.append(action_str[4])
            while backtrack_node.parent is not None:
                child_node = backtrack_node
                backtrack_node = backtrack_node.parent
                plans.append(backtrack_node.state_tuble())
                actions.append(action_str[action_space.index(child_node.direction)])

            break

        for i in range(0,len(action_space)):
            action = action_space[i]
            offset = action_value[i]
            new_node = transition(current_node[1],action,grid)
            
            if new_node is not None and new_node.xy_tuble() not in explored_set:
                new_node.parent = current_node[1]
                new_node.direction = action
                tovisit.append((int(current_node[0]) - 1 - offset,new_node))

    plans.reverse()
    actions.reverse()
    return (actions,plans,explored)


def breadth_first_search( ###############################################
        grid: np.ndarray
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    facing_map = ['N','E','S','W'] # to determine facing based on value 20 30 40 50

    action_space = ['W','N','E','S'] # action space
    action_str = ['Move West','Move North','Move East','Move South','Stop']


    # declare initial state
    (ax,ay) = find_agent(grid)
    init_node = node(ax,ay,facing_map[grid[ay][ax]//10 - 2],None)

    explored = []
    explored_set = set()
    tovisitnext = []

    actions = []
    plans = []

    # layout (node)
    explored.append(init_node.state_tuble())
    explored_set.add(init_node.xy_tuble())
    tovisitnext.append(init_node)


    while len(tovisitnext) != 0:
        
        current_node = tovisitnext.pop(0)

        if is_goal(current_node,grid):
            # Backtracking
            backtrack_node = current_node
            child_node = None
            plans.append(backtrack_node.state_tuble())
            actions.append(action_str[4])
            while backtrack_node.parent is not None:
                child_node = backtrack_node
                backtrack_node = backtrack_node.parent
                plans.append(backtrack_node.state_tuble())
                actions.append(action_str[action_space.index(child_node.direction)])
            break
        
        for action in action_space:
            new_node = transition(current_node,action,grid)
            if new_node is not None and new_node.xy_tuble() not in explored_set:
                new_node.parent = current_node
                new_node.direction = action
                explored.append(new_node.state_tuble())
                explored_set.add(new_node.xy_tuble())
                tovisitnext.append(new_node)
        # print(len(tovisitnext))

    plans.reverse()
    actions.reverse()
    return (actions,plans,explored)

def cost_search(grid: np.ndarray,strategy) -> Tuple[ ###########################################################
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    
    facing_map = ['N','E','S','W'] # to determine facing based on value 20 30 40 50

    action_space = ['W','N','E','S'] # action space
    action_str = ['Move West','Move North','Move East','Move South','Stop']


    # declare initial state
    (ax,ay) = find_agent(grid)
    init_node = node(ax,ay,facing_map[grid[ay][ax]//10 - 2],None)

    # declare goal state
    goal_node = node(len(grid) - 2,len(grid[0]) - 2,'W',None)

    explored = []

    # (x,y)
    explored_set = set()

    # (node)
    open_list = [] # store cell that have been visited but not selected yet

    backtrack_node = None
    actions = []
    plans = []

    open_list.append(init_node)
    init_node.assign_cost_sum(0)

    while len(open_list) != 0:
        min_cost_sum = np.inf if strategy == 'A*' or strategy == 'UCS' else 0
        min_heuristic = np.inf if strategy == 'A*' or strategy == 'GS' else 0
        selected_node = None

        # find new node O(len(open_list))
        for current_node in open_list:
            cost_sum = current_node.cost_sum if strategy == 'A*' or strategy == 'UCS' else 0
            heuristic_value = heuristic(current_node,goal_node) if strategy == 'A*' or strategy == 'GS' else 0
            if cost_sum + heuristic_value < min_cost_sum + min_heuristic:
                min_cost_sum = cost_sum
                min_heuristic = heuristic_value
                selected_node = current_node

        # set x,y to visited/explore once selected
        explored.append(selected_node.state_tuble())
        explored_set.add(selected_node.xy_tuble())

        # check if the goal is met
        if is_goal(selected_node,grid):
            backtrack_node = selected_node
            break
        
        # remove node that has x,y as the same as selected node from open list O(len(open_list))
        keep_node = []
        for _node in open_list:
            if _node.xy_tuble() != selected_node.xy_tuble():
                keep_node.append(_node)
        open_list = keep_node
        # open_list.remove(selected_node)

        # relax node cost_sum or add node into the open list
        for action in action_space:
            adjacent_node = transition(selected_node,action,grid)
            if adjacent_node is not None and adjacent_node.xy_tuble() not in explored_set:
                adjacent_node.assign_cost_sum(selected_node.cost_sum + cost(selected_node,action,grid))
                adjacent_node.direction = action
                adjacent_node.parent = selected_node
                open_list.append(adjacent_node) # 2 nodes may have duplicate x,y value but parent not the same, (this does not affect the algorithm but affect space complexity)
    
    # backtracking 
    child_node = None
    plans.append(backtrack_node.state_tuble())
    actions.append(action_str[4])
    while backtrack_node.parent is not None:
        child_node = backtrack_node
        backtrack_node = backtrack_node.parent
        plans.append(backtrack_node.state_tuble())
        actions.append(action_str[action_space.index(child_node.direction)])
    actions.reverse()
    plans.reverse()


    return (actions,plans,explored)




        
        
        
            

        
        














    
    
