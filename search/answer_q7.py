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
    if(grid[state.y][state.x] > 19 and grid[state.y][state.x] < 60):
        return 0
    return grid[state.y][state.x]


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
        
        return cost_search(grid)


class node:
    def __init__(self,x,y,direction,parent) -> None:
        self.x = x
        self.y = y
        self.direction = direction
        self.parent = parent

    def __eq__(self, __o: object) -> bool: # for comparison 
        return True

    def xy_tuble(self):
        return (int(self.x),int(self.y))
    
    def state_tuble(self):
        return (int(self.x),int(self.y),self.direction)


def depth_first_search(
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
    action_str = ['Move West','Move North','Move East','Move South','At the start']
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
            while backtrack_node.parent is not None:
                plans.append(backtrack_node.state_tuble())
                actions.append(action_str[action_space.index(backtrack_node.direction)])
                backtrack_node = backtrack_node.parent
            plans.append(backtrack_node.state_tuble())
            actions.append(action_str[action_space.index(backtrack_node.direction)])
            actions[-1] = action_str[4]

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


def breadth_first_search(
        grid: np.ndarray
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    facing_map = ['N','E','S','W'] # to determine facing based on value 20 30 40 50

    action_space = ['W','N','E','S'] # action space
    action_str = ['Move West','Move North','Move East','Move South','At the start']


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
            while backtrack_node.parent is not None:
                plans.append(backtrack_node.state_tuble())
                actions.append(action_str[action_space.index(backtrack_node.direction)])
                backtrack_node = backtrack_node.parent
            plans.append(backtrack_node.state_tuble())
            actions.append(action_str[action_space.index(backtrack_node.direction)])
            actions[-1] = action_str[4]
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


def cost_search(grid: np.ndarray) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    facing_map = ['N','E','S','W'] # to determine facing based on value 20 30 40 50

    action_space = ['W','N','E','S'] # action space
    action_str = ['Move West','Move North','Move East','Move South','At the start']


    # declare initial state
    (ax,ay) = find_agent(grid)
    init_node = node(ax,ay,facing_map[grid[ay][ax]//10 - 2],None)

    # declare goal state
    goal_node = node(len(grid),len(grid[0]),'W',None)

    explored = []

    # (x,y)
    explored_set = set()

    # (node)
    open_set = set() # store cell that have been visited but not selected yet

    actions = []
    plans = []

    # store cost sum which can be changed due to cost relaxation protocol
    cost_sum_grid = np.ndarray((len(grid),len(grid[0])))
    cost_sum_grid.fill(np.inf)

    # layout (node)
    explored.append(init_node.state_tuble())
    explored_set.add(init_node.xy_tuble())
    open_set.add(init_node.xy_tuble())
    cost_sum_grid[init_node.y][init_node.x] = 0 # initial node cost sum is 0

    parent_dict = {} # to trace back to parent using key in the format "x,y"
    parent_dict[str(init_node.x).join(',').join(str(init_node.y))] = None
    i = 0
    while len(open_set) != 0:
        if(i == 300):
            break
        min_cost_sum = np.inf
        min_heuristic = np.inf
        selected_node = None
        # selecting node to expand
        for _node in open_set:
            (nx,ny) = _node
            if cost_sum_grid[ny][nx] + heuristic(node(nx,ny,'w',None),goal_node) < min_cost_sum + min_heuristic:
                min_cost_sum = cost_sum_grid[ny][nx]
                min_heuristic = heuristic(node(nx,ny,'w',None),goal_node)
                selected_node = node(nx,ny,'w',None)

        # print(open_set)
        open_set.remove(selected_node.xy_tuble())
        # print(open_set)

        # end case
        # print(selected_node.state_tuble())
        if is_goal(selected_node,grid):
            # print('leaving loop')
            break
        
        for action in action_space:
            new_node = transition(selected_node,action,grid)
            if new_node is None:
                continue
            action_cost = cost(new_node,action,grid)
            if new_node.xy_tuble() in open_set:
                # update the cost_sum / parent (relaxation)
                if cost_sum_grid[new_node.y][new_node.x] < cost_sum_grid[selected_node.y][selected_node.x] + action_cost:
                    cost_sum_grid[new_node.y][new_node.x] = cost_sum_grid[selected_node.y][selected_node.x] + action_cost
                    parent_dict[str(new_node.x).join(',').join(str(new_node.y))] = selected_node.xy_tuble()
            elif new_node.xy_tuble() not in explored_set:
                # print('add')
                # assign cost_sum / parent
                # add to open_set
                cost_sum_grid[new_node.y][new_node.x] = cost_sum_grid[selected_node.y][selected_node.x] + action_cost
                parent_dict[str(new_node.x).join(',').join(str(new_node.y))] = selected_node.xy_tuble()
                open_set.add(new_node.xy_tuble())
                explored.append((new_node.x,new_node.y,action))
                explored_set.add(new_node.xy_tuble())
        # i = i + 1
        # print('i is ', i)
    return (actions,plans,explored)


        
        
        
            

        
        














    
    
