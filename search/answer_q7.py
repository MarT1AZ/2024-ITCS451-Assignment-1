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
    pass


def is_goal(state: State, grid: np.ndarray) -> bool:
    """Return whether the state is a goal state."""
    # TODO
    pass


def cost(state: State, actoin: str, grid: np.ndarray) -> float:
    """Return a cost of an action on the state."""
    # TODO
    # Place the following lines with your own implementation
    # State (x,y,f)
    return grid[state[1]][state[0]]


#===============================================================================
# 7.2 SEARCH
#===============================================================================


def heuristic(state: State, goal_state: State) -> float:
    """Return the heuristic value of the state."""
    # TODO
    pass


def graph_search(
        grid: np.ndarray,
        strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    # TODO
    # Replace the lines below with your own implementation

    actions = []
    plans = []
    explored = []


    # find agent
    ax,ay = find_agent(grid)
    ax = int(ax)
    ay = int(ay)
    # '^', '>', 'v', '<'
    #  2   3     4    5
    # agent positions are 0-indexed so plus one to make it look like 1 - indexed
    
    # Symbol mapping:
    # -  0: ' ', empty (passable)
    # -  1: '#', wall (not passable)
    # -  2x: '^', agent is facing up (north)
    # -  3x: '>', agent is facing right (east)
    # -  4x: 'v', agent is facing down (south)
    # -  5x: '<', agent is facing left (west)
    # -  6: 'G', goal
    # -  7: '~', mud (passable, but cost more)
    # -  8: '.', grass (passable, but cost more)
    # """
    t = 0
    if strategy == 'DFS':
        dir_str_map = ['N','E','S','W']
        dir_map = [(1,0),(0,1),(-1,0),(0,-1)]
        dict_assign_index = 0
        parent_dict = {}
        # dict does not contain -1 as a key
        # state format --> (level * -1,x,y,dir,parent_index) while x and y are 1 - indexed
        init_state = (-1,ax, ay, dir_str_map[(grid[ay][ax] // 10) - 2],-1)
        # parent_dict[str(dict_assign_index)] = init_state
        # dict_assign_index = 1
        tovisit = []
        visited = set()
        tovisit.append(init_state)
        found_flag = False
        goal_node = None
        while not found_flag:
            heapq.heapify(tovisit)
            gonext = heapq.heappop(tovisit)
            parent_dict[str(dict_assign_index)] = gonext
            visited.add((gonext[1],gonext[2]))
            # check neighbore cell
            for dx,dy in dir_map:
                if grid[gonext[2] + dy][gonext[1] + dx] != 1 and (gonext[1] + dx,gonext[2] + dy) not in visited:
                    
                    tovisit.append((gonext[0] - 1,gonext[1] + dx,gonext[2] + dy,'W',dict_assign_index)) # use 'W' but will find the actual dir later
                    if(grid[gonext[2] + dy][gonext[1] + dx] == 6):
                        goal_node = (gonext[0] - 1,gonext[1] + dx,gonext[2] + dy,'W',dict_assign_index)
                        found_flag = True
                        break
            dict_assign_index = dict_assign_index + 1

        # backtracking
        nextx,nexty = goal_node[1],goal_node[2]
        parent = goal_node[4]
        goal_node = (goal_node[1],goal_node[2],'W')
        actions.append('at the end')
        plans.append(goal_node)
        while parent != -1:
            plans.append(parent_dict.get(str(parent)))
            dx,dy = nextx - plans[-1][1],nexty - plans[-1][2]
            assign_dir = ''
            if(dx == 1 and dy == 0):
                assign_dir = 'E'
                actions.append('move EAST')
            elif(dx == 0 and dy == 1):
                assign_dir = 'S'
                actions.append('move SOUTH')
            elif(dx == -1 and dy == 0):
                assign_dir = 'W'
                actions.append('move WEST')
            else:
                assign_dir = 'N'
                actions.append('move NORTH')
            parent = plans[-1][4]
            nextx,nexty = plans[-1][1],plans[-1][2]
            # reformatting
            plans[-1] = (plans[-1][1],plans[-1][2],assign_dir)

        explored = list(visited)
        for i in range(len(explored)):
            explored[i] = (explored[i][0],explored[i][1],'W')
        actions.reverse()
        plans.reverse()
        return (actions,plans,explored)

        
        

    # init_state = (1, 1, 'W')
    # return (
    #     ['move west', 'move east', 'move east', 'move south'], actions of the plan'
    #     [(1, 1, 'S'), (2, 1, 'S'), (3, 1, 'S'), (4, 1, 'S')], 'states of the plan'
    #     [(1, 1, 'W'), (1, 1, 'E'), (1, 1, 'S'), (1, 2, 'S'), (1, 3, 'S'), (2, 1, 'E'), (2, 1, 'S'),(5,4,'W')] 'explored states'
    # )

 