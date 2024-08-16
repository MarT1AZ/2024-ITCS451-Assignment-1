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
    if(grid[state[1]][state[0]] > 19 and grid[state[1]][state[0]] < 60):
        return 0
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
    dir_str_map = ['N','E','S','W']
    dir_map = [(1,0),(0,1),(-1,0),(0,-1)]

    # use for backtracking from goal to start cell
    dict_assign_index = 0
    parent_dict = {}
    # use for backtracking from goal to start cell

    



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
    if strategy == 'DFS':
        
        # use for tracking visted cell and choosing where to go next (for DFS)
        tovisit = []
        visited = set()
        # use for tracking visted cell and choosing where to go next (for DFS)

        # dict does not contain -1 as a key
        # state format --> (level * -1,x,y,dir,parent_index) while x and y are 1 - indexed
        init_state = (-1,ax, ay, dir_str_map[(grid[ay][ax] // 10) - 2],-1)
        # parent_dict[str(dict_assign_index)] = init_state
        # dict_assign_index = 1
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
        

    elif strategy == 'BFS':

        found_flag = False
        #  keep track of explored cell and where to go next (for BFS)
        visited = set()
        tovisit = set()
        tovisitnext = set()
        #  keep track of explored cell and where to go next (for BFS)

        # keep track of minimum weight sum of each cell
        weight_grid = np.ndarray((len(grid),len(grid[0])))
        weight_grid.fill(np.inf)
        weight_grid[ay][ax] = 0 #initial weight at agent position (weight = 0)
        # keep track of minimum weight sum of each cell
        
        visited.add((ax,ay))
        tovisit.add((ax + 1,ay))
        tovisit.add((ax,ay + 1))
        while not found_flag:
            
            for cell in tovisit:
                if found_flag:
                    break
                if(grid[cell[1]][cell[0]] != 1):
                    visited.add(cell)
                    # finding visited neighbor cell woth lowest weight sum
                    # atleast one neighbor cell must contain a sum
                    # assign the weight into the weight matrix
                    min_weight = np.inf
                    for dx,dy in dir_map:
                        nbcell = (cell[0] + dx,cell[1] + dy)
                        if nbcell in visited and weight_grid[nbcell[1]][nbcell[0]] != np.inf and weight_grid[nbcell[1]][nbcell[0]] < min_weight:
                            min_weight = weight_grid[nbcell[1]][nbcell[0]]
                            weight_grid[cell[1]][cell[0]] = min_weight + grid[cell[1]][cell[0]] + 1
                            if grid[cell[1]][cell[0]] == 6:
                                found_flag = True
            if found_flag:
                break

            for cell in tovisit:
                # find neightbor cell that has not been visited
                for dx,dy in dir_map:
                    nbcell = (cell[0] + dx,cell[1] + dy)
                    if nbcell not in visited and grid[nbcell[1]][nbcell[0]] != 1:
                        tovisitnext.add(nbcell)

            tovisit.clear()
            tovisit = tovisit.union(tovisitnext)
            tovisitnext.clear()

        # backtracking (Dijkstar,BFS)
        plans.append((len(grid) - 2,len(grid[0]) - 2,'W'))
        actions.append('at the end')
        current_cell = (len(grid) - 2,len(grid[0]) - 2)
        while current_cell != (ax,ay):
            next_cell = None
            min_weight = np.inf
            for dx,dy in dir_map:
                nbcell = (current_cell[0] + dx,current_cell[1] + dy)
                if weight_grid[nbcell[1]][nbcell[0]] != np.inf and min_weight > weight_grid[nbcell[1]][nbcell[0]]:
                    min_weight = weight_grid[nbcell[1]][nbcell[0]]
                    next_cell = (nbcell[0],nbcell[1])
            dx,dy = current_cell[0] - next_cell[0],current_cell[1] - next_cell[1]
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
            current_cell = next_cell
            plans.append((current_cell[0],current_cell[1],assign_dir))
        
        explored = list(visited)
        for i in range(len(explored)):
            explored[i] = (explored[i][0],explored[i][1],'W')
        plans.reverse()
        actions.reverse()


    return (actions,plans,explored)


        
        

    # init_state = (1, 1, 'W')
    # return (
    #     ['move west', 'move east', 'move east', 'move south'], actions of the plan'
    #     [(1, 1, 'S'), (2, 1, 'S'), (3, 1, 'S'), (4, 1, 'S')], 'states of the plan'
    #     [(1, 1, 'W'), (1, 1, 'E'), (1, 1, 'S'), (1, 2, 'S'), (1, 3, 'S'), (2, 1, 'E'), (2, 1, 'S'),(5,4,'W')] 'explored states'
    # )

 