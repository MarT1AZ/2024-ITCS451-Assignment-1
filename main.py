import time

import pygame
import numpy as np

from search import env
from search import answer_q7

# During the development you can reduce this GRID_SIZE
GRID_SIZE = 20
CELL_SIZE = 40
WINDOW_SIZE = (CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE)

grid = env.gen_maze(GRID_SIZE, add_mud=True, add_grass=True)


pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Pathfinding')
env.render_maze_game(grid, screen, CELL_SIZE)

# Start the search
plan_actions, plan_states, explored_states = answer_q7.graph_search(grid,'A*')

# Visualization
overlay = np.zeros_like(grid)
total_cost = 0.0
for state in explored_states:
    x, y, f = state
    print('Explored: ', x, y, f)
    overlay[y, x] += 1
    env.render_overlay(overlay, screen, CELL_SIZE)
    time.sleep(0.01)

overlay = np.zeros_like(grid)
total_cost = 0.0

for action, state in zip(plan_actions, plan_states):
    print(action)
    print(state)
    total_cost += answer_q7.cost(state, action, grid)
    x, y, f = state
    overlay[y, x] += 101
    env.render_overlay(overlay, screen, CELL_SIZE)
    time.sleep(0.01)

print('The plan with :', ', '.join(plan_actions))
print('The plan cost:', total_cost)
print('Total explored states (iterations):', len(explored_states))

pygame.display.flip()
time.sleep(3)
pygame.quit()

# input('Hit `Enter` to end the program.')
