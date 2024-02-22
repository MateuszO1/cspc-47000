#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: szczurpi

This program implements the uniform cost search algorithm for solving a maze
(1 point cost for each move)
"""

import numpy as np
import queue # Needed for frontier queue
from heapq import heapify


class MazeState():
    """ Stores information about each visited state within the search """
    # Define constants
    SPACE = 0
    WALL = 1
    EXIT = 2
    VISITED = 3
    PATH = 4
    START_MARK = 5
    END_MARK = 6

    MAZE_FILE = 'maze2024.txt'
    maze = np.loadtxt(MAZE_FILE, dtype=np.int32)  
    start = tuple(np.array(np.where(maze==5)).flatten())
    ends = np.where(maze==2)
    move_num = 0 # Used by show_path() to count moves in the solution path
    
    def reset_state():
        """ Resets the static variables to prepare for a new search """
        MazeState.maze = np.loadtxt(MazeState.MAZE_FILE, dtype=np.int32)  
        MazeState.start = tuple(np.array(np.where(MazeState.maze==5)).flatten())
        MazeState.ends = np.where(MazeState.maze==2)
        MazeState.move_num = 0
    
    def __init__(self, conf=start, g=0, pred_state=None, pred_action=None):
        """ Initializes the state with information passed from the arguments """
        self.pos = conf         # Configuration of the state - current coordinates
        self.gcost = g          # Path cost
        self.pred = pred_state  # Predecesor state
        self.action_from_pred = pred_action  # Action from predecesor state to current state
    
    def __hash__(self):
        """ Returns a hash code so that it can be stored in a set data structure """
        return self.pos.__hash__()
    
    def is_goal(self):
        """ Returns true if current position is same as the exit position """
        return self.maze[self.pos] == MazeState.EXIT
    
    def __eq__(self, other):
        """ Checks for equality of states by positions only """
        return self.pos == other.pos
    
    def __lt__(self, other):
        """ Allows for ordering the states by the path (g) cost """
        return self.gcost < other.gcost
    
    def __str__(self):
        """ Returns the maze representation of the state """
        a = np.array(self.maze)
        a[self.start] = MazeState.START_MARK
        a[self.ends] = MazeState.EXIT
        return str(a)

    def show_path(self):
        """ Recursively outputs the list of moves and states along path """
        if self.pred is not None:
            self.pred.show_path()
        
        if MazeState.move_num==0:
            print('START')
        else:
            print('Move',MazeState.move_num, 'ACTION:', self.action_from_pred)
        MazeState.move_num = MazeState.move_num + 1
        self.maze[self.pos] = MazeState.PATH
    
    def get_new_pos(self, move):
        """ Returns a new position from the current position and the specified move """
        if move=='up':
            new_pos = (self.pos[0]-1, self.pos[1])
        elif move=='down':
            new_pos = (self.pos[0]+1, self.pos[1])
        elif move=='left':
            new_pos = (self.pos[0], self.pos[1]-1)
        elif move=='right':
            new_pos = (self.pos[0], self.pos[1]+1)
        else:
            raise('wrong direction for checking move')
        
        # Wrap around functionality 
        # modulo returns the new_pos above if its less than the height/width of the maze
        # modulo returns the wrapped x or y or both if greater than height/width of maze
        new_pos = (new_pos[0] % self.maze.shape[0], new_pos[1] % self.maze.shape[1])

        return new_pos
        
    def can_move(self, move):
        """ Returns true if agent can move in the given direction """
        new_pos = self.get_new_pos(move)
        if new_pos[0] < 0 or new_pos[0] >= self.maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.maze.shape[1]:
            return False
        else:
            return self.maze[new_pos]!=MazeState.WALL
                    
    def gen_next_state(self, move):
        """ Generates a new MazeState object by taking move from current state """
        new_pos = self.get_new_pos(move)
        if self.maze[new_pos] != MazeState.EXIT:
            self.maze[new_pos] = MazeState.VISITED
        return MazeState(new_pos, self.gcost+1, self, move)
            
# Display the heading info
print('Artificial Intelligence')
print('MP1: Robot navigation')
print('SEMESTER: Spring 2024')
print('NAME: Mateusz Obrochta')
print()

print('INITIAL MAZE')

# load start state onto frontier priority queue
frontier = queue.PriorityQueue() # This does best-first search
#frontier = queue.LifoQueue() # This would do depth-first search
#frontier = queue.Queue() # This would do breadth-first search

start_state = MazeState()
frontier.put(start_state)
print(start_state)
# Keep a closed set of states to which optimal path was already found
closed_set = set()

# Expand state (up to 4 moves possible)
possible_moves = ['left', 'right', 'down', 'up']

# Variables for keeping tracking of best disabled move and its path length
best_disabled_move = None
best_disabled_move_length = float('inf')

# Loop that disables one possible move each time, if solution exists it prints the states visited and best path length
for possible_move in possible_moves:
    print(f'\nSOLUTION AFTER DISABLED MOVE: {possible_move}')
    
    # Reset the maze and other static variables for a new search
    MazeState.reset_state()

    # Remove one possible_move from disabled_moves list
    disabled_moves = ['left', 'right', 'down', 'up']
    if possible_moves:
        disabled_moves.remove(possible_move)

    
    num_states = 0
    frontier = queue.PriorityQueue()
    start_state = MazeState()
    frontier.put(start_state)
    # Keep a closed set of states to which optimal path was already found
    closed_set = set()

    while not frontier.empty():
        # Choose state at front of priority queue
        next_state = frontier.get()
        num_states += 1

        # If goal then quit and return path
        if next_state.is_goal():
            next_state.show_path()
            break

        # Add state chosen for expansion to closed_set
        closed_set.add(next_state)

        # Expanding the node
        for move in disabled_moves:
            if next_state.can_move(move):
                neighbor = next_state.gen_next_state(move)
                if neighbor in closed_set:
                    continue
                if neighbor not in frontier.queue:
                    frontier.put(neighbor)
                else:
                    if neighbor.gcost < frontier.queue[frontier.queue.index(neighbor)].gcost:
                        frontier.queue[frontier.queue.index(neighbor)] = neighbor
                        heapify(frontier.queue)

    print(start_state)
    # Print no solution if no moves were taken, otherwise print states visited and shortest path
    if MazeState.move_num == 0:
        print(f'No solution')
    else:      
        print('\nNumber of states visited =', num_states)
        move_path_length = MazeState.move_num-1
        print('Length of shortest path = ', move_path_length)

        # Compare and update the best disabled move
        if move_path_length < best_disabled_move_length:
            best_disabled_move = possible_move
            best_disabled_move_length = move_path_length

# Display best disabled move and its shortest path
print('\nBEST MOVE: disable ', best_disabled_move)
print('SHORTEST PATH LENGTH FOR BEST MOVE:', best_disabled_move_length)
