from enum import IntEnum

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

### Constants for states in the maze
WALL = 0
PATH = 1 
TARGET = 2 
START = 3
PLAYER = 4

