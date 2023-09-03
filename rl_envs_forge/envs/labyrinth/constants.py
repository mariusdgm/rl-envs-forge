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

### Display relevant constants
# Colors mapped to maze objects for easy reference
COLORS = {
    WALL: (0, 0, 0),
    PATH: (255, 255, 255),
    PLAYER: (255, 0, 0),
    TARGET: (0, 255, 0),
    START: (0, 0, 255)
}

CELL_SIZE = 40
