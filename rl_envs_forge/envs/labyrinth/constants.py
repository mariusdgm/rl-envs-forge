from enum import IntEnum

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class CorridorMoveStatus(IntEnum):
    VALID_MOVE = 1
    MAZE_BOUNDARY = 2
    ROOM_BOUNDARY = 3
    INVALID = 4

### Constants for states in the maze
WALL = 0
PATH = 1 
TARGET = 2 
START = 3
PLAYER = 4

