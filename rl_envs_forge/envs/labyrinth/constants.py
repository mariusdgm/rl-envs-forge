from enum import IntEnum

# Constants for actions
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


### Constants for corridor generation
class CorridorMoveStatus(IntEnum):
    VALID_MOVE = 1
    MAZE_BOUNDARY = 2
    ROOM_BOUNDARY = 3
    INVALID = 4


class CorridorCosts(IntEnum):
    BASE = 1
    CORRIDOR = 0
    ACCESS_POINT = 0
    ADJACENT_ROOM = 100
    ADJACENT_ACCESS_POINT = 0
    ROOM_CELL = 100000
