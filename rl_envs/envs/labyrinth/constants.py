### Constants for states in the maze
WALL = 0
PATH = 1 
TARGET = 2 
START = 3
AGENT = 4

### Display relevant constants
# Colors mapped to maze objects for easy reference
COLORS = {
    WALL: (0, 0, 0),
    PATH: (255, 255, 255),
    AGENT: (255, 0, 0),
    TARGET: (0, 255, 0),
    START: (0, 0, 255)
}

CELL_SIZE = 40