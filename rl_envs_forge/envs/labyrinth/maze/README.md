# Labyrinth Maze and Room
## Introduction

The Labyrinth Maze module provides functionality for generating and manipulating maze layouts. It consists of two main components: the `Maze` class and the `Room` class.

The `Maze` class represents a maze layout and provides various methods for generating and manipulating mazes. The `Room` class represents a room in the maze and provides methods for generating and manipulating rooms.

To instantiate a maze or a room, you can make use of the corresponding factory classes: `MazeFactory` and `RoomFactory`. These factory classes provide a convenient way to create instances of `Maze` and `Room` with different configurations and settings.

## Contents

- [Maze](#maze)
- [Room](#room)

## Maze

The `Maze` class represents a maze layout and provides various methods for generating and manipulating mazes.

### Maze Generation

The maze generation process involves the following steps:

1. **Initialize Maze**: The maze dimensions are specified, and the initial walls are set up.

2. **Place Rooms**: Rooms are randomly placed in the maze based on the desired number of rooms and the global room ratio.

3. **Place Start and End Positions**: The start and end positions are randomly placed in the maze.

4. **Generate Corridor Maze**: Corridors are generated to connect the rooms and create a connected maze structure.

5. **Connect Rooms to Paths**: The rooms are connected to the main paths of the maze.

6. **Post-process Maze**: Additional modifications are made to the maze, such as removing dead ends or adding loops.

### Maze Corridor Algorithms

The following algorithms are available for generating the corridors in the maze:

 - Prim's algorithm

The corridor layout is also affected by the *grid_connect_corridors* parameter in the following way:

- **Non-grid Connect**: Corridors are not grid-connected, allowing for more organic and irregular corridor paths.
- **Grid Connect**: Corridors are grid-connected, ensuring a strict grid structure.

## Room

The `Room` class represents a room in the maze and provides methods for generating and manipulating rooms.

### Room Types

The `Room` class allows for the implementation of different room types. Currently, the following room types are available:

- **Rectangular Room**: A basic rectangular room.

To add new room types, simply create a new class that inherits from the `Room` class and implements the necessary methods.

