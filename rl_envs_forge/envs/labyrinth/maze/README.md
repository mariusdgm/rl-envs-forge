# Labyrinth Maze and Room

## Introduction

The Labyrinth Maze module provides functionality for generating and manipulating maze layouts. It consists of two main components: the `Maze` class and the `Room` class.

The `Maze` class represents a maze layout and provides various methods for generating and manipulating mazes. `Maze` also uses a `CorridorBuilder` component to handle the path generation.

The `Room` class represents a room in the maze and provides methods for generating and manipulating rooms.

To instantiate a maze or a room, you can make use of the corresponding factory classes: `MazeFactory` and `RoomFactory`. These factory classes provide a convenient way to create instances of `Maze` and `Room` with different configurations and settings.

## Contents

- [Maze](#maze)
- [Room](#room)
- [CorridorBuilder](#corridorbuilder)

## Maze

The `Maze` class represents a maze layout and provides various methods for generating and manipulating mazes.

### Maze Generation

The maze generation process involves the following steps:

1. **Initialize Maze**: The maze dimensions are specified, and the initial walls are set up.

2. **Place Rooms**: Rooms are randomly placed in the maze based on the desired number of rooms and the global room ratio.

3. **Place Start and End Positions**: The start and end positions are randomly placed in the maze.

4. **Generate Corridor Maze**: Corridors are generated to connect the rooms and create a connected maze structure.

## Room

The `Room` class represents a room in the maze and provides methods for generating and manipulating rooms.

### Room Types

The `Room` class allows for the implementation of different room types. Currently, the following room types are available:

- **Rectangle Room**: A basic rectangular room.
- **Oval Room**: A basic oval room.
- **Donut Room**: A donut shaped room, built with an outer and an inner shape. These can be either rectangular or oval
- **T Shape Room**: A T shaped room.
- **L Shape Room**: An L shaped room.

To add new room types, simply create a new class that inherits from the `Room` class and implements the necessary methods.

## CorridorBuilder

Because the `Maze` starts by being populated with rooms, the maze generation is done with constraints defined by the room layouts.

`CorridorBuilder` encapsulates the functions relevant to corridor generation, and is a strongly coupled component of `Maze`.

A `CorridorBuilder` object is instantiated at `Maze` instantiation, and the functions have access to the internals of `Maze` and may modify them.

### Maze Corridor Algorithms

The following algorithms are available for generating the corridors in the maze:

- Prim's (prim): generates maze like corridors
- A* (astar): generates more direct corridors, should find the shortest paths
- Greedy Best First Search (gbfs): Unoptimal A*, much faster. Generates very direct paths.

Warning:
A* is ~ 100 times slower than Prim's
GBFS is ~ 4.5 times slower than Prim's

When the corridor generation algorithm is Prim, the corridor layout is also affected by the following parameters:

corridor_grid_connect_option:

- **True**: Corridors are grid-connected, ensuring a strict grid structure.
- **False**: Corridors are not grid-connected, allowing for more organic and irregular corridor paths.

post_process_option:

- **True**: An extra post-processing step is done after the corridor generation. This step randomly adds some extra walls in available spaces.

For A* and GBFS, the following parameters affect the layout:

maze_sort_access_points_option:

- **True**: Sorts the access points in a way that orders them so they are closer to each other (should lead to direct paths)
- **False**: The access points list is shuffled, leading to longer corridors on average.
