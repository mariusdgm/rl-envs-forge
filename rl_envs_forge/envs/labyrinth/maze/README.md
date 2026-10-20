# Maze generation

This package builds the layouts used by [Labyrinth](../README.md). A maze
starts with rooms, then adds corridors and chooses start and target positions.
You can use the generators directly when you only need a layout.

## Contents

- [Creating a maze](#creating-a-maze)
- [Rooms](#rooms)
- [Corridors](#corridors)

## Creating a maze

```python
from rl_envs_forge.envs.labyrinth.maze.maze import MazeFactory

factory = MazeFactory(
    rows=30,
    cols=30,
    nr_desired_rooms=5,
    global_room_ratio=0.4,
    room_types=["rectangle", "oval"],
    corridor_algorithm="prim",
    seed=0,
)
maze = factory.create_maze()
layout = maze.grid
start = maze.start_position
target = maze.target_position
```

Generated mazes need at least 10 rows and 10 columns. `grid` stores walls as
zero and paths as one; start and target positions are separate attributes.
The requested room count is a placement target: `nr_placed_rooms` records how
many rooms fit.

`global_room_ratio` sets the share of the maze reserved for rooms. For variable
layouts, leave `nr_desired_rooms` or `global_room_ratio` unset and use their
corresponding `_range` arguments. `room_ratio` controls room proportions,
while `access_points_per_room` controls corridor entrances.

## Rooms

`RoomFactory` creates individual rooms:

```python
from rl_envs_forge.envs.labyrinth.maze.room import RoomFactory

room_factory = RoomFactory(room_types=["rectangle", "oval"], seed=0)
room = room_factory.create_room(rows=8, cols=10)
layout = room.grid
entrances = room.access_points
```

Available `room_types` are `rectangle`, `oval`, `donut`, `t_shape`, `l_shape`,
and `triangle`. You can pass dimensions or a `desired_area`; an area takes
precedence over dimensions. Set `access_points_nr` for a fixed number of
entrances, or use `access_points_range` to sample it.

To add a room shape, implement a `Room` subclass, add a creation method to
`RoomFactory`, and register its name in the factory's available types and
creation dispatch.

## Corridors

`CorridorBuilder` works on the `Maze` that owns it and writes to its corridor
grid. The factory selects an algorithm through `corridor_algorithm`:

| Value | Layout |
| --- | --- |
| `prim` | Branching, maze-like corridors. |
| `astar` | Connects access points using A* search. |
| `gbfs` | Connects access points using greedy best-first search. |
| `random` | Chooses one of the three algorithms. |

For Prim, `corridor_grid_connect_option` controls how neighboring corridors
can connect. `corridor_post_process_option=True` extends paths near borders
and rooms where the local layout allows it.

For A* and GBFS, `corridor_sort_access_points_option=True` orders entrances
by proximity before connecting them; `False` shuffles them. Grid connection
and access-point sorting also accept `"random"`.

When configuring a `Labyrinth` environment rather than a `MazeFactory`, use
the corresponding `maze_` arguments, such as `maze_corridor_algorithm` and
`maze_corridor_sort_access_points_option`. The environment calls the factory
for you.
