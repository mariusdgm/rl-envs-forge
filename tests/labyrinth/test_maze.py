import pytest
from unittest.mock import Mock
import numpy as np
import random

from rl_envs.envs.labyrinth.maze import Maze, MazeFactory
from rl_envs.envs.labyrinth.constants import WALL, PATH


class TestMaze:
    def test_initialization(self):
        rows, cols = 50, 50
        maze = Maze(rows, cols)
        assert maze.grid.shape == (rows, cols)

    def test_room_placement(self):
        maze = Maze(50, 50, nr_desired_rooms=5)

        # Count number of PATH cells
        total_path_cells = np.sum(maze.room_grid == PATH)

        # It should be positive if rooms are added
        assert total_path_cells > 0

        # It should not exceed the expected ratio
        assert total_path_cells <= maze.global_room_ratio * maze.room_grid.size

    def test_levy_flight(self):
        maze = Maze(50, 50, nr_desired_rooms=1)
        step_size = maze.levy_step_size()

        # Lévy flight should occasionally produce larger jumps
        assert step_size >= 1

    def test_infinite_loop_break(self):
        maze = Maze(
            50, 50, nr_desired_rooms=20, global_room_ratio=1
        )  # this is impossible to fully build
        assert maze.nr_placed_rooms > 0

    def test_dimensions_too_small(self):
        with pytest.raises(ValueError):
            maze = Maze(1, 1)

    def test_start_point_inaccessible(self):
        maze = Maze(20, 20, nr_desired_rooms=1)
        # Surround the start point with walls
        x, y = maze.start_position
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= x + dx < maze.rows and 0 <= y + dy < maze.cols:
                maze.grid[x + dx][y + dy] = WALL

        # Validate that the maze is now invalid
        assert maze.is_valid_maze() is False

    def test_room_inaccessible(self):
        maze = Maze(20, 20, nr_desired_rooms=1)

        # Choose a room randomly
        room = random.choice(maze.rooms)

        # Surround the room with walls on the nearest perimeter
        for x, y in room.get_perimeter_cells():
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= x + dx < maze.rows and 0 <= y + dy < maze.cols:
                    maze.grid[x + dx][y + dy] = WALL

        # Validate that the maze is now invalid
        assert maze.is_valid_maze() is False

    def test_no_valid_target_position(self):
        # Initialize the maze (adjust the logic as per your initialization requirements)
        maze = Maze(
            rows=20, cols=20, nr_desired_rooms=1, global_room_ratio=0.2
        )  # Or however you initialize your maze

        room = maze.rooms[0]

        # Fill the cells inside the perimeter with walls
        perimeter_cells = room.get_perimeter_cells()
        for row in range(room.rows):
            for col in range(room.cols):
                if (row, col) not in perimeter_cells:
                    room.grid[row, col] = WALL

        # Check that the function raises the expected error
        with pytest.raises(
            ValueError, match="Could not find a valid target position in any room."
        ):
            maze.choose_target_position()

    def test_rooms_are_placed_extreme_min_coverage(self):
        maze = Maze(
            rows=20, cols=20, nr_desired_rooms=20, global_room_ratio=0.1
        )  # Or however you initialize your maze
        assert len(maze.rooms) > 0

    def test_grow_path_from(self):
        """Testing the fall back mechanism, this code is not covered in other tests..."""
        # Initialize the maze
        maze = Maze(rows=10, cols=10)

        # Clear any pre-existing rooms and grids
        maze.rooms = []
        maze.grid.fill(WALL)
        maze.room_grid.fill(WALL)  
        maze.corridor_grid.fill(WALL)
        maze.start_position = (maze.rows-1, maze.cols-1)

        # Insert our specific test room
        room = Mock()
        room.rows = 3
        room.cols = 3
        room.global_position = (1, 1)
        room.access_points = [(0, 0)]
        room.grid = np.ones((room.rows, room.cols), dtype=int)
        maze.rooms.append(room)

        # Overlay the room onto maze.room_grid and maze.grid
        for i in range(room.rows):
            for j in range(room.cols):
                maze.room_grid[room.global_position[0] + i, room.global_position[1] + j] = room.grid[i, j]
        maze.grid = np.where(maze.room_grid == PATH, PATH, maze.grid)

        # Run the corridor generation function
        maze.generate_corridor_maze()
        maze.generate_corridor_maze()
        maze.connect_rooms_to_paths()
        maze.grid = np.where(maze.corridor_grid == PATH, PATH, maze.grid)

        assert maze.is_valid_maze() is True

        # Here, create conditions that would trigger the grow function.
        # For instance, we can add a wall barrier between the room's access points and the corridors.
        for i in range(4):
            maze.corridor_grid[i, 4] = WALL
            maze.corridor_grid[4, i] = WALL

            maze.grid[i, 4] = WALL
            maze.grid[4, i] = WALL

        assert maze.is_valid_maze() is False

        # Connect the rooms to paths
        maze.grow_path_from((room.access_points[0]))

        # # Overlay the corridors onto self.grid
        maze.grid = np.where(maze.corridor_grid == PATH, PATH, maze.grid)

        assert maze.is_valid_maze() is True



class TestMazeFactory:
    @pytest.fixture
    def default_factory(self):
        return MazeFactory(rows=20, cols=20)

    def test_initialization(self, default_factory):
        assert default_factory is not None

    def test_maze_creation(self, default_factory):
        maze = default_factory.create_maze()
        assert maze is not None

    def test_num_rooms_within_range(self, default_factory):
        maze = default_factory.create_maze()
        assert (
            default_factory.nr_desired_rooms_range[0]
            <= maze.nr_desired_rooms
            <= default_factory.nr_desired_rooms_range[1]
        )

    def test_global_room_ratio_within_range(self, default_factory):
        maze = default_factory.create_maze()
        assert (
            default_factory.global_room_ratio_range[0]
            <= maze.global_room_ratio
            <= default_factory.global_room_ratio_range[1]
        )

    def test_invalid_global_room_ratio(self):
        with pytest.raises(ValueError):
            maze_factory = MazeFactory(rows=20, cols=20, global_room_ratio=1.1)
            maze_factory.create_maze()

    def test_invalid_global_room_ratio_range(self):
        with pytest.raises(ValueError):
            maze_factory = MazeFactory(
                rows=20, cols=20, global_room_ratio_range=(0.1, 1.9)
            )
            maze_factory.create_maze()

    def test_custom_num_rooms(self):
        maze_factory = MazeFactory(rows=20, cols=20, nr_desired_rooms=3)
        maze_factory.create_maze()
        assert maze_factory.nr_desired_rooms == 3

    def test_custom_global_room_ratio(self):
        maze_factory = MazeFactory(rows=20, cols=20, global_room_ratio=0.6)
        maze_factory.create_maze()
        assert maze_factory.global_room_ratio == 0.6

    @pytest.mark.skip(reason="Testing other things")
    def test_bulk_mazes_are_valid(self):
        maze_factory = MazeFactory(rows=20, cols=20)
        for _ in range(1000):
            maze = maze_factory.create_maze()
            assert maze.is_valid_maze(), "Invalid maze detected!"
