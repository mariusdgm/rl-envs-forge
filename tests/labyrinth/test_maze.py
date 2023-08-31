import pytest
import numpy as np
import random

from rl_envs.envs.labyrinth.maze import Maze
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
        maze = Maze(50, 50, nr_desired_rooms=20, global_room_ratio=1) # this is impossible to fully build
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
        with pytest.raises(ValueError):
            maze.is_valid_maze()

    def test_room_inaccessible(self):
        maze = Maze(20, 20, nr_desired_rooms=1)

        # Choose a room randomly
        room = random.choice(maze.rooms)

        # Surround the room with walls on the nearest perimeter
        for (x, y) in room.get_perimeter_cells():
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= x + dx < maze.rows and 0 <= y + dy < maze.cols:
                    maze.grid[x + dx][y + dy] = WALL

        # Validate that the maze is now invalid
        with pytest.raises(ValueError):
            maze.is_valid_maze()

    def test_no_valid_target_position(self):
        # Initialize the maze (adjust the logic as per your initialization requirements)
        maze = Maze(rows=20, cols=20, nr_desired_rooms=1, global_room_ratio=0.2)  # Or however you initialize your maze

        room = maze.rooms[0]

        # Fill the cells inside the perimeter with walls
        perimeter_cells = room.get_perimeter_cells()
        for row in range(room.rows):
            for col in range(room.cols):
                if (row, col) not in perimeter_cells:
                    room.grid[row, col] = WALL

        # Check that the function raises the expected error
        with pytest.raises(ValueError, match="Could not find a valid target position in any room."):
            maze.choose_target_position()

    
    