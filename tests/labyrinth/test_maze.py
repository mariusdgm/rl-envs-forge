import pytest
import numpy as np
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
    