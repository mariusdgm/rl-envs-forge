import pytest
import numpy as np
import random

from rl_envs_forge.envs.labyrinth.maze.maze import Maze, MazeFactory
from rl_envs_forge.envs.labyrinth.constants import WALL, PATH


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
        experiments = 5  # Number of experiments to perform
        total_step_size = 0

        # Perform multiple experiments to gather a more reliable average
        for _ in range(experiments):
            step_size = maze.levy_step_size()
            total_step_size += step_size

        average_step_size = total_step_size / experiments

        # Check if the average step size meets the expected criteria
        # The criteria might need to be adjusted based on what 'larger jumps' means quantitatively
        assert average_step_size >= 1, f"Average step size should be at least 1, was {average_step_size}"

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
        maze = Maze(
            rows=20,
            cols=20,
            nr_desired_rooms=1,
            global_room_ratio=0.2,
            room_types=["rectangle"],
        )

        maze.rooms = []

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

    def test_grid_connect_option_true(self):
        total_paths_with_option = 0
        total_paths_without_option = 0

        num_runs = 10

        for i in range(num_runs):
            maze = Maze(rows=10, cols=10, corridor_grid_connect_option=True, seed=i)
            total_paths_with_option += maze.corridor_grid.sum()

            maze_without_option = Maze(
                rows=10, cols=10, corridor_grid_connect_option=False, seed=i
            )
            total_paths_without_option += maze_without_option.corridor_grid.sum()

        assert total_paths_with_option > total_paths_without_option

    def test_grid_connect_option_false_consistency(self):
        for i in range(20):
            maze = Maze(
                rows=10,
                cols=10,
                corridor_algorithm="prim",
                corridor_grid_connect_option=False,
                seed=i,
            )

            # Generate another maze with the same seed and corridor_grid_connect_option=False
            another_maze = Maze(
                rows=10,
                cols=10,
                corridor_algorithm="prim",
                corridor_grid_connect_option=False,
                seed=i,
            )

            # Assert that the corridor grids are the same for both mazes with the same seed and option
            assert np.array_equal(maze.corridor_grid, another_maze.corridor_grid)


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

    @pytest.mark.slow
    def test_bulk_mazes_are_valid(self):
        """For multiple mazes test that all mazes are valid.
        This test is much slower than the rest of the testing codebase.
        Skip it by calling pytest as  [pytest tests -m 'not slow']."""
        for seed in range(5):
            random.seed(seed)
            rows = random.randint(10, 50)
            cols = random.randint(10, 50)
            post_process_option = random.choice([True, False])

            maze_factory = MazeFactory(
                rows=rows,
                cols=cols,
                corridor_post_process_option=post_process_option,
            )

            for generation in range(5):
                maze = maze_factory.create_maze()
                assert (
                    maze.is_valid_maze()
                ), f"Invalid maze detected! Seed: {seed}, Generation: {generation}"

    @pytest.mark.slow
    def test_bulk_mazes_are_valid_prim(self):
        """For multiple mazes test that all mazes are valid.
        This test is much slower than the rest of the testing codebase.
        Skip it by calling pytest as  [pytest tests -m 'not slow']."""
        for seed in range(100):
            random.seed(seed)
            rows = random.randint(10, 50)
            cols = random.randint(10, 50)
            post_process_option = random.choice([True, False])

            maze_factory = MazeFactory(
                rows=rows,
                cols=cols,
                corridor_algorithm="prim",
                corridor_post_process_option=post_process_option,
            )

            for generation in range(10):
                maze = maze_factory.create_maze()
                assert (
                    maze.is_valid_maze()
                ), f"Invalid maze detected! Seed: {seed}, Generation: {generation}"

    @pytest.mark.slow
    def test_bulk_mazes_are_valid_astar(self):
        """For multiple mazes test that all mazes are valid.
        This test is much slower than the rest of the testing codebase.
        Skip it by calling pytest as  [pytest tests -m 'not slow']."""
        for seed in range(2):
            random.seed(seed)
            rows = random.randint(10, 50)
            cols = random.randint(10, 50)

            maze_factory = MazeFactory(
                rows=rows,
                cols=cols,
                corridor_algorithm="astar",
            )

            for generation in range(3):
                maze = maze_factory.create_maze()
                assert (
                    maze.is_valid_maze()
                ), f"Invalid maze detected! Seed: {seed}, Generation: {generation}"

    @pytest.mark.slow
    def test_bulk_mazes_are_valid_gbfs(self):
        """For multiple mazes test that all mazes are valid.
        This test is much slower than the rest of the testing codebase.
        Skip it by calling pytest as  [pytest tests -m 'not slow']."""
        for seed in range(75):
            random.seed(seed)
            rows = random.randint(10, 50)
            cols = random.randint(10, 50)

            maze_factory = MazeFactory(
                rows=rows,
                cols=cols,
                corridor_algorithm="gbfs",
            )

            for generation in range(5):
                maze = maze_factory.create_maze()
                assert (
                    maze.is_valid_maze()
                ), f"Invalid maze detected! Seed: {seed}, Generation: {generation}"

    def test_grid_connect_option_random(self):
        factory = MazeFactory(rows=20, cols=20, seed=42)

        paths = []
        for _ in range(10):  # Running 10 tests to make randomness more evident
            factory.corridor_grid_connect_option = "random"
            maze = factory.create_maze()
            paths_count = maze.corridor_grid.sum()
            paths.append(paths_count)

        assert len(set(paths)) > 1

    def test_invalid_grid_connect_option(self):
        with pytest.raises(ValueError):
            factory = MazeFactory(
                rows=10, cols=10, corridor_grid_connect_option="invalid_value"
            )

        with pytest.raises(ValueError):
            factory = MazeFactory(rows=10, cols=10)
            factory.corridor_grid_connect_option = "invalid_value"
            factory.create_maze()

    def test_invalid_corridor_algorithm_option(self):
        with pytest.raises(ValueError):
            factory = MazeFactory(rows=10, cols=10, corridor_algorithm="invalid_value")

        with pytest.raises(ValueError):
            factory = MazeFactory(rows=10, cols=10)
            factory.corridor_algorithm = "invalid_value"
            factory.create_maze()
