import pytest
from unittest.mock import Mock
import numpy as np
import random

from rl_envs_forge.envs.labyrinth.maze.corridor import CustomPriorityQueue
from rl_envs_forge.envs.labyrinth.maze.maze import Maze, MazeFactory
from rl_envs_forge.envs.labyrinth.maze.room import RectangleRoom
from rl_envs_forge.envs.labyrinth.constants import WALL, PATH, CorridorMoveStatus


class TestCorridorBuilder:
    @pytest.fixture
    def blank_maze(self):
        maze = Maze(rows=10, cols=10)
        maze.rooms = []
        maze.grid = np.full((10, 10), WALL)
        maze.room_grid = np.full((10, 10), WALL)
        maze.room_inner_area_grid = np.full((10, 10), WALL)
        maze.corridor_grid = np.full((10, 10), WALL)
        return maze

    # helper function
    def add_room_to_maze(self, maze, room):
        maze.rooms.append(room)
        maze.room_grid[
            room.global_position[0] : room.global_position[0] + room.rows,
            room.global_position[1] : room.global_position[1] + room.cols,
        ] = PATH
        maze.room_inner_area_grid[
            room.global_position[0] : room.global_position[0] + room.rows,
            room.global_position[1] : room.global_position[1] + room.cols,
        ] = PATH

    def test_is_valid_position(self, blank_maze):
        maze = blank_maze
        room = RectangleRoom(5, 5, top_left_coord=(4, 4))
        self.add_room_to_maze(maze, room)

        # Test case where new position is valid and not adjacent to any room
        result, status = maze.corridor_builder.is_valid_next_position(
            maze.grid, (0, 0), (1, 0)
        )
        assert result
        assert status == CorridorMoveStatus.VALID_MOVE

        # Test case where new position is outside maze boundaries
        result, status = maze.corridor_builder.is_valid_next_position(
            maze.grid, (0, 0), (-1, 0)
        )
        assert not result
        assert status == CorridorMoveStatus.MAZE_BOUNDARY

        # Test case where new position is adjacent to room boundary
        result, status = maze.corridor_builder.is_valid_next_position(
            maze.grid, (4, 2), (0, 1)
        )
        assert not result
        assert status == CorridorMoveStatus.ROOM_BOUNDARY

        # Test case where new position is invalid (going into room on grid)
        result, status = maze.corridor_builder.is_valid_next_position(
            maze.grid, (6, 3), (0, 3)
        )
        assert not result
        assert status == CorridorMoveStatus.INVALID

    def test_is_adjacent_to_room(self, blank_maze):
        maze = blank_maze
        room = RectangleRoom(3, 3, top_left_coord=(4, 4))
        self.add_room_to_maze(maze, room)

        # Test case where position is not adjacent to room
        result = maze.corridor_builder.is_adjacent_to_room((0, 0))
        assert not result

        # Test case where position is adjacent to room
        result = maze.corridor_builder.is_adjacent_to_room((3, 4))
        assert result

        # Test case where position is adjacent to room
        result = maze.corridor_builder.is_adjacent_to_room((4, 3))
        assert result

    def test_grow_path_from(self, blank_maze):
        """Testing the fall back mechanism,
        this code is not covered in other tests because the
        standard generation procedure seems to already fully connect the
        tested mazes. Probably could help more in small mazes (under 10x10),
        but those are not recommended and 10x10 is enforced as minimum size at the moment.
        """
        maze = blank_maze
        maze.start_position = (maze.rows - 1, maze.cols - 1)

        # Insert our specific test room
        room = Mock()
        room.rows = 3
        room.cols = 3
        room.global_position = (1, 1)
        room.access_points = [(0, 0)]
        room.grid = np.ones((room.rows, room.cols), dtype=int)
        room.generate_inner_area_mask = Mock(
            return_value=np.ones((room.rows, room.cols), dtype=int)
        )
        maze.rooms.append(room)

        # Overlay the room onto maze.room_grid and maze.grid
        for i in range(room.rows):
            for j in range(room.cols):
                maze.room_grid[
                    room.global_position[0] + i, room.global_position[1] + j
                ] = room.grid[i, j]
                maze.room_inner_area_grid[
                    room.global_position[0] + i, room.global_position[1] + j
                ] = room.grid[i, j]
        maze.grid = np.where(maze.room_grid == PATH, PATH, maze.grid)

        # Run the corridor generation function
        maze.corridor_grid = maze.corridor_builder.generate_corridor_prim()
        maze.corridor_builder.connect_rooms_to_paths()
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
        maze.corridor_builder.grow_path_from((room.access_points[0]))

        # # Overlay the corridors onto self.grid
        maze.grid = np.where(maze.corridor_grid == PATH, PATH, maze.grid)

        assert maze.is_valid_maze() is True

    def test_is_line_segment_intersecting_room(self, blank_maze):
        maze = blank_maze
        room = RectangleRoom(3, 3, top_left_coord=(4, 4))
        self.add_room_to_maze(maze, room)

        # Test case 1: Line segment that clearly passes through the room
        p1 = (3, 3)
        p2 = (7, 7)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert result

        # Test case 2: Line segment that starts in a room cell but doesn't intersect any other
        p1 = (5, 5)
        p2 = (5, 3)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert result

        # Test case 3: Line segment entirely outside the room, no intersection
        p1 = (0, 0)
        p2 = (3, 3)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert not result

        # Test case 4: Line segment that starts outside and ends in a room, but doesn't intersect any other cell
        p1 = (3, 3)
        p2 = (4, 4)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert result

        # Test case 5: Line segment that starts and ends in a room but doesn't intersect any other cell
        p1 = (4, 4)
        p2 = (6, 6)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert result

        # Test case 6: Horizontal line segment crossing the room
        p1 = (4, 2)
        p2 = (4, 7)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert result

        # Test case 7: Vertical line segment crossing the room
        p1 = (3, 5)
        p2 = (7, 5)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert result

        # Test case 8: Line starts in room, goes outsied and does not intersect room cells
        # We want to allow this case so we build connections from access points to corridor
        p1 = (4, 4)
        p2 = (2, 4)
        result = maze.corridor_builder.is_line_segment_intersecting_room(p1, p2)
        assert not result

    def test_generate_corridor_prim_and_connect_rooms_to_paths(self, blank_maze):
        maze = blank_maze
        maze.start_position = (0, 0)
        room = RectangleRoom(3, 3, nr_access_points=3, top_left_coord=(4, 4))
        self.add_room_to_maze(maze, room)

        maze.corridor_grid = maze.corridor_builder.generate_corridor_prim()
        maze.corridor_builder.connect_rooms_to_paths()
        maze.grid = np.where(maze.corridor_grid == PATH, PATH, maze.grid)

        assert maze.is_valid_maze() is True

    def test_is_next_to_access_point(self, blank_maze):
        ### Corner access points
        maze = blank_maze
        maze.start_position = (0, 0)
        room = RectangleRoom(3, 3, nr_access_points=3, top_left_coord=(4, 4))
        room.access_points = [(0, 0), (0, 2), (2, 0), (2, 2)]  # local coordinates
        self.add_room_to_maze(maze, room)
        perimeter_cells_to_check = room.get_perimeter_cells(padding=1)
        access_adjacent = [
            (0, -1),
            (-1, 0),
            (2, -1),
            (3, 0),
            (-1, 2),
            (0, 3),
            (2, 3),
            (3, 2),
        ]

        access_adjacent = [
            (x + room.global_position[0], y + room.global_position[1])
            for x, y in access_adjacent
        ]

        for perimeter_cell in perimeter_cells_to_check:
            if perimeter_cell in access_adjacent:
                assert maze.corridor_builder.is_next_to_access_point(perimeter_cell)
            else:
                assert not maze.corridor_builder.is_next_to_access_point(perimeter_cell)

        ### Middle access points
        maze = blank_maze
        maze.start_position = (0, 0)
        room = RectangleRoom(3, 3, nr_access_points=3, top_left_coord=(4, 4))
        room.access_points = [(0, 1), (2, 1), (1, 2), (1, 0)]  # local coordinates
        self.add_room_to_maze(maze, room)
        perimeter_cells_to_check = room.get_perimeter_cells(padding=1)
        access_adjacent = [
            (1, -1),
            (1, 3),
            (3, 1),
            (1, -1),
        ]

        access_adjacent = [
            (x + room.global_position[0], y + room.global_position[1])
            for x, y in access_adjacent
        ]

        for perimeter_cell in perimeter_cells_to_check:
            if perimeter_cell in access_adjacent:
                assert maze.corridor_builder.is_next_to_access_point(perimeter_cell)
            else:
                assert not maze.corridor_builder.is_next_to_access_point(perimeter_cell)

    def test_corridor_generation_prim(self):
        for _ in range(10):
            post_process_option = random.choice([True, False])
            maze_factory = MazeFactory(
                rows=20,
                cols=20,
                corridor_algorithm="prim",
                corridor_post_process_option=post_process_option,
            )
            maze = maze_factory.create_maze()

            assert maze.is_valid_maze()

    def test_corridor_generation_astar(self):
        for _ in range(10):
            maze_factory = MazeFactory(rows=20, cols=20, corridor_algorithm="astar")
            maze = maze_factory.create_maze()


class TestCustomPriorityQueue:
    def test_priority_queue_behavior(self):
        q = CustomPriorityQueue()
        q.put(8, (0, 16))
        q.put(10, (0, 18))
        q.put(8, (1, 17))

        assert not q.empty()
        assert q.get() == (0, 16)
        assert q.get() == (1, 17)
        assert q.get() == (0, 18)
        assert q.empty()
