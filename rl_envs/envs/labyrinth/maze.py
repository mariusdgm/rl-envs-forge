import numpy as np
import random
from collections import deque
from typing import List, Tuple, Union, Optional


from .room import RoomFactory
from .constants import WALL, PATH, START, TARGET

from ..common.grid_functions import on_line


class MazeFactory:
    def __init__(
        self,
        rows: int,
        cols: int,
        nr_desired_rooms: Optional[int] = None,
        nr_desired_rooms_range: Tuple[int, int] = (1, 8),
        global_room_ratio: Optional[float] = None,
        global_room_ratio_range: Tuple[float, float] = (0.1, 0.8),
        access_points_per_room: Optional[int] = None,
        access_points_per_room_range: Tuple[int, int] = (1, 4),
        room_types: Optional[List[str]] = None,
        room_ratio: Optional[Union[int, float]] = None,
        room_ratio_range: Tuple[Union[int, float], Union[int, float]] = (0.5, 1.5),
        seed: Optional[int] = None,
    ):
        """
        Maze builder to randomize the generated maze.

        Arguments num_rooms, global_room_ratio, access_points_per_room
        are used to fix specific values, otherwise the values are drawn from the minimum and maximum distibutions.

        room_types is used to determine what types of rooms are to be added in the maze, if None, the random seletion
        considers all the implemented room types.

        Args:
            rows (int): The number of rows in the maze.
            cols (int): The number of columns in the maze.
            nr_desired_rooms (int, optional): The desired number of rooms in the maze. Defaults to None.
            nr_desired_rooms_range (tuple, optional): The range of desired number of rooms. Defaults to (1, 8).
            global_room_ratio (float, optional): The global room ratio in the maze. Defaults to None.
            global_room_ratio_range (tuple, optional): The range of global room ratio. Defaults to (0.1, 0.8).
            access_points_per_room (int, optional): The number of access points in each room. Defaults to None.
            access_points_per_room_range (tuple, optional): The range of access points per room. Defaults to (1, 4).
            room_types (list, optional): The types of rooms to be added in the maze. Defaults to None.
            room_ratio (float, optional): The room ratio. Defaults to None.
            room_ratio_range (tuple, optional): The range of room ratio. Defaults to (0.5, 1.5).
            seed (int, optional): The seed to use for generating random numbers. Defaults to None.

        """

        self.rows = rows
        self.cols = cols

        self.nr_desired_rooms = nr_desired_rooms
        self.nr_desired_rooms_range = nr_desired_rooms_range

        self.global_room_ratio = global_room_ratio
        self.global_room_ratio_range = global_room_ratio_range

        self.access_points_per_room = access_points_per_room
        self.access_points_per_room_range = access_points_per_room_range

        self.room_types = room_types

        self.room_ratio = room_ratio
        self.room_ratio_range = room_ratio_range

        self.seed = seed
        if self.seed is None:
            self.seed = random.randint(0, 1e6)
        self.py_random = random.Random(self.seed)
        self.np_random = np.random.RandomState(self.seed)

    def create_maze(self):
        # Decide the number of rooms
        if self.nr_desired_rooms:
            num_rooms = self.nr_desired_rooms
        else:
            num_rooms = self.py_random.randint(
                self.nr_desired_rooms_range[0], self.nr_desired_rooms_range[1]
            )

        # Decide the global room ratio
        if self.global_room_ratio:
            if self.global_room_ratio > 1:
                raise ValueError(
                    "Global room ratio must be less than 1, but got {self.global_room_ratio}."
                )
            global_room_ratio = self.global_room_ratio
        else:
            if (
                self.global_room_ratio_range[0] < 0
                or self.global_room_ratio_range[1] > 1
            ):
                raise ValueError(
                    f"Global room ratio range must be between 0 and 1, but got {self.global_room_ratio_range}."
                )
            global_room_ratio = self.py_random.uniform(
                self.global_room_ratio_range[0], self.global_room_ratio_range[1]
            )

        maze_seed = self.seed + 1
        maze = Maze(
            rows=self.rows,
            cols=self.cols,
            nr_desired_rooms=num_rooms,
            global_room_ratio=global_room_ratio,
            access_points_per_room=self.access_points_per_room,
            access_points_per_room_range=self.access_points_per_room_range,
            room_types=self.room_types,
            room_ratio=self.room_ratio,
            room_ratio_range=self.room_ratio_range,
            seed=maze_seed,
        )
        return maze


class Maze:
    def __init__(
        self,
        rows: int,
        cols: int,
        nr_desired_rooms: int = 3,
        global_room_ratio: float = 0.5,
        access_points_per_room: Optional[int] = None,
        access_points_per_room_range: Tuple[int, int] = (1, 4),
        room_types: Optional[List[str]] = None,
        room_ratio: Optional[Union[int, float]] = None,
        room_ratio_range: Tuple[Union[int, float], Union[int, float]] = (0.5, 1.5),
        seed: Optional[int] = None,
    ) -> None:
        """Construct a Maze object representing a maze layout.

        Args:
            rows (int): The number of rows in the maze.
            cols (int): The number of columns in the maze.
            nr_desired_rooms (int, optional): The desired number of rooms in the maze. Defaults to 3.
            global_room_ratio (float, optional): The global room ratio in the maze. Defaults to 0.5.
            access_points_per_room (int, optional): The number of access points in each room. Defaults to None.
            access_points_per_room_range (tuple, optional): The range of access points per room. Defaults to (1, 4).
            room_types (list, optional): The types of rooms to be added in the maze. Defaults to None.
            room_ratio (float, optional): The room ratio. Defaults to None.
            room_ratio_range (tuple, optional): The range of room ratio. Defaults to (0.5, 1.5).
            seed (int, optional): The seed to use for generating random numbers. Defaults to None.
        """
        if rows < 10 or cols < 10:
            raise ValueError(
                "Maze dimensions must be at least 10x10, otherwise the maze generation logic risks breaking."
            )

        self.rows = rows
        self.cols = cols

        self.seed = seed
        if self.seed is None:
            self.seed = random.randint(0, 1e6)
        self.py_random = random.Random(self.seed)

        self.grid = np.ones((rows, cols), dtype=int) * WALL
        self.room_grid = np.ones((rows, cols), dtype=int) * WALL
        self.corridor_grid = np.ones((rows, cols), dtype=int) * WALL

        self.rooms = []
        self.nr_desired_rooms = nr_desired_rooms

        # in some cases we won't be able to place all rooms we want so track them
        self.nr_placed_rooms = 0
        self.global_room_ratio = global_room_ratio

        self.start_position = None
        self.target_position = None

        # Make Room Factory
        room_factory_seed = self.seed + 1
        self.room_factory = RoomFactory(
            access_points_nr=access_points_per_room,
            access_points_range=access_points_per_room_range,
            room_types=room_types,
            ratio=room_ratio,
            ratio_range=room_ratio_range,
            seed=room_factory_seed,
        )

        self._build_maze()

    def _build_maze(self):
        self.place_rooms()
        self.grid = np.where(self.room_grid == PATH, PATH, self.grid)

        self.place_start_end_positions()
        self.generate_corridor_maze()
        self.connect_rooms_to_paths()
        self.grid = np.where(self.corridor_grid == PATH, PATH, self.grid)

    ##### Room generation and placement #####
    def levy_flight_place(self, room):
        """Seek a random position for a room in such a way
        that it does not overlap with other rooms."""
        max_attempts = 100
        attempt = 0

        margin_padding = 2

        # define range from 2 to len - (minsize + 2) because we don't want to place rooms too close to edge of maze
        row_padding = room.rows + margin_padding
        col_padding = room.cols + margin_padding

        bottom_right_possible_row_coord = self.grid.shape[0] - row_padding
        bottom_right_possible_col_coord = self.grid.shape[1] - col_padding

        if (
            bottom_right_possible_row_coord < margin_padding
            or bottom_right_possible_col_coord < margin_padding
        ):
            # This proposed room is too big
            return False

        #### Attempt to find a viable position ####
        position = (
            self.py_random.randint(margin_padding, bottom_right_possible_row_coord),
            self.py_random.randint(margin_padding, bottom_right_possible_col_coord),
        )
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while attempt < max_attempts:
            if self.is_valid_room_position(room, position[0], position[1]):
                self.rooms.append(room)
                room.global_position = position
                self.materialize_room(room, position[0], position[1])
                return True
            else:
                step_size = self.levy_step_size()
                direction = self.py_random.choice(directions)
                position = (
                    position[0] + direction[0] * step_size,
                    position[1] + direction[1] * step_size,
                )
                attempt += 1

        return False

    def place_rooms(self):
        """Randomly generate rooms and attempt to place them in the maze grid."""
        total_room_path_area_covered = 0
        target_path_area = self.global_room_ratio * self.grid.size
        start_avg_room_path_area = target_path_area / self.nr_desired_rooms
        desired_room_path_area = start_avg_room_path_area

        # Because of the desired_room_path_area = max(9, desired_room_path_area),
        # we need to also check for a maximum attempt counter
        attempts = 0
        max_attempts = 100

        room_sizes = []
        while (
            self.nr_placed_rooms < self.nr_desired_rooms
            and total_room_path_area_covered < target_path_area
            and attempts < max_attempts
        ):
            # In some cases when the global ratio is small and we want many rooms,
            # the start_avg is small from the very start so we need to enforce minimum area
            desired_room_path_area = max(9, desired_room_path_area)

            room = self.room_factory.create_room(desired_area=desired_room_path_area)
            room_sizes.append((room.rows, room.cols))
            if self.levy_flight_place(room):
                total_room_path_area_covered += room.rows * room.cols
                self.nr_placed_rooms += 1

            # Each time a room is added, or a fail happens, decrease the desired area of the next room trials
            desired_room_path_area = int(0.9 * desired_room_path_area)
            attempts += 1

    def is_valid_room_position(self, room, start_row, start_col):
        end_row, end_col = start_row + room.rows, start_col + room.cols

        # Extract sub-grid and check boundaries
        if (
            start_row < 1
            or start_col < 1
            or end_row > self.room_grid.shape[0] - 1
            or end_col > self.room_grid.shape[1] - 1
        ):
            return False

        sub_grid = self.room_grid[
            start_row - 1 : end_row + 1, start_col - 1 : end_col + 1
        ]

        # Check if placing the room would overlap with another room or not maintain separation
        return np.sum(sub_grid == PATH) == 0

    def materialize_room(self, room, start_row, start_col):
        self.room_grid[
            start_row : start_row + room.rows, start_col : start_col + room.cols
        ] = room.grid

    def levy_step_size(self):
        r = self.py_random.random()  # r is between 0 and 1
        return int(
            1 / ((r + 0.001) ** 0.5)
        )  # This will give us a step length L according to inverse square distribution.

    ##### Corridor generation #####

    def choose_start_position(self):
        """Choose a random position on the perimeter of the maze grid."""
        side = self.py_random.choice(["top", "bottom", "left", "right"])

        if side == "top":
            return (0, self.py_random.randint(0, self.grid.shape[1] - 1))
        elif side == "bottom":
            return (
                self.grid.shape[0] - 1,
                self.py_random.randint(0, self.grid.shape[1] - 1),
            )
        elif side == "left":
            return (self.py_random.randint(0, self.grid.shape[0] - 1), 0)
        else:  # 'right'
            return (
                self.py_random.randint(0, self.grid.shape[0] - 1),
                self.grid.shape[1] - 1,
            )

    def choose_target_position(self):
        """Choose a random target position in one of the generated rooms."""

        # Shuffle the rooms to ensure random selection without replacement
        shuffled_rooms = self.py_random.sample(self.rooms, len(self.rooms))

        for random_room in shuffled_rooms:
            room_top_left_row, room_top_left_col = random_room.global_position

            # Create a list of all possible cell positions in the room
            all_positions = [
                (row, col)
                for row in range(random_room.rows)
                for col in range(random_room.cols)
            ]
            # Shuffle these positions to randomly select without replacement
            self.py_random.shuffle(all_positions)

            for row, col in all_positions:
                # Target must be on an empty room tile and not in the perimeter
                if (random_room.grid[row, col] == PATH) and (
                    (row, col) not in random_room.get_perimeter_cells()
                ):
                    return (room_top_left_row + row, room_top_left_col + col)

        # If the function hasn't returned by this point, then no valid target position was found in any room
        raise ValueError(f"Could not find a valid target position in any room.")

    def place_start_end_positions(self):
        self.start_position = self.choose_start_position()
        self.target_position = self.choose_target_position()

    def generate_corridor_maze(self):
        grid = np.full((self.rows, self.cols), WALL)
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        def is_valid_position(position, direction):
            new_pos = (position[0] + direction[0], position[1] + direction[1])
            if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                if (
                    grid[new_pos[0], new_pos[1]] == WALL
                    and self.room_grid[new_pos[0], new_pos[1]] == WALL
                ):
                    # Check for room padding
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if (
                                0 <= new_pos[0] + dx < self.rows
                                and 0 <= new_pos[1] + dy < self.cols
                                and self.room_grid[new_pos[0] + dx, new_pos[1] + dy]
                                == PATH
                            ):
                                return False
                    return True
            return False

        # Initialize the starting position
        current_position = self.start_position
        grid[current_position] = PATH

        walls = []

        for d in directions:
            if is_valid_position(current_position, d):
                walls.append((current_position, d))

        while walls:
            # Randomly select a wall from the list
            current_position, direction = self.py_random.choice(walls)

            next_pos = (
                current_position[0] + direction[0],
                current_position[1] + direction[1],
            )

            if grid[next_pos[0], next_pos[1]] == WALL:
                # Carve a passage through the wall to join the two cells
                grid[
                    current_position[0] + direction[0] // 2,
                    current_position[1] + direction[1] // 2,
                ] = PATH
                grid[next_pos[0], next_pos[1]] = PATH

                # Add the neighboring walls of the new cell to the wall list
                for d in directions:
                    if is_valid_position(next_pos, d):
                        walls.append((next_pos, d))

            # Remove the wall from the list
            walls.remove((current_position, direction))

        self.corridor_grid = grid

    def connect_rooms_to_paths(self):
        """Connect the rooms to the corridors via the shortest path."""

        for room in self.rooms:
            for access_point in room.access_points:
                global_access_point = (
                    room.global_position[0] + access_point[0],
                    room.global_position[1] + access_point[1],
                )
                connection = self.bfs_to_find_closest_path(global_access_point)

                if not connection:
                    # If BFS failed to find a connection, try growing a path
                    connection = self.grow_path_from(global_access_point)

                # If we found a connection, draw a path to it
                if connection:
                    self.plot_path_from_to(global_access_point, connection)

    def bfs_to_find_closest_path(self, access_point):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        visited = np.zeros((self.rows, self.cols), dtype=bool)
        queue = deque([access_point])

        # If the access point is already on a corridor path
        if self.corridor_grid[access_point[0], access_point[1]] == PATH:
            return access_point

        while queue:
            current = queue.popleft()

            for d in directions:
                new_pos = (current[0] + d[0], current[1] + d[1])

                # Check boundaries
                if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                    # If the new position is an existing path, we found a connection
                    if self.corridor_grid[new_pos[0], new_pos[1]] == PATH:
                        return new_pos

                    # If not visited, not a room cell, not inside any room (excluding the access point), and not yet part of the corridor
                    if (
                        not visited[new_pos[0], new_pos[1]]
                        and not self.is_inside_any_room(new_pos, exception=access_point)
                        and self.corridor_grid[new_pos[0], new_pos[1]] == WALL
                    ):
                        visited[new_pos[0], new_pos[1]] = True
                        queue.append(new_pos)

        return None

    def plot_path_from_to(self, p1, p2):
        # Directly connect if the cells are adjacent and not diagonally so
        diff_x = abs(p1[0] - p2[0])
        diff_y = abs(p1[1] - p2[1])
        if diff_x <= 1 and diff_y <= 1 and not (diff_x == 1 and diff_y == 1):
            self.corridor_grid[p2[0], p2[1]] = PATH
            return

        while p1 != p2:
            # First adjust horizontally
            if p1[1] != p2[1]:
                p1 = (p1[0], p1[1] + np.sign(p2[1] - p1[1]))
            # Then adjust vertically
            else:
                p1 = (p1[0] + np.sign(p2[0] - p1[0]), p1[1])

            if not self.is_inside_any_room(p1, exception=None):
                self.corridor_grid[p1[0], p1[1]] = PATH

    def is_line_segment_intersecting_room(self, p1, p2):
        """Check if the line segment p1p2 intersects any room cell."""
        for room in self.rooms:
            for i in range(room.rows):
                for j in range(room.cols):
                    if room.grid[i][j] == PATH:
                        global_cell_pos = (
                            room.global_position[0] + i,
                            room.global_position[1] + j,
                        )

                        # Skip if it's the starting point
                        if global_cell_pos == p1:
                            continue

                        if on_line(p1, global_cell_pos, p2):
                            return True
        return False

    def direction_cost(self, current_pos, direction, target_pos):
        new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
        base_cost = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])

        # Penalize if the new position is close to a room.
        if self.is_inside_any_room(new_pos):
            base_cost += 10

        return base_cost

    def grow_path_from(
        self, start_pos, max_attempts=5000
    ):  # Temporarily increased max_attempts
        """Grow a path from start_pos until it reaches a corridor or max_attempts are reached."""

        current_pos = start_pos
        attempt = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        visited = set()
        visited.add(start_pos)

        # Get a list of all corridor positions
        corridor_positions = np.argwhere(self.corridor_grid == PATH)

        while attempt < max_attempts:
            potential_directions = []

            # Get the closest corridor position to the current position
            distances = [
                np.linalg.norm(np.array(current_pos) - np.array(corridor_pos))
                for corridor_pos in corridor_positions
            ]
            closest_corridor_pos = corridor_positions[np.argmin(distances)]

            for d in directions:
                new_pos = (current_pos[0] + d[0], current_pos[1] + d[1])

                if new_pos in visited:
                    continue

                # Check boundaries
                if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
                    continue  # Go to the next direction if this is out of bounds

                # If the new position is an existing path, validate and then return
                if self.corridor_grid[new_pos[0], new_pos[1]] == PATH:
                    diff_x = new_pos[0] - current_pos[0]
                    diff_y = new_pos[1] - current_pos[1]

                    # If this is not diagonal, then validate
                    if not (abs(diff_x) == 1 and abs(diff_y) == 1):
                        if not self.is_line_segment_intersecting_room(
                            start_pos, new_pos
                        ):
                            return new_pos
                    continue  # Go to the next direction

                # If not inside any room and not yet part of the corridor
                if (
                    not self.is_inside_any_room(new_pos)
                    and self.corridor_grid[new_pos[0], new_pos[1]] == WALL
                ):
                    potential_directions.append(
                        (
                            new_pos,
                            self.direction_cost(current_pos, d, closest_corridor_pos),
                        )
                    )

            # Sort the potential directions by their cost
            potential_directions.sort(key=lambda x: x[1])

            if not potential_directions:
                break

            # Select the best direction
            next_pos = potential_directions[0][0]
            visited.add(next_pos)

            # Update the grid
            self.corridor_grid[next_pos[0], next_pos[1]] = PATH
            current_pos = next_pos
            attempt += 1

        # If we reached here, we didn't find a corridor
        return None

    def is_inside_any_room(self, pos, exception=None):
        """Check if a position is inside any of the rooms in the maze."""

        for room in self.rooms:
            # Check if the position is within the global boundaries of the room
            if (
                room.global_position[0] <= pos[0] < room.global_position[0] + room.rows
                and room.global_position[1]
                <= pos[1]
                < room.global_position[1] + room.cols
            ):
                # If the position is the exception, consider it not inside the room
                if exception and pos == exception:
                    continue
                # Otherwise, it's inside this room
                return True

        # If we reached here, the position isn't inside any room
        return False

    #### Validate maze ####
    def is_valid_maze(self):
        """Validate if all access points are reachable from the starting point.
        Sanity check function, was not included in the constructor but can be used after generation.
        """
        start_point = self.start_position
        visited = set([start_point])
        queue = deque([start_point])

        while queue:
            current_point = queue.popleft()
            for neighbor in self.get_neighbors(*current_point):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check if all PATH cells have been visited
        all_path_cells = {
            (x, y)
            for x in range(self.rows)
            for y in range(self.cols)
            if self.grid[x][y] == PATH
        }

        return visited == all_path_cells

    def get_neighbors(self, x, y):
        """Get neighboring PATH cells of a given cell (x, y)."""
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.rows
                and 0 <= ny < self.cols
                and self.grid[nx][ny] == PATH
            ):
                neighbors.append((nx, ny))
        return neighbors
