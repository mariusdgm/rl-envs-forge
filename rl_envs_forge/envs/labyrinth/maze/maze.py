import numpy as np
import random
from collections import deque
from typing import List, Tuple, Union, Optional
from enum import Enum

from .room import RoomFactory
from .corridor import CorridorBuilder
from ..constants import WALL, PATH, START, TARGET, CorridorMoveStatus

from ...common.grid_functions import on_line


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
        corridor_algorithm: Optional[str] = "random",
        corridor_grid_connect_option: Optional[Union[bool, str]] = "random",
        corridor_post_process_option: bool = True,
        corridor_sort_access_points_option: Optional[Union[bool, str]] = "random",
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
            corridor_algorithm (str, optional): The algorithm to use for generating the maze. Defaults to 'random'. Can be one of: 'prim', 'astar', 'gbfs' or 'random'. if 'random', randomly selects algorithm.
            corridor_grid_connect_option (Union[bool, str], optional): Option to decide the nature of corridor paths connectivity.
                - If `True`, corridors will be grid-connected.
                - If `False`, corridors will not be grid-connected.
                - If `"random"`, the choice will be made randomly.
                Defaults to False.
            corridor_post_process_option (bool, optional): Whether to post-process the maze. Defaults to True.
            corridor_sort_access_points_option (Union[bool, str], str, optional): Whether to sort the access points. Defaults to "random". If "random" will randomly select between True and False.
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

        self.corridor_algorithm = corridor_algorithm
        self._check_corridor_algorithm_option()

        self.corridor_grid_connect_option = corridor_grid_connect_option
        self._check_corridor_grid_connect_option()

        self.corridor_post_process_option = corridor_post_process_option

        self.corridor_sort_access_points_option = corridor_sort_access_points_option
        self._check_corridor_sort_access_points_option()

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

        # Randomize or fix other parameters
        self._check_corridor_algorithm_option()
        if self.corridor_algorithm == "random":
            corridor_algorithm = self.py_random.choice(["prim", "astar", "gbfs"])
        else:
            corridor_algorithm = self.corridor_algorithm

        self._check_corridor_grid_connect_option()
        if self.corridor_grid_connect_option == "random":
            corridor_grid_connect_option = self.py_random.choice([True, False])
        else:
            corridor_grid_connect_option = self.corridor_grid_connect_option

        self._check_corridor_sort_access_points_option()
        if self.corridor_sort_access_points_option == "random":
            corridor_sort_access_points_option = self.py_random.choice([True, False])
        else:
            corridor_sort_access_points_option = self.corridor_sort_access_points_option

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
            corridor_algorithm=corridor_algorithm,
            corridor_grid_connect_option=corridor_grid_connect_option,
            corridor_post_process_option=self.corridor_post_process_option,
            corridor_sort_access_points_option=corridor_sort_access_points_option,
            seed=maze_seed,
        )
        return maze

    def _check_corridor_grid_connect_option(self):
        if self.corridor_grid_connect_option not in [True, False, "random"]:
            raise ValueError(
                f"Invalid value for corridor_grid_connect_option. Got {self.corridor_grid_connect_option}, expected [True, False, 'random']."
            )

    def _check_corridor_algorithm_option(self):
        if self.corridor_algorithm not in ["prim", "astar", "gbfs", "random"]:
            raise ValueError(
                f"Invalid value for corridor_algorithm. Got {self.corridor_algorithm}, expected ['prim', 'astar', 'gbfs', 'random']."
            )

    def _check_corridor_sort_access_points_option(self):
        if self.corridor_sort_access_points_option not in [True, False, "random"]:
            raise ValueError(
                f"Invalid value for corridor_sort_access_points_option. Got {self.corridor_sort_access_points_option}, expected [True, False, 'random']."
            )


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
        corridor_algorithm: Optional[str] = "prim",
        corridor_grid_connect_option: Optional[bool] = False,
        corridor_post_process_option: Optional[bool] = True,
        corridor_sort_access_points_option: Optional[bool] = False,
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
            algorithm (str, optional): The algorithm to use for generating the maze. Defaults to 'prim'. Can be one of: 'prim', 'astar'.
            corridor_grid_connect_option (bool, optional): Option to decide the nature of corridor paths connectivity.
                - If `True`, corridors will be grid-connected.
                - If `False`, corridors will not be grid-connected.
                Defaults to False. Only affects 'prim' algorithm.
            corridor_grid_connect_option (bool, optional): Whether to post-process the maze. Defaults to True. Only affects 'prim' algorithm.
            corridor_sort_access_points_option (bool, optional): Whether to sort the access points in corridor generation. Defaults to False.
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

        # Corridor generationg settings
        self.corridor_algorithm = corridor_algorithm
        self.corridor_grid_connect_option = corridor_grid_connect_option
        self.corridor_post_process_option = corridor_post_process_option
        self.corridor_sort_access_points_option = corridor_sort_access_points_option

        # Make Corridor Builder
        self.corridor_builder = CorridorBuilder(self)

        self._build_maze()

    def _build_maze(self):
        self.make_rooms()
        self.grid = np.where(self.room_grid == PATH, PATH, self.grid)

        self.make_corridors()
        self.grid = np.where(self.corridor_grid == PATH, PATH, self.grid)

    def make_corridors(self):
        self.place_start_end_positions()

        if self.corridor_algorithm == "prim":
            self.corridor_grid = self.corridor_builder.generate_corridor_prim()
            self.corridor_builder.connect_rooms_to_paths()
            if self.corridor_post_process_option:
                self.corridor_builder.post_process_maze()

        elif self.corridor_algorithm == "astar":
            self.corridor_grid = self.corridor_builder.generate_corridor_a_star()

        elif self.corridor_algorithm == "gbfs":
            self.corridor_grid = self.corridor_builder.generate_corridor_gbfs()

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

    def make_rooms(self):
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

        # Basically add a padding of 1
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

        # Preference to place the target in a non perimeter cell
        for random_room in shuffled_rooms:
            room_top_left_row, room_top_left_col = random_room.global_position

            # Create a list of all possible cell positions in the room
            non_perimeter_mask = random_room.get_non_perimeter_inner_cells()
            path_mask = random_room.grid == PATH
            combined_mask = non_perimeter_mask * path_mask
            candidates = np.argwhere(combined_mask == 1).tolist()

            # Shuffle these positions to randomly select without replacement
            self.py_random.shuffle(candidates)

            if len(candidates) > 0:
                row, col = candidates[0]
                return (room_top_left_row + row, room_top_left_col + col)
            
        # if no such cel could not be found, place target on a non access perimeter
        for random_room in shuffled_rooms:
            perimeter_cells = random_room.get_perimeter_cells()
            non_acces_perimeter_cells = [cell for cell in perimeter_cells if cell not in random_room.access_points]
            if len(non_acces_perimeter_cells) > 0:
                self.py_random.shuffle(non_acces_perimeter_cells)
                return non_acces_perimeter_cells[0]         

        raise ValueError(f"Could not find a valid target position in any room.")

    def place_start_end_positions(self):
        self.start_position = self.choose_start_position()
        self.target_position = self.choose_target_position()
        # self.target_position = self.py_random.choice(self.rooms).global_position

    def is_inside_any_room(self, pos, exception=None):
        """Check if a position is inside any of the rooms in the maze."""
        for room in self.rooms:
            room_mask = room.generate_inner_area_mask()

            # Convert the local room position to a global position
            global_row = pos[0] - room.global_position[0]
            global_col = pos[1] - room.global_position[1]

            # Check if the global position is within the bounds of the room's grid
            if (0 <= global_row < room_mask.shape[0]) and (
                0 <= global_col < room_mask.shape[1]
            ):
                # Check if the position is inside the room using the inner area mask
                if room_mask[global_row, global_col] == 1:
                    # If the position is the exception, consider it not inside the room
                    if exception and pos == exception:
                        continue
                    # Otherwise, it's inside this room
                    return True
        # If we reached here, the position isn't inside any room
        return False

    def generate_global_room_mask(self):
        global_room_mask = np.zeros((self.rows, self.cols), dtype=bool)

        for room in self.rooms:
            local_mask = room.generate_inner_area_mask()
            global_position = room.global_position

            for i in range(local_mask.shape[0]):
                for j in range(local_mask.shape[1]):
                    global_room_mask[
                        global_position[0] + i, global_position[1] + j
                    ] |= local_mask[i, j]

        return global_room_mask

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
