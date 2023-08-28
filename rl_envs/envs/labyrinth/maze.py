import numpy as np
import random
from .room import RoomFactory
from .constants import WALL, PATH


class Maze:
    def __init__(
        self,
        rows,
        cols,
        nr_desired_rooms=3,
        global_room_ratio=0.5,
        min_room_rows=5,
        min_room_cols=5,
    ):
        self.grid = np.ones((rows, cols), dtype=int) * WALL
        self.nr_desired_rooms = nr_desired_rooms
        self.nr_placed_rooms = 0 # in some cases we won't be able to place all rooms we want
        self.global_room_ratio = global_room_ratio
        self.min_room_rows = min_room_rows
        self.min_room_cols = min_room_cols
        self.placed_rooms = []
        self.place_rooms()

    def is_valid_room_position(self, room, start_row, start_col):
        end_row, end_col = start_row + room.rows, start_col + room.cols

        # Extract sub-grid and check boundaries
        if start_row < 1 or start_col < 1 or end_row > self.grid.shape[0]-1 or end_col > self.grid.shape[1]-1:
            return False
        
        sub_grid = self.grid[start_row-1:end_row+1, start_col-1:end_col+1]
        
        # Check if placing the room would overlap with another room or not maintain separation
        return np.sum(sub_grid == PATH) == 0

    def place_room(self, room, start_row, start_col):
        self.grid[start_row: start_row + room.rows, start_col: start_col + room.cols] = room.grid

    def levy_step_size(self):
        r = random.random()  # r is between 0 and 1
        return int(
            1 / (r**0.5)
        )  # This will give us a step length L according to inverse square distribution.

    def levy_flight_place(self, room):
        max_attempts = 100
        attempt = 0
        position = (
            random.randint(0, self.grid.shape[0] - 1),
            random.randint(0, self.grid.shape[1] - 1),
        )
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while attempt < max_attempts:
            if self.is_valid_room_position(room, position[0], position[1]):
                self.place_room(room, position[0], position[1])
                return True
            else:
                step_size = self.levy_step_size()
                direction = random.choice(directions)
                position = (
                    position[0] + direction[0] * step_size,
                    position[1] + direction[1] * step_size,
                )
                attempt += 1

        return False

    def place_rooms(self):
        total_room_path_area_covered = 0
        target_path_area = self.global_room_ratio * self.grid.size
        start_avg_room_path_area = target_path_area / self.nr_desired_rooms
        desired_room_path_area = start_avg_room_path_area

        room_factory = RoomFactory()

        while (
            self.nr_placed_rooms < self.nr_desired_rooms
            and total_room_path_area_covered < target_path_area
        ):
            room = room_factory.create_room(desired_area=desired_room_path_area)

            if self.levy_flight_place(
                room
            ):  # Use levy_flight_place instead of random_walk_place
                total_room_path_area_covered += room.rows * room.cols
                self.nr_placed_rooms += 1

            # Each time a room is added, or a fail happens, decrease the desired area of the next room trials
            desired_room_path_area = int(0.9 * desired_room_path_area)

            # Optional: Add a condition to break the loop if total_path_area exceeds some threshold.
            # This prevents infinite loops if it's impossible to fit any more rooms.
            if (
                total_room_path_area_covered >= self.grid.size * self.global_room_ratio
            ) or (desired_room_path_area <= 9):
                break

