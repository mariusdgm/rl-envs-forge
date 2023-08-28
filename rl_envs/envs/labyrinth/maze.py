import numpy as np
import random

from .room import RectangularRoom

WALL = 0
PATH = 1

class Maze:
    def __init__(self, rows, cols, num_rooms=3, padding=1, global_room_ratio=0.5, max_room_generate_retries=10, max_reposition_retries=10, seed=None):
        if seed:
            random.seed(seed)
        self.rows = rows
        self.cols = cols
        self.num_rooms = num_rooms
        self.padding = padding
        self.grid = [[WALL for _ in range(cols)] for _ in range(rows)]
        self.rooms = []
        self.global_room_ratio = global_room_ratio
        self.max_room_generate_retries = max_room_generate_retries
        self.max_reposition_retries = max_reposition_retries

        self.generate_rooms()

    def generate_rooms(self):
        room_ratios = self.generate_room_ratios()
      
        total_maze_area = self.rows * self.cols

        for i in range(self.num_rooms):
            desired_room_area = room_ratios[i] * total_maze_area
            self.generate_and_place_room(desired_room_area)
        

    def generate_and_place_room(self, desired_room_area):
        # Outer retry loop for regenerating a room
        for _ in range(self.max_room_generate_retries):
            room = RectangularRoom(area=desired_room_area, ratio=random.uniform(0.2, 1.8))
            
            success = False  # Inner retry loop success flag
            placement_retries = 0
            # Inner retry loop for repositioning the same room
            while not success and placement_retries < self.max_reposition_retries:
                success = self.place_room(room)
                if success:
                    self.rooms.append(room)
                    return True
                placement_retries += 1
            
    def place_room(self, room):
        # This function places a room on the grid ensuring there's a padding.
        room_rows, room_cols = room.get_shape()
        start_row = random.randint(0, self.rows - room_rows - self.padding)
        start_col = random.randint(0, self.cols - room_cols - self.padding)

        # Check if the room can be placed without overlapping another room
        can_place = True
        for r in range(start_row - self.padding, start_row + room_rows + self.padding):
            for c in range(start_col - self.padding, start_col + room_cols + self.padding):
                if self.grid[r][c]:
                    can_place = False
                    break
            if not can_place:
                break

        if can_place:
            for r in range(start_row, start_row + room_rows):
                for c in range(start_col, start_col + room_cols):
                    self.grid[r][c] = PATH
            
            return True
        
        return False

    def generate_room_ratios(self):
        total_ratio = self.global_room_ratio
        # Ensure the given constraints make sense
        if total_ratio < self.num_rooms * 0.05:
            raise ValueError("The total ratio is too small for the given number of rooms.")

        # Initialize variables
        min_room_ratio = max(0.05, total_ratio / (self.num_rooms * 2))
        max_room_ratio = total_ratio - min_room_ratio * (self.num_rooms - 1)
        individual_ratios = []

        # Distribute ratios for all rooms except the last one
        for _ in range(self.num_rooms - 1):
            current_ratio = random.uniform(min_room_ratio, max_room_ratio)
            individual_ratios.append(current_ratio)

            # Update total_ratio and max_room_ratio for next iteration
            total_ratio -= current_ratio
            max_room_ratio = total_ratio - min_room_ratio * (self.num_rooms - len(individual_ratios) - 1)

        # Assign remaining ratio to the last room
        individual_ratios.append(total_ratio)

        return individual_ratios

    def display(self):
        for row in self.grid:
            print("".join(["#" if cell else "." for cell in row]))

    def _generate_corridors(self):
        # This method should generate corridors similar to your previous logic
        # ... (The original logic for corridors)
        pass


if __name__ == "__main__":
    # Testing the Maze with RectangularRoom integration
    m = Maze(50, 50, seed=42)
    m.display()
