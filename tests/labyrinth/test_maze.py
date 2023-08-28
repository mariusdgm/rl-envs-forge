import pytest
import numpy as np
from rl_envs.envs.labyrinth.maze import Maze
from rl_envs.envs.labyrinth.constants import WALL, PATH

def test_maze_initialization():
    m = Maze(50, 50, num_rooms=5)
    assert m.grid.shape == (50, 50)
    assert len(m.compartments) == 5

def test_room_placement_within_bounds():
    m = Maze(50, 50, num_rooms=5)
    for compartment in m.compartments:
        room = compartment.room
        if room is not None:
            # Ensure all rows and columns of the room are within maze bounds
            assert room.shape[0] <= compartment.rows
            assert room.shape[1] <= compartment.cols

def test_rooms_dont_overlap():
    m = Maze(50, 50, num_rooms=5)
    all_room_coords = []

    for compartment in m.compartments:
        room = compartment.room
        room_coords = set()
        start_row = compartment.start_row + 1
        start_col = compartment.start_col + 1
        for row in range(room.shape[0]):
            for col in range(room.shape[1]):
                if m.grid[start_row + row][start_col + col] == PATH:
                    coord = (start_row + row, start_col + col)
                    room_coords.add(coord)
        all_room_coords.append(room_coords)
    
    # Check overlapping coordinates between rooms
    for i in range(len(all_room_coords)):
        for j in range(i+1, len(all_room_coords)):
            overlapping_coords = all_room_coords[i].intersection(all_room_coords[j])
            assert len(overlapping_coords) == 0

def test_maze_display():
    m = Maze(50, 50, num_rooms=5)
    display_output = m.display()
    assert "#" in display_output  # The display output should contain walls
    assert "." in display_output  # The display output should contain paths