import pytest
import numpy as np
from rl_envs_forge.envs.labyrinth.room import RectangularRoom
from rl_envs_forge.envs.labyrinth.constants import *


class TestRectangularRoom:
    def test_generate_room_layout(self):
        room = RectangularRoom(rows=5, cols=5)
        assert room.rows == 5
        assert room.cols == 5
        assert room.shape == (5, 5)
        assert np.all(room.grid == PATH)

    def test_get_perimeter_cells(self):
        room = RectangularRoom(rows=5, cols=5)
        perimeter = room.get_perimeter_cells()
        assert (0, 0) in perimeter
        assert (4, 4) in perimeter

    def test_set_access_points(self):
        room = RectangularRoom(rows=5, cols=5, nr_access_points=3)
        assert len(room.access_points) == 3
        for point in room.access_points:
            assert point in room.get_perimeter_cells()

    def test_room_area(self):
        room = RectangularRoom(rows=5, cols=5)
        assert room.area == 25

    def test_global_position(self):
        room = RectangularRoom(rows=5, cols=5)
        assert room.global_position == (0, 0)
        room.global_position = (1, 1)
        assert room.global_position == (1, 1)

    def test_generate_inner_area_mask(self):
        # Given a RectangularRoom of size 5x5.
        room = RectangularRoom(rows=5, cols=5)
        expected_mask = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        # When we get the inner area mask.
        inner_area_mask = room.generate_inner_area_mask()

        # Then the inner area mask should match the expected mask.
        assert np.array_equal(inner_area_mask, expected_mask)

    def test_get_perimeter_cells_with_padding_2(self):
        room = RectangularRoom(rows=3, cols=4)
        perimeter_cells = set(room.get_perimeter_cells(padding=2))
        
        expected_perimeter = set()

        # Top and bottom with padding
        for col in range(-2, room.cols + 2):
            expected_perimeter.add((-2, col))
            expected_perimeter.add((room.rows + 1, col))
            
        # Left and right with padding (excluding the corners)
        for row in range(1 - 2, room.rows - 1 + 2):
            expected_perimeter.add((row, -2))
            expected_perimeter.add((row, room.cols + 1))
        
        assert perimeter_cells == expected_perimeter
