import pytest
import numpy as np
from rl_envs.envs.labyrinth.room import RectangularRoom
from rl_envs.envs.labyrinth.constants import *


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
        # ... additional perimeter checks if necessary.

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