import numpy as np

from rl_envs_forge.envs.labyrinth.maze.room import LShapeRoom
from rl_envs_forge.envs.labyrinth.constants import *

class TestLShapeRoom:
    def test_default_lshape(self):
        room = LShapeRoom()
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        )

    def test_top_right_lshape(self):
        room = LShapeRoom(corner="top_right")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        )

    def test_bottom_left_lshape(self):
        room = LShapeRoom(corner="bottom_left")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
        )

    def test_bottom_right_lshape(self):
        room = LShapeRoom(corner="bottom_right")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
        )

    def test_min_carve_ratios_min_size(self):
        room = LShapeRoom(vertical_carve=0, horizontal_carve=0)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 4

    def test_max_carve_ratios_min_size(self):
        room = LShapeRoom(vertical_carve=1, horizontal_carve=1)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 4

    def test_6_4_ratio_075(self):
        room = LShapeRoom(rows=6, cols=4, vertical_carve=0.75, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 6

    def test_6_4_ratio_1(self):
        room = LShapeRoom(rows=6, cols=4, vertical_carve=1, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 8

    def test_6_4_ratio_075_bottom_right(self):
        room = LShapeRoom(rows=6, cols=4, corner="bottom_right", vertical_carve=0.75, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 6

    def test_6_4_ratio_1_bottom_right(self):
        room = LShapeRoom(rows=6, cols=4, corner="bottom_right", vertical_carve=1, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 8