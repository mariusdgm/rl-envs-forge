import numpy as np

from rl_envs_forge.envs.labyrinth.maze.room import TShapeRoom
from rl_envs_forge.envs.labyrinth.constants import *


class TestTShapeRoom:
    def test_default_tshape(self):
        room = TShapeRoom()
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0]])
        )

    def test_left_tshape(self):
        room = TShapeRoom(rotation="left")
        grid = room.generate_room_layout()
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]])
        )

    def test_right_tshape(self):
        room = TShapeRoom(rotation="right")
        grid = room.generate_room_layout()
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]])
        )

    def test_down_tshape(self):
        room = TShapeRoom(rotation="down")
        grid = room.generate_room_layout()
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid, np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        )

    def test_min_carve_ratios_min_size(self):
        room = TShapeRoom(vertical_carve=0, horizontal_carve=0)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 4

    def test_max_carve_ratios_min_size(self):
        room = TShapeRoom(vertical_carve=1, horizontal_carve=1)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 4

    def test_6_4_ratio_075(self):
        room = TShapeRoom(rows=6, cols=4, vertical_carve=0.75, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 6

    def test_6_4_ratio_1(self):
        room = TShapeRoom(rows=6, cols=4, vertical_carve=1, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 8

    def test_6_4_ratio_075_left(self):
        room = TShapeRoom(rows=6, cols=4, rotation="left", vertical_carve=0.75, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 8

    def test_6_4_ratio_1_left(self):
        room = TShapeRoom(rows=6, cols=4,rotation="left", vertical_carve=1, horizontal_carve=0.5)
        grid = room.generate_room_layout()
        assert np.sum(grid == WALL) == 8
