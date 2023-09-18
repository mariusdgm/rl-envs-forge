import numpy as np

from rl_envs_forge.envs.labyrinth.maze.room import TriangleRoom
from rl_envs_forge.envs.labyrinth.constants import *


class TestTriangleRoom:
    def test_default_triangle(self):
        room = TriangleRoom()
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [WALL, WALL, WALL, PATH],
                    [WALL, WALL, PATH, PATH],
                    [WALL, PATH, PATH, PATH],
                    [PATH, PATH, PATH, PATH],
                ]
            ),
        )

    def test_top_right_triangle(self):
        room = TriangleRoom(corner="top_right")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [PATH, WALL, WALL, WALL],
                    [PATH, PATH, WALL, WALL],
                    [PATH, PATH, PATH, WALL],
                    [PATH, PATH, PATH, PATH],
                ]
            ),
        )

    def test_down_left_triangle(self):
        room = TriangleRoom(corner="down_left")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [PATH, PATH, PATH, PATH],
                    [WALL, PATH, PATH, PATH],
                    [WALL, WALL, PATH, PATH],
                    [WALL, WALL, WALL, PATH],
                ]
            ),
        )

    def test_down_right_triangle(self):
        room = TriangleRoom(corner="down_right")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [PATH, PATH, PATH, PATH],
                    [PATH, PATH, PATH, WALL],
                    [PATH, PATH, WALL, WALL],
                    [PATH, WALL, WALL, WALL],
                ]
            ),
        )

    def test_8_4_triangle_top_left(self):
        room = TriangleRoom(rows=8, cols=4, corner="top_left")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
        )

    def test_8_4_triangle_top_right(self):
        room = TriangleRoom(rows=8, cols=4, corner="top_right")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                ]
            ),
        )

    def test_4_8_triangle_down_left(self):
        room = TriangleRoom(rows=4, cols=8, corner="down_left")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ]
            ),
        )

    def test_4_8_triangle_down_right(self):
        room = TriangleRoom(rows=4, cols=8, corner="down_right")
        grid = room.generate_room_layout()
        assert np.array_equal(
            grid,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )
