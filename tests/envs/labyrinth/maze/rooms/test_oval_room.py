import pytest


from rl_envs_forge.envs.labyrinth.maze.room import OvalRoom
from rl_envs_forge.envs.labyrinth.maze.utils import is_perimeter
from rl_envs_forge.envs.labyrinth.constants import *


class TestOvalRoom:
    def test_default_oval_room(self):
        room = OvalRoom()
        perimeter_cells = room.get_perimeter_cells()
        for cell in perimeter_cells:
            assert is_perimeter(room.grid, cell)

    def test_large_oval_room(self):
        room = OvalRoom(rows=10, cols=10)
        perimeter_cells = room.get_perimeter_cells()
        for cell in perimeter_cells:
            assert is_perimeter(room.grid, cell)

    def test_5x5_padding_1(self):
        room = OvalRoom(rows=5, cols=5)
        expected_perimeter = {
            (-1, 2),
            (0, 1),
            (0, 3),
            (1, 0),
            (1, 4),
            (2, -1),
            (2, 5),
            (3, 0),
            (3, 4),
            (4, 1),
            (4, 3),
            (5, 2),
        }
        perimeter_cells = room.get_perimeter_cells(padding=1)
        assert set(perimeter_cells) == expected_perimeter

    def test_5x5_padding_2(self):
        room = OvalRoom(rows=5, cols=5)
        expected_perimeter = {
            (-2, 2),
            (-1, 1),
            (-1, 3),
            (0, 0),
            (0, 4),
            (1, -1),
            (1, 5),
            (2, -2),
            (2, 6),
            (3, -1),
            (3, 5),
            (4, 0),
            (4, 4),
            (5, 1),
            (5, 3),
            (6, 2),
        }
        perimeter_cells = room.get_perimeter_cells(padding=2)
        assert set(perimeter_cells) == expected_perimeter

    def test_non_square_room(self):
        room = OvalRoom(rows=7, cols=5)
        perimeter_cells = room.get_perimeter_cells()
        for cell in perimeter_cells:
            assert is_perimeter(room.grid, cell)
