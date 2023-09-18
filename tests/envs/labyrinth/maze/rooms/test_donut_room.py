import pytest
import numpy as np

from rl_envs_forge.envs.labyrinth.maze.room import DonutRoom
from rl_envs_forge.envs.labyrinth.constants import *


class TestDonutRoom:
    @pytest.fixture
    def donut_room(self):
        room = DonutRoom(rows=7, cols=7)
        return room

    def test_room_layout(self, donut_room):
        """Test if the room layout is correctly generated."""
        expected_outer = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        assert np.array_equal(donut_room.grid, expected_outer)

    def test_generate_inner_area_mask(self, donut_room):
        """Test if the inner area mask is correctly generated."""
        mask = donut_room.generate_inner_area_mask()
        assert mask.shape == (7, 7)
        assert np.all(mask >= 0)
        assert np.all(mask <= 1)

    def test_min_dims(self):
        """Test if the minimum dimensions are correctly calculated."""
        room = DonutRoom(rows=3, cols=3)
        assert room.rows == 7
        assert room.cols == 7

    def test_rectangle_outer_rectangle_inner(self):
        room = DonutRoom(
            rows=7, cols=7, inner_shape="rectangle", outer_shape="rectangle"
        )
        assert room.outer_shape == "rectangle"
        assert room.inner_shape == "rectangle"
        assert len(room.get_perimeter_cells()) > 1

        non_perimeter_mask = room.get_non_perimeter_inner_cells()
        path_mask = room.grid == PATH
        combined_mask = non_perimeter_mask * path_mask
        assert np.sum(combined_mask) > 1

    def test_oval_outer_rectangle_inner(self):
        room = DonutRoom(
            rows=7, cols=7, inner_shape="rectangle", outer_shape="oval"
        )
        assert room.outer_shape == "oval"
        assert room.inner_shape == "rectangle"
        assert len(room.get_perimeter_cells()) > 1

        non_perimeter_mask = room.get_non_perimeter_inner_cells()
        path_mask = room.grid == PATH
        combined_mask = non_perimeter_mask * path_mask
        assert np.sum(combined_mask) > 1

    def test_rectangle_outer_oval_inner(self):
        room = DonutRoom(
            rows=7, cols=7, inner_shape="oval", outer_shape="rectangle"
        )
        assert room.outer_shape == "rectangle"
        assert room.inner_shape == "oval"
        assert len(room.get_perimeter_cells()) > 1

        non_perimeter_mask = room.get_non_perimeter_inner_cells()
        path_mask = room.grid == PATH
        combined_mask = non_perimeter_mask * path_mask
        assert np.sum(combined_mask) > 1