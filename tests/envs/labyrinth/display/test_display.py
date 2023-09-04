import pytest
from unittest.mock import Mock, patch

from rl_envs_forge.envs.labyrinth.display.display import EnvDisplay
from rl_envs_forge.envs.labyrinth.constants import WALL, PATH


@pytest.fixture
def mock_pygame():
    with patch("rl_envs_forge.envs.labyrinth.display.display.pygame") as mock:
        yield mock


@pytest.fixture
def env_display(mock_pygame):
    return EnvDisplay(5, 5, labyrinth=Mock())


class TestEnvDisplay:
    def test_initialization(self, env_display):
        assert env_display.rows == 5
        assert env_display.cols == 5
        assert env_display.window_width == 800
        assert env_display.window_height == 800

    def test_compute_sizes_and_paddings(self, env_display):
        env_display._compute_sizes_and_paddings()
        assert env_display.cell_width == (800 - 2 * EnvDisplay.BORDER_PADDING) // 5
        assert env_display.cell_height == (800 - 2 * EnvDisplay.BORDER_PADDING) // 5

    def test_adjust_coords_for_padding(self, env_display): 
        x, y = env_display._adjust_coords_for_padding(1, 1)
        expected_x = env_display.additional_padding_x + env_display.cell_width
        expected_y = env_display.additional_padding_y + env_display.cell_height
        assert x == expected_x
        assert y == expected_y

    def test_compute_sizes_and_paddings_additional_paddings(self, env_display):
        env_display._compute_sizes_and_paddings()
        
        total_cell_width = round(env_display.cell_width * env_display.cols)
        total_cell_height = round(env_display.cell_height * env_display.rows)
        
        expected_padding_x = (
            env_display.window_width - total_cell_width - 2 * EnvDisplay.BORDER_PADDING
        )
        expected_padding_y = (
            env_display.window_height - total_cell_height - 2 * EnvDisplay.BORDER_PADDING
        )

        assert env_display.additional_padding_x == expected_padding_x
        assert env_display.additional_padding_y == expected_padding_y
