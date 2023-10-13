import pytest
from unittest.mock import patch, MagicMock

from rl_envs_forge.envs.labyrinth.constants import Action
from rl_envs_forge.envs.labyrinth.display.player import PlayerDisplayer


class TestPlayerDisplayer:
    @pytest.fixture
    def mock_sprite_sheet(self):
        mock_sheet = MagicMock()
        mock_sheet.get_height.return_value = 2
        mock_sheet.get_width.return_value = 2
        mock_sheet.subsurface.return_value = MagicMock()
        return mock_sheet

    @pytest.fixture
    def player_displayer(self, mock_sprite_sheet):
        with patch(
            "rl_envs_forge.envs.labyrinth.display.player.pygame.image.load",
            return_value=mock_sprite_sheet,
        ):
            entity_mock = MagicMock()
            entity_mock.moving = False
            entity_mock.face_orientation = Action.LEFT
            return PlayerDisplayer(entity_mock)

    def test_get_sprite_idle(self, player_displayer):
        sprite = player_displayer.get_sprite()
        assert sprite == player_displayer.idle_sprite

    def test_get_sprite_moving(self, player_displayer):
        player_displayer.entity.moving = True
        sprite = player_displayer.get_sprite()
        assert sprite == player_displayer.moving_sprite

    @patch("rl_envs_forge.envs.labyrinth.display.player.pygame.transform.flip")
    def test_get_sprite_flipped(self, mock_flip, player_displayer):
        player_displayer.entity.face_orientation = Action.RIGHT
        player_displayer.get_sprite()
        mock_flip.assert_called_once()
