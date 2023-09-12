import pytest
from rl_envs_forge.envs.labyrinth.constants import Action
from rl_envs_forge.envs.labyrinth.entities.player import Player 

class TestPlayer:
    def test_player_potential_next_position(self):
        player = Player(start_position=(5, 5))

        assert player.potential_next_position(Action.UP) == (4, 5)
        assert player.potential_next_position(Action.RIGHT) == (5, 6)
        assert player.potential_next_position(Action.DOWN) == (6, 5)
        assert player.potential_next_position(Action.LEFT) == (5, 4)
        class InvalidAction:
            """Dummy class to emulate an invalid Action."""
            pass

        assert player.potential_next_position(InvalidAction()) == (5, 5)

    def test_move_render_position(self):
        player = Player((5, 5))
        player.position = (6, 7)
        
        # The player's rendered position should move closer to (6, 7) but not reach it in one step.
        player.move_render_position()
        assert player._positions_are_close(player.rendered_position, (5.25, 5.5), threshold=0.1)

        # If we call it a few more times, it should get closer but not fully reach the target position.
        for _ in range(10):
            player.move_render_position()
        assert player._positions_are_close(player.rendered_position, (6, 7), threshold=0.1)


    def test_positions_are_close(self):
        player = Player((2, 3))

        assert player._positions_are_close((2, 3), (2.01, 3.01))  # True
        assert not player._positions_are_close((2, 3), (2.2, 3.2))  # False
        assert not player._positions_are_close((2, 3), (3, 4))  # False