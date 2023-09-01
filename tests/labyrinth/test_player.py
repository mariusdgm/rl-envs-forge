import pytest
from rl_envs.envs.labyrinth.constants import Action
from rl_envs.envs.labyrinth.player import Player  # Adjust the import based on your folder structure

def test_player_potential_next_position():
    player = Player(start_position=(5, 5))

    # Test Up movement
    assert player.potential_next_position(Action.UP) == (4, 5)
    
    # Test Right movement
    assert player.potential_next_position(Action.RIGHT) == (5, 6)

    # Test Down movement
    assert player.potential_next_position(Action.DOWN) == (6, 5)

    # Test Left movement
    assert player.potential_next_position(Action.LEFT) == (5, 4)

    # Test invalid movement
    class InvalidAction:
        """Dummy class to emulate an invalid Action."""
        pass

    assert player.potential_next_position(InvalidAction()) == (5, 5)