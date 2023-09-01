import pytest
import numpy as np
from rl_envs.envs.labyrinth.labyrinth import Labyrinth
from rl_envs.envs.labyrinth.constants import *

@pytest.fixture
def labyrinth():
    return Labyrinth(31, 31)

def test_initialization(labyrinth):
    assert labyrinth.state is not None
    assert labyrinth.player is not None
    assert labyrinth.maze is not None

def test_valid_move(labyrinth):
    initial_position = labyrinth.player.position
    labyrinth.step(Action.RIGHT)  # Assuming right is a valid move
    assert labyrinth.player.position != initial_position

def test_invalid_move(labyrinth):
    initial_position = labyrinth.player.position
    # Assuming UP leads to a wall or out of the grid
    labyrinth.step(Action.UP)
    assert labyrinth.player.position == initial_position

def test_reaching_target(labyrinth):
    # Set the player's position near the target for simplicity
    labyrinth.player.position = (labyrinth.maze.target_position[0] - 1, labyrinth.maze.target_position[1])
    _, reward, done, _, _ = labyrinth.step(Action.DOWN)  # Assuming down leads to target
    assert reward == labyrinth.reward_schema["target_reached_reward"]
    assert done

def test_reset(labyrinth):
    initial_state = labyrinth.state.copy()
    labyrinth.reset()
    assert not np.array_equal(labyrinth.state, initial_state)

def test_seeding():
    env1 = Labyrinth(31, 31, seed=42)
    env2 = Labyrinth(31, 31, seed=42)
    assert np.array_equal(env1.state, env2.state)