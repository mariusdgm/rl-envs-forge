import pytest
from rl_envs_forge.envs.grid_world.grid_world import (
    GridWorld,
    Action,
)  # Replace with the actual import path


@pytest.fixture
def default_env():
    """Fixture to create a default GridWorld environment for each test."""
    return GridWorld()


class TestGridWorld:
    def test_initialization(self, default_env):
        assert default_env.rows == 5
        assert default_env.cols == 5
        assert default_env.start_state == (0, 0)

    def test_actions(self, default_env):
        new_state, _, _, _, _ = default_env.step(Action.DOWN)
        assert new_state == (1, 0)

    def test_rewards(self, default_env):
        _, reward, _, _, _ = default_env.step(Action.RIGHT)
        assert reward == -0.1

    def test_reset(self, default_env):
        default_env.step(Action.DOWN)
        state = default_env.reset()
        assert state == (0, 0)

    def test_special_transition(default_env):
        # Set up the special transition
        special_state = (0, 1)  # Replace with actual coordinates
        jump_state = (4, 4)
        special_reward = 0.5
        default_env.special_transitions[((special_state, Action.UP))] = (
            jump_state,
            special_reward,
        )

        # Test the special transition
        default_env.state = special_state
        new_state, reward, _, _._ = default_env.step(Action.DOWN)
        assert new_state == jump_state
        assert reward == special_reward
