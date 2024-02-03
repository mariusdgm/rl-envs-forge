import pytest
import matplotlib.pyplot as plt
import numpy as np
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

    def test_special_transition(self, default_env):
        # Set up the special transition
        special_state = (0, 0)  # Actually the default start state
        special_action = Action.DOWN
        jump_state = (4, 4)
        special_reward = 0.5
        default_env.add_special_transition(
            from_state=special_state, 
            action=special_action, 
            to_state=jump_state, 
            reward=special_reward
        )

        # Test the special transition
        default_env.state = special_state
        new_state, reward, _, _, _ = default_env.step(special_action)
        assert reward == special_reward
        assert new_state == jump_state
        
    def test_render(self, default_env, monkeypatch):
        """
        Test if the violin_plot mode of the render function produces a figure.
        """
        # Mock plt.show() to prevent actual rendering
        monkeypatch.setattr(plt, "show", lambda: None)
        default_env.render()

        # Check if a figure has been generated
        assert plt.get_fignums()

    def test_random_movements_with_constant_action(self):
        """
        Test that with a random_move_frequency of 1, performing the same action
        repeatedly results in different states, indicating random movements.
        """
        random_states = set()
        action = Action.RIGHT  # Use a constant action for all iterations

        # Perform the action and reset/instantiate the environment each time
        for _ in range(10):
            env = GridWorld(random_move_frequency=1, rows=5, cols=5, start_state=(0, 0))
            
            # Perform the action
            new_state, _, _, _, _ = env.step(action)
            random_states.add(new_state)  # Add the new state to a set for uniqueness

        # Since random_move_frequency is 1, all moves should be random.
        # This should lead to a diverse set of states despite the same action being taken.
        # Assert that we have multiple unique states from performing the same action,
        # indicating randomness in movement.
        assert len(random_states) > 1, "Expected random movements to result in multiple unique states when performing the same action repeatedly."