import pytest
import matplotlib.pyplot as plt
import numpy as np
from rl_envs_forge.envs.grid_world.grid_world import (
    GridWorld,
    Action,
)


@pytest.fixture
def default_env():
    """Fixture to create a default GridWorld environment for each test."""
    return GridWorld()


@pytest.fixture
def slippery_env():
    """Fixture to create a GridWorld environment with slippage."""
    return GridWorld(p_success=0.5, seed=42)


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
            reward=special_reward,
        )

        # Test the special transition
        default_env.state = special_state
        new_state, reward, _, _, _ = default_env.step(special_action)
        assert new_state == jump_state
        assert reward == special_reward

    def test_render(self, default_env, monkeypatch):
        """
        Test if the violin_plot mode of the render function produces a figure.
        """
        # Mock plt.show() to prevent actual rendering
        monkeypatch.setattr(plt, "show", lambda: None)
        default_env.render()

        # Check if a figure has been generated
        assert plt.get_fignums()

    def test_slippage_occurs(self, slippery_env):
        env = slippery_env
        np.random.seed(42)  # Ensuring reproducibility

        # Perform a series of actions
        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        intended_states = []
        actual_states = []

        for action in actions:
            intended_next_state = env.calculate_next_state(env.state, action)
            intended_states.append(intended_next_state)
            new_state, _, _, _, _ = env.step(
                action.value
            )  # Using action.value if your step function expects int
            actual_states.append(new_state)
            env.reset()  # Resetting to start state after each action

        # Check if at least one action resulted in slippage
        slippage_occurred = any(
            intended != actual
            for intended, actual in zip(intended_states, actual_states)
        )
        assert (
            slippage_occurred
        ), "Expected slippage to occur, but all actions resulted in intended states."

    def test_probability_matrix(self):
        # Create a GridWorld environment with walls, a terminal state, and a special transition
        env = GridWorld(
            rows=3,
            cols=3,
            start_state=(0, 0),
            terminal_states={(2, 2): 1.0},  # Terminal state at bottom right
            walls={(1, 1)},  # Wall in the center
            special_transitions={
                ((0, 0), Action.RIGHT): ((0, 2), 0.5)
            },  # Special transition from (0, 0) right to (0, 2)
            p_success=1.0,
        )  # Ensure deterministic behavior for this test

        # Expected dimensions of the P matrix: 9 states x 9 states x 4 actions
        assert env.P.shape == (9, 9, 4), "P matrix has incorrect shape."

        # Check terminal states loop back to themselves with a probability of 1 for all actions
        terminal_index = 2 * 3 + 2  # Convert 2D state (2, 2) to 1D index
        for action in range(4):
            assert (
                env.P[terminal_index, terminal_index, action] == 1.0
            ), "Terminal state transition probability is incorrect."

        # Corrected check for wall states: Expecting self-transition probability of 1
        wall_index = 1 * 3 + 1  # Convert 2D state (1, 1) to 1D index
        for action in range(4):
            assert (
                env.P[wall_index, wall_index, action] == 1.0
            ), "Wall state self-transition probability is incorrect."

        # Check special transition is correctly represented
        special_from_index = 0 * 3 + 0  # Convert 2D state (0, 0) to 1D index
        special_to_index = 0 * 3 + 2  # Convert 2D state (0, 2) to 1D index
        assert (
            env.P[special_from_index, special_to_index, Action.RIGHT] == 1.0
        ), "Special transition probability is incorrect."

        # Check normal action leads to expected state with correct probability
        # For example, moving DOWN from (0, 0) should lead to (1, 0) with probability 1.0
        # Note: This is under the condition that p_success=1.0 for simplicity
        normal_from_index = 0 * 3 + 0  # (0, 0) state index
        normal_to_index = 1 * 3 + 0  # (1, 0) state index
        assert (
            env.P[normal_from_index, normal_to_index, Action.DOWN] == 1.0
        ), "Normal action transition probability is incorrect."
