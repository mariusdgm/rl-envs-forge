import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")
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


@pytest.fixture
def limited_length_env():
    """Fixture to create a GridWorld environment with an episode length limit."""
    return GridWorld(episode_length_limit=10)


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

    def test_episode_length_limit(self, limited_length_env):
        env = limited_length_env
        env.reset()
        for _ in range(env.episode_length_limit - 1):
            _, _, done, truncated, _ = env.step(Action.RIGHT)
            assert not done, "Episode ended earlier than the limit"
            assert not truncated, "Episode was truncated earlier than the limit"

        _, _, done, truncated, _ = env.step(Action.RIGHT)
        assert not done, "Episode did not end when the limit was reached"
        assert (
            truncated
        ), "Episode was not marked as truncated when the limit was reached"

    @pytest.mark.parametrize(
        "kwargs, expected_exception",
        [
            ({"rows": "5"}, ValueError),  # rows must be a positive integer
            ({"cols": -1}, ValueError),  # cols must be a positive integer
            (
                {"start_state": "0,0"},
                ValueError,
            ),  # start_state must be a tuple of two integers
            (
                {"terminal_states": {"3,4": 1.0}},
                ValueError,
            ),  # terminal_states must be a dictionary with keys as tuples of two integers and values as integers or floats
            (
                {"walls": ["1,1"]},
                ValueError,
            ),  # walls must be a set of tuples of two integers
            (
                {"special_transitions": {((0, 0), "DOWN"): ((4, 4), 0.5)}},
                ValueError,
            ),  # special_transitions must be a dictionary with keys as tuples where the first element is a tuple of two integers and the second element is an Action, and values as tuples where the first element is a tuple of two integers and the second element is an integer or float
            (
                {"rewards": {1: -0.1}},
                ValueError,
            ),  # rewards must be a dictionary with keys as strings and values as integers or floats
            ({"seed": "42"}, ValueError),  # seed must be an integer or None
            (
                {"slip_distribution": {Action.UP: "0.1"}},
                ValueError,
            ),  # slip_distribution must be a dictionary with keys as Actions and values as integers or floats
            (
                {"p_success": 2.0},
                ValueError,
            ),  # p_success must be a float between 0 and 1
            (
                {"episode_length_limit": "10"},
                ValueError,
            ),  # episode_length_limit must be an integer or None
            ({"walls": set()}, None),  # walls can be an empty set
            (
                {"special_transitions": {}},
                None,
            ),  # special_transitions can be an empty dictionary
            ({"rewards": {}}, None),  # rewards can be an empty dictionary
            (
                {"slip_distribution": {}},
                None,
            ),  # slip_distribution can be an empty dictionary
        ],
    )
    def test_invalid_args(self, kwargs, expected_exception):
        if expected_exception:
            with pytest.raises(expected_exception):
                GridWorld(**kwargs)
        else:
            # Should not raise an exception
            GridWorld(**kwargs)

    def test_reset_and_move(self, default_env):
        # Reset and move to ensure we get the expected new cell
        new_start_state = (2, 3)
        default_env.reset(new_start_state=new_start_state)
        new_state, _, _, _, _ = default_env.step(Action.RIGHT)
        expected_state = (2, 4)
        assert (
            new_state == expected_state
        ), f"Expected new state to be {expected_state}, but got {new_state}"

    def test_reset_to_new_starting_state(self, default_env):
        # Reset to a new starting state and ensure it's correctly set
        new_start_state = (5, 5)
        initial_state = default_env.reset(new_start_state=new_start_state)
        assert (
            initial_state == new_start_state
        ), f"Expected initial state to be {new_start_state}, but got {initial_state}"

    def test_render_rgb_array(self, default_env):
        """
        Test if the rgb_array mode of the render function returns an image array.
        """
        img = default_env.render(mode="rgb_array")
        assert isinstance(img, np.ndarray)
        assert img.shape[-1] == 3  # Check if the image has 3 color channels (RGB)

    def test_mdp_size(self, default_env):
        """
        Test if the size of the MDP is correct. It should account for all viable cells
        (total cells minus walls) and ensure each viable cell has transitions for each action.
        """
        viable_cells = [
            (r, c)
            for r in range(default_env.rows)
            for c in range(default_env.cols)
            if (r, c) not in default_env.walls
        ]
        expected_size = len(viable_cells) * len(Action)
        actual_size = len(default_env.mdp)
        assert (
            actual_size == expected_size
        ), f"Expected MDP size {expected_size}, but got {actual_size}"
