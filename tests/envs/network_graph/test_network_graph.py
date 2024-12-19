import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph
from scipy.linalg import expm


@pytest.fixture
def default_env():
    """Fixture to create a default NetworkGraph environment for each test."""
    return NetworkGraph()


@pytest.fixture
def weighted_env():
    """Fixture to create a NetworkGraph environment with weighted edges."""
    return NetworkGraph(use_weighted_edges=True, weight_range=(0.1, 5.0))


class TestNetworkGraph:
    def test_initialization(self, default_env):
        assert default_env.num_agents == 10
        assert default_env.budget == 10.0
        assert default_env.tau == 1.0
        assert default_env.max_steps == 100
        assert default_env.current_step == 0
        assert default_env.total_spent == 0.0

    def test_initialization_with_weighted_edges(self, weighted_env):
        assert weighted_env.use_weighted_edges is True
        assert weighted_env.weight_range == (0.1, 5.0)
        assert weighted_env.num_agents == 10

    def test_reset(self, default_env):
        initial_opinions = default_env.reset()
        assert len(initial_opinions) == default_env.num_agents
        assert default_env.current_step == 0
        assert default_env.total_spent == 0.0

    def test_reset_with_initial_opinions(self, default_env):
        initial_opinions = np.array([0.1, 0.2, 0.3])
        default_env = NetworkGraph(num_agents=3, initial_opinions=initial_opinions)
        opinions = default_env.reset()
        np.testing.assert_array_equal(opinions, initial_opinions)

    def test_step(self, default_env):
        default_env.reset()
        action = np.array([0.1] * default_env.num_agents, dtype=np.float32)
        state, reward, done, truncated, info = default_env.step(action)
        assert len(state) == default_env.num_agents
        assert isinstance(reward, float)
        assert not done
        assert not truncated
        assert info["current_step"] == 1

    def test_step_budget_exceeded(self, default_env):
        default_env = NetworkGraph(
            num_agents=3, budget=0.05, control_resistance=np.array([1.0, 1.0, 1.0])
        )
        default_env.reset()
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        state, reward, done, truncated, info = default_env.step(action)
        assert done
        assert not truncated

    def test_step_max_steps_exceeded(self, default_env):
        default_env = NetworkGraph(num_agents=3, max_steps=1)
        default_env.reset()
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        state, reward, done, truncated, info = default_env.step(action)
        assert not done
        assert truncated

    @patch("rl_envs_forge.envs.network_graph.visualize.plt.show")  # Patch plt.show
    def test_render(self, mock_show, default_env):
        default_env.reset()
        default_env.render(mode="matplotlib")
        assert (
            mock_show.called
        ), "plt.show should be called when rendering in 'matplotlib' mode"

    def test_render_output(self, default_env):
        default_env.reset()
        with patch("builtins.print") as mock_print:
            default_env.render(mode="human")
            mock_print.assert_called()

    def test_centrality_computation_with_weighted_edges(self, weighted_env):
        L = weighted_env.L
        centralities = weighted_env.centralities
        assert (
            np.sum(centralities) > 0
        ), "Centralities should be computed correctly for weighted edges"

        # Adjust the check to ensure the Laplacian is valid
        assert np.all(np.diag(L) >= 0), "Diagonal of Laplacian should be non-negative"
        assert np.all(
            L - np.diag(np.diag(L)) <= 0
        ), "Off-diagonal elements of Laplacian should be non-positive"

    def test_saturation_of_control_and_opinions(self):
        # Set up the environment with specific parameters
        num_agents = 3
        max_u = 0.1  # Maximum control input
        initial_opinions = np.array(
            [0.0, 0.5, 1.0]
        )  # Initial opinions ranging from 0 to 1

        # Create an environment with these parameters
        env = NetworkGraph(
            num_agents=num_agents,
            max_u=max_u,
            initial_opinions=initial_opinions,
            budget=10.0,
            desired_opinion=1.0,
            tau=1.0,
            max_steps=100,
        )

        env.reset()

        # Define an action that exceeds the maximum control input
        action = np.array([0.2, 0.15, 0.25], dtype=np.float32)

        # Step the environment with the action
        state, reward, done, truncated, info = env.step(action)

        # Retrieve the clipped action from the info dictionary
        applied_action = info["action_applied"]

        # Check that the action was clipped to the max_u
        np.testing.assert_array_less(
            applied_action,
            np.full(action.shape, max_u + 1e-5),
            "Control inputs should be clipped to max_u",
        )

        # Check that the state (opinions) remains within [0, 1]
        assert np.all(state >= 0.0) and np.all(
            state <= 1.0
        ), "Opinions should be within the range [0, 1]"

    def test_step_max_vs_zero_action(self, default_env):

        # Set initial opinions to be a linspace between 0 and 1
        num_agents = default_env.num_agents
        initial_opinions = np.linspace(0, 1, num_agents)

        # Initialize the environment with the specified initial opinions
        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            max_u=0.1,  # Define max control input for testing
            desired_opinion=1.0,
            tau=1.0,
            budget=10.0,
            max_steps=100,
        )

        env.reset()

        # Define zero action (no control) and max action (full control)
        zero_action = np.zeros(num_agents, dtype=np.float32)
        max_action = np.full(num_agents, env.max_u, dtype=np.float32)

        # Step with zero action
        zero_action_state, _, _, _, _ = env.step(zero_action)

        # Step with max action
        env.reset()  # Reset the environment again to start from the same initial state
        max_action_state, _, _, _, _ = env.step(max_action)

        # Compute the average opinions for both actions
        avg_zero_action_state = np.mean(zero_action_state)
        avg_max_action_state = np.mean(max_action_state)

        # Check that the average state resulting from max action is greater than the average state from zero action
        assert (
            avg_max_action_state >= avg_zero_action_state
        ), "The average state with max action should be greater than or equal to the average state with zero action"

    def test_step_with_negative_step_duration(self, default_env):
        # Reset the environment
        default_env.reset()

        # Define an action
        action = np.array(
            [default_env.max_u] * default_env.num_agents, dtype=np.float32
        )

        # Define a negative step_duration
        step_duration = -0.5

        # Expect a ValueError to be raised
        with pytest.raises(ValueError) as excinfo:
            default_env.step(action, step_duration=step_duration)
        assert "step_duration must be positive" in str(excinfo.value)

    def test_step_with_large_step_duration(self, default_env):
        # Reset the environment
        default_env.reset()

        # Define an action (no action applied)
        action = np.zeros(default_env.num_agents, dtype=np.float32)

        # Define a large step_duration
        step_duration = 10.0  # Simulate over a long time without action

        # Step the environment
        state, reward, done, truncated, info = default_env.step(
            action, step_duration=step_duration
        )

        # Check that the state has evolved only according to the network dynamics
        # Since the Laplacian tends to drive opinions towards consensus, the opinions should be closer together

        # Check that the state is updated correctly
        assert len(state) == default_env.num_agents
        assert isinstance(reward, float)
        assert not done
        assert not truncated
        assert info["current_step"] == 1

    def test_dynamics_function_no_state_modification(self, default_env):
        # Set the initial conditions for testing
        initial_opinions = np.linspace(0, 1, default_env.num_agents)
        action = np.full(
            default_env.num_agents, default_env.max_u / 2
        )  # Use half of max_u for control
        step_duration = 1.0  # Set a step duration

        # Set the initial state of the environment and save its original state
        default_env.opinions = initial_opinions.copy()
        initial_step = default_env.current_step
        initial_total_spent = default_env.total_spent

        # Call the dynamics function directly without modifying the env state
        next_state = default_env.compute_dynamics(
            initial_opinions, action, step_duration
        )

        # Verify that the output is in the expected range
        assert np.all(next_state >= 0) and np.all(
            next_state <= 1
        ), "Next state should be within [0, 1]"

        # Check that the internal metrics of the environment have not changed
        np.testing.assert_array_equal(
            default_env.opinions, initial_opinions, "Opinions should remain unchanged"
        )
        assert (
            default_env.current_step == initial_step
        ), "Current step should not be modified"
        assert (
            default_env.total_spent == initial_total_spent
        ), "Total spent should not be modified"

    def test_control_resistance_effect(self):
        """Test that control resistance affects the resulting opinions."""
        num_agents = 5
        initial_opinions = np.linspace(0.2, 0.8, num_agents)
        action = np.full(num_agents, 0.5)

        # Environment with no control resistance
        env_no_resistance = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            control_resistance=np.zeros(num_agents),
        )
        env_no_resistance.reset()
        state_no_resistance, _, _, _, _ = env_no_resistance.step(action)

        # Environment with control resistance
        control_resistance = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        env_with_resistance = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            control_resistance=control_resistance,
        )
        env_with_resistance.reset()
        state_with_resistance, _, _, _, _ = env_with_resistance.step(action)

      
        mean_no_resistance = np.mean(state_no_resistance)
        mean_with_resistance = np.mean(state_with_resistance)
        assert mean_with_resistance < mean_no_resistance, "Mean opinion with resistance should be smaller."

    def test_reward_function_effect(self):
        """Test that the reward function returns higher rewards when closer to desired opinions."""
        num_agents = 3
        initial_opinions = np.array([0.1, 0.5, 0.9])
        action = np.array([0.1, 0.2, 0.3])
        desired_opinion = 1.0
        beta = 0.4

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            control_beta=beta,
        )
        env.reset()

        # Step the environment and compute the reward
        _, reward_initial, _, _, _ = env.step(action)

        # Manually set opinions closer to the desired value
        env.opinions = np.array([0.8, 0.9, 1.0])
        _, reward_final, _, _, _ = env.step(action)

        # Assert that the final reward is greater (less negative) than the initial reward
        assert reward_final > reward_initial, (
            "Reward should improve as opinions get closer to the desired value."
        )