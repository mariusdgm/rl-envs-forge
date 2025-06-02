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
        assert default_env.budget is None
        assert default_env.tau == 1.0
        assert default_env.max_steps == 100
        assert default_env.current_step == 0
        assert default_env.total_spent == 0.0

    def test_initialization_with_weighted_edges(self, weighted_env):
        assert weighted_env.use_weighted_edges is True
        assert weighted_env.weight_range == (0.1, 5.0)
        assert weighted_env.num_agents == 10

    def test_reset(self, default_env):
        initial_opinions, _ = default_env.reset()
        assert len(initial_opinions) == default_env.num_agents
        assert default_env.current_step == 0
        assert default_env.total_spent == 0.0

    def test_reset_with_initial_opinions(self, default_env):
        initial_opinions = np.array([0.1, 0.2, 0.3])
        default_env = NetworkGraph(num_agents=3, initial_opinions=initial_opinions)
        opinions, _ = default_env.reset()
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

    def test_step_max_steps_exceeded(self):
        env = NetworkGraph(num_agents=3, 
                           max_steps=1, 
                           initial_opinions=[0, 0, 0],
                           tau=0.1)
        env.reset()
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
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

        # Check that the **original** action (before saturation) is what we passed
        np.testing.assert_array_almost_equal(
            info["action_applied_raw"],
            action,
            err_msg="The environment should record the original (unsaturated) action correctly."
        )

        # Check that the **applied** action was **saturated** correctly
        np.testing.assert_array_less(
            info["action_applied_clipped"],
            env.max_u + 1e-5,
            err_msg="The saturated control action should be clipped to max_u."
        )

        # Check that the **opinions** are still within valid range [0, 1]
        assert np.all(state >= 0.0) and np.all(
            state <= 1.0
        ), "Opinions should be within [0, 1] after applying dynamics."

      

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
            seed=42
        )
        env_no_resistance.reset()
        state_no_resistance, _, _, _, _ = env_no_resistance.step(action)

        # Environment with control resistance
        control_resistance = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        env_with_resistance = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            control_resistance=control_resistance,
            seed=42
        )
        env_with_resistance.reset()
        state_with_resistance, _, _, _, _ = env_with_resistance.step(action)

        mean_no_resistance = np.mean(state_no_resistance)
        mean_with_resistance = np.mean(state_with_resistance)
        assert (
            mean_with_resistance < mean_no_resistance
        ), "Mean opinion with resistance should be smaller."

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
        assert (
            reward_final > reward_initial
        ), "Reward should improve as opinions get closer to the desired value."

    def test_termination_based_on_average_opinion(self):
        """Test that the environment terminates when the average opinion is within tolerance of the desired opinion."""
        num_agents = 5
        initial_opinions = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        desired_opinion = 0.5
        tolerance = 0.05

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            opinion_end_tolerance=tolerance,
        )
        env.reset()

        # Set opinions manually to be within the tolerance range
        env.opinions = np.array(
            [0.48, 0.49, 0.50, 0.51, 0.52]
        )  # Average = 0.5, within tolerance

        # Take a step with zero action to trigger the termination check
        _, _, done, _, _ = env.step(np.zeros(num_agents, dtype=np.float32))

        # Check if the episode is correctly marked as done
        assert (
            done
        ), "Episode should terminate when the average opinion is within the tolerance range."

    def test_state_getter(self, default_env):
        default_env.opinions = np.array([0.1, 0.2, 0.3])
        assert np.array_equal(default_env.state, np.array([0.1, 0.2, 0.3]))

    def test_state_setter(self, default_env):
        new_opinions = np.array([0.4, 0.5, 0.6])
        default_env.state = new_opinions
        assert np.array_equal(default_env.opinions, new_opinions)

    def test_opinions_and_state_sync(self, default_env):
        """Ensure that opinions and state are always synchronized."""
        new_opinions = np.array([0.7, 0.8, 0.9])
        default_env.state = new_opinions
        assert np.array_equal(default_env.opinions, default_env.state)

        default_env.opinions = np.array([0.2, 0.3, 0.4])
        assert np.array_equal(default_env.opinions, default_env.state)


    def test_normalized_vs_non_normalized_rewards(self):
        """Test consistency between normalized and non-normalized rewards as opinions approach the desired value."""
        num_agents = 5
        initial_opinions = np.zeros(num_agents)
        desired_opinion = 1.0
        max_u = 0.1
        control_beta = 0.4
        action = np.full(num_agents, max_u)

        # Environment 1: Normalized rewards
        env_norm = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            max_u=max_u,
            control_beta=control_beta,
            normalize_reward=True,
        )
        env_norm.reset()

        # Environment 2: Raw rewards
        env_raw = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            max_u=max_u,
            control_beta=control_beta,
            normalize_reward=False,
        )
        env_raw.reset()

        # Step both envs once
        _, reward_norm_initial, _, _, _ = env_norm.step(action)
        _, reward_raw_initial, _, _, _ = env_raw.step(action)

        # Move opinions closer to desired
        env_norm.opinions = np.full(num_agents, 0.9)
        env_raw.opinions = np.full(num_agents, 0.9)

        # Step both envs again
        _, reward_norm_later, _, _, _ = env_norm.step(action)
        _, reward_raw_later, _, _, _ = env_raw.step(action)

        # Checks for normalized rewards
        assert reward_norm_initial <= 0.0, "Normalized reward should be <= 0 initially"
        assert reward_norm_later <= 0.0, "Normalized reward should be <= 0 after improvement"
        assert reward_norm_later > reward_norm_initial, "Normalized reward should improve as opinions approach desired"

        # Checks for raw rewards
        assert reward_raw_initial <= 0.0, "Raw reward should be <= 0 initially"
        assert reward_raw_later <= 0.0, "Raw reward should be <= 0 after improvement"
        assert reward_raw_later > reward_raw_initial, "Raw reward should improve as opinions approach desired"

        # Cross-check: normalized reward should be a scaled version of raw reward
        scale_factor = env_norm.num_agents  # Since normalization divides by num_agents
        np.testing.assert_allclose(
            reward_norm_initial, reward_raw_initial / scale_factor, rtol=1e-5,
            err_msg="Normalized initial reward should match raw reward divided by number of agents."
        )
        np.testing.assert_allclose(
            reward_norm_later, reward_raw_later / scale_factor, rtol=1e-5,
            err_msg="Normalized later reward should match raw reward divided by number of agents."
        )
        
    def test_terminal_reward_added_when_done(self):
        """Test that terminal reward is added when the environment reaches the done state."""
        num_agents = 5
        desired_opinion = 0.5
        tolerance = 0.01
        terminal_reward = 1.0

        # Set initial opinions to match desired (triggering immediate termination)
        initial_opinions = np.full(num_agents, desired_opinion)

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            opinion_end_tolerance=tolerance,
        )
        env.terminal_reward = terminal_reward  # set terminal reward directly
        env.reset()

        # Step with zero action; should immediately trigger `done = True`
        action = np.zeros(num_agents, dtype=np.float32)
        _, reward, done, truncated, info = env.step(action)

        # Assert that we hit terminal condition and reward includes the terminal bonus
        assert done, "Environment should be done when opinions match desired within tolerance"
        assert reward > 0.0, "Reward should include terminal bonus and be greater than 0"
        assert info.get("terminal_reward_applied", False), "Info should flag terminal reward as applied"
        
    @pytest.mark.parametrize("mode,params", [
        ("uniform", (0.2, 0.5)),
        ("normal", {"mean": 0.4, "std": 0.1}),
        ("exponential", {"scale": 2.0})
    ])
    def test_graph_generation_modes(self, mode, params):
        """Test that graph generation modes create reasonable graphs."""
        num_agents = 20

        # Test with bidirectional probability 1.0 → symmetric
        env_symmetric = NetworkGraph(
            num_agents=num_agents,
            graph_connection_distribution=mode,
            graph_connection_params=params,
            bidirectional_prob=1.0
        )
        adj_sym = env_symmetric.connectivity_matrix
        assert np.allclose(adj_sym, adj_sym.T), "Adjacency matrix should be symmetric when bidirectional=1.0"

        # Test with bidirectional probability 0.0 → asymmetric (directed)
        env_asymmetric = NetworkGraph(
            num_agents=num_agents,
            graph_connection_distribution=mode,
            graph_connection_params=params,
            bidirectional_prob=0.0
        )
        adj_asym = env_asymmetric.connectivity_matrix
        assert not np.allclose(adj_asym, adj_asym.T), "Adjacency matrix should not be symmetric when bidirectional=0.0"

        # In both cases, graph should have at least some edges
        assert np.any(adj_sym > 0), f"Graph with mode='{mode}' should have edges (symmetric)"
        assert np.any(adj_asym > 0), f"Graph with mode='{mode}' should have edges (asymmetric)"
    
    def test_no_budget_removes_remaining_budget_from_info(self):
        """Test that when budget=None, remaining_budget is not included in info."""
        env = NetworkGraph(num_agents=5, budget=None)
        env.reset()
        action = np.zeros(5, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "remaining_budget" not in info, "Info should not contain remaining_budget if budget is None"
        
    def test_action_raw_vs_clipped_behavior(self):
        """Test that action_applied_raw and action_applied_clipped are handled correctly."""
        num_agents = 4
        max_u = 0.1

        env = NetworkGraph(num_agents=num_agents, max_u=max_u)
        env.reset()

        action = np.array([0.2, 0.05, 0.15, 0.09], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        np.testing.assert_array_almost_equal(info["action_applied_raw"], action)
        assert np.all(info["action_applied_clipped"] <= env.max_u + 1e-5), "Clipped action must be <= max_u"

        # At least one value should have been clipped
        assert np.any(info["action_applied_raw"] > info["action_applied_clipped"]), "Some values should have been clipped"
        
    def test_normalized_reward_with_exceeding_action(self):
        """Test that normalized reward handles exceeding actions and stays in [-1, 0]."""
        num_agents = 6
        max_u = 0.1
        env = NetworkGraph(
            num_agents=num_agents,
            max_u=max_u,
            desired_opinion=1.0,
            normalize_reward=True,
        )
        env.reset()

        # Apply action that exceeds allowed max_u
        action = np.full(num_agents, 2 * max_u, dtype=np.float32)  # double max_u
        _, reward, _, _, _ = env.step(action)

        assert -1.0 <= reward <= 0.0, "Normalized reward should be within [-1, 0] even with large action"
        
    def test_step_reward_computed_from_previous_state(self):
        """Test that reward is computed based on the previous state, not next."""
        env = NetworkGraph(
            num_agents=3,
            max_steps=5,
            initial_opinions=[0.1, 0.1, 0.1],
            tau=0.1,
            desired_opinion=1.0,
            max_u=0.4,
            control_beta=0.5,  # some penalty on action
            normalize_reward=False,
            terminal_reward=10.0  # make terminal reward big if successful
        )
        env.reset()

        # Save original opinions for manual reward calculation
        old_opinions = env.opinions.copy()

        # Define a moderate action
        action = np.array([0.2, 0.0, 0.2], dtype=np.float32)

        # Step
        state, reward, done, truncated, info = env.step(action)

        # MANUALLY compute expected reward:
        expected_raw_reward = -np.abs(env.desired_opinion - old_opinions).sum() - env.control_beta * np.sum(action)
        expected_terminal = False  # Not done yet
        if expected_terminal:
            expected_raw_reward += env.terminal_reward

        # Check reward
        assert np.isclose(reward, expected_raw_reward, atol=1e-5), f"Reward mismatch: {reward} vs {expected_raw_reward}"
        
    def test_reset_randomize_opinions(self):
        env = NetworkGraph(num_agents=4, initial_opinions=np.array([0.1, 0.2, 0.3, 0.4]))
        opinions_fixed, info_fixed = env.reset(randomize_opinions=False)
        opinions_random, info_random = env.reset(randomize_opinions=True)

        # Should differ because one is random
        assert not np.allclose(opinions_fixed, opinions_random), "Randomized reset should produce different opinions"
        assert info_fixed["random_opinions"] is False
        assert info_random["random_opinions"] is True
        
    def test_graph_reproducibility_with_seeds(self):
        # Same seed -> same graph
        env1 = NetworkGraph(num_agents=5, graph_connection_distribution="uniform", seed=123)
        env2 = NetworkGraph(num_agents=5, graph_connection_distribution="uniform", seed=123)

        np.testing.assert_array_equal(
            env1.connectivity_matrix,
            env2.connectivity_matrix,
            err_msg="Environments with the same seed should produce identical connectivity matrices"
        )

        # Different seed -> likely different graph
        env3 = NetworkGraph(num_agents=5, graph_connection_distribution="uniform", seed=456)

        assert not np.allclose(
            env1.connectivity_matrix, env3.connectivity_matrix
        ), "Environments with different seeds should likely produce different connectivity matrices"
        
    def test_random_graph_centralities_vary(self):
        """Ensure that centralities vary in a directed random graph."""
        env = NetworkGraph(
            num_agents=10,
            graph_connection_distribution="uniform",
            connection_prob_range=(0.4, 0.6),
            bidirectional_prob=0.3,  # allow directionality
            seed=42,
        )
        centralities = env.centralities

        # Check that centralities sum to 1 and vary across nodes
        assert np.isclose(np.sum(centralities), 1.0), "Centralities should be normalized to sum to 1"
        assert np.std(centralities) > 0.01, f"Centralities should vary across nodes, got std={np.std(centralities)}"
    
    def test_dynamics_step_zero_behavior(self):
        """Test that compute_dynamics with step_duration=0 only applies control and no propagation."""

        num_agents = 5
        initial_opinions = np.linspace(0.1, 0.9, num_agents)
        action = np.zeros(num_agents)
        action[1] = 0.5  # Apply control to just one node

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=1.0,
            control_resistance=np.zeros(num_agents),  # No resistance for clarity
        )
        env.reset()

        result = env.compute_dynamics(current_state=initial_opinions, control_action=action, step_duration=0.0)

        # Manually compute expected outcome
        expected = initial_opinions.copy()
        expected[1] = (1 - action[1]) * initial_opinions[1] + action[1] * env.desired_opinion

        np.testing.assert_allclose(result, expected, rtol=1e-6,
            err_msg="Only the controlled node should be affected when step_duration=0")
        
    def test_disable_termination_flag(self):
        """Test that when terminate_when_converged is False, the environment never sets done=True."""
        num_agents = 5
        desired_opinion = 0.5
        tolerance = 0.01

        # Opinions close enough to desired_opinion to normally trigger done=True
        initial_opinions = np.full(num_agents, desired_opinion)

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            opinion_end_tolerance=tolerance,
            terminate_when_converged=False  
        )
        env.reset()

        # Step with zero action (would normally converge)
        action = np.zeros(num_agents, dtype=np.float32)
        _, reward, done, truncated, info = env.step(action)

        assert not done, "Environment should not terminate when terminate_when_converged is False"
        assert not truncated, "Episode should not be truncated"
        assert "terminal_reward_applied" in info
        assert info["terminal_reward_applied"] is False, "Terminal reward should not be applied"