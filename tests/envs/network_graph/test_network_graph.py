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
        assert default_env.t_campaign == 1
        assert default_env.t_s == 0.1
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
        env = NetworkGraph(num_agents=3, max_steps=1, initial_opinions=[0, 0, 0])
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
            err_msg="The environment should record the original (unsaturated) action correctly.",
        )

        # Check that the **applied** action was **saturated** correctly
        np.testing.assert_array_less(
            info["action_applied_clipped"],
            env.max_u + 1e-5,
            err_msg="The saturated control action should be clipped to max_u.",
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

        # Expect a ValueError to be raised
        with pytest.raises(ValueError) as excinfo:
            default_env.step(action, t_s=-0.5)
        assert "t_s must be positive" in str(excinfo.value)

    def test_step_with_large_step_duration(self, default_env):
        # Reset the environment
        default_env.reset()

        # Define an action (no action applied)
        action = np.zeros(default_env.num_agents, dtype=np.float32)

        # Step the environment
        state, reward, done, truncated, info = default_env.step(action, t_campaign=10.0)

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

        # Set the initial state of the environment and save its original state
        default_env.opinions = initial_opinions.copy()
        initial_step = default_env.current_step
        initial_total_spent = default_env.total_spent

        # Call the dynamics function directly without modifying the env state
        next_state, _ = default_env.compute_dynamics(initial_opinions, action)

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
            seed=42,
        )
        env_no_resistance.reset()
        state_no_resistance, _, _, _, _ = env_no_resistance.step(action)

        # Environment with control resistance
        control_resistance = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        env_with_resistance = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            control_resistance=control_resistance,
            seed=42,
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
        assert (
            reward_norm_later <= 0.0
        ), "Normalized reward should be <= 0 after improvement"
        assert (
            reward_norm_later > reward_norm_initial
        ), "Normalized reward should improve as opinions approach desired"

        # Checks for raw rewards
        assert reward_raw_initial <= 0.0, "Raw reward should be <= 0 initially"
        assert reward_raw_later <= 0.0, "Raw reward should be <= 0 after improvement"
        assert (
            reward_raw_later > reward_raw_initial
        ), "Raw reward should improve as opinions approach desired"

        # Cross-check: normalized reward should be a scaled version of raw reward
        scale_factor = env_norm.num_agents  # Since normalization divides by num_agents
        np.testing.assert_allclose(
            reward_norm_initial,
            reward_raw_initial / scale_factor,
            rtol=1e-5,
            err_msg="Normalized initial reward should match raw reward divided by number of agents.",
        )
        np.testing.assert_allclose(
            reward_norm_later,
            reward_raw_later / scale_factor,
            rtol=1e-5,
            err_msg="Normalized later reward should match raw reward divided by number of agents.",
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
        assert (
            done
        ), "Environment should be done when opinions match desired within tolerance"
        assert (
            reward > 0.0
        ), "Reward should include terminal bonus and be greater than 0"
        assert info.get(
            "terminal_reward_applied", False
        ), "Info should flag terminal reward as applied"

    @pytest.mark.parametrize(
        "mode,params",
        [
            ("uniform", (0.2, 0.5)),
            ("normal", {"mean": 0.4, "std": 0.1}),
            ("exponential", {"scale": 2.0}),
        ],
    )
    def test_graph_generation_modes(self, mode, params):
        """Test that graph generation modes create reasonable graphs."""
        num_agents = 20

        # Test with bidirectional probability 1.0 → symmetric
        env_symmetric = NetworkGraph(
            num_agents=num_agents,
            graph_connection_distribution=mode,
            graph_connection_params=params,
            bidirectional_prob=1.0,
        )
        adj_sym = env_symmetric.connectivity_matrix
        assert np.allclose(
            adj_sym, adj_sym.T
        ), "Adjacency matrix should be symmetric when bidirectional=1.0"

        # Test with bidirectional probability 0.0 → asymmetric (directed)
        env_asymmetric = NetworkGraph(
            num_agents=num_agents,
            graph_connection_distribution=mode,
            graph_connection_params=params,
            bidirectional_prob=0.0,
        )
        adj_asym = env_asymmetric.connectivity_matrix
        assert not np.allclose(
            adj_asym, adj_asym.T
        ), "Adjacency matrix should not be symmetric when bidirectional=0.0"

        # In both cases, graph should have at least some edges
        assert np.any(
            adj_sym > 0
        ), f"Graph with mode='{mode}' should have edges (symmetric)"
        assert np.any(
            adj_asym > 0
        ), f"Graph with mode='{mode}' should have edges (asymmetric)"

    def test_no_budget_removes_remaining_budget_from_info(self):
        """Test that when budget=None, remaining_budget is not included in info."""
        env = NetworkGraph(num_agents=5, budget=None)
        env.reset()
        action = np.zeros(5, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert (
            "remaining_budget" not in info
        ), "Info should not contain remaining_budget if budget is None"

    def test_action_raw_vs_clipped_behavior(self):
        """Test that action_applied_raw and action_applied_clipped are handled correctly."""
        num_agents = 4
        max_u = 0.1

        env = NetworkGraph(num_agents=num_agents, max_u=max_u)
        env.reset()

        action = np.array([0.2, 0.05, 0.15, 0.09], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        np.testing.assert_array_almost_equal(info["action_applied_raw"], action)
        assert np.all(
            info["action_applied_clipped"] <= env.max_u + 1e-5
        ), "Clipped action must be <= max_u"

        # At least one value should have been clipped
        assert np.any(
            info["action_applied_raw"] > info["action_applied_clipped"]
        ), "Some values should have been clipped"

    def test_normalized_reward_with_exceeding_action():
        num_agents = 6
        max_u = 0.1
        control_beta = 0.4

        env = NetworkGraph(
            num_agents=num_agents,
            max_u=max_u,
            desired_opinion=1.0,
            control_beta=control_beta,
            normalize_reward=True,
            use_delta_shaping=False,  # keep absolute-only for predictable bound
        )
        env.reset()

        action = np.full(num_agents, 2 * max_u, dtype=np.float32)  # double max_u
        _, reward, _, _, _ = env.step(action)

        # Bounds: abs-term ∈ [-1, 0], cost-term = -(beta * sum(u))/N
        cost_per_agent = control_beta * np.sum(action) / num_agents
        lower_bound = -1.0 - cost_per_agent
        upper_bound = 0.0  # best case: zero distance, zero cost (not here but still an upper bound)

        assert lower_bound - 1e-6 <= reward <= upper_bound + 1e-6, \
            f"Normalized reward {reward} should be in [{lower_bound}, {upper_bound}]"

    def test_delta_shaping_increases_reward_on_improvement():
        num_agents = 5
        desired_opinion = 1.0
        initial_opinions = np.linspace(0.1, 0.5, num_agents)
        action = np.full(num_agents, 0.2, dtype=np.float32)  # positive push toward 1.0

        # Env A: no delta shaping
        env_abs = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            control_beta=0.4,
            normalize_reward=False,
            use_delta_shaping=False,
        )
        env_abs.reset()
        _, r_abs, _, _, _ = env_abs.step(action)

        # Env B: same setup, with delta shaping
        env_delta = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            control_beta=0.4,
            normalize_reward=False,
            use_delta_shaping=True,
            delta_lambda=0.1,   # small bonus
        )
        env_delta.reset()
        _, r_delta, _, _, _ = env_delta.step(action)

        # Because action improves toward d, delta shaping should make reward less negative (greater)
        assert r_delta > r_abs, f"Delta-shaped reward ({r_delta}) should exceed absolute-only ({r_abs}) when improving."
    
    def test_step_reward_computed_from_next_state():
        env = NetworkGraph(
            num_agents=3,
            max_steps=5,
            initial_opinions=[0.1, 0.1, 0.1],
            t_campaign=1,
            t_s=0.1,
            desired_opinion=1.0,
            max_u=0.4,
            control_beta=0.5,
            normalize_reward=False,
            terminal_reward=10.0,
            # IMPORTANT: disable delta shaping for this test
            use_delta_shaping=False,
        )
        env.reset()

        x_prev = env.opinions.copy()
        action = np.array([0.2, 0.0, 0.2], dtype=np.float32)

        # Compute x_next independently
        clipped = np.clip(action, 0, env.max_u)
        x_next_manual, _ = env.compute_dynamics(x_prev, clipped, env.t_campaign, env.t_s)

        # Step the env
        _, reward, done, truncated, info = env.step(action)

        # Expected reward uses x_next and ORIGINAL (unclipped) action for cost
        expected_raw_reward = -np.abs(env.desired_opinion - x_next_manual).sum() \
                            - env.control_beta * np.sum(action)
        # Terminal bonus only if we truly terminated (rare here)
        if done and not truncated:
            expected_raw_reward += env.terminal_reward

        assert np.isclose(reward, expected_raw_reward, atol=1e-5), \
            f"Reward mismatch: {reward} vs {expected_raw_reward}"

    def test_reset_randomize_opinions(self):
        env = NetworkGraph(
            num_agents=4, initial_opinions=np.array([0.1, 0.2, 0.3, 0.4])
        )
        opinions_fixed, info_fixed = env.reset(randomize_opinions=False)
        opinions_random, info_random = env.reset(randomize_opinions=True)

        # Should differ because one is random
        assert not np.allclose(
            opinions_fixed, opinions_random
        ), "Randomized reset should produce different opinions"
        assert info_fixed["random_opinions"] is False
        assert info_random["random_opinions"] is True

    def test_graph_reproducibility_with_seeds(self):
        # Same seed -> same graph
        env1 = NetworkGraph(
            num_agents=5, graph_connection_distribution="uniform", seed=123
        )
        env2 = NetworkGraph(
            num_agents=5, graph_connection_distribution="uniform", seed=123
        )

        np.testing.assert_array_equal(
            env1.connectivity_matrix,
            env2.connectivity_matrix,
            err_msg="Environments with the same seed should produce identical connectivity matrices",
        )

        # Different seed -> likely different graph
        env3 = NetworkGraph(
            num_agents=5, graph_connection_distribution="uniform", seed=456
        )

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
        assert np.isclose(
            np.sum(centralities), 1.0
        ), "Centralities should be normalized to sum to 1"
        assert (
            np.std(centralities) > 0.01
        ), f"Centralities should vary across nodes, got std={np.std(centralities)}"

    def test_dynamics_step_zero_behavior(self):
        """Test that compute_dynamics with t_campaign=0 only applies control and no propagation."""

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

        result, _ = env.compute_dynamics(
            current_state=initial_opinions, control_action=action, t_campaign=0.0
        )

        # Manually compute expected outcome
        expected = initial_opinions.copy()
        expected[1] = (1 - action[1]) * initial_opinions[1] + action[
            1
        ] * env.desired_opinion

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-6,
            err_msg="Only the controlled node should be affected when step_duration=0",
        )

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
            terminate_when_converged=False,
        )
        env.reset()

        # Step with zero action (would normally converge)
        action = np.zeros(num_agents, dtype=np.float32)
        _, reward, done, truncated, info = env.step(action)

        assert (
            not done
        ), "Environment should not terminate when terminate_when_converged is False"
        assert not truncated, "Episode should not be truncated"
        assert "terminal_reward_applied" in info
        assert (
            info["terminal_reward_applied"] is False
        ), "Terminal reward should not be applied"

    def test_coca_dynamics_behavior(self):
        """Test that COCA dynamics run and change opinions in expected nonlinear way."""
        num_agents = 3
        initial_opinions = np.array([0.1, 0.5, 0.9])  # more asymmetrical
        desired_opinion = 1.0

        # Fully connected undirected graph
        connectivity_matrix = np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=desired_opinion,
            t_campaign=1,
            t_s=0.1,
            connectivity_matrix=connectivity_matrix,
            dynamics_model="coca",
        )
        env.reset()

        action = np.zeros(num_agents, dtype=np.float32)
        state_before = env.opinions.copy()
        state_after, _, _, _, _ = env.step(action)

        # Ensure opinions changed
        assert not np.allclose(
            state_before, state_after
        ), "Opinions should update under COCA dynamics"

        # Ensure outputs remain valid
        assert np.all(
            (state_after >= 0) & (state_after <= 1)
        ), "Opinions must be within [0, 1]"

        # Ensure expected direction of movement (toward average)
        avg_before = np.mean(state_before)
        for i in range(num_agents):
            if state_before[i] < avg_before:
                assert (
                    state_after[i] > state_before[i]
                ), f"Agent {i} should move upward toward average"
            elif state_before[i] > avg_before:
                assert (
                    state_after[i] < state_before[i]
                ), f"Agent {i} should move downward toward average"

    def test_step_returns_intermediate_states(self, default_env):
        default_env.reset()
        action = np.zeros(default_env.num_agents, dtype=np.float32)
        _, _, _, _, info = default_env.step(action)

        # Check that intermediate_states is in info
        assert (
            "intermediate_states" in info
        ), "step() should return 'intermediate_states' in info"

        intermediates = info["intermediate_states"]
        expected_steps = (
            int(default_env.t_campaign / default_env.t_s) + 1
        )  # +1 for the post-control state
        assert intermediates.shape == (
            expected_steps,
            default_env.num_agents,
        ), f"Expected shape {(expected_steps, default_env.num_agents)}, got {intermediates.shape}"

    def test_coca_dynamics_respects_time_step_scaling(self):
        num_agents = 3
        connectivity = np.ones((num_agents, num_agents)) - np.eye(num_agents)

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=[0.1, 0.5, 0.9],
            desired_opinion=1.0,
            connectivity_matrix=connectivity,
            dynamics_model="coca",
            t_campaign=1.0,
            t_s=0.5,
        )
        env.reset()

        # No control, only propagation
        action = np.zeros(num_agents, dtype=np.float32)

        state_tiny_step, _, _, _, info_small = env.step(action, t_campaign=1.0, t_s=0.1)
        state_big_step, _, _, _, info_big = env.step(action, t_campaign=1.0, t_s=0.5)

        # Expect: more frequent sampling (smaller t_s) → slightly different results, but not wildly divergent
        np.testing.assert_allclose(
            state_tiny_step,
            state_big_step,
            atol=0.1,
            err_msg="COCA integration with different t_s values should produce similar qualitative behavior",
        )

    def test_impulse_only_no_propagation(self):
        num_agents = 3
        initial_opinions = np.array([0.0, 0.5, 1.0])
        action = np.array([0.5, 0.0, 0.0])

        env = NetworkGraph(
            num_agents=num_agents,
            initial_opinions=initial_opinions,
            desired_opinion=1.0,
            t_campaign=0.0,  # No propagation
            t_s=0.1,
            max_u=0.5,
        )
        env.reset()

        assert np.all(
            env.opinions == initial_opinions
        ), "Environment should start with initial opinions"

        next_state, _, _, _, info = env.step(action, t_campaign=0.0)

        expected_state = initial_opinions.copy()
        effective_control = action[0] * (1 - env.control_resistance[0])
        expected_state[0] = (
            effective_control * env.desired_opinion_vector[0]
            + (1 - effective_control) * expected_state[0]
        )

        np.testing.assert_allclose(
            next_state,
            expected_state,
            atol=1e-6,
            err_msg="Impulse-only step should only apply control, no dynamics",
        )

        # Check that intermediate_states includes only one state (the impulse result)
        assert (
            info["intermediate_states"].shape[0] == 1
        ), "Impulse-only step should have exactly one intermediate state"
