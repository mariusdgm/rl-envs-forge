import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph

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
        default_env = NetworkGraph(num_agents=3, budget=0.05, impulse_resistance=np.array([1.0, 1.0, 1.0]))
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
        assert mock_show.called, "plt.show should be called when rendering in 'matplotlib' mode"

    def test_render_output(self, default_env):
        default_env.reset()
        with patch('builtins.print') as mock_print:
            default_env.render(mode="human")
            mock_print.assert_called()



    def test_centrality_computation_with_weighted_edges(self, weighted_env):
        L = weighted_env.L
        centralities = weighted_env.centralities
        assert np.sum(centralities) > 0, "Centralities should be computed correctly for weighted edges"
        
        # Adjust the check to ensure the Laplacian is valid
        assert np.all(np.diag(L) >= 0), "Diagonal of Laplacian should be non-negative"
        assert np.all(L - np.diag(np.diag(L)) <= 0), "Off-diagonal elements of Laplacian should be non-positive"

   
    def test_saturation_of_control_and_opinions(self, default_env):
        # Set up the environment with specific parameters
        num_agents = 3
        max_u = 0.1  # Maximum control input
        initial_opinions = np.array([0.0, 0.5, 1.0])  # Initial opinions ranging from 0 to 1
        
        # Create an environment with these parameters
        env = NetworkGraph(
            num_agents=num_agents,
            max_u=max_u,
            initial_opinions=initial_opinions,
            budget=10.0,
            
            desired_opinion=1.0,
            tau=1.0,
            max_steps=100
        )
        
        env.reset()

        # Define an action that exceeds the maximum control input
        action = np.array([0.2, 0.15, 0.25], dtype=np.float32)
        
        # Step the environment with the action
        state, reward, done, truncated, info = env.step(action)
        
        # Retrieve the clipped action from the info dictionary
        applied_action = info["action_applied"]
        
        # Check that the action was clipped to the max_u
        np.testing.assert_array_less(applied_action, np.full(action.shape, max_u + 1e-5), 
                                     "Control inputs should be clipped to max_u")
        
        # Check that the state (opinions) remains within [0, 1]
        assert np.all(state >= 0.0) and np.all(state <= 1.0), "Opinions should be within the range [0, 1]"
