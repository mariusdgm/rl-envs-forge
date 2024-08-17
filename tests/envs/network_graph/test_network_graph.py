import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


@pytest.fixture
def default_env():
    """Fixture to create a default NetworkGraph environment for each test."""
    return NetworkGraph()

@pytest.fixture
def custom_adjacency_matrix_env():
    """Fixture to create a NetworkGraph environment with a custom adjacency matrix."""
    custom_matrix = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    return NetworkGraph(custom_adjacency_matrix=custom_matrix)

@pytest.fixture
def centrality_connectivity_env():
    """Fixture to create a NetworkGraph environment using connectivity and centrality."""
    connectivity_matrix = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    desired_centrality = np.array([0.3, 0.4, 0.3])
    return NetworkGraph(connectivity_matrix=connectivity_matrix, desired_centrality=desired_centrality)

class TestNetworkGraph:
    def test_initialization(self, default_env):
        assert default_env.num_agents == 100
        assert default_env.budget == 10.0
        assert default_env.tau == 1.0
        assert default_env.max_steps == 100
        assert default_env.current_step == 0
        assert default_env.total_spent == 0.0

    def test_initialization_with_custom_adjacency_matrix(self, custom_adjacency_matrix_env):
        assert custom_adjacency_matrix_env.num_agents == 3
        np.testing.assert_array_equal(custom_adjacency_matrix_env.adjacency_matrix, np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]))

    def test_initialization_with_connectivity_and_centrality(self, centrality_connectivity_env):
        assert centrality_connectivity_env.num_agents == 3
        assert centrality_connectivity_env.adjacency_matrix is not None

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

    def test_render(self, default_env):
        with patch('your_module.draw_network_graph') as mock_draw_network_graph:
            default_env.reset()
            default_env.render()
            mock_draw_network_graph.assert_called_once()

    def test_render_output(self, default_env):
        default_env.reset()
        with patch('builtins.print') as mock_print:
            default_env.render()
            mock_print.assert_called()

    def test_invalid_initialization(self):
        with pytest.raises(ValueError):
            NetworkGraph(
                custom_adjacency_matrix=np.array([
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ]),
                connectivity_matrix=np.array([
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ]),
                desired_centrality=np.array([0.3, 0.4, 0.3])
            )
