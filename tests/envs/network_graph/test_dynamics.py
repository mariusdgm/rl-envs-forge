import pytest
import numpy as np
from scipy.linalg import expm

from rl_envs_forge.envs.network_graph.dynamics import rollout_dynamics
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph
from rl_envs_forge.envs.network_graph.graph_utils import compute_laplacian


@pytest.fixture
def small_adjacency():
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


class TestDynamics:
    def test_degroot_alias_matches_laplacian_rollout(self, small_adjacency):
        A = small_adjacency
        x0 = np.array([0.1, 0.6, 0.9], dtype=float)
        u0 = np.zeros(3, dtype=float)
        d = np.ones(3, dtype=float)
        r = np.zeros(3, dtype=float)
        t_s = 0.1
        t_campaign = 0.3

        L = compute_laplacian(A)
        expL_ts = expm(-L * t_s)

        x_lap, inter_lap = rollout_dynamics(
            x0,
            u0,
            dynamics_model="laplacian",
            adjacency_matrix=A,
            L=L,
            t_campaign=t_campaign,
            t_s=t_s,
            desired_opinion_vector=d,
            control_resistance=r,
            expL_ts=expL_ts,
            default_t_s=t_s,
        )
        x_deg, inter_deg = rollout_dynamics(
            x0,
            u0,
            dynamics_model="degroot",
            adjacency_matrix=A,
            L=L,
            t_campaign=t_campaign,
            t_s=t_s,
            desired_opinion_vector=d,
            control_resistance=r,
            expL_ts=expL_ts,
            default_t_s=t_s,
        )

        np.testing.assert_allclose(x_deg, x_lap, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(inter_deg, inter_lap, atol=1e-12, rtol=1e-12)

    def test_friedkinjohnsen_one_step_matches_formula(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.2, 0.8], dtype=float)
        u0 = np.zeros(2, dtype=float)
        d = np.ones(2, dtype=float)
        r = np.zeros(2, dtype=float)
        prejudice = np.array([0.1, 0.9], dtype=float)
        fj_lambda = np.array([0.75, 0.25], dtype=float)

        x1, inter = rollout_dynamics(
            x0,
            u0,
            dynamics_model="friedkinjohnsen",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=1.0,
            t_s=1.0,
            desired_opinion_vector=d,
            control_resistance=r,
            fj_lambda=fj_lambda,
            prejudice=prejudice,
        )

        expected = fj_lambda * (A @ x0) + (1.0 - fj_lambda) * prejudice
        np.testing.assert_allclose(x1, expected, atol=1e-12, rtol=1e-12)
        assert inter.shape == (2, 2)

    def test_hegselmannkrause_filters_far_neighbors(self):
        A = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        x0 = np.array([0.10, 0.15, 0.90], dtype=float)
        u0 = np.zeros(3, dtype=float)
        d = np.ones(3, dtype=float)
        r = np.zeros(3, dtype=float)

        x1, inter = rollout_dynamics(
            x0,
            u0,
            dynamics_model="hegselmannkrause",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=1.0,
            t_s=1.0,
            desired_opinion_vector=d,
            control_resistance=r,
            hk_epsilon=0.10,
            hk_include_self=True,
        )

        expected = np.array([(0.10 + 0.15) / 2.0, (0.15 + 0.10) / 2.0, 0.90])
        np.testing.assert_allclose(x1, expected, atol=1e-12, rtol=1e-12)
        assert inter.shape == (2, 3)

    def test_friedkinjohnsen_requires_lambda(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.2, 0.8], dtype=float)
        u0 = np.zeros(2, dtype=float)
        d = np.ones(2, dtype=float)
        r = np.zeros(2, dtype=float)

        with pytest.raises(ValueError, match="fj_lambda"):
            rollout_dynamics(
                x0,
                u0,
                dynamics_model="friedkinjohnsen",
                adjacency_matrix=A,
                L=compute_laplacian(A),
                t_campaign=1.0,
                t_s=1.0,
                desired_opinion_vector=d,
                control_resistance=r,
                fj_lambda=None,
                prejudice=np.array([0.1, 0.9], dtype=float),
            )

    def test_friedkinjohnsen_requires_prejudice(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.2, 0.8], dtype=float)
        u0 = np.zeros(2, dtype=float)
        d = np.ones(2, dtype=float)
        r = np.zeros(2, dtype=float)

        with pytest.raises(ValueError, match="prejudice"):
            rollout_dynamics(
                x0,
                u0,
                dynamics_model="friedkinjohnsen",
                adjacency_matrix=A,
                L=compute_laplacian(A),
                t_campaign=1.0,
                t_s=1.0,
                desired_opinion_vector=d,
                control_resistance=r,
                fj_lambda=0.9,
                prejudice=None,
            )

    def test_hegselmannkrause_requires_epsilon(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.2, 0.8], dtype=float)
        u0 = np.zeros(2, dtype=float)
        d = np.ones(2, dtype=float)
        r = np.zeros(2, dtype=float)

        with pytest.raises(ValueError, match="hk_epsilon"):
            rollout_dynamics(
                x0,
                u0,
                dynamics_model="hegselmannkrause",
                adjacency_matrix=A,
                L=compute_laplacian(A),
                t_campaign=1.0,
                t_s=1.0,
                desired_opinion_vector=d,
                control_resistance=r,
                hk_epsilon=None,
            )

    def test_networkgraph_accepts_degroot_name(self, small_adjacency):
        env = NetworkGraph(
            num_agents=3,
            connectivity_matrix=small_adjacency,
            initial_opinions=np.array([0.1, 0.6, 0.9]),
            dynamics_model="degroot",
            t_campaign=0.2,
            t_s=0.1,
        )
        env.reset()
        out, inter = env.compute_dynamics(
            env.opinions.copy(),
            np.zeros(3),
            t_campaign=0.2,
            t_s=0.1,
        )
        assert out.shape == (3,)
        assert inter.shape == (3, 3)

    def test_networkgraph_friedkinjohnsen_uses_custom_prejudice_after_reset(self):
        prejudice = np.array([0.9, 0.1], dtype=float)
        env = NetworkGraph(
            num_agents=2,
            connectivity_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            initial_opinions=None,
            dynamics_model="friedkinjohnsen",
            fj_lambda=np.array([0.0, 0.0], dtype=float),
            fj_prejudice=prejudice,
            seed=123,
            t_campaign=1.0,
            t_s=1.0,
        )
        env.reset(randomize_opinions=True)
        out, _ = env.compute_dynamics(
            env.opinions.copy(),
            np.zeros(2),
            t_campaign=1.0,
            t_s=1.0,
        )

        np.testing.assert_allclose(out, prejudice, atol=1e-12, rtol=1e-12)

    def test_networkgraph_hegselmannkrause_step_runs(self):
        env = NetworkGraph(
            num_agents=3,
            connectivity_matrix=np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            initial_opinions=np.array([0.10, 0.15, 0.90]),
            dynamics_model="hegselmannkrause",
            hk_epsilon=0.10,
            hk_include_self=True,
            t_campaign=1.0,
            t_s=1.0,
        )
        env.reset()
        state, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert state.shape == (3,)
        assert isinstance(reward, float)
        assert "intermediate_states" in info
        assert info["intermediate_states"].shape == (2, 3)
