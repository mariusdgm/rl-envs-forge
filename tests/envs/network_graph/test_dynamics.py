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
        x1, inter = rollout_dynamics(
            x0,
            np.zeros(2),
            dynamics_model="friedkinjohnsen",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=1.0,
            t_s=1.0,
            desired_opinion_vector=np.ones(2),
            control_resistance=np.zeros(2),
            fj_lambda=np.array([0.75, 0.25]),
            prejudice=np.array([0.1, 0.9], dtype=float),
        )
        expected = np.array([0.75, 0.25]) * (A @ x0) + (
            1.0 - np.array([0.75, 0.25])
        ) * np.array([0.1, 0.9])
        np.testing.assert_allclose(x1, expected, atol=1e-12, rtol=1e-12)
        assert inter.shape == (2, 2)

    def test_hegselmannkrause_filters_far_neighbors(self):
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=float)
        x0 = np.array([0.10, 0.15, 0.90], dtype=float)
        x1, _ = rollout_dynamics(
            x0,
            np.zeros(3),
            dynamics_model="hegselmannkrause",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=1.0,
            t_s=1.0,
            desired_opinion_vector=np.ones(3),
            control_resistance=np.zeros(3),
            hk_epsilon=0.10,
            hk_include_self=True,
        )
        expected = np.array([(0.10 + 0.15) / 2.0, (0.15 + 0.10) / 2.0, 0.90])
        np.testing.assert_allclose(x1, expected, atol=1e-12, rtol=1e-12)

    def test_nonlinearinfluence_uses_tanh_pairwise_update(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.2, 0.8], dtype=float)
        beta = 2.0
        t_s = 0.1
        x1, _ = rollout_dynamics(
            x0,
            np.zeros(2),
            dynamics_model="nonlinearinfluence",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=t_s,
            t_s=t_s,
            desired_opinion_vector=np.ones(2),
            control_resistance=np.zeros(2),
            nonlinear_beta=beta,
        )
        diff = x0[1] - x0[0]
        expected = np.array(
            [
                x0[0] + t_s * np.tanh(beta * diff),
                x0[1] + t_s * np.tanh(beta * (-diff)),
            ]
        )
        np.testing.assert_allclose(
            x1, np.clip(expected, 0.0, 1.0), atol=1e-12, rtol=1e-12
        )

    def test_repulsion_repels_when_difference_exceeds_threshold(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.2, 0.8], dtype=float)
        t_s = 0.1
        x1, _ = rollout_dynamics(
            x0,
            np.zeros(2),
            dynamics_model="repulsion",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=t_s,
            t_s=t_s,
            desired_opinion_vector=np.ones(2),
            control_resistance=np.zeros(2),
            repulsion_epsilon=0.25,
            repulsion_strength=0.5,
        )
        assert x1[0] < x0[0]
        assert x1[1] > x0[1]

    def test_repulsion_attracts_when_difference_within_threshold(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        x0 = np.array([0.4, 0.5], dtype=float)
        t_s = 0.1
        x1, _ = rollout_dynamics(
            x0,
            np.zeros(2),
            dynamics_model="repulsion",
            adjacency_matrix=A,
            L=compute_laplacian(A),
            t_campaign=t_s,
            t_s=t_s,
            desired_opinion_vector=np.ones(2),
            control_resistance=np.zeros(2),
            repulsion_epsilon=0.25,
            repulsion_strength=0.5,
        )
        assert x1[0] > x0[0]
        assert x1[1] < x0[1]

    def test_networkgraph_accepts_new_dynamics_names(self, small_adjacency):
        for model, kwargs in [
            ("degroot", {}),
            ("friedkinjohnsen", {"fj_lambda": 0.8}),
            ("hegselmannkrause", {"hk_epsilon": 0.2}),
            ("nonlinearinfluence", {"nonlinear_beta": 2.0}),
            ("repulsion", {"repulsion_epsilon": 0.2, "repulsion_strength": 0.5}),
        ]:
            env = NetworkGraph(
                num_agents=3,
                connectivity_matrix=small_adjacency,
                initial_opinions=np.array([0.1, 0.6, 0.9]),
                dynamics_model=model,
                t_campaign=0.2,
                t_s=0.1,
                **kwargs,
            )
            env.reset()
            out, inter = env.compute_dynamics(
                env.opinions.copy(), np.zeros(3), t_campaign=0.2, t_s=0.1
            )
            assert out.shape == (3,)
            assert inter.shape[1] == 3
