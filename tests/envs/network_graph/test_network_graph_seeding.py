"""NetworkGraph's constructor and episode reset reproducibility contracts."""

import copy

import numpy as np
import pytest

from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


@pytest.mark.parametrize("seed", [0, 42, 999])
@pytest.mark.parametrize(
    "graph_config",
    [
        {"graph_connection_distribution": "uniform", "bidirectional_prob": 0.0},
        {"graph_connection_distribution": "normal", "bidirectional_prob": 1.0},
        {"graph_connection_distribution": "exponential"},
        {"graph_model": "barabasi_albert", "use_weighted_edges": True},
        {
            "connectivity_matrix": np.ones((10, 10)) - np.eye(10),
            "use_weighted_edges": True,
        },
    ],
    ids=["directed", "undirected", "exponential", "weighted-ba", "provided"],
)
def test_reset_seed_only_restarts_opinion_sampling(seed, graph_config):
    env = NetworkGraph(seed=seed, **copy.deepcopy(graph_config))
    matrices = {
        name: getattr(env, name).copy()
        for name in ("connectivity_matrix", "L", "expL_ts", "centralities")
    }
    expected_rng = np.random.default_rng(seed)

    for reset_seed in (seed, None):
        observation, info = env.reset(seed=reset_seed)
        np.testing.assert_array_equal(
            observation,
            expected_rng.uniform(*env.initial_opinion_range, size=env.num_agents),
        )
        assert info == {"random_opinions": True}
        for name, expected in matrices.items():
            np.testing.assert_array_equal(getattr(env, name), expected)
        np.testing.assert_equal(env.np_random.bit_generator.state, expected_rng.bit_generator.state)


@pytest.mark.parametrize(
    "dynamics_model",
    [
        "laplacian",
        "coca",
        "friedkinjohnsen",
        "hegselmannkrause",
        "nonlinearinfluence",
        "repulsion",
    ],
)
@pytest.mark.parametrize("custom_prejudice", [False, True])
def test_seeded_reset_replays_controlled_trajectory(dynamics_model, custom_prejudice):
    prejudice = np.linspace(0.1, 0.9, 10) if custom_prejudice else None
    graph_config = (
        {"connectivity_matrix": (np.ones((10, 10)) - np.eye(10)) / 9}
        if dynamics_model == "friedkinjohnsen" else {}
    )
    env = NetworkGraph(
        seed=999,
        dynamics_model=dynamics_model,
        initial_opinions=np.linspace(0, 1, 10),
        fj_prejudice=prejudice,
        max_steps=3,
        terminate_when_converged=False,
        use_delta_shaping=True,
        delta_lambda=0.5,
        **graph_config,
    )
    actions = [np.full(10, value, dtype=np.float32) for value in (0.025, 0.075, 0.05)]

    initial, info = env.reset(seed=0, options={"randomize_opinions": True})
    expected_prejudice = prejudice if custom_prejudice else initial.copy()
    np.testing.assert_array_equal(env.fj_prejudice, expected_prejudice)
    expected = [copy.deepcopy(env.step(action)) for action in actions]
    assert env.current_step == 3
    assert env.total_spent > 0
    assert expected[-1][3]

    repeated, repeated_info = env.reset(seed=0, randomize_opinions=True)
    np.testing.assert_array_equal(repeated, initial)
    assert repeated_info == info
    assert env.current_step == 0
    assert env.total_spent == 0
    np.testing.assert_array_equal(env.fj_prejudice, expected_prejudice)
    np.testing.assert_equal([env.step(action) for action in actions], expected)
