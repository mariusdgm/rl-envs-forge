"""FJ mathematical assumptions and graph acceptance contracts."""

import numpy as np
import networkx as nx
import pytest
from gymnasium.utils.env_checker import check_env

from rl_envs_forge.envs.network_graph.dynamics import friedkin_johnsen_step, rollout_dynamics
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


ALIASES = ["friedkinjohnsen", "friedkin_johnsen", "fj", "FJ"]
A = np.array([[0.0, 1.0], [1.0, 0.0]])


@pytest.mark.parametrize("alias", ALIASES)
def test_fj_aliases_match_hand_calculated_one_step(alias):
    env = NetworkGraph(
        connectivity_matrix=A, dynamics_model=alias, initial_opinions=[0.2, 0.8],
        fj_lambda=[0.75, 0.25], fj_prejudice=[0.1, 0.9],
        t_campaign=1, t_s=1, seed=0,
    )
    observation, *_ = env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(observation, [0.625, 0.725], rtol=0, atol=1e-15)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("constant", [0.0, 0.37, 1.0])
def test_fj_constant_vector_invariance(seed, constant):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(size=(5, 5))
    matrix /= matrix.sum(axis=1, keepdims=True)  # Explicit preparation by the caller.
    env = NetworkGraph(
        connectivity_matrix=matrix, dynamics_model="fj", seed=seed,
        initial_opinions=np.full(5, constant), desired_opinion=constant,
        fj_lambda=rng.uniform(size=5), t_campaign=1, t_s=0.1,
        terminate_when_converged=False,
    )
    for _ in range(5):
        observation, *_ = env.step(rng.uniform(0, 0.1, 5).astype(np.float32))
        np.testing.assert_allclose(observation, constant, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_fj_boundedness_and_python_reference_with_control(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(size=(4, 4))
    matrix /= matrix.sum(axis=1, keepdims=True)
    initial, prejudice, lam = rng.uniform(size=(3, 4))
    lam[0], lam[-1] = 0, 1
    resistance, target = rng.uniform(size=(2, 4))
    env = NetworkGraph(
        connectivity_matrix=matrix, dynamics_model="fj", initial_opinions=initial,
        fj_lambda=lam, fj_prejudice=prejudice, control_resistance=resistance,
        desired_opinion=target, t_campaign=0.3, t_s=0.1, seed=seed,
        terminate_when_converged=False,
    )
    expected = list(initial)
    for _ in range(10):
        action = rng.uniform(0, 0.1, 4).astype(np.float32)
        expected = [float(action[i]) * (1-resistance[i]) * target[i]
                    + (1-float(action[i]) * (1-resistance[i])) * expected[i]
                    for i in range(4)]
        for _ in range(3):
            expected = [lam[i] * sum(float(matrix[i, j]) * expected[j] for j in range(4))
                        + (1-lam[i]) * prejudice[i] for i in range(4)]
        observation, *_ = env.step(action)
        np.testing.assert_allclose(observation, expected, rtol=0, atol=2e-15)
        assert np.all((observation >= 0) & (observation <= 1))


@pytest.mark.parametrize("matrix", [
    np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 3)), np.array([1, 1]),
    np.array([[0, -1], [1, 0]]), np.array([[0, np.nan], [1, 0]]),
    np.array([[0, np.inf], [1, 0]]), np.array([[0, 1j], [1, 0]]),
])
@pytest.mark.parametrize("alias", ALIASES)
def test_fj_invalid_matrices_are_rejected_without_modification(matrix, alias):
    original = matrix.copy()
    with pytest.raises(ValueError, match="FJ"):
        NetworkGraph(connectivity_matrix=matrix, dynamics_model=alias, seed=0)
    np.testing.assert_equal(matrix, original)
    with pytest.raises(ValueError, match="FJ"):
        friedkin_johnsen_step(np.array([0.2, 0.8]), adjacency_matrix=matrix,
                             fj_lambda=0.9, prejudice=np.array([0.2, 0.8]))


@pytest.mark.parametrize("kwargs", [
    {"fj_lambda": -0.1}, {"fj_lambda": 1.1}, {"fj_lambda": np.nan}, {"fj_lambda": np.inf},
    {"fj_lambda": [0.2]}, {"fj_lambda": [[0.2], [0.8]]},
    {"fj_prejudice": [0.5]}, {"fj_prejudice": [0, np.inf]}, {"fj_prejudice": [0, np.nan]},
    {"fj_prejudice": [-0.1, 0.5]}, {"fj_prejudice": [0, 1.1]},
    {"initial_opinions": [0.2]}, {"initial_opinions": [0, 1.1]},
    {"initial_opinions": [0.2j, 0.3]}, {"initial_opinion_range": [-0.1, 1]},
    {"initial_opinion_range": [0.9, 0.1]}, {"initial_opinion_range": [0, np.inf]},
    {"max_u": 1.1}, {"max_u": [-0.1, 0.1]}, {"max_u": [0.1]},
    {"desired_opinion": 1.1}, {"control_resistance": [0, 1.1]},
])
def test_fj_parameter_ranges_and_dimensions_are_explicit(kwargs):
    with pytest.raises(ValueError, match="FJ"):
        NetworkGraph(connectivity_matrix=A, dynamics_model="fj", seed=0, **kwargs)


def test_invalid_lambda_is_not_clipped_and_direct_rollout_validates_zero_duration():
    with pytest.raises(ValueError, match="fj_lambda"):
        friedkin_johnsen_step(np.array([0.2, 0.8]), adjacency_matrix=A,
                             fj_lambda=1.1, prejudice=np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="control_action"):
        rollout_dynamics(
            np.array([0.2, 0.8]), np.array([2.0, 0.0]), dynamics_model="fj",
            adjacency_matrix=A, L=np.eye(2)-A, t_campaign=0, t_s=1,
            desired_opinion_vector=np.ones(2), control_resistance=np.zeros(2),
            fj_lambda=0.9, prejudice=np.array([0.2, 0.8]),
        )


def test_fj_mutated_parameters_and_adjacency_are_revalidated():
    env = NetworkGraph(connectivity_matrix=A, dynamics_model="fj", seed=0)
    env.fj_lambda = 2
    with pytest.raises(ValueError, match="fj_lambda"):
        env.reset(seed=0)
    env.fj_lambda = 0.9
    env.connectivity_matrix[0, 1] = 2
    with pytest.raises(ValueError, match="row-stochastic"):
        env.step(np.zeros(2))


def test_fj_validation_follows_the_selected_model_after_a_model_change():
    env = NetworkGraph(connectivity_matrix=np.ones((2, 2)), seed=0)
    env.dynamics_model = "FJ"
    with pytest.raises(ValueError, match="row-stochastic"):
        env.reset(seed=0)
    env.dynamics_model = "coca"
    env.reset(seed=0)
    observation, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("distribution", ["uniform", "normal", "exponential"])
def test_random_generation_produces_valid_fj_input_or_explicitly_rejects(distribution, weighted):
    valid = NetworkGraph(
        num_agents=2, graph_connection_distribution=distribution,
        bidirectional_prob=1, dynamics_model="fj", seed=0, use_weighted_edges=weighted,
    )
    np.testing.assert_array_equal(valid.connectivity_matrix.sum(axis=1), 1)
    with pytest.raises(ValueError, match="incompatible with FJ"):
        NetworkGraph(num_agents=6, dynamics_model="fj", seed=0,
                     graph_connection_distribution=distribution, use_weighted_edges=weighted)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("criterion", ["paper", "no_singleton_sink", "strong", "strongly_connected"])
def test_ba_generates_valid_fj_inputs_and_replays(weighted, criterion, strict):
    config = dict(num_agents=6, graph_model="barabasi_albert", ba_m=2,
                  ba_prune_max_frac=0, ba_connectivity=criterion,
                  use_weighted_edges=weighted, dynamics_model="fj", seed=42,
                  strict_connectivity=strict)
    first, second = NetworkGraph(**config), NetworkGraph(**config)
    np.testing.assert_array_equal(first.connectivity_matrix, second.connectivity_matrix)
    np.testing.assert_array_equal(first.opinions, second.opinions)
    np.testing.assert_allclose(first.connectivity_matrix.sum(axis=1), 1, rtol=0, atol=1e-12)
    check_env(first, skip_render_check=True)


def test_ba_zero_row_fallback_is_rejected_by_fj_even_in_best_effort_mode():
    with pytest.raises(ValueError, match="incompatible with FJ"):
        NetworkGraph(num_agents=4, graph_model="barabasi_albert", ba_max_tries=0,
                     dynamics_model="fj", seed=0)


def test_fj_validates_the_result_of_supplied_matrix_weighting():
    original = A.astype(np.int64)
    with pytest.raises(ValueError, match="row-stochastic"):
        NetworkGraph(connectivity_matrix=original, use_weighted_edges=True,
                     dynamics_model="fj", seed=0)
    np.testing.assert_array_equal(original, A)
    valid = NetworkGraph(
        connectivity_matrix=original, use_weighted_edges=True,
        weight_range=(1, 1), dynamics_model="fj", seed=0,
    )
    np.testing.assert_array_equal(valid.connectivity_matrix, A)


@pytest.mark.parametrize("strict", [False, True])
def test_disconnected_random_graph_can_have_no_isolates(strict, capsys):
    # Seeded isolated-node repair gives multiple components; no resampling.
    env = NetworkGraph(num_agents=4, connection_prob_range=(0, 0),
                       bidirectional_prob=1, seed=7, strict_connectivity=strict)
    graph = nx.from_numpy_array(env.connectivity_matrix)
    assert min(dict(graph.degree()).values()) > 0
    assert not nx.is_connected(graph)
    assert "does not require weak connectivity" in capsys.readouterr().out


@pytest.mark.parametrize("criterion", ["weak", "strong"])
def test_strict_additional_requirement_rejects_disconnected_random_graph(criterion):
    with pytest.raises(ValueError, match=criterion):
        NetworkGraph(num_agents=4, connection_prob_range=(0, 0),
                     bidirectional_prob=1, seed=7, strict_connectivity=True,
                     connectivity_requirement=criterion)


def test_directed_weak_and_strong_criteria_are_distinct():
    chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    NetworkGraph(connectivity_matrix=chain, connectivity_requirement="weak",
                 strict_connectivity=True)
    with pytest.raises(ValueError, match="strong"):
        NetworkGraph(connectivity_matrix=chain, connectivity_requirement="strong",
                     strict_connectivity=True)
    # Row Laplacian has one zero, but the unique sink SCC is a singleton.
    env = NetworkGraph(connectivity_matrix=chain)
    graph = nx.from_numpy_array(chain, create_using=nx.DiGraph)
    assert env._passes_connectivity_filter(graph, chain, "paper")
    assert not env._passes_connectivity_filter(graph, chain, "no_singleton_sink")
    sink_pair = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    graph = nx.from_numpy_array(sink_pair, create_using=nx.DiGraph)
    assert env._passes_connectivity_filter(graph, sink_pair, "paper")
    assert env._passes_connectivity_filter(graph, sink_pair, "no_singleton_sink")
    assert not env._passes_connectivity_filter(graph, sink_pair, "strong")


@pytest.mark.parametrize("criterion", ["paper", "no_singleton_sink", "strong"])
def test_ba_retry_exhaustion_warns_for_selected_filter_and_strict_rejects(criterion, capsys):
    # Zero pruning leaves a valid sample; an intentionally large spectral
    # tolerance forces failure for spectral criteria. Strong uses a seeded
    # fully-pruned sample instead (deterministic seed discovered independently).
    config = dict(num_agents=3, ba_m=1, graph_model="barabasi_albert",
                  ba_connectivity=criterion, ba_max_tries=1, seed=0)
    if criterion == "strong":
        config.update(ba_prune_max_frac=1)
    else:
        config.update(ba_qsc_tol=10, ba_prune_max_frac=0)
    env = NetworkGraph(**config)
    assert criterion in capsys.readouterr().out
    assert not env._passes_ba_connectivity_filter(
        nx.from_numpy_array(env.connectivity_matrix, create_using=nx.DiGraph),
        env.connectivity_matrix,
    )
    with pytest.raises(ValueError, match="exhausted 1 attempts"):
        NetworkGraph(**config, strict_connectivity=True)


def test_ba_zero_attempt_fallback_is_preserved_and_strict_rejects(capsys):
    env = NetworkGraph(num_agents=4, graph_model="barabasi_albert", ba_max_tries=0, seed=0)
    np.testing.assert_array_equal(env.connectivity_matrix, np.zeros((4, 4)))
    assert "zero matrix (no attempts)" in capsys.readouterr().out
    with pytest.raises(ValueError, match="exhausted 0 attempts"):
        NetworkGraph(num_agents=4, graph_model="barabasi_albert", ba_max_tries=0,
                     seed=0, strict_connectivity=True)


def test_erdos_renyi_remains_unsupported_and_unknown_criteria_are_rejected():
    with pytest.raises(ValueError, match="Unknown graph_model"):
        NetworkGraph(graph_model="erdos_renyi")
    with pytest.raises(ValueError, match="connectivity_requirement"):
        NetworkGraph(connectivity_requirement="connected-ish")
    with pytest.raises(ValueError, match="ba_connectivity"):
        NetworkGraph(graph_model="barabasi_albert", ba_connectivity="connected-ish")


@pytest.mark.parametrize("distribution", ["uniform", "normal", "exponential"])
@pytest.mark.parametrize("bidirectional", [0, 1])
def test_random_graph_strict_default_preserves_graph_opinions_and_rng(distribution, bidirectional):
    config = dict(num_agents=6, seed=42, graph_connection_distribution=distribution,
                  bidirectional_prob=bidirectional)
    first = NetworkGraph(**config)
    second = NetworkGraph(**config, strict_connectivity=True)
    np.testing.assert_array_equal(first.connectivity_matrix, second.connectivity_matrix)
    np.testing.assert_array_equal(first.opinions, second.opinions)
    np.testing.assert_equal(first.np_random.bit_generator.state, second.np_random.bit_generator.state)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_valid_fj_matrix_dtypes_preserve_values(dtype):
    matrix = A.astype(dtype)
    env = NetworkGraph(connectivity_matrix=matrix, dynamics_model="fj", seed=0)
    assert env.connectivity_matrix.dtype == dtype
    np.testing.assert_array_equal(env.connectivity_matrix, matrix)


def test_boolean_endpoint_parameters_remain_valid_fj_inputs():
    env = NetworkGraph(
        connectivity_matrix=A, dynamics_model="fj", seed=0,
        initial_opinions=[False, True], control_resistance=[True, False],
        desired_opinion=True, max_u=True, fj_lambda=True,
        fj_prejudice=[False, True], t_campaign=1, t_s=1,
    )
    observed, *_ = env.step(np.array([0.7, 0.2], dtype=np.float32))
    np.testing.assert_array_equal(observed, [1, 0])


@pytest.mark.parametrize("lam,expected", [
    (0, [0.1, 0.9]), (1, [0.8, 0.2]), (0.5, [0.45, 0.55]),
])
def test_scalar_lambda_endpoints_and_broadcasting(lam, expected):
    observed = friedkin_johnsen_step(
        np.array([0.2, 0.8]), adjacency_matrix=A,
        fj_lambda=lam, prejudice=np.array([0.1, 0.9]),
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("model", [
    "laplacian", "degroot", "coca", "hegselmannkrause", "nonlinearinfluence", "repulsion",
])
def test_other_models_match_independent_two_node_reference(model):
    import math

    initial = [0.2, 0.8]
    expected = list(initial)
    if model in ("laplacian", "degroot"):
        delta = 0.3 * math.exp(-1.4 * 0.3)
        expected = [0.5-delta, 0.5+delta]
    else:
        for _ in range(3):
            updated = []
            for i in range(2):
                x, peer = expected[i], expected[1-i]
                if model == "coca":
                    value = x + 0.1 * x * (1-x) * (peer-x)
                elif model == "hegselmannkrause":
                    value = (x + 0.7 * peer) / 1.7
                elif model == "nonlinearinfluence":
                    value = x + 0.1 * 0.7 * math.tanh(2 * (peer-x))
                else:
                    influence = peer-x if abs(peer-x) <= 0.25 else -0.5 * (peer-x)
                    value = x + 0.1 * 0.7 * influence
                updated.append(min(1.0, max(0.0, value)))
            expected = updated
    env = NetworkGraph(
        connectivity_matrix=np.array([[0, 0.7], [0.7, 0]]),
        initial_opinions=initial, dynamics_model=model, hk_epsilon=1,
        t_campaign=0.3, t_s=0.1, terminate_when_converged=False, seed=0,
    )
    observed, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
    assert reward == pytest.approx(-1)
    assert not terminated and not truncated
