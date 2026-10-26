"""Repeated FJ recurrences, valid reset domains and manual-layout diagnostics."""

import copy

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
from scipy.special import softmax

from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem
from rl_envs_forge.envs.grid_world.grid_world import GridWorld
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.network_graph.dynamics import friedkin_johnsen_step
from rl_envs_forge.envs.network_graph.graph_utils import normalize_adjacency_matrix
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


def matrix_for(kind):
    raw = np.random.default_rng(0).uniform(size=(7, 7))
    if kind == "float32":
        return softmax(raw.astype(np.float32), axis=1)
    if kind == "float64":
        return softmax(raw, axis=1)
    return np.roll(np.eye(7), 1, axis=1)


@pytest.mark.parametrize("alias", ["fj", "FJ", "friedkinjohnsen", "friedkin_johnsen"])
@pytest.mark.parametrize("kind", ["float32", "float64", "exact"])
@pytest.mark.parametrize("endpoint", [0.0, 1.0])
def test_repeated_fj_endpoint_updates_match_scalar_reference(alias, kind, endpoint):
    matrix = matrix_for(kind)
    original = matrix.copy()
    env = NetworkGraph(
        connectivity_matrix=matrix, dynamics_model=alias, seed=0,
        initial_opinions=np.full(7, endpoint), fj_lambda=1,
        t_campaign=0.4, t_s=0.1, max_steps=75, terminate_when_converged=False,
    )
    initial, _ = env.reset(seed=0)
    assert env.observation_space.contains(initial)
    space = env.observation_space
    expected = initial.copy()
    maximum_seen = endpoint
    for step in range(75):
        for _ in range(4):
            expected = np.array([sum(float(matrix[i, j]) * expected[j] for j in range(7))
                                 for i in range(7)])
        observation, reward, terminated, truncated, info = env.step(np.zeros(7, np.float32))
        np.testing.assert_allclose(observation, expected, rtol=0, atol=1e-13)
        assert env.observation_space is space
        assert space.contains(observation)
        assert np.all(np.isfinite(info["intermediate_states"]))
        maximum_seen = max(maximum_seen, info["intermediate_states"].max())
        assert np.isfinite(reward)
        assert not terminated
        assert truncated == (step == 74)
    if kind == "exact" or endpoint == 0:
        np.testing.assert_array_equal(observation, np.full(7, endpoint))
    if kind == "float32" and endpoint == 1:
        # This softmax matrix overshoots initially but can later drift downward.
        assert maximum_seen > 1 + 1e-12
    np.testing.assert_array_equal(matrix, original)
    np.testing.assert_array_equal(env.connectivity_matrix, original)
    np.testing.assert_array_equal(env.reset(seed=0)[0], initial)


@pytest.mark.parametrize("dtype,delta", [(np.float32, 2 * np.finfo(np.float32).eps),
                                        (np.float64, 5e-13)])
def test_superstochastic_roundoff_accumulates_instead_of_being_clipped(dtype, delta):
    matrix = np.eye(2, dtype=dtype) * dtype(1 + delta)
    env = NetworkGraph(connectivity_matrix=matrix, dynamics_model="fj", seed=0,
                       initial_opinions=np.ones(2), fj_lambda=1, t_s=1,
                       t_campaign=256, max_steps=16, terminate_when_converged=False)
    for _ in range(16):
        observation, *_ = env.step(np.zeros(2, np.float32))
        assert env.observation_space.contains(observation)
    expected = float(matrix[0, 0]) ** 4096
    np.testing.assert_allclose(observation, expected, rtol=4096 * np.finfo(float).eps, atol=0)
    assert np.all(observation > 1 + 1e-12)


def test_explicit_float64_preparation_reduces_error_without_implicit_normalization():
    matrix = matrix_for("float32")
    original = matrix.copy()
    prepared = normalize_adjacency_matrix(matrix.astype(np.float64))
    assert abs(prepared.sum(axis=1) - 1).max() < abs(matrix.sum(axis=1, dtype=np.float64) - 1).max()
    for candidate in (matrix, prepared):
        env = NetworkGraph(connectivity_matrix=candidate, dynamics_model="fj", seed=0,
                           initial_opinions=np.ones(7), fj_lambda=1,
                           t_s=1, t_campaign=10, terminate_when_converged=False)
        for _ in range(30):
            assert env.observation_space.contains(env.step(np.zeros(7, np.float32))[0])
        np.testing.assert_array_equal(env.connectivity_matrix, candidate)
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("kind", ["float32", "float64", "exact"])
def test_endpoint_fj_gymnasium_checks(kind):
    check_env(NetworkGraph(connectivity_matrix=matrix_for(kind), dynamics_model="fj",
                           initial_opinions=np.ones(7), fj_lambda=1, seed=0),
              skip_render_check=True)


def test_fj_controls_and_susceptibility_endpoints_remain_finite_and_nonnegative():
    env = NetworkGraph(connectivity_matrix=matrix_for("float32"), dynamics_model="fj", seed=0,
                       initial_opinions=np.ones(7), max_u=1,
                       fj_lambda=[0, 1, 0.5, 1, 0, 0.5, 1], fj_prejudice=np.ones(7),
                       desired_opinion=[0, 1, 0, 1, 0, 1, 0],
                       control_resistance=[0, 1, 0, 1, 0, 1, 0],
                       t_campaign=2, t_s=1, terminate_when_converged=False)
    for _ in range(40):
        observation, reward, *_ = env.step(np.ones(7, np.float32))
        assert env.observation_space.contains(observation)
        assert np.all(np.isfinite(observation)) and np.isfinite(reward)


def test_exact_adjacency_with_float32_impulse_roundoff_repeats_without_projection():
    resistance = np.zeros(3, dtype=np.float32)
    action = np.full(3, 0.2, dtype=np.float32)
    env = NetworkGraph(connectivity_matrix=np.eye(3), dynamics_model="fj", seed=0,
                       initial_opinions=np.ones(3), fj_lambda=1,
                       control_resistance=resistance, max_u=1, desired_opinion=1,
                       t_s=1, t_campaign=1, terminate_when_converged=False)
    # The existing float32 impulse coefficients sum to slightly more than one
    # when used in float64 state arithmetic. Do not replace them or clip output.
    effective = action[0] * (np.float32(1) - resistance[0])
    retained = np.float32(1) - effective
    assert float(effective) + float(retained) > 1
    expected = 1.0
    for _ in range(100):
        expected = float(effective) + float(retained) * expected
        observation, *_ = env.step(action)
        np.testing.assert_array_equal(observation, np.full(3, expected))
        assert env.observation_space.contains(observation)
    assert observation.min() > 1 + 1e-12


def test_fj_initial_range_is_distinct_from_the_propagated_state_domain():
    with pytest.raises(ValueError, match="initial_opinions"):
        NetworkGraph(connectivity_matrix=np.eye(2), dynamics_model="fj", initial_opinions=[1.01, 0])
    env = NetworkGraph(connectivity_matrix=np.eye(2), dynamics_model="fj", seed=0)
    with pytest.raises(ValueError, match="state"):
        env.state = [1.01, 0]
    env.initial_opinions = [1.01, 0]
    with pytest.raises(ValueError, match="initial_opinions"):
        env.reset()
    # The standalone affine kernel accepts finite real propagated states.
    x = np.array([-0.1, 1.1])
    np.testing.assert_array_equal(friedkin_johnsen_step(x, adjacency_matrix=np.eye(2),
                                  fj_lambda=1, prejudice=np.zeros(2)), x)


@pytest.mark.parametrize("state", [[np.nan, 0], [0, np.inf], [1j, 0], ["x", "y"], [[0, 1]]])
def test_fj_nonfinite_or_nonreal_states_still_raise(state):
    with pytest.raises(ValueError, match="state"):
        friedkin_johnsen_step(state, adjacency_matrix=np.eye(2), fj_lambda=1, prejudice=np.zeros(2))


def test_fj_overflow_raises_before_environment_state_or_counters_change():
    env = NetworkGraph(connectivity_matrix=np.eye(2) * (1 + 5e-13), dynamics_model="fj",
                       t_s=1, t_campaign=1, seed=0)
    env.opinions = np.full(2, np.finfo(float).max)
    original = env.opinions.copy()
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(FloatingPointError, match="overflow|nonfinite"):
        env.step(np.zeros(2, np.float32))
    np.testing.assert_array_equal(env.opinions, original)
    assert env.current_step == env.total_spent == 0
    assert env.np_random.bit_generator.state == rng


def test_model_changes_refresh_the_space_without_changing_other_models():
    env = NetworkGraph(connectivity_matrix=np.eye(2), seed=0)
    assert np.isposinf(env.observation_space.high).all()
    env.dynamics_model = "fj"
    assert env.observation_space.contains(env.step(np.zeros(2, np.float32))[0])
    assert np.isposinf(env.observation_space.high).all()
    env.dynamics_model = "coca"
    env.reset(seed=0)
    assert np.all(env.observation_space.high == 1)


@pytest.mark.parametrize("start", [(-1, 0), (0, -1), (5, 0), (0, 5), (5, 5), (1.0, 1), [1, 1]])
def test_grid_constructor_rejects_out_of_space_starts(start):
    with pytest.raises(ValueError, match="start_state"):
        GridWorld(start_state=start)


@pytest.mark.parametrize("via_options", [False, True])
@pytest.mark.parametrize("start", [(-1, 0), (5, 5), (1.0, 1), [1, 1]])
def test_grid_invalid_reset_preserves_state_configuration_and_rng(start, via_options):
    env = GridWorld(seed=0, p_success=0.5)
    env.step(1)
    before = (env.state, env.start_state, env.episode_length_counter, env.seed,
              copy.deepcopy(env.np_random.bit_generator.state))
    kwargs = {"options": {"new_start_state": start}} if via_options else {"new_start_state": start}
    with pytest.raises(ValueError, match="new_start_state"):
        env.reset(seed=42, **kwargs)
    assert before == (env.state, env.start_state, env.episode_length_counter, env.seed,
                      env.np_random.bit_generator.state)


@pytest.mark.parametrize("start", [(0, 0), (4, 4), (3, 4)])
def test_grid_valid_corner_and_terminal_resets_keep_persistence(start):
    env = GridWorld()
    assert env.reset(options={"new_start_state": start}) == (start, {})
    assert env.reset() == (start, {})
    assert env.observation_space.contains(start)


@pytest.mark.parametrize("start", [-1, 11, 1.5, np.nan, np.inf, None, "1", (1,)])
def test_gambler_constructor_rejects_out_of_space_capital(start):
    with pytest.raises(ValueError, match="start_capital"):
        GamblersProblem(goal_amount=10, start_capital=start)


@pytest.mark.parametrize("start", [-1, 11, 1.5, None])
def test_gambler_invalid_reset_preserves_state_and_rng(start):
    env = GamblersProblem(goal_amount=10, start_capital=5)
    env.reset(seed=0)
    env.step(1)
    before = (env.state, copy.deepcopy(env.np_random.bit_generator.state))
    env.start_capital = start
    with pytest.raises(ValueError, match="start_capital"):
        env.reset(seed=42)
    assert before == (env.state, env.np_random.bit_generator.state)


@pytest.mark.parametrize("start", [0, 10, np.int64(3)])
def test_gambler_valid_initial_capitals_include_terminal_states(start):
    env = GamblersProblem(goal_amount=10, start_capital=start)
    observation, _ = env.reset(seed=0)
    assert observation == start and env.observation_space.contains(observation)


@pytest.mark.parametrize("vision", [None, 1])
def test_manual_labyrinth_setup_and_diagnostics(vision):
    layout = np.ones((5, 6), dtype=np.uint8)
    env = Labyrinth(maze_layout_predetermined=layout, state_vision_range=vision, seed=0)
    if vision is None:
        observation, _ = env.reset(seed=0)
        np.testing.assert_array_equal(observation, layout)
    else:
        with pytest.raises(RuntimeError, match="uninitialized.*after each reset"):
            env.reset(seed=0)
    for operation in (lambda: env.step(1), env.render):
        with pytest.raises(RuntimeError, match="env.set_state.*env.maze.target_position"):
            operation()
    if vision is not None:
        with pytest.raises(RuntimeError, match="uninitialized"):
            _ = env.state
    env.set_state((1, 1))
    with pytest.raises(RuntimeError, match="uninitialized"):
        env.step(1)
    env.maze.target_position = (1, 3)
    assert env.observation_space.contains(env.state)
    assert env.step(1)[2:4] == (False, False)
    assert env.step(1)[2:4] == (True, False)
    np.testing.assert_array_equal(layout, np.ones((5, 6), np.uint8))
