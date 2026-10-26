"""H1/H2/M1: unsigned inputs and observation metadata, without new dynamics."""

import copy

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
from scipy.linalg import expm

from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental
from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem
from rl_envs_forge.envs.network_graph.graph_utils import compute_laplacian
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


UNSIGNED_TYPES = (np.uint8, np.uint16, np.uint32, np.uint64)
INTEGER_TYPES = (int, np.int8, np.int16, np.int32, np.int64, *UNSIGNED_TYPES)


def assert_same_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", UNSIGNED_TYPES)
@pytest.mark.parametrize("size", [2, 3, 6])
@pytest.mark.parametrize("model", ["laplacian", "degroot"])
def test_unsigned_binary_laplacian_matches_float64(dtype, size, model):
    adjacency = (np.ones((size, size)) - np.eye(size)).astype(dtype)
    original = adjacency.copy()
    expected = np.diag(adjacency.astype(float).sum(axis=1)) - adjacency.astype(float)
    laplacian = compute_laplacian(adjacency)
    assert_same_bits(laplacian, expected)
    assert np.all(laplacian[~np.eye(size, dtype=bool)] == -1)
    np.testing.assert_array_equal(laplacian.sum(axis=1, dtype=float), np.zeros(size))

    kwargs = dict(initial_opinions=np.linspace(.2, .8, size), seed=42,
                  dynamics_model=model, terminate_when_converged=False)
    unsigned = NetworkGraph(connectivity_matrix=adjacency, **kwargs)
    floating = NetworkGraph(connectivity_matrix=adjacency.astype(float), **kwargs)
    assert unsigned.connectivity_matrix.dtype == dtype
    for name in ("L", "expL_ts", "centralities"):
        assert_same_bits(getattr(unsigned, name), getattr(floating, name))
    for value in (0., .02, .07, .01):
        action = np.full(size, value, np.float32)
        actual = unsigned.step(action)
        reference = floating.step(action)
        assert_same_bits(actual[0], reference[0])
        assert_same_bits(actual[4]["intermediate_states"], reference[4]["intermediate_states"])
        assert actual[1:4] == reference[1:4]
        assert np.isfinite(actual[0]).all() and np.isfinite(actual[1])
        assert unsigned.observation_space.contains(actual[0])
    assert unsigned.np_random.bit_generator.state == floating.np_random.bit_generator.state
    assert_same_bits(adjacency, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64])
def test_unsigned_promotion_precedes_degree_sum_and_subtraction(dtype):
    weight = np.iinfo(dtype).max
    adjacency = np.array([[0, weight, weight], [weight, 0, weight],
                          [weight, weight, 0]], dtype=dtype)
    floating = adjacency.astype(np.float64)
    expected = np.diag(floating.sum(axis=1)) - floating
    actual = compute_laplacian(adjacency)
    assert_same_bits(actual, expected)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.sum(axis=1), np.zeros(3))


@pytest.mark.parametrize("model", ["laplacian", "degroot"])
def test_unsigned_two_agent_consensus_matches_closed_form(model):
    env = NetworkGraph(connectivity_matrix=np.array([[0, 1], [1, 0]], np.uint64),
                       initial_opinions=[.2, .8], dynamics_model=model,
                       terminate_when_converged=False, seed=0)
    for step in range(1, 11):
        observation, reward, *_ = env.step(np.zeros(2, np.float32))
        difference = .3 * np.exp(-2 * step)
        np.testing.assert_allclose(observation, [.5 - difference, .5 + difference],
                                   rtol=0, atol=2e-15)
        assert np.isfinite(reward)


@pytest.mark.parametrize("model", ["laplacian", "degroot"])
@pytest.mark.parametrize("size", [1, 2, 5, 6])
@pytest.mark.parametrize("endpoint,control", [(0., 0.), (1., 0.), (1., .2), (0., 1.)])
@pytest.mark.parametrize("campaign", [0., .2, 1.])
def test_consensus_space_contains_raw_repeated_endpoint_updates(model, size, endpoint,
                                                               control, campaign):
    adjacency = np.ones((size, size)) - np.eye(size)
    resistance = np.zeros(size, np.float32)
    env = NetworkGraph(connectivity_matrix=adjacency, dynamics_model=model,
                       initial_opinions=np.full(size, endpoint),
                       control_resistance=resistance, max_u=1, seed=0,
                       t_campaign=campaign, t_s=.1, max_steps=30,
                       terminate_when_converged=False)
    initial, _ = env.reset(seed=42)
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    space = env.observation_space
    assert space.dtype == np.float64 and space.contains(initial)
    assert np.all(space.low == 0) and np.isposinf(space.high).all()

    # Reproduce the original arithmetic independently of package helpers. This
    # checks raw intermediate states and rewards, not merely space membership.
    reference_laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    propagator = expm(-reference_laplacian * .1)
    expected = initial.copy()
    action = np.full(size, control, np.float32)
    assert env.action_space.contains(action)
    maximum = endpoint
    for step in range(30):
        effective = action * (1 - resistance)
        expected = effective * np.ones(size) + (1 - effective) * expected
        intermediates = [expected.copy()]
        for _ in range(round(campaign / .1)):
            expected = propagator @ expected
            intermediates.append(expected.copy())
        observation, reward, terminated, truncated, info = env.step(action)
        assert_same_bits(observation, expected)
        assert_same_bits(info["intermediate_states"], np.array(intermediates))
        assert reward == -np.abs(1. - expected).sum() - .4 * np.sum(action)
        assert space is env.observation_space and space.contains(observation)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert not terminated and truncated == (step == 29)
        maximum = max(maximum, observation.max())
    if endpoint == 1 and control == .2 and size == 2:
        assert maximum > 1 + 1e-9  # Preserve impulse roundoff; do not project it.
    assert env.np_random.bit_generator.state == rng
    assert_same_bits(env.reset(seed=42)[0], initial)


@pytest.mark.parametrize("model", ["laplacian", "degroot"])
@pytest.mark.parametrize("size", [2, 5, 6])
def test_consensus_endpoint_and_unsigned_graph_pass_gymnasium(model, size):
    adjacency = (np.ones((size, size)) - np.eye(size)).astype(np.uint64)
    check_env(NetworkGraph(connectivity_matrix=adjacency, dynamics_model=model,
                           initial_opinions=np.ones(size), max_u=1,
                           control_resistance=np.zeros(size, np.float32), seed=0),
              skip_render_check=True)


@pytest.mark.parametrize("state_type", INTEGER_TYPES)
@pytest.mark.parametrize("action_type", INTEGER_TYPES)
def test_car_model_integer_types_match_native_rows_and_rewards(state_type, action_type):
    env = JacksCarRental(max_cars=3, max_move_cars=1, init_state_option="equal")
    env.reset(seed=42)
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    state_before = env.state
    model = env.build_transition_model()
    for a in range(4):
        for b in range(4):
            for action in range(3):
                # Include mixed Python/NumPy states, as well as homogeneous pairs.
                for state in ((state_type(a), state_type(b)), (state_type(a), b)):
                    actual = model[(state, action_type(action))]
                    assert actual == model[((a, b), action)]
                    assert model.expected_reward(state, action_type(action)) == model.expected_reward((a, b), action)
                    assert all(np.isfinite(outcome[1]) for outcome in actual)
    assert env.state == state_before and env.np_random.bit_generator.state == rng


@pytest.mark.parametrize("key", [((1., 1), 1), ((1.5, 1), 1), ((1, 1.), 1),
                                 ((1, 1), 1.), ((1, 1), 1.5),
                                 ((np.uint64(2**63), 1), 1), ((1, 1), np.uint64(2**63))])
def test_car_model_does_not_truncate_or_wrap_invalid_integer_inputs(key):
    model = JacksCarRental(max_cars=3, max_move_cars=1).build_transition_model()
    with pytest.raises(KeyError):
        model[key]
    with pytest.raises(KeyError):
        model.expected_reward(*key)


@pytest.mark.parametrize("dtype", INTEGER_TYPES)
@pytest.mark.parametrize("goal", [5, 10, 100])
def test_gambler_integer_helpers_agree_with_native_stakes_and_mdp(dtype, goal):
    env = GamblersProblem(goal_amount=goal, start_capital=dtype(2))
    env.reset(seed=42)
    state_before = env.state
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    model = env.build_mdp()
    for state in range(goal + 1):
        expected = tuple(range(1, min(state, goal - state) + 1))
        assert env.feasible_actions(dtype(state)) == expected
        assert tuple(np.flatnonzero(env.action_mask(dtype(state)))) == expected
        assert {stake for capital, stake in model if capital == state} == set(expected)
    assert env.feasible_actions() == env.feasible_actions(2)
    np.testing.assert_array_equal(env.action_mask(), env.action_mask(2))
    assert env.state is state_before  # Helpers do not change stored observation types.
    assert env.np_random.bit_generator.state == rng


@pytest.mark.parametrize("state", [2., 2.5, np.float32(2), np.float64(2.5), np.uint64(2**63)])
def test_gambler_helpers_do_not_truncate_or_wrap_invalid_capital(state):
    env = GamblersProblem(goal_amount=5)
    for helper in (env.feasible_actions, env.action_mask):
        with pytest.raises(ValueError, match="capital"):
            helper(state)
