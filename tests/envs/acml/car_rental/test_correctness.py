"""Independent inventory, Poisson, and stochastic-planning references."""

import copy
import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental


def set_initial_inventory(env, state, seed=0):
    env.reset(seed=seed)
    env.state = state


@pytest.mark.parametrize("state,signed,expected,cost", [
    ((4, 0), 2, (2, 2), 4), ((0, 4), -2, (2, 2), 4),
    ((4, 3), 2, (3, 4), 2), ((3, 4), -2, (4, 3), 2),
    ((0, 0), 2, (0, 0), 0), ((0, 0), -2, (0, 0), 0),
    ((4, 4), 2, (4, 4), 0), ((4, 4), -2, (4, 4), 0),
    ((1, 0), 2, (0, 1), 2), ((0, 1), -2, (1, 0), 2),
    ((1, 3), 0, (1, 3), 0),
])
def test_zero_events_clip_transfers_and_charge_actual_movement(state, signed, expected, cost):
    env = JacksCarRental(
        max_cars=4, max_move_cars=2, request_lambda=[0, 0], return_lambda=[0, 0],
    )
    set_initial_inventory(env, state)
    assert env.step(signed + 2) == (expected, -cost, False, False, {})
    assert sum(expected) == sum(state)
    assert env.observation_space.contains(expected)


def test_all_zero_event_inventory_transfers_conserve_cars():
    env = JacksCarRental(
        max_cars=3, max_move_cars=2, request_lambda=[0, 0], return_lambda=[0, 0],
    )
    for state in product(range(4), repeat=2):
        for action in range(5):
            set_initial_inventory(env, state)
            destination, reward, terminated, truncated, _ = env.step(action)
            assert sum(destination) == sum(state)
            assert env.observation_space.contains(destination)
            moved = abs(destination[0] - state[0])
            assert moved <= min(abs(action - 2), 2)
            assert reward == -2 * moved  # No income when there are no requests.
            assert not terminated and not truncated
            if action > 2:
                assert destination[0] <= state[0]
            elif action < 2:
                assert destination[0] >= state[0]


@pytest.mark.parametrize("seed,expected_state", [(13, (4, 4)), (11, (0, 0))])
@pytest.mark.parametrize("action", [0, 4])
def test_06a_06b_public_seeded_full_or_empty_locations_do_not_lose_cars(seed, expected_state, action):
    env = JacksCarRental(
        max_cars=4, max_move_cars=2, request_lambda=[0, 0], return_lambda=[0, 0],
    )
    assert env.reset(seed=seed)[0] == expected_state
    assert env.step(action) == (expected_state, 0, False, False, {})


@pytest.mark.parametrize("requests,action,expected", [
    ([0, 100], 4, (4, 0)), ([100, 0], 0, (0, 4)),
])
def test_06c_high_requests_cannot_rent_temporary_overflow(requests, action, expected):
    env = JacksCarRental(
        max_cars=4, max_move_cars=2, request_lambda=requests, return_lambda=[0, 0],
    )
    assert env.reset(seed=13)[0] == (4, 4)
    assert env.step(action) == (expected, 40, False, False, {})


@pytest.mark.parametrize("state", [(0, 0), (1, 2), (3, 3)])
def test_returns_fill_only_remaining_capacity(state):
    env = JacksCarRental(
        max_cars=3, request_lambda=[0, 0], return_lambda=[100, 100],
    )
    set_initial_inventory(env, state)
    assert env.step(env.max_move_cars) == ((3, 3), 0, False, False, {})


def test_05a_default_legacy_model_keeps_structure_without_support_index_error():
    env = JacksCarRental()
    model = env.build_mdp()
    assert isinstance(model, dict)
    assert len(model) == 21 * 21 * 11
    assert len(env.poisson_probs["request"]["A"]) == 13
    assert ((10, 10), -5) in model
    for representative, reward, done in model.values():
        assert env.observation_space.contains(representative)
        assert math.isfinite(reward)
        assert done is False


@pytest.mark.parametrize("support", [0, 1, 12])
def test_05b_05c_one_car_exact_tail_income_and_complete_distribution(support):
    env = JacksCarRental(
        max_cars=1, max_move_cars=1, request_lambda=[1, 0],
        return_lambda=[0, 0], poisson_max_value=support,
    )
    legacy = env.build_mdp()
    model = env.build_transition_model()
    assert isinstance(model, Mapping)
    assert legacy[((1, 0), 0)] == ((1, 0), 10 * (1 - math.exp(-1)), False)
    outcomes = model[((1, 0), 1)]
    assert len(outcomes) == 2
    assert outcomes[0][:3] == ((0, 0), 10, False)
    assert outcomes[0][3] == pytest.approx(1 - math.exp(-1), abs=1e-15)
    assert outcomes[1][:3] == ((1, 0), 0, False)
    assert outcomes[1][3] == pytest.approx(math.exp(-1), abs=1e-15)


def test_one_car_joint_destination_and_conditional_reward_closed_form():
    env = JacksCarRental(
        max_cars=1, max_move_cars=1, request_lambda=[1, 0], return_lambda=[1, 0],
    )
    model = env.build_transition_model()
    outcomes = {state: (reward, probability) for state, reward, _, probability in model[((1, 0), 1)]}
    no_event = math.exp(-1)
    event = 1 - no_event
    assert outcomes[(0, 0)][1] == pytest.approx(event * no_event)
    assert outcomes[(0, 0)][0] == pytest.approx(10)
    assert outcomes[(1, 0)][1] == pytest.approx(1 - event * no_event)
    assert outcomes[(1, 0)][0] == pytest.approx(10 * event**2 / (1 - event * no_event))
    assert sum(reward * probability for reward, probability in outcomes.values()) == pytest.approx(10 * event)


def test_05b_zero_stock_returns_have_four_next_states_without_changing_legacy_format():
    env = JacksCarRental(
        max_cars=1, max_move_cars=1, request_lambda=[0, 0], return_lambda=[1, 1],
    )
    assert env.build_mdp()[((0, 0), 0)] == ((0, 0), 0, False)
    model = env.build_transition_model()
    no_return = math.exp(-1)
    some_returns = 1 - no_return
    expected = {
        (0, 0): no_return**2, (0, 1): no_return * some_returns,
        (1, 0): some_returns * no_return, (1, 1): some_returns**2,
    }
    assert len(model[((0, 0), 1)]) == 4
    for destination, reward, done, probability in model[((0, 0), 1)]:
        assert reward == 0 and done is False
        assert probability == pytest.approx(expected[destination], abs=1e-15)


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("requests,returns", [
    ([0, 0], [0, 0]), ([0.7, 1.3], [1.1, 0.4]), ([0, 1], [2, 0]),
])
def test_every_small_model_row_has_mass_one_and_bounded_states(capacity, requests, returns):
    env = JacksCarRental(
        max_cars=capacity, max_move_cars=2, request_lambda=requests, return_lambda=returns,
    )
    model = env.build_transition_model()
    legacy = env.build_mdp()
    assert len(model) == (capacity + 1)**2 * 5
    assert len(set(model)) == len(model)
    for (state, action) in model:
        outcomes = model[(state, action)]
        assert outcomes
        assert len(outcomes) == len({row[0] for row in outcomes})
        probabilities = np.array([row[3] for row in outcomes])
        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0)
        assert probabilities.sum() == pytest.approx(1, abs=3e-15)
        for destination, reward, done, _ in outcomes:
            assert env.observation_space.contains(destination)
            assert math.isfinite(reward)
            assert done is False
        expected_reward = sum(reward * p for _, reward, _, p in outcomes)
        assert expected_reward == pytest.approx(model.expected_reward(state, action), abs=1e-12)
        assert expected_reward == pytest.approx(legacy[(state, action - 2)][1], abs=1e-12)


def poisson_count_series(rate, cutoff=40):
    """Independent finite series; no scipy or production tail probabilities."""
    return [math.exp(-rate) * rate**count / math.factorial(count)
            for count in range(cutoff)]


def location_event_reference(inventory, capacity, request_rate, return_rate):
    outcomes = defaultdict(float)
    for requests, request_p in enumerate(poisson_count_series(request_rate)):
        for returns, return_p in enumerate(poisson_count_series(return_rate)):
            rented = min(inventory, requests)
            destination = min(capacity, inventory - rented + returns)
            outcomes[(destination, rented)] += request_p * return_p
    return outcomes


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_joint_probabilities_and_reward_moments_against_independent_event_series(capacity):
    requests, returns = [0.7, 1.3], [1.1, 0.4]
    env = JacksCarRental(
        max_cars=capacity, max_move_cars=1, request_lambda=requests,
        return_lambda=returns, rental_credit=7, move_car_cost=3,
    )
    model = env.build_transition_model()
    references = [
        [location_event_reference(stock, capacity, req, ret) for stock in range(capacity + 1)]
        for req, ret in zip(requests, returns)
    ]
    # Chernoff union bound for all four omitted Poisson tails at count >= 40.
    tail_bound = sum(math.exp(-rate) * (math.e * rate / 40)**40
                     for rate in requests + returns)
    assert tail_bound < 1e-40
    for state in product(range(capacity + 1), repeat=2):
        for action in range(3):
            signed = action - 1
            if signed > 0 and state[0] > 0 and state[1] < capacity:
                moved, cost = (state[0] - 1, state[1] + 1), 3
            elif signed < 0 and state[1] > 0 and state[0] < capacity:
                moved, cost = (state[0] + 1, state[1] - 1), 3
            else:
                moved, cost = state, 0
            probabilities, moments = defaultdict(float), defaultdict(float)
            for (next_a, rent_a), p_a in references[0][moved[0]].items():
                for (next_b, rent_b), p_b in references[1][moved[1]].items():
                    probability = p_a * p_b
                    probabilities[(next_a, next_b)] += probability
                    moments[(next_a, next_b)] += probability * (7 * (rent_a + rent_b) - cost)
            for destination, reward, _, probability in model[(state, action)]:
                assert probability == pytest.approx(probabilities[destination], abs=3e-15)
                assert probability * reward == pytest.approx(moments[destination], abs=1e-12)


@pytest.mark.parametrize("state,signed,moved,requests,returns", [
    ((1, 0), 0, (1, 0), [1, 0], [0, 0]),
    ((3, 1), 1, (2, 2), [0.7, 1.3], [1.1, 0.4]),
    ((1, 3), -1, (2, 2), [1.3, 0.7], [0, 2]),
    ((3, 3), 2, (3, 3), [2, 0.5], [0.5, 1.5]),
    ((0, 1), 2, (0, 1), [0, 1], [1, 0]),
    ((0, 0), -2, (0, 0), [0, 0], [1, 2]),
])
def test_complete_model_matches_independently_seeded_daily_samples(state, signed, moved, requests, returns):
    env = JacksCarRental(
        max_cars=3, max_move_cars=2, request_lambda=requests, return_lambda=returns,
    )
    model = env.build_transition_model()
    outcomes = model[(state, signed + 2)]
    expected_probabilities = np.zeros((4, 4))
    for destination, _, _, probability in outcomes:
        expected_probabilities[destination] = probability
    observed_counts = np.zeros((4, 4))
    rewards = []
    samples = 3000
    for seed in range(samples):
        set_initial_inventory(env, state, seed=seed)
        destination, reward, done, truncated, _ = env.step(signed + 2)
        assert not done and not truncated
        observed_counts[destination] += 1
        rewards.append(reward)

    # Two-sided Bernstein bounds, unioned across 16 destinations + the reward
    # and all six cases. Family-wise error bound <= 1e-6; no fitted tolerances.
    log_factor = math.log(2 * 17 * 6 / 1e-6)
    tolerance = np.sqrt(
        2 * expected_probabilities * (1 - expected_probabilities) * log_factor / samples
    ) + 2 * log_factor / (3 * samples)
    assert np.all(np.abs(observed_counts / samples - expected_probabilities) <= tolerance)

    variance = 0
    for stock, rate in zip(moved, requests):
        tails = [1 - math.exp(-rate) * sum(rate**j / math.factorial(j) for j in range(k))
                 for k in range(1, stock + 1)]
        mean = sum(tails)
        second_moment = sum((2 * k - 1) * tail for k, tail in enumerate(tails, 1))
        variance += 100 * (second_moment - mean**2)
    reward_range = 10 * sum(moved)
    reward_tolerance = math.sqrt(2 * max(variance, 0) * log_factor / samples)
    reward_tolerance += 2 * reward_range * log_factor / (3 * samples)
    assert abs(np.mean(rewards) - model.expected_reward(state, signed + 2)) <= reward_tolerance


def test_model_is_an_immutable_mapping_snapshot_with_no_rng_or_inventory_effect():
    env = JacksCarRental(max_cars=3, request_lambda=[1, 2], return_lambda=[2, 1])
    env.reset(seed=42)
    state, rng = env.state, copy.deepcopy(env.np_random.bit_generator.state)
    model = env.build_transition_model()
    before = model[((2, 1), 5)]
    reward = model.expected_reward((2, 1), 5)
    np.testing.assert_equal(env.np_random.bit_generator.state, rng)
    assert env.state == state
    with pytest.raises(TypeError):
        model[((2, 1), 5)] = ()
    with pytest.raises(TypeError):
        before[0] = ()
    env.request_lambda[0] = 10
    env.return_lambda[1] = 10
    env.rental_credit = 100
    env.move_car_cost = 100
    env.state = (0, 0)
    env.poisson_probs["request"]["A"][0] = 0
    assert model[((2, 1), 5)] == before
    assert model.expected_reward((2, 1), 5) == reward
    assert env.build_transition_model()[((2, 1), 5)] != before


@pytest.mark.parametrize("key", [None, ((0, 0), -1), ((0, 0), 11), ((21, 0), 5), ((0.5, 0), 5)])
def test_model_rejects_invalid_state_or_action_keys(key):
    with pytest.raises(KeyError):
        JacksCarRental().build_transition_model()[key]


@pytest.mark.parametrize("rates", [[-1, 0], [np.nan, 1], [1, np.inf], [1]])
def test_planning_rejects_invalid_poisson_rates(rates):
    env = JacksCarRental(init_state_option="equal")
    env.request_lambda = rates
    with pytest.raises(ValueError, match="Poisson rates"):
        env.build_transition_model()


def test_zero_capacity_has_one_continuing_state_and_zero_rewards():
    env = JacksCarRental(max_cars=0)
    model = env.build_transition_model()
    for key in model:
        assert model[key] == (((0, 0), 0, False, 1.0),)
    assert env.step(0) == ((0, 0), 0, False, False, {})


@pytest.mark.parametrize("seed", [0, 42])
def test_corrected_daily_trajectories_replay_on_reseed(seed):
    env = JacksCarRental(max_cars=3, max_move_cars=2)
    actions = [4, 0, 2, 3, 1] * 3
    initial = env.reset(seed=seed)
    expected = [env.step(action) for action in actions]
    assert env.reset(seed=seed) == initial
    assert [env.step(action) for action in actions] == expected


@pytest.mark.parametrize("seed,actions,expected", [
    (0, [6, 4, 5, 7, 3, 5], [
        ((14, 12), 18), ((14, 15), 48), ((15, 11), 80),
        ((13, 11), 46), ((16, 7), 66), ((16, 3), 140),
    ]),
    (42, [6, 4, 5, 7], [
        ((9, 7), 88), ((8, 5), 88), ((7, 0), 80), ((4, 1), 66),
    ]),
])
def test_feasible_seeded_trajectories_match_original_head(seed, actions, expected):
    # Recorded from 0bec7114c03735797eb64ce18f2f9620b0f6d92b.
    env = JacksCarRental()
    env.reset("equal", seed=seed)
    for action, expected_transition in zip(actions, expected):
        signed = action - env.max_move_cars
        source, destination = env.state if signed > 0 else env.state[::-1]
        assert abs(signed) <= min(source, env.max_cars - destination)
        assert env.step(action)[:2] == expected_transition


def test_default_and_small_environment_pass_gymnasium_checks():
    check_env(JacksCarRental(), skip_render_check=True)
    check_env(JacksCarRental(max_cars=1), skip_render_check=True)
