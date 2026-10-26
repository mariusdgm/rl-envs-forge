"""Stochastic movement, canonical outcomes, and the audit's cases 04a--04d."""

import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_envs_forge.envs.grid_world.grid_world import Action, GridWorld


def assert_consistent_models(env):
    """Check every tensor row against the reward-bearing canonical outcomes."""
    expected_P = np.zeros_like(env.P)
    expected_keys = {
        ((row, col), action)
        for row in range(env.rows)
        for col in range(env.cols)
        for action in Action
    }
    assert env.mdp.keys() == env.transition_probs.keys() == expected_keys
    for (state, action), outcomes in env.mdp.items():
        assert env.calculate_outcomes(state, action) == outcomes
        probabilities = np.array([outcome[3] for outcome in outcomes])
        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0)
        assert probabilities.sum() == pytest.approx(1, abs=1e-14)
        aggregated = {}
        for destination, _, _, probability in outcomes:
            assert env.observation_space.contains(destination)
            aggregated[destination] = aggregated.get(destination, 0) + probability
            expected_P[
                state[0] * env.cols + state[1],
                destination[0] * env.cols + destination[1],
                action,
            ] += probability
        assert env.transition_probs[(state, action)] == aggregated
    assert np.all(np.isfinite(env.P))
    assert np.all(env.P >= 0)
    np.testing.assert_allclose(env.P.sum(axis=1), 1, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(env.P, expected_P)
    np.testing.assert_array_equal(env.create_probability_matrix(), expected_P)
    assert env.default_transition_probs() == env.transition_probs


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
@pytest.mark.parametrize(
    "slips",
    [None, {Action.UP: 0.25, Action.RIGHT: 0.25,
            Action.DOWN: 0.25, Action.LEFT: 0.25},
     {Action.UP: 0.75, Action.DOWN: 0.25}],
    ids=["default", "explicit-uniform", "directional"],
)
def test_all_models_share_normalized_outcomes(p_success, slips):
    env = GridWorld(
        rows=4, cols=4, start_state=(1, 1), terminal_states={(3, 3): 10},
        walls={(1, 2)}, special_transitions={
            ((1, 1), Action.RIGHT): ((3, 3), 0.75),
            ((2, 1), Action.UP): ((0, 3), 2),
        },
        rewards={"wall_collision": -7, "out_of_bounds": -3, "valid_move": -0.1},
        p_success=p_success, slip_distribution=slips, seed=0,
    )
    assert_consistent_models(env)


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_04a_special_jump_only_on_success(p_success):
    env = GridWorld(
        start_state=(1, 1), terminal_states={(3, 3): 10},
        special_transitions={((1, 1), Action.RIGHT): ((3, 3), 0.75)},
        p_success=p_success, seed=0,
    )
    expected = [((3, 3), 0.75, True, p_success)]
    if p_success < 1:
        expected += [
            (state, -0.1, False, (1 - p_success) / 3)
            for state in ((0, 1), (2, 1), (1, 0))
        ]
    assert env.mdp[((1, 1), Action.RIGHT)] == expected
    # Both sides of the success threshold are exercised without frequency tests.
    for seed in (0, 2):
        env.reset(seed=seed)
        index = np.random.default_rng(seed).choice(
            len(expected), p=[outcome[3] for outcome in expected]
        )
        assert env.step(Action.RIGHT) == (*expected[index][:3], False, {})
    assert_consistent_models(env)


def test_04b_custom_slips_exclude_forbidden_directions():
    # The original singleton global distribution cannot specify an alternative
    # for intended UP. It is now rejected instead of silently ignored.
    with pytest.raises(ValueError, match="alternative"):
        GridWorld(p_success=0.5, slip_distribution={Action.UP: 1.0})

    env = GridWorld(
        start_state=(2, 2), p_success=0.5, seed=4,
        slip_distribution={Action.UP: 0.75, Action.DOWN: 0.25},
    )
    assert env.transition_probs[((2, 2), Action.RIGHT)] == {
        (2, 3): 0.5, (1, 2): 0.375, (3, 2): 0.125, (2, 1): 0.0,
    }
    # Previously seed 4 slipped LEFT, outside the specified support.
    assert env.step(Action.RIGHT)[0] == (3, 2)


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_04c_default_slips_are_conditional_and_not_scaled_twice(p_success):
    env = GridWorld(start_state=(2, 2), p_success=p_success, seed=0)
    expected = {(2, 3): p_success}
    if p_success < 1:
        expected.update({state: (1 - p_success) / 3
                         for state in ((1, 2), (3, 2), (2, 1))})
    assert env.transition_probs[((2, 2), Action.RIGHT)] == expected
    assert sum(expected.values()) == pytest.approx(1)


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_04d_mutation_refreshes_all_models(p_success):
    caller_transitions = {}
    env = GridWorld(
        start_state=(1, 1), p_success=p_success, seed=0,
        special_transitions=caller_transitions,
    )
    rng_state = copy.deepcopy(env.np_random.bit_generator.state)
    for destination, reward in (((3, 3), 0.75), ((3, 4), 5)):
        env.add_special_transition((1, 1), Action.RIGHT, destination, reward)
        assert env.special_transitions[((1, 1), Action.RIGHT)] == (destination, reward)
        assert env.calculate_next_state((1, 1), Action.RIGHT) == destination
        assert env.calculate_transition((1, 1), Action.RIGHT) == (
            destination, reward, destination in env.terminal_states,
        )
        assert env.P[6, destination[0] * env.cols + destination[1], Action.RIGHT] == p_success
        assert_consistent_models(env)
    np.testing.assert_equal(env.np_random.bit_generator.state, rng_state)
    assert caller_transitions == {}


@pytest.mark.parametrize("intended", list(Action))
def test_global_distribution_is_conditioned_on_an_alternative(intended):
    slips = {Action.UP: 0.4, Action.RIGHT: 0.1,
             Action.DOWN: 0.2, Action.LEFT: 0.3}
    env = GridWorld(
        start_state=(2, 2), p_success=0.5, slip_distribution=slips, seed=0,
    )
    destinations = {Action.UP: (1, 2), Action.RIGHT: (2, 3),
                    Action.DOWN: (3, 2), Action.LEFT: (2, 1)}
    distribution = env.transition_probs[((2, 2), intended)]
    assert distribution[destinations[intended]] == 0.5
    denominator = sum(prob for action, prob in slips.items() if action != intended)
    for action in Action:
        if action != intended:
            assert distribution[destinations[action]] == pytest.approx(
                0.5 * slips[action] / denominator
            )
    slips[Action.UP] = 0  # Caller mutation must not change the configuration.
    assert env.slip_distribution[Action.UP] == 0.4


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_failed_actions_ignore_other_special_transitions(p_success):
    env = GridWorld(
        start_state=(2, 2), p_success=p_success, seed=0,
        special_transitions={((2, 2), Action.UP): ((0, 0), 99)},
    )
    right_outcomes = env.calculate_outcomes((2, 2), Action.RIGHT)
    if p_success < 1:
        assert right_outcomes[1] == ((1, 2), -0.1, False, (1 - p_success) / 3)
    assert all(destination != (0, 0) for destination, *_ in right_outcomes)
    assert env.calculate_transition((2, 2), Action.UP) == ((0, 0), 99, False)


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
@pytest.mark.parametrize("special", [False, True])
def test_same_destination_can_retain_distinct_rewards(p_success, special):
    env = GridWorld(
        rows=3, cols=3, terminal_states={}, walls={(0, 1)},
        p_success=p_success, seed=0,
        rewards={"wall_collision": -7, "out_of_bounds": -3, "valid_move": -0.1},
        special_transitions={((0, 0), Action.RIGHT): ((0, 0), 17)} if special else None,
    )
    intended_reward = 17 if special else -7
    outcomes = [((0, 0), intended_reward, False, p_success)]
    if p_success < 1:
        outcomes += [
            ((0, 0), -3, False, (1 - p_success) / 3),
            ((1, 0), -0.1, False, (1 - p_success) / 3),
            ((0, 0), -3, False, (1 - p_success) / 3),
        ]
    assert env.mdp[((0, 0), Action.RIGHT)] == outcomes
    for seed in (0, 2):
        env.reset(seed=seed)
        index = np.random.default_rng(seed).choice(
            len(outcomes), p=[outcome[3] for outcome in outcomes]
        )
        assert env.step(Action.RIGHT) == (*outcomes[index][:3], False, {})
    assert_consistent_models(env)


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_terminal_and_wall_rows_are_absorbing_in_every_model(p_success):
    env = GridWorld(
        rows=3, cols=3, start_state=(1, 1), terminal_states={(1, 2): 10},
        walls={(0, 1)}, p_success=p_success,
        special_transitions={((1, 2), Action.RIGHT): ((2, 2), 99)}, seed=0,
    )
    for action in Action:
        assert env.mdp[((1, 2), action)] == [((1, 2), 0, True, 1.0)]
        assert env.mdp[((0, 1), action)] == [((0, 1), 0, False, 1.0)]
    env.reset(new_start_state=(1, 2), seed=0)
    assert env.step(Action.RIGHT) == ((1, 2), 0, True, False, {})
    assert_consistent_models(env)


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_ordinary_terminal_entries_keep_reward_and_termination(p_success):
    env = GridWorld(
        rows=3, cols=3, start_state=(1, 1), terminal_states={(1, 2): 10},
        p_success=p_success, seed=0,
    )
    assert env.calculate_outcomes((1, 1), Action.RIGHT)[0] == (
        (1, 2), 10, True, p_success,
    )
    if p_success < 1:
        assert env.calculate_outcomes((1, 1), Action.DOWN)[2] == (
            (1, 2), 10, True, (1 - p_success) / 3,
        )


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_special_and_slip_can_reach_same_terminal_with_different_rewards(p_success):
    env = GridWorld(
        rows=3, cols=3, start_state=(1, 1), terminal_states={(1, 2): 10},
        special_transitions={((1, 1), Action.DOWN): ((1, 2), 17)},
        p_success=p_success, seed=0,
    )
    outcomes = env.mdp[((1, 1), Action.DOWN)]
    assert outcomes[0] == ((1, 2), 17, True, p_success)
    if p_success < 1:
        assert outcomes[2] == ((1, 2), 10, True, (1 - p_success) / 3)
    for seed in (0, 2, 5):
        env.reset(seed=seed)
        index = np.random.default_rng(seed).choice(
            len(outcomes), p=[outcome[3] for outcome in outcomes]
        )
        assert env.step(Action.DOWN) == (*outcomes[index][:3], False, {})
    assert_consistent_models(env)


@pytest.mark.parametrize("p_success,expected", [
    (0, [(2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (4, 4), (4, 4)]),
    (0.5, [(2, 1), (1, 1), (2, 1), (1, 1), (0, 1), (1, 1), (1, 2), (0, 2)]),
    (1, [(2, 3), (3, 3), (3, 2), (2, 2), (2, 3), (3, 3), (3, 2), (2, 2)]),
])
def test_default_seeded_trajectories_match_phase_one(p_success, expected):
    # Recorded from commit 97e240486ca444742328b772786918d2a5377612.
    env = GridWorld(start_state=(2, 2), p_success=p_success, seed=4)
    actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP] * 2
    assert [env.step(action)[0] for action in actions] == expected
    env.reset(seed=4)
    assert [env.step(action)[0] for action in actions] == expected


@pytest.mark.parametrize("p_success", [np.nan, np.inf, -np.inf, -0.1, 1.1])
def test_invalid_success_probabilities_are_rejected(p_success):
    with pytest.raises(ValueError, match="p_success"):
        GridWorld(p_success=p_success)


@pytest.mark.parametrize("slips", [
    {}, [], {"UP": 0.5, Action.DOWN: 0.5}, {4: 1},
    {Action.UP: "0.5", Action.DOWN: 0.5},
    {Action.UP: np.nan, Action.DOWN: 0.5},
    {Action.UP: np.inf, Action.DOWN: 0.5},
    {Action.UP: -0.1, Action.DOWN: 1.1},
    {Action.UP: 0.4, Action.DOWN: 0.4},
    {Action.UP: 0.6, Action.DOWN: 0.6},
    {Action.UP: 1.0},
])
@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_invalid_slip_distributions_are_rejected(slips, p_success):
    with pytest.raises(ValueError, match="slip_distribution"):
        GridWorld(slip_distribution=slips, p_success=p_success)


def test_rebuild_and_default_special_arguments_refresh_models():
    env = GridWorld(walls={(0, 1)}, rewards={"wall_collision": -7}, seed=0)
    env.add_special_transition((0, 0), Action.RIGHT)
    assert env.special_transitions[((0, 0), Action.RIGHT)] == ((0, 0), 0)
    assert env.step(Action.RIGHT) == ((0, 0), 0, False, False, {})
    env.special_transitions[((0, 0), Action.RIGHT)] = ((1, 1), 2)
    env.build_mdp()
    assert_consistent_models(env)
    assert env.P[0, 6, Action.RIGHT] == 1


@pytest.mark.parametrize("source,destination,reward", [
    ((-1, 0), (1, 1), 1), ((0, 0), (5, 0), 1),
    ((0, 0), (1, 1), np.nan), ((0, 0), (1, 1), np.inf),
])
def test_invalid_special_transition_leaves_models_unchanged(source, destination, reward):
    env = GridWorld(seed=0)
    original = copy.deepcopy((env.special_transitions, env.mdp, env.transition_probs, env.P))
    with pytest.raises(ValueError, match="special transition"):
        env.add_special_transition(source, Action.RIGHT, destination, reward)
    np.testing.assert_equal(
        (env.special_transitions, env.mdp, env.transition_probs, env.P), original
    )
    with pytest.raises(ValueError, match="special transition"):
        GridWorld(special_transitions={(source, Action.RIGHT): (destination, reward)})


@pytest.mark.parametrize("p_success", [0, 0.5, 1])
def test_step_and_transition_helper_sample_same_canonical_outcomes(p_success):
    env = GridWorld(
        start_state=(1, 1), p_success=p_success,
        slip_distribution={Action.UP: 0.75, Action.DOWN: 0.25},
        special_transitions={((1, 1), Action.RIGHT): ((3, 3), 5)}, seed=0,
    )
    env.reset(seed=4)
    expected = env.step(Action.RIGHT)[0]
    env.reset(seed=4)
    assert env.make_transition((1, 1), Action.RIGHT) == expected
    assert env.state == (1, 1)
    check_env(env, skip_render_check=True)
