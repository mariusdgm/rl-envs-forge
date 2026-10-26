"""State-feasible stakes agree with the textbook planning model."""

import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem


@pytest.mark.parametrize("goal", [5, 10, 100])
def test_all_feasible_stakes_masks_and_mdp_keys_agree(goal):
    env = GamblersProblem(goal_amount=goal)
    model = env.build_mdp()
    keys = set()
    for state in range(goal + 1):
        expected = tuple(range(1, min(state, goal-state) + 1))
        assert env.feasible_actions(state) == expected
        mask = env.action_mask(state)
        assert mask.shape == (goal + 1,) and mask.dtype == bool
        assert tuple(np.flatnonzero(mask)) == expected
        keys.update((state, action) for action in expected)
        mask[:] = False
        assert tuple(np.flatnonzero(env.action_mask(state))) == expected
    assert model.keys() == keys
    assert env.action_space.n == goal + 1


@pytest.mark.parametrize("probability", [0, 1])
def test_every_feasible_stake_matches_deterministic_win_or_loss_model(probability):
    for state in range(1, 8):
        env = GamblersProblem(goal_amount=8, start_capital=state,
                             win_probability=probability, strict_actions=True)
        model = env.build_mdp()
        for action in env.feasible_actions():
            env.reset(seed=0)
            outcome = model[(state, action)]["win" if probability else "loss"]
            assert env.step(action) == (*outcome[:3], False, {})
            assert outcome[3] == 1


@pytest.mark.parametrize("probability", [0, 1])
@pytest.mark.parametrize("action", [0, 5])
def test_default_keeps_zero_stakes_and_overbets_permissive(probability, action):
    env = GamblersProblem(goal_amount=5, start_capital=2, win_probability=probability)
    env.reset(seed=0)
    expected = 2 if action == 0 else (5 if probability else 0)
    assert env.step(action) == (expected, int(expected == 5), expected in (0, 5), False, {})


@pytest.mark.parametrize("action", [0, 3, 5])
def test_strict_validation_is_opt_in_and_rejection_has_no_rng_or_state_effect(action):
    env = GamblersProblem(goal_amount=5, start_capital=2, strict_actions=True)
    env.reset(seed=0)
    rng_state = copy.deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(ValueError, match="not feasible"):
        env.step(action)
    assert env.state == 2
    np.testing.assert_equal(env.np_random.bit_generator.state, rng_state)


@pytest.mark.parametrize("state", [-1, 101, 0.5, "10"])
def test_feasible_helper_validates_capital(state):
    with pytest.raises(ValueError, match="capital"):
        GamblersProblem().feasible_actions(state)


def test_mask_inspection_is_read_only_and_strict_seeded_runs_replay():
    env = GamblersProblem(goal_amount=10, start_capital=5, strict_actions=True)
    initial = env.reset(seed=42)
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    assert env.feasible_actions() == (1, 2, 3, 4, 5)
    env.action_mask()
    np.testing.assert_equal(env.np_random.bit_generator.state, rng)
    assert env.state == 5
    expected = []
    for _ in range(20):
        expected.append(env.step(1))
        if expected[-1][2]:
            break
    assert env.reset(seed=42) == initial
    assert [env.step(1) for _ in expected] == expected


def test_permissive_gambler_remains_gymnasium_compatible():
    check_env(GamblersProblem(start_capital=50), skip_render_check=True)
