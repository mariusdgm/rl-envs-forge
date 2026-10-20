"""Regression tests for Gymnasium reset arguments and independent RNG streams."""

import copy
import random

import numpy as np
import pytest

from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental
from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem
from rl_envs_forge.envs.grid_world.grid_world import Action, GridWorld
from rl_envs_forge.envs.inverted_pendulum.cart_pole.cart_pole import CartPole
from rl_envs_forge.envs.inverted_pendulum.pendulum_disk.pendulum_disk import PendulumDisk
from rl_envs_forge.envs.k_armed_bandit.k_armed_bandit import KArmedBandit
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


@pytest.fixture(
    params=[
        pytest.param(CartPole, id="cart-pole"),
        pytest.param(PendulumDisk, id="pendulum-disk"),
        pytest.param(JacksCarRental, id="car-rental"),
        pytest.param(lambda: GamblersProblem(start_capital=50), id="gambler"),
        pytest.param(lambda: GridWorld(p_success=0.5), id="grid-world"),
        pytest.param(KArmedBandit, id="bandit"),
        pytest.param(Labyrinth, id="labyrinth"),
        pytest.param(NetworkGraph, id="network-graph"),
    ]
)
def env_factory(request):
    return request.param


def rollout(env):
    if isinstance(env, (CartPole, PendulumDisk, NetworkGraph)):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
    elif isinstance(env, JacksCarRental):
        action = env.max_move_cars
    elif isinstance(env, GamblersProblem):
        action = 1
    else:
        action = 0

    transitions = []
    for _ in range(4):
        transition = copy.deepcopy(env.step(action))
        transitions.append(transition)
        if transition[2] or transition[3]:
            break
    return transitions


@pytest.mark.parametrize("seed", [0, 42])
def test_seeded_reset_replays_episode(env_factory, seed):
    env = env_factory()
    initial = copy.deepcopy(env.reset(seed=seed, options={}))
    expected = rollout(env)
    rollout(env)

    np.testing.assert_equal(env.reset(seed=seed, options={}), initial)
    np.testing.assert_equal(rollout(env), expected)


def test_unseeded_reset_continues_rng_stream(env_factory):
    env = env_factory()
    env.reset(seed=0)
    rng = env.np_random
    state = copy.deepcopy(rng.bit_generator.state)
    env.reset(seed=None, options={})

    assert env.np_random is rng
    continued_sample = env.np_random.random()
    rng.bit_generator.state = state
    # A rewind to the original seed must not have happened, even for resets
    # that do not themselves sample a random initial observation.
    next_sample = env.np_random.random()
    if isinstance(env, (GridWorld, GamblersProblem, Labyrinth)):
        assert continued_sample == next_sample
    else:
        assert continued_sample != next_sample


def test_simulation_does_not_consume_global_randomness(env_factory):
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    env = env_factory()
    env.reset(seed=0, options={})
    rollout(env)
    env.reset()

    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state


@pytest.mark.parametrize("env_type", [CartPole, PendulumDisk])
def test_pendulum_initial_state_option_and_legacy_argument(env_type):
    env = env_type()
    state = np.full(env.observation_space.shape, 0.005)
    observation, info = env.reset(seed=0, options={"initial_state": state})
    np.testing.assert_array_equal(observation, state)
    assert info == {}
    np.testing.assert_array_equal(env.reset(initial_state=state)[0], state)


def test_grid_start_state_option_and_legacy_argument():
    env = GridWorld()
    assert env.reset(seed=0, options={"new_start_state": (1, 1)}) == ((1, 1), {})
    assert env.reset((2, 2)) == ((2, 2), {})


def test_grid_transition_helper_uses_seeded_mdp_without_mutating_state():
    env = GridWorld(p_success=0.5)
    env.reset(seed=0)
    expected = [env.make_transition((2, 2), Action.RIGHT) for _ in range(20)]
    env.reset(seed=0)
    assert [env.make_transition((2, 2), Action.RIGHT) for _ in range(20)] == expected
    assert env.state == env.start_state
    assert len(set(expected)) > 1
    assert env.make_transition((3, 4), Action.RIGHT) == (3, 4)


def test_car_rental_option_and_legacy_argument():
    env = JacksCarRental()
    expected = ((10, 10), {"Reset Option": "equal"})
    assert env.reset(seed=0, options={"option": "equal"}) == expected
    assert env.reset("equal") == expected
    with pytest.raises(ValueError, match="option"):
        env.reset(options={"option": "unknown"})


def test_bandit_reset_restarts_parameter_schedule():
    env = KArmedBandit(
        k=1,
        arm_params={
            0: {
                "distribution": "normal",
                "mean": 0,
                "std": 0,
                "param_functions": [
                    {"function": lambda timestep: timestep, "target_param": "mean"}
                ],
            }
        },
    )
    env.reset(seed=0)
    expected = [env.step(0)[1] for _ in range(4)]
    env.reset(seed=0)
    assert env.timestep == 0
    assert [env.step(0)[1] for _ in range(4)] == expected


def test_labyrinth_seed_zero_and_same_seed_option():
    env = Labyrinth(seed=0)
    initial = env.state.copy()
    np.testing.assert_array_equal(env.reset(seed=0)[0], initial)
    np.testing.assert_array_equal(env.reset(options={"same_seed": True})[0], initial)
    env.reset()
    assert env.seed == 1


def test_labyrinth_copy_has_independent_matching_rng():
    env = Labyrinth(seed=0)
    copied = copy.deepcopy(env)
    assert env.np_random is not copied.np_random
    np.testing.assert_array_equal(env.np_random.random(10), copied.np_random.random(10))


def test_network_reset_options_preserve_topology_and_fixed_opinions():
    env = NetworkGraph(initial_opinions=np.linspace(0, 1, 10), seed=0)
    topology = env.connectivity_matrix.copy()
    randomized, info = env.reset(seed=0, options={"randomize_opinions": True})
    assert info == {"random_opinions": True}
    assert not np.array_equal(randomized, env.initial_opinions)
    np.testing.assert_array_equal(env.reset(True, seed=0)[0], randomized)
    np.testing.assert_array_equal(env.connectivity_matrix, topology)
    np.testing.assert_array_equal(env.reset()[0], env.initial_opinions)
