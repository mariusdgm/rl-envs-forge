"""Physical failure and external horizons are independent ending conditions."""

import numpy as np
import pytest

from rl_envs_forge.envs.inverted_pendulum.cart_pole.cart_pole import CartPole
from rl_envs_forge.envs.inverted_pendulum.pendulum_disk.pendulum_disk import PendulumDisk


@pytest.mark.parametrize("env_type,state,kwargs", [
    (CartPole, [2.39, 1, 0, 0], {}),  # x_next = 2.39 + .02 > 2.4.
    (CartPole, [0, 0, 0.19, 1], {}),  # theta_next = .21 > .2.
    (CartPole, [0, 0, 0.11, 0], {"angle_termination": 0.1}),
    (PendulumDisk, [np.pi / 2, 47], {}),  # Gravity pushes velocity above 15*pi.
    (PendulumDisk, [0.2, 0], {"angle_termination": 0.1}),
])
@pytest.mark.parametrize("horizon", [1, 2])
def test_physical_failure_is_retained_at_and_before_horizon(env_type, state, kwargs, horizon):
    env = env_type(episode_length_limit=horizon, **kwargs)
    env.reset(seed=0, options={"initial_state": np.array(state, dtype=float)})
    _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert terminated
    assert truncated == (horizon == 1)
    assert info["truncated"] == truncated
    assert info["steps"] == 1


@pytest.mark.parametrize("env_type,size", [(CartPole, 4), (PendulumDisk, 2)])
@pytest.mark.parametrize("horizon", [1, 2])
def test_timeout_only_and_ordinary_transition(env_type, size, horizon):
    env = env_type(episode_length_limit=horizon)
    env.reset(seed=0, options={"initial_state": np.zeros(size)})
    observation, reward, terminated, truncated, _ = env.step(np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(observation, np.zeros(size))
    assert reward == 1
    assert not terminated
    assert truncated == (horizon == 1)
