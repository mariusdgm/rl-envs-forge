import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from rl_envs_forge.envs.inverted_pendulum.pendulum_disk.pendulum_disk import PendulumDisk


@pytest.fixture
def default_env():
    """Fixture to create a default PendulumDisk environment for each test."""
    return PendulumDisk()


@pytest.fixture
def continuous_reward_env():
    """Fixture to create a PendulumDisk environment with continuous reward."""
    return PendulumDisk(continuous_reward=True)


@pytest.fixture
def limited_length_env():
    """Fixture to create a PendulumDisk environment with an episode length limit."""
    return PendulumDisk(episode_length_limit=10)


@pytest.fixture
def angle_termination_env():
    """Fixture to create a PendulumDisk environment with angle termination."""
    return PendulumDisk(angle_termination=0.1)


@pytest.fixture
def nonlinear_reward_env():
    """Fixture to create a PendulumDisk environment with nonlinear reward."""
    return PendulumDisk(continuous_reward=True, nonlinear_reward=True, reward_decay_rate=30.0)


@pytest.fixture
def discrete_reward_env():
    """Fixture to create a PendulumDisk environment with discrete reward."""
    return PendulumDisk(continuous_reward=False)


class TestPendulumDisk:
    def test_initialization(self, default_env):
        assert default_env.tau == 0.005
        assert default_env.continuous_reward is False
        assert default_env.episode_length_limit == 1000
        assert default_env.angle_termination is None
        assert default_env.initial_state is None
        assert default_env.nonlinear_reward is False
        assert default_env.curve_param == 30.0

    def test_reset(self, default_env):
        default_env.reset()
        assert len(default_env.state) == 2
        assert default_env.current_step == 0

    def test_reset_with_initial_state(self, default_env):
        initial_state = [0.1, 0.1]
        state = default_env.reset(initial_state=initial_state)
        assert np.array_equal(state, initial_state)
        assert np.array_equal(default_env.state, initial_state)

    def test_step(self, default_env):
        default_env.reset()
        action = np.array([1.0], dtype=np.float32)  # Ensure action is of correct dtype
        state, reward, done, truncated, info = default_env.step(action)
        assert len(state) == 2
        assert reward == 1.0
        assert done is False
        assert info["truncated"] is False

    def test_continuous_reward(self, continuous_reward_env):
        continuous_reward_env.reset()
        action = np.array([1.0], dtype=np.float32)  # Ensure action is of correct dtype
        continuous_reward_env.state = [0.0, 0.1]  # Set state for testing
        state, reward, done, truncated, info = continuous_reward_env.step(action)
        expected_reward = 1.0 - abs(0.1) / np.pi
        print(f"Expected reward: {expected_reward}, Actual reward: {reward}")
        assert reward == pytest.approx(expected_reward, rel=1e-5)

    def test_nonlinear_reward(self, nonlinear_reward_env):
        nonlinear_reward_env.reset()
        action = np.array([1.0], dtype=np.float32)  # Ensure action is of correct dtype
        nonlinear_reward_env.state = [0.1, 0.0]  # Set state for testing
        state, reward, done, truncated, info = nonlinear_reward_env.step(action)
        expected_reward = 1.0 - (
            1.0 - np.exp(-nonlinear_reward_env.curve_param * abs(0.1))
        ) / (1.0 - np.exp(-nonlinear_reward_env.curve_param * np.pi))
        print(f"Expected reward: {expected_reward}, Actual reward: {reward}")
        assert reward == pytest.approx(expected_reward, rel=1e-5)

    def test_discrete_reward(self, discrete_reward_env):
        discrete_reward_env.reset()
        action = np.array([1.0], dtype=np.float32)  # Ensure action is of correct dtype
        discrete_reward_env.state = [0.0, 0.0]  # Set state for testing
        state, reward, done, truncated, info = discrete_reward_env.step(action)
        assert reward == 1.0  # In the upright position

        discrete_reward_env.state = [0.2, 0.0]  # Set state outside of upright position
        state, reward, done, truncated, info = discrete_reward_env.step(action)
        assert reward == 0.0  # Not in the upright position

    def test_episode_length_limit(self, limited_length_env):
        limited_length_env.reset()
        action = np.array([1.0], dtype=np.float32)  # Ensure action is of correct dtype
        for step in range(limited_length_env.episode_length_limit - 1):
            state, reward, done, truncated, info = limited_length_env.step(action)
            print(f"Step: {step}, Done: {done}, Truncated: {truncated}")
            assert not done
            assert not truncated
        state, reward, done, truncated, info = limited_length_env.step(action)
        print(
            f"Final Step: {limited_length_env.episode_length_limit}, Done: {done}, Truncated: {truncated}"
        )
        assert not done
        assert truncated

    def test_angle_termination(self, angle_termination_env):
        angle_termination_env.reset()
        action = np.array([1.0], dtype=np.float32)  # Ensure action is of correct dtype
        angle_termination_env.state = [0.11, 0.0]  # Set state for testing
        state, reward, done, truncated, info = angle_termination_env.step(action)
        print(
            f"Done: {done}, Truncated: {truncated}, Alpha: {angle_termination_env.state[0]}"
        )
        assert done
        assert not truncated

    @patch("rl_envs_forge.envs.inverted_pendulum.pendulum_disk.pendulum_disk.pygame.quit")
    def test_close(self, mock_quit, default_env):
        """
        Test if the close function terminates the viewer.
        """
        default_env.render()
        default_env.close()
        assert default_env.viewer is None

    def test_normalize_angle(self, default_env):
        angles = [0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi, -3*np.pi]
        normalized_angles = [default_env.normalize_angle(angle) for angle in angles]
        expected_angles = [0, np.pi, np.pi, 0, 0, np.pi, np.pi]
        for angle, expected in zip(normalized_angles, expected_angles):
            assert angle == pytest.approx(expected, rel=1e-5)
            
    def test_state_initialization(self, default_env):
        initial_state = default_env._initialize_state()
        assert len(initial_state) == 2
        assert np.all(initial_state >= -0.01) and np.all(initial_state <= 0.01)
        
    def test_step_with_invalid_action(self, default_env):
        default_env.reset()
        invalid_action = np.array([5.0], dtype=np.float32)  # Out of action space bounds
        with pytest.raises(AssertionError):
            default_env.step(invalid_action)
