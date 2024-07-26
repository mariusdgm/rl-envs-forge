import pytest
import numpy as np

from rl_envs_forge.envs.inverted_pendulum.cart_pole.cart_pole import (
    CartPole
)


@pytest.fixture
def default_env():
    """Fixture to create a default CartPole environment for each test."""
    return CartPole()

@pytest.fixture
def continuous_reward_env():
    """Fixture to create a CartPole environment with continuous reward."""
    return CartPole(continuous_reward=True)

@pytest.fixture
def limited_length_env():
    """Fixture to create a CartPole environment with an episode length limit."""
    return CartPole(episode_length_limit=10)

@pytest.fixture
def angle_termination_env():
    """Fixture to create a CartPole environment with angle termination."""
    return CartPole(angle_termination=0.1)

class TestCartPole:
    def test_initialization(self, default_env):
        assert default_env.tau == 0.02
        assert default_env.continuous_reward is False
        assert default_env.episode_length_limit == 1000
        assert default_env.angle_termination is None
        assert default_env.initial_state is None

    def test_reset(self, default_env):
        default_env.reset()
        assert len(default_env.state) == 4
        assert default_env.current_step == 0

    def test_reset_with_initial_state(self, default_env):
        initial_state = [0.1, 0.1, 0.1, 0.1]
        state = default_env.reset(initial_state=initial_state)
        assert np.array_equal(state, initial_state)
        assert np.array_equal(default_env.state, initial_state)

    def test_step(self, default_env):
        default_env.reset()
        action = np.array([1.0])
        state, reward, done, info = default_env.step(action)
        assert len(state) == 4
        assert reward == 1.0
        assert done is False
        assert info == {"truncated": False}

    def test_continuous_reward(self, continuous_reward_env):
        continuous_reward_env.reset()
        action = np.array([1.0])
        continuous_reward_env.state = [0.0, 0.0, 0.1, 0.0]  # Set state for testing
        state, reward, done, info = continuous_reward_env.step(action)
        expected_reward = 1.0 - abs(0.1) / 0.2
        assert reward == expected_reward

    def test_episode_length_limit(self, limited_length_env):
        limited_length_env.reset()
        action = np.array([1.0])
        for _ in range(limited_length_env.episode_length_limit - 1):
            state, reward, done, info = limited_length_env.step(action)
            assert not done
            assert not info["truncated"]
        state, reward, done, info = limited_length_env.step(action)
        assert not done
        assert info["truncated"]

    def test_angle_termination(self, angle_termination_env):
        angle_termination_env.reset()
        action = np.array([1.0])
        angle_termination_env.state = [0.0, 0.0, 0.05, 0.0]  # Set state for testing
        state, reward, done, info = angle_termination_env.step(action)
        assert done
        assert not info["truncated"]

    def test_render(self, default_env, monkeypatch):
        """
        Test if the render function produces an output without errors.
        """
        # Mock pygame.init and pygame.display.set_mode to prevent actual rendering
        monkeypatch.setattr("pygame.init", lambda: None)
        monkeypatch.setattr("pygame.display.set_mode", lambda *args, **kwargs: None)
        monkeypatch.setattr("pygame.font.Font", lambda *args, **kwargs: None)
        default_env.render()

    def test_render_with_state(self, default_env, monkeypatch):
        """
        Test if the render function displays state information correctly.
        """
        # Mock pygame.init and pygame.display.set_mode to prevent actual rendering
        monkeypatch.setattr("pygame.init", lambda: None)
        monkeypatch.setattr("pygame.display.set_mode", lambda *args, **kwargs: None)
        monkeypatch.setattr("pygame.font.Font", lambda *args, **kwargs: None)
        
        # Set a known state
        default_env.reset()
        default_env.state = [1.0, 0.5, 0.1, 0.05]
        
        default_env.render()

    def test_close(self, default_env, monkeypatch):
        """
        Test if the close function terminates the viewer.
        """
        # Mock pygame.quit to prevent actual quitting
        monkeypatch.setattr("pygame.quit", lambda: None)
        default_env.render()
        default_env.close()
        assert default_env.viewer is None
