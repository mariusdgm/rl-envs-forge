from typing import Optional
import numpy as np
import gym

class KArmedBandit(gym.Env):
    def __init__(self, 
                 k: int = 10, 
                 reward_distributions: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        k-armed bandit environment for reinforcement learning.

        Args:
            k (int): Number of arms of the bandit. Defaults to 10.
            reward_distributions (numpy.ndarray, optional): Mean reward for each arm. If not provided, 
                rewards are drawn from a standard normal distribution. Size should be (k,).
            seed (int, optional): Seed for reproducibility. Defaults to None.
        """
        
        super().__init__()
        
        self.k = k
        
        self.seed = seed
        self.np_random = np.random.RandomState(self.seed)
        
        if reward_distributions is not None:
            assert len(reward_distributions) == k, "Mismatch between k and reward_distributions size."
            self.true_values = reward_distributions
        else:
            # Drawing from a standard normal distribution for each arm.
            self.true_values = self.np_random.normal(0, 1, k)
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(k)
        # As the bandit problem doesn't provide traditional state information, 
        # the observation space can be a single value (the chosen arm in most implementations).
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, action: int):
        """
        Take a step using the chosen action.

        Args:
            action (int): The chosen arm.

        Returns:
            tuple: observation, reward, done, info
        """
        assert 0 <= action < self.k, "Invalid action!"

        # Drawing a reward from a normal distribution with mean = true value of the chosen action
        # and a standard deviation of 1.
        reward = self.np_random.normal(self.true_values[action], 1)
        
        done = True  # In k-armed bandit problem, each action ends the episode.
        return action, reward, done, {}

    def reset(self):
        """Reset the environment."""
        # For k-armed bandit, reset doesn't change the environment state.
        # But we can return an initial observation.
        return 0

    def render(self, mode='human'):
        """Render the environment."""
        # Rendering can be as simple as printing the true values for each arm.
        print("True values of arms:", self.true_values)