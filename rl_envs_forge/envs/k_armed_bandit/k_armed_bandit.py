from typing import Optional, List
import numpy as np
import gymnasium as gym

class KArmedBandit(gym.Env):
    def __init__(self, 
                 k: int = 10, 
                 reward_distributions: Optional[np.ndarray] = None,
                 reward_std: float = 1.0,
                 seed: Optional[int] = None):
        """
        k-armed bandit environment for reinforcement learning.

        Args:
            k (int): Number of arms of the bandit. Defaults to 10.
            reward_distributions (numpy.ndarray, optional): Mean reward for each arm. If not provided, 
                rewards are drawn from a standard normal distribution. Size should be (k,).
            reward_std (float): Standard deviation of the reward distribution. Defaults to 1.0.
            seed (int, optional): Seed for reproducibility. Defaults to None.
        """
        super().__init__()

        self.k = k
        self.reward_std = reward_std
        self.seed = seed
        self.np_random = np.random.RandomState(self.seed)

        if reward_distributions is not None:
            assert len(reward_distributions) == k, "Mismatch between k and reward_distributions size."
            self.true_values = reward_distributions
        else:
            self.true_values = self.np_random.normal(0, 1, k)

        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, action: int):
        """
        Take a step using the chosen action.

        Args:
            action (int): The chosen arm.

        Returns:
            tuple: observation (always 0 in this environment), reward, done (always True), info (empty dict)
        """
        assert 0 <= action < self.k, "Invalid action, must be between 0 and k-1."
        reward = self.np_random.normal(self.true_values[action], self.reward_std)
        done = True  
        return 0, reward, done, {}

    def reset(self):
        """
        Reset the environment. 

        For k-armed bandit, the environment state remains unchanged between resets.
        
        Returns:
            int: An initial observation, which is always 0.
        """
        return 0

    def render(self, mode='human'):
        """
        Render the environment.

        For now, it prints the true values of the arms when in 'human' mode.

        Args:
            mode (str): The mode for rendering. Only 'human' is supported currently.
        """
        if mode == 'human':
            print("True values of arms:", self.true_values)
        else:
            super().render(mode=mode)  # Just call the super class implementation, which will raise an exception

    def close(self):
        """
        Close the environment. 

        For this simple environment, there's nothing to close, but it's added for API consistency.
        """
        pass