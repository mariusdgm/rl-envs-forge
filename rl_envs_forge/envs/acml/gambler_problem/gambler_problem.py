import gymnasium as gym
import numpy as np
from typing import Tuple


class GamblersProblem(gym.Env):
    """
    Gambler's Problem environment.
    """

    def __init__(
        self,
        goal_amount: int = 100,
        win_probability: float = 0.4,
        start_capital: int = 1,
    ):
        """
        Initialize the environment.

        Args:
            goal_amount (int): The goal capital amount to be achieved.
            win_probability (float): Probability of winning a bet.
        """
        self.goal_amount = goal_amount
        self.win_probability = win_probability
        self.start_capital = start_capital

        # Action space: bet an amount from 0 to the current capital
        # Observation space: current capital amount
        self.action_space = gym.spaces.Discrete(goal_amount + 1)
        self.observation_space = gym.spaces.Discrete(goal_amount + 1)

        self.state = None
        self.reset()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute one time step within the environment.

        Args:
            action (int): Amount of capital to bet.

        Returns:
            Tuple containing the new state, reward, done flag, and additional info.
        """
        assert self.action_space.contains(action), "Invalid action"

        done = False
        reward = 0

        # Win case
        if np.random.rand() < self.win_probability:
            self.state += action
            if self.state >= self.goal_amount:
                self.state = self.goal_amount
                reward = 1
                done = True
        else:
            # Lose case
            self.state -= action
            if self.state <= 0:
                self.state = 0
                done = True

        truncated = False
        return self.state, reward, done, truncated, {}

    def reset(self) -> int:
        """
        Reset the environment to an initial state.

        Returns:
            The initial state.
        """
        self.state = self.start_capital
        return self.state

    def render(self, mode="human") -> str:
        """
        Render the environment.
        """
        if mode == "human":
            print(f"Current Capital: ${self.state}")
        elif mode == "ansi":
            return f"Current Capital: ${self.state}"
        else:
            raise NotImplementedError("Render mode not supported: " + mode)

    def build_mdp(self) -> dict:
        """
        Build the MDP representation for the Gambler's Problem.
        """
        t_r_dict = {}

        for state in range(1, self.goal_amount):  # States from 1 to goal_amount - 1
            for action in range(
                1, min(state, self.goal_amount - state) + 1
            ):  # Possible bets
                # Winning scenario
                win_state = min(state + action, self.goal_amount)
                win_reward = 1 if win_state == self.goal_amount else 0
                win_prob = self.win_probability

                # Losing scenario
                lose_state = max(state - action, 0)
                lose_reward = 0
                lose_prob = 1 - self.win_probability

                t_r_dict[(state, action)] = {
                    "win": (
                        win_state,
                        win_reward,
                        win_state == self.goal_amount,
                        win_prob,
                    ),
                    "loss": (lose_state, lose_reward, lose_state == 0, lose_prob),
                }

        return t_r_dict
