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
        strict_actions: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            goal_amount (int): The goal capital amount to be achieved.
            win_probability (float): Probability of winning a bet.
            strict_actions (bool): Reject stakes outside feasible_actions().
                Defaults to the existing permissive action behavior.
        """
        self.goal_amount = goal_amount
        self.win_probability = win_probability
        self.start_capital = start_capital
        self.strict_actions = strict_actions

        # Fixed global action space; state-feasible stakes are exposed separately.
        # Observation space: current capital amount
        self.action_space = gym.spaces.Discrete(goal_amount + 1)
        self.observation_space = gym.spaces.Discrete(goal_amount + 1)

        self.state = None
        self.reset()

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        Execute one time step within the environment.

        Args:
            action (int): Amount of capital to bet.

        Returns:
            Tuple containing the new state, reward, done flag, and additional info.
        """
        assert self.action_space.contains(action), "Invalid action"
        if self.strict_actions and action not in self.feasible_actions():
            raise ValueError(f"Stake {action} is not feasible at capital {self.state}")

        done = False
        reward = 0

        # Win case
        if self.np_random.random() < self.win_probability:
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

    def reset(self, *, seed=None, options=None) -> Tuple[int, dict]:
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): Reseed the environment's random stream.
            options (dict, optional): Reserved for reset options.

        Returns:
            The initial state and additional info.
        """
        if not self.observation_space.contains(self.start_capital):
            raise ValueError("start_capital must be an integer between zero and goal_amount")
        # Validation precedes reseeding so rejected resets do not alter the RNG.
        super().reset(seed=seed)
        self.state = self.start_capital
        return self.state, {}

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
            for action in self.feasible_actions(state):
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

    def feasible_actions(self, state=None) -> tuple:
        """Textbook stakes at a capital level; terminal states have no stakes.

        Zero stakes and overbets remain accepted by step() unless strict_actions
        is enabled. This helper uses the same feasible set as build_mdp().
        """
        state = self.state if state is None else state
        if not isinstance(state, (int, np.integer)) or not 0 <= state <= self.goal_amount:
            raise ValueError("capital must be an integer between zero and goal_amount")
        state = int(state)
        return tuple(range(1, min(state, self.goal_amount - state) + 1))

    def action_mask(self, state=None) -> np.ndarray:
        """Return a fresh boolean mask over the unchanged global action space."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[list(self.feasible_actions(state))] = True
        return mask
