import gymnasium as gym
import numpy as np
from enum import IntEnum
from typing import Tuple, Dict, Set, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Constants for actions
class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridWorld(gym.Env):
    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        start_state: Tuple[int, int] = (0, 0),
        terminal_states: Dict[Tuple[int, int], float] = {(3, 4): 1.0},
        transition_probs: Optional[Dict] = None,
        walls: Set[Tuple[int, int]] = None,  # Add this line
        special_transitions: Dict[
            Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]
        ] = None,
        rewards: Dict[str, float] = None,
        seed: Optional[int] = None,
    ):
        """
        Grid world environment for reinforcement learning.

        Args:
            width (int): Width of the grid world.
            height (int): Height of the grid world.
            start_state (Tuple[int, int]): Starting position in the grid.
            terminal_states (Dict[Tuple[int, int], float]): Terminal states and their rewards.
            transition_probs (Dict, optional): Transition probabilities for the environment dynamics.
            seed (int, optional): Seed for reproducibility.
        """
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.transition_probs = transition_probs or self._default_transition_probs()
        self.state = start_state
        self.seed = seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.walls = walls or set()  # Initialize walls
        self.special_transitions = special_transitions or {}
        self.rewards = rewards or {
            "valid_move": -0.1,
            "wall_collision": -1,
            "out_of_bounds": -1,  # Added penalty for out-of-bounds
            "terminal_state": 5,  # Added reward for reaching terminal state
            "default": 0.0,
        }

        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.rows), gym.spaces.Discrete(self.cols))
        )

    def _default_transition_probs(self):
        """
        Default deterministic transition probabilities for grid world.
        """
        return {
            Action.UP: lambda s: (max(s[0] - 1, 0), s[1]),
            Action.DOWN: lambda s: (min(s[0] + 1, self.rows - 1), s[1]),
            Action.LEFT: lambda s: (s[0], max(s[1] - 1, 0)),
            Action.RIGHT: lambda s: (s[0], min(s[1] + 1, self.cols - 1)),
        }

    def _custom_transition_probs(self):
        """
        Custom transition probabilities for grid world, considering walls and special transitions.
        """

        def transition(state, action):
            # Calculate the proposed new state based on the action
            if action == Action.UP:
                new_state = (state[0], max(state[1] - 1, 0))
            elif action == Action.DOWN:
                new_state = (state[0], min(state[1] + 1, self.rows - 1))
            elif action == Action.LEFT:
                new_state = (max(state[0] - 1, 0), state[1])
            elif action == Action.RIGHT:
                new_state = (min(state[0] + 1, self.cols - 1), state[1])

            # If the new state is a wall, return the original state
            if new_state in self.walls:
                return state

            # If there is a special transition defined, return the new state specified by the special transition
            if (state, action) in self.special_transitions:
                return self.special_transitions[(state, action)][0]

            return new_state

        # Return a dictionary that maps each action to the transition function
        return {
            Action.UP: lambda s: transition(s, Action.UP),
            Action.DOWN: lambda s: transition(s, Action.DOWN),
            Action.LEFT: lambda s: transition(s, Action.LEFT),
            Action.RIGHT: lambda s: transition(s, Action.RIGHT),
        }

    def step(self, action: int):
        """
        Take an action in the grid world, with transitions modified for walls and special transitions.

        Args:
            action (int): The action to take.

        Returns:
            tuple: Observation (state), reward, done, info
        """
        # Apply the action to the current state
        new_state = self.transition_probs[action](self.state)

        # Check if the new state is a terminal state
        done = new_state in self.terminal_states

        # Special case: if we have a special transition, get the reward from there
        if (self.state, action) in self.special_transitions:
            reward = self.special_transitions[(self.state, action)][1]
        else:
            reward = self.terminal_states.get(new_state, self.rewards["terminal_state"])

        if new_state == self.state:
            reward = (
                self.rewards["wall_collision"]
                if new_state in self.walls
                else self.rewards["out_of_bounds"]
            )

        # Update the current state
        self.state = new_state if not done else self.start_state

        truncated = False
        return self.state, reward, done, truncated, {}

    def reset(self):
        """
        Reset the environment to the starting state.

        Returns:
            Tuple[int, int]: An initial observation.
        """
        self.state = self.start_state
        return self.state

    def render(self, mode="human"):
        """
        Render the environment with matplotlib, with red for the terminal state,
        black gridlines, and an 'A' where the agent is located.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)

        # Draw the black gridlines
        for y in range(self.rows + 1):
            ax.axhline(y, color='black', lw=1)
        for x in range(self.cols + 1):
            ax.axvline(x, color='black', lw=1)

        # Draw the terminal state cell(s) in red
        for terminal_state, _ in self.terminal_states.items():
            ax.add_patch(
                plt.Rectangle(
                    (terminal_state[1], self.rows - terminal_state[0] - 1),
                    1,
                    1,
                    facecolor='red'
                )
            )

        # Draw the walls in dark gray
        for wall in self.walls:
            ax.add_patch(
                plt.Rectangle(
                    (wall[1], self.rows - wall[0] - 1),
                    1,
                    1,
                    facecolor='darkgray'
                )
            )

        # Indicate agent's position with an 'A'
        agent_pos = self.state
        ax.text(
            agent_pos[1] + 0.5,
            self.rows - agent_pos[0] - 0.5,
            'A',
            color='black',
            ha='center',
            va='center',
            fontsize=12
        )

        # Remove the axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

    def close(self):
        """
        Close the environment.
        """
        pass
