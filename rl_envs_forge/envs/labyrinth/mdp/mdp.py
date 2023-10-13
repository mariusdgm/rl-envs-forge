"""
Functions to generate a Markov Decision Process view of the Labyrinth environment.
"""
from typing import Tuple
import numpy as np
import copy

from ..constants import PATH


class LabyrinthMDP:
    def position_to_key(self, position: Tuple[int, ...]) -> np.ndarray:
        """Converts position to a tuple that can be used as a dictionary key."""
        return tuple(position)

    def key_to_position(self, key: Tuple[int, ...]) -> np.ndarray:
        """Converts a tuple key back to a position."""
        return np.array(key).astype(int)

    def build_mdp(self, env: "Labyrinth") -> Tuple[dict, set]:
        """
        Builds an MDP for the given Labyrinth environment using the maze layout.

        Parameters:
            env (Labyrinth): The Labyrinth environment to build the MDP for.

        Returns:
            Tuple[dict, set]: A tuple containing the transition and reward dictionary and the set of explored positions.
        """
        num_actions = env.action_space.n
        transition_reward_dict = {}
        explored_positions = set()

        # Iterate through each position in the grid where the value is PATH
        for position in np.argwhere(env.maze.grid == PATH):
            position_key = self.position_to_key(tuple(position))

            # If the position is terminal, skip it
            if env.maze.target_position == position_key:
                continue

            explored_positions.add(position_key)

            # For each action, determine the transition and reward
            for action in range(num_actions):
                env.set_state(tuple(position))  # Set the state for the current position
                next_state, reward, done, _, _ = env.step(action)

                # We are using env.player.position as state
                next_position_key = self.position_to_key(env.player.position)
                transition_reward_dict[(position_key, action)] = (
                    next_position_key,
                    reward,
                    done,
                )

        return transition_reward_dict, explored_positions

    def build_mdp_dfs(self, env: "Labyrinth") -> Tuple[dict, set]:
        """
        Builds an MDP (Markov Decision Process) for the given Labyrinth environment using DFS.

        WARNING: if a state is unreachable because the terminal state is blocking it, then the MDP will not be valid.

        Parameters:
            env (Labyrinth): The Labyrinth environment to build the MDP for.

        Returns:
            Tuple[dict, set]: A tuple containing the transition and reward dictionary and the set of explored positions.
        """
        num_actions = env.action_space.n

        # Dictionary to store transition and reward information
        transition_reward_dict = {}

        # Stack to keep track of state-action pairs to explore
        to_explore = []
        explored_positions = set()

        # Start exploring from the initial state
        initial_position = self.position_to_key(env.player.position)
        for action in range(num_actions):
            to_explore.append((initial_position, action))

        while to_explore:
            position_key, action = to_explore.pop()

            if (position_key, action) in transition_reward_dict:
                continue  # Skip already explored state-action pairs

            # Set the environment state using the position
            env_copy = copy.deepcopy(env)
            player_position = self.key_to_position(position_key)
            env_copy.set_state(player_position)

            # Simulate step on the environment copy
            next_state, reward, done, _, _ = env_copy.step(action)

            # Update the transition and reward dictionary
            next_position_key = self.position_to_key(env_copy.player.position)
            transition_reward_dict[(position_key, action)] = (
                next_position_key,
                reward,
                done,
            )

            # Check if the position has been explored already
            if next_position_key not in explored_positions:
                if not done:  # If not done, keep exploring
                    explored_positions.add(next_position_key)

                    # Add neighboring state-action pairs to the exploration stack
                    for next_action in range(num_actions):
                        to_explore.append((next_position_key, next_action))

        return transition_reward_dict, explored_positions
