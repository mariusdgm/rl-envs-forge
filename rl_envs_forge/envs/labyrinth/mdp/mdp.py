"""
Functions to generate a Markov Decision Process view of the Labyrinth environment.
"""
import numpy as np
import copy


class LabyrinthMDP:
    def position_to_key(self, position):
        """Converts position to a tuple that can be used as a dictionary key."""
        return tuple(position)

    def key_to_position(self, key):
        """Converts a tuple key back to a position."""
        return np.array(key).astype(int)

    def build_mdp(self, env):
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
            transition_reward_dict[(position_key, action)] = (next_position_key, reward)

            # Check if the position has been explored already
            if next_position_key not in explored_positions:
                explored_positions.add(next_position_key)

                if not done:  # If not done, keep exploring
                    # Add neighboring state-action pairs to the exploration stack
                    for next_action in range(num_actions):
                        to_explore.append((next_position_key, next_action))

        return transition_reward_dict, explored_positions
