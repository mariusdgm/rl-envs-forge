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
            print(len(to_explore))
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
            if (next_position_key not in explored_positions) and (not done):
                explored_positions.add(next_position_key)

                # Add neighboring state-action pairs to the exploration stack
                for next_action in range(num_actions):
                    to_explore.append((next_position_key, next_action))

        return transition_reward_dict, explored_positions


# # Build MDP
# transition_reward_dict = build_mdp(env)
# transition_reward_dict


# def build_index_maps(explored_states, num_actions):
#     state_key_to_index = {
#         state_key: index for index, state_key in enumerate(explored_states)
#     }
#     action_to_index = {
#         action: action for action in range(num_actions)
#     }  # action indices are the same as action values in this case
#     return state_key_to_index, action_to_index


# def create_matrices(transition_reward_dict, num_states, num_actions):
#     # Initialize matrices
#     P = np.zeros((num_states, num_actions, num_states))
#     R = np.zeros((num_states, num_actions))

#     # Assume a function state_key_to_index that maps a state key to an index
#     # and a function action_to_index that maps an action to an index
#     for (state_key, action), (next_state_key, reward) in transition_reward_dict.items():
#         state_index = state_key_to_index(state_key)
#         action_index = action_to_index(action)
#         next_state_index = state_key_to_index(next_state_key)

#         P[
#             state_index, action_index, next_state_index
#         ] = 1  # assuming deterministic transitions
#         R[state_index, action_index] = reward

#     return P, R


# # Now create the matrices
# P, R = create_matrices(transition_reward_dict, num_states, num_actions)


# # Initialize empty dictionaries for P and R
# P = {}  # Transition probabilities
# R = {}  # Rewards

# # Iterate through all possible states and actions to fill in P and R
# for state in all_possible_states:
#     state_id = state_to_key(state)
#     P[state_id] = {}  # Initialize empty dict for this state in P
#     R[state_id] = {}  # Initialize empty dict for this state in R
#     for action in range(4):  # Assuming 4 actions: 0: up, 1: right, 2: down, 3: left
#         # Reset the environment to the current state
#         env.reset()
#         env.set_state(
#             state
#         )  # Assuming set_state is a method to set the environment state
#         next_state, reward, done, _ = env.step(action)
#         next_state_id = state_to_key(next_state)
#         # Since the environment is deterministic, transition probability is 1
#         P[state_id][action] = {next_state_id: 1}
#         R[state_id][action] = {next_state_id: reward}

# # Now P and R are filled in with the transition probabilities and rewards for all state-action pairs
