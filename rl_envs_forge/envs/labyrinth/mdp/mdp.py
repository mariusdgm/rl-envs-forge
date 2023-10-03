"""
Functions to generate a Markov Decision Process view of the Labyrinth environment.
"""
import numpy as np


class LabyrinthMDP:
    def state_to_key(self, grid):
        """Converts grid to a tuple that can be used as a dictionary key."""
        return tuple(grid.flatten())

    def key_to_state(self, key, shape):
        """Converts a tuple key back to a grid."""
        return np.array(key).reshape(shape)

    def build_mdp(self, env):
        num_actions = env.action_space.n

        # Dictionary to store transition and reward information
        transition_reward_dict = {}

        # Stack to keep track of state-action pairs to explore
        to_explore = []
        explored_states = set()

        # Start exploring from the initial state
        initial_state = self.state_to_key(env.state)
        for action in range(num_actions):
            to_explore.append((initial_state, action))

        while to_explore:
            state_key, action = to_explore.pop()

            if (state_key, action) in transition_reward_dict:
                continue  # Skip already explored state-action pairs

            # Reset environment to the current state
            env.reset(same_seed=True)
            env.state = self.key_to_state(state_key, env.state.shape)

            # Simulate step
            next_state, reward, _, _, _ = env.step(action)

            # Update the transition and reward dictionary
            next_state_key = self.state_to_key(next_state)
            transition_reward_dict[(state_key, action)] = (next_state_key, reward)
            explored_states.add(next_state_key)

            # Add neighboring state-action pairs to the exploration stack
            for next_action in range(num_actions):
                to_explore.append((next_state_key, next_action))

        # Build P and R matrices from the transition and reward dictionary
        # Due to the complexity of the state, constructing P and R in a traditional way may not be straightforward.
        # One way to handle this is to keep them in dictionary form, or explore other data structures to hold this information.
        return transition_reward_dict, explored_states


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
