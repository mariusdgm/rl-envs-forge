"""
Functions to generate a Markov Decision Process view of the GridWorld environment.
"""

from typing import Dict

class GridWorldMDP:
   
    def build_mdp(self, env: "GridWorld") -> Dict:
        
        num_actions = env.action_space.n
        t_r_dict = {}
        for row in range(env.rows):
            for col in range(env.cols):
                state = (row, col)

                # Skip processing if the current state is a wall
                if state in env.walls or state in env.terminal_states:
                    continue

                for action in range(num_actions):
                    # Temporarily set the environment state
                    env.state = state

                    # Perform the action and get the next state, reward, and done flag
                    next_state, reward, done, _, _ = env.step(action)

                    # Reset the environment state to original after stepping
                    env.state = state

                    # Record the transition
                    t_r_dict[(state, action)] = (next_state, reward, done)
                    
        return t_r_dict

 