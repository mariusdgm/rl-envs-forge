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
        walls: Set[Tuple[int, int]] = None,
        special_transitions: Dict[
            Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]
        ] = None,
        rewards: Dict[str, float] = None,
        seed: Optional[int] = None,
        slip_distribution: Optional[Dict[Action, float]] = None,
        p_success: float = 1.0,
    ):
        super().__init__()
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.rows, self.cols = rows, cols
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.walls = walls or set()
        self.special_transitions = special_transitions or {}
        self.state = start_state
        self.seed = seed
        self.p_success = p_success
        self.slip_distribution = slip_distribution or {
            action: (1.0 - p_success) / (len(Action) - 1) for action in Action
        }
        self.transition_probs = self.default_transition_probs()
        self.rewards = rewards or {
            "valid_move": -0.1,
            "wall_collision": -1,
            "out_of_bounds": -1,  # Added penalty for out-of-bounds
            "default": 0.0,
        }

        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.rows), gym.spaces.Discrete(self.cols))
        )

        self.build_mdp()

    def add_special_transition(
        self,
        from_state: Tuple[int, int],
        action: Action,
        to_state: Optional[Tuple[int, int]] = None,
        reward: Optional[float] = None,
    ):
        if to_state is None:
            to_state = self.calculate_next_state(
                from_state, action, check_special=False
            )
        if reward is None:
            reward = self.rewards["default"]

        # Update or insert the special transition with its intended probability
        done = to_state in self.terminal_states
        # Retain original action probability for the special transition
        self.mdp[(from_state, action)] = [(to_state, reward, done, self.p_success)]

    def build_mdp(self):
        self.mdp = {}
        for state in [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if ((r, c) not in self.walls) and ((r, c) not in self.terminal_states)
        ]:
            for action in Action:
                self.mdp[(state, action)] = self.calculate_outcomes(state, action)

        # Now integrate special transitions directly into the MDP
        for (from_state, action), (
            to_state,
            reward,
        ) in self.special_transitions.items():
            done = to_state in self.terminal_states
            # Override any existing outcomes with the special transition
            self.mdp[(from_state, action)] = [(to_state, reward, done, self.p_success)]

    def calculate_outcomes(self, state, action):
        outcomes = []
        # Handle intended action
        intended_next_state, intended_reward, intended_done = self.calculate_transition(
            state, action
        )
        outcomes.append(
            (intended_next_state, intended_reward, intended_done, self.p_success)
        )

        # Handle slippage
        if self.p_success < 1.0:
            for slip_action in Action:
                if slip_action != action:
                    slip_next_state, slip_reward, slip_done = self.calculate_transition(
                        state, slip_action
                    )
                    slip_prob = (1.0 - self.p_success) / (len(Action) - 1)
                    outcomes.append(
                        (slip_next_state, slip_reward, slip_done, slip_prob)
                    )

        return outcomes

    def calculate_transition(self, state, action):
        # This method calculates the next state and reward, factoring in walls, out-of-bounds, and special transitions
        if (state, action) in self.special_transitions:
            # Handle special transition
            to_state, reward = self.special_transitions[(state, action)]
            done = to_state in self.terminal_states
            return to_state, reward, done
        else:
            to_state = self.default_transition(state, action)
            if to_state in self.walls:
                to_state = state  # Stay in place if the next state is a wall
            reward = self.calculate_reward(state, action, to_state)
            done = to_state in self.terminal_states
            return to_state, reward, done

    def default_transition(self, state, action):
        if action == Action.UP:
            new_state = (max(state[0] - 1, 0), state[1])
        elif action == Action.DOWN:
            new_state = (min(state[0] + 1, self.rows - 1), state[1])
        elif action == Action.LEFT:
            new_state = (state[0], max(state[1] - 1, 0))
        elif action == Action.RIGHT:
            new_state = (state[0], min(state[1] + 1, self.cols - 1))

        return new_state

    def default_transition_probs(self):
        transition_probs = {}

        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                if state in self.terminal_states or state in self.walls:
                    continue

                for action in Action:
                    outcomes = {}
                    # Success case
                    next_state = self.calculate_next_state(state, action)
                    outcomes[next_state] = outcomes.get(next_state, 0) + self.p_success

                    # Slip case
                    for slip_action, slip_prob in self.slip_distribution.items():
                        slip_next_state = self.calculate_next_state(state, slip_action)
                        if slip_next_state not in outcomes:
                            outcomes[slip_next_state] = 0
                        outcomes[slip_next_state] += (1 - self.p_success) * slip_prob

                    transition_probs[(state, action)] = outcomes
        return transition_probs

    def calculate_next_state(self, state, action, check_special=True):
        if check_special and (state, action) in self.special_transitions:
            return self.special_transitions[(state, action)][
                0
            ]  # Return the special next state directly

        # Regular movement logic considering walls and bounds
        row, col = state
        if action == Action.UP:
            next_state = (max(row - 1, 0), col)
        elif action == Action.DOWN:
            next_state = (min(row + 1, self.rows - 1), col)
        elif action == Action.LEFT:
            next_state = (row, max(col - 1, 0))
        elif action == Action.RIGHT:
            next_state = (row, min(col + 1, self.cols - 1))
        else:
            next_state = state

        # Check for wall or out-of-bounds to stay in place
        if next_state in self.walls or next_state == state:
            return state
        return next_state

    def calculate_reward(self, current_state, action, next_state):
        # If the next state is a terminal state, return its associated reward
        if next_state in self.terminal_states:
            return self.terminal_states[next_state]

        # If the action leads to staying in place due to a wall or the boundary, return the 'wall_collision' or 'out_of_bounds' reward
        if next_state == current_state:
            return self.rewards.get("out_of_bounds", -1)

        # If moving into a wall, specifically
        if next_state in self.walls:
            return self.rewards.get("wall_collision", -1)

        # For any other movement, return the 'valid_move' reward
        return self.rewards.get("valid_move", -0.1)

    def step(self, action: int):
        action_enum = Action(action)
        outcomes = self.mdp.get((self.state, action_enum), [])
        if not outcomes:
            raise ValueError(
                f"No outcomes defined for state {self.state} and action {action_enum}"
            )

        # Extracting probabilities for random choice
        probabilities = [outcome[3] for outcome in outcomes]
        selected_index = self.np_random.choice(len(outcomes), p=probabilities)
        next_state, reward, done, _ = outcomes[selected_index]

        self.state = next_state if not done else self.start_state
        return self.state, reward, done, False, {}

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
            ax.axhline(y, color="black", lw=1)
        for x in range(self.cols + 1):
            ax.axvline(x, color="black", lw=1)

        # Draw the terminal state cell(s) in red
        for terminal_state, _ in self.terminal_states.items():
            ax.add_patch(
                plt.Rectangle(
                    (terminal_state[1], self.rows - terminal_state[0] - 1),
                    1,
                    1,
                    facecolor="red",
                )
            )

        # Draw the walls in dark gray
        for wall in self.walls:
            ax.add_patch(
                plt.Rectangle(
                    (wall[1], self.rows - wall[0] - 1), 1, 1, facecolor="darkgray"
                )
            )

        # Indicate agent's position with an 'A'
        agent_pos = self.state
        ax.text(
            agent_pos[1] + 0.5,
            self.rows - agent_pos[0] - 0.5,
            "A",
            color="black",
            ha="center",
            va="center",
            fontsize=12,
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
