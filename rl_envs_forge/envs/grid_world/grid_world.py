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
        """
        Initializes a new instance of the class.

        Args:
            rows (int, optional): The number of rows in the grid. Defaults to 5.
            cols (int, optional): The number of columns in the grid. Defaults to 5.
            start_state (Tuple[int, int], optional): The starting state of the agent. Defaults to (0, 0).
            terminal_states (Dict[Tuple[int, int], float], optional): A dictionary of terminal states and their corresponding probabilities. Defaults to {(3, 4): 1.0}.
            walls (Set[Tuple[int, int]], optional): A set of walls in the grid. Defaults to None.
            special_transitions (Dict[Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]], optional): A dictionary of special transitions and their corresponding probabilities. Defaults to None.
            rewards (Dict[str, float], optional): A dictionary of rewards and their corresponding values. Defaults to None.
            seed (Optional[int], optional): The seed for random number generation. Defaults to None.
            slip_distribution (Optional[Dict[Action, float]], optional): A dictionary of slip probabilities for each action. Defaults to None.
            p_success (float, optional): The success probability for slip actions. Defaults to 1.0.

        Returns:
            None
        """
        super().__init__()
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.rows, self.cols = rows, cols
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.walls = walls or set()  # Initialize walls
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
        self.P = self.create_probability_matrix()

    def add_special_transition(
        self,
        from_state: Tuple[int, int],
        action: Action,
        to_state: Optional[Tuple[int, int]] = None,
        reward: Optional[float] = None,
    ):
        """
        Add a special transition to the Markov Decision Process (MDP) model.

        Args:
            from_state (Tuple[int, int]): The current state in the MDP.
            action (Action): The action taken from the current state.
            to_state (Optional[Tuple[int, int]]): The next state in the MDP (default is None).
            reward (Optional[float]): The reward for the transition (default is None).

        Returns:
            None
        """
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
        """
        Builds the Markov Decision Process (MDP) based on the current environment state.
        Integrates special transitions directly into the MDP.
        """
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

    def create_probability_matrix(self):
        """
        Initialize the probability matrix with zeros
        The matrix size is (rows * cols, rows * cols, len(Action)) to accommodate all state transitions for each action
        """
        # Initialize the probability matrix with zeros
        # The matrix size is (rows * cols, rows * cols, len(Action)) to accommodate all state transitions for each action
        P = np.zeros((self.rows * self.cols, self.rows * self.cols, len(Action)))

        for state in [(r, c) for r in range(self.rows) for c in range(self.cols)]:
            state_index = state[0] * self.cols + state[1]

            if state in self.terminal_states or state in self.walls:
                # For terminal states or walls, set the transition probability to stay in the same state
                for action in Action:
                    P[state_index, state_index, action] = 1.0
            else:
                for action in Action:
                    outcomes = self.mdp.get((state, action), [])
                    for outcome in outcomes:
                        next_state, _, _, prob = outcome
                        next_state_index = next_state[0] * self.cols + next_state[1]
                        P[state_index, next_state_index, action] += prob

        return P

    def make_transition(self, state, action):
        if self.np_random.rand() < self.random_move_frequency:
            action = self.np_random.choice(list(Action))

        # Check for a special transition first
        if (state, action) in self.special_transitions:
            return self.special_transitions[(state, action)][0]

        for state in [(r, c) for r in range(self.rows) for c in range(self.cols)]:
            state_index = state[0] * self.cols + state[1]

            if state in self.terminal_states or state in self.walls:
                # For terminal states or walls, set the transition probability to stay in the same state
                for action in Action:
                    P[state_index, state_index, action] = 1.0
            else:
                for action in Action:
                    outcomes = self.mdp.get((state, action), [])
                    for outcome in outcomes:
                        next_state, _, _, prob = outcome
                        next_state_index = next_state[0] * self.cols + next_state[1]
                        P[state_index, next_state_index, action] += prob

        return P

    def calculate_outcomes(self, state, action):
        """
        Calculate the possible outcomes for taking a given action in a given state.

        Args:
            state: The current state of the environment.
            action: The action to be taken in the current state.

        Returns:
            A list of tuples, each containing the next state, reward, and a flag indicating whether the episode is done, along with the probability of the outcome.
        """
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
        """
        This method calculates the next state and reward, factoring in walls, out-of-bounds, and special transitions
        """
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
        """
        Perform a default transition based on the given state and action.

        Parameters:
            state (tuple): The current state represented as a tuple (row, column).
            action (Action): The action to be taken.

        Returns:
            tuple: The new state after applying the transition.
        """
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
        """
        Calculate the transition probabilities for each state-action pair in the environment.

        Returns:
            dict: A dictionary containing the transition probabilities for each state-action pair.
        """
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
        """
        Calculate the next state based on the current state and the action taken.

        Args:
            state: tuple, the current state coordinates
            action: Action, the action taken
            check_special: bool, flag to check for special transitions (default True)

        Returns:
            tuple: the next state coordinates
        """
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
        """
        Calculate the reward based on the current state, action, and next state.

        Parameters:
            self (object): The object itself
            current_state (int): The current state
            action (str): The action taken
            next_state (int): The next state

        Returns:
            float: The reward value based on the conditions
        """
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
        """
        A method to take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            tuple: A tuple containing the next state, reward, and a flag indicating if the episode is done.
        """
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

        # Update state only if not done, else keep the terminal state until reset is called
        if not done:
            self.state = next_state
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
