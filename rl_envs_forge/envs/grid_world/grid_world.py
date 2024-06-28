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
        walls: Optional[Set[Tuple[int, int]]] = None,
        special_transitions: Optional[
            Dict[Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]]
        ] = None,
        rewards: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        slip_distribution: Optional[Dict[Action, float]] = None,
        p_success: float = 1.0,
        episode_length_limit: Optional[int] = None,
    ):
        """
        Initializes a new instance of the GridWorld class.

        Args:
            rows (int, optional): The number of rows in the grid world. Defaults to 5.
            cols (int, optional): The number of columns in the grid world. Defaults to 5.
            start_state (Tuple[int, int], optional): The starting state of the agent in the grid world. Defaults to (0, 0).
            terminal_states (Dict[Tuple[int, int], float], optional): A dictionary mapping terminal states to their corresponding rewards. Defaults to {(3, 4): 1.0}.
            walls (Optional[Set[Tuple[int, int]]], optional): A set of walls in the grid world. Defaults to None.
            special_transitions (Optional[Dict[Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]]], optional): A dictionary mapping special transitions to their corresponding destinations and probabilities. Defaults to None.
            rewards (Optional[Dict[str, float]], optional): A dictionary mapping reward names to their corresponding values. Defaults to None.
            seed (Optional[int], optional): The seed for random number generation. Defaults to None.
            slip_distribution (Optional[Dict[Action, float]], optional): A dictionary mapping actions to their corresponding slip probabilities. Defaults to None.
            p_success (float, optional): The success probability for actions. Defaults to 1.0.
            episode_length_limit (Optional[int], optional): The maximum length of an episode. Defaults to None.

        Returns:
            None
        """
        self.validate_args(
            rows,
            cols,
            start_state,
            terminal_states,
            walls,
            special_transitions,
            rewards,
            seed,
            slip_distribution,
            p_success,
            episode_length_limit,
        )

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

        self.episode_length_limit = episode_length_limit
        self.episode_length_counter = 0

        self.build_mdp()
        self.P = self.create_probability_matrix()

    @staticmethod
    def validate_args(
        rows: int,
        cols: int,
        start_state: Tuple[int, int],
        terminal_states: Dict[Tuple[int, int], float],
        walls: Optional[Set[Tuple[int, int]]],
        special_transitions: Optional[
            Dict[Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]]
        ],
        rewards: Optional[Dict[str, float]],
        seed: Optional[int],
        slip_distribution: Optional[Dict[Action, float]],
        p_success: float,
        episode_length_limit: Optional[int],
    ):
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if not isinstance(cols, int) or cols <= 0:
            raise ValueError("cols must be a positive integer")
        if not (
            isinstance(start_state, tuple)
            and len(start_state) == 2
            and all(isinstance(x, int) for x in start_state)
        ):
            raise ValueError("start_state must be a tuple of two integers")
        if not (
            isinstance(terminal_states, dict)
            and all(
                isinstance(k, tuple)
                and len(k) == 2
                and all(isinstance(x, int) for x in k)
                and isinstance(v, (int, float))
                for k, v in terminal_states.items()
            )
        ):
            raise ValueError(
                "terminal_states must be a dictionary with keys as tuples of two integers and values as integers or floats"
            )
        if walls is not None and not (
            isinstance(walls, set)
            and (
                len(walls) == 0
                or all(
                    isinstance(w, tuple)
                    and len(w) == 2
                    and all(isinstance(x, int) for x in w)
                    for w in walls
                )
            )
        ):
            raise ValueError(
                "walls must be a set of tuples of two integers or an empty set"
            )
        if special_transitions is not None and not (
            isinstance(special_transitions, dict)
            and (
                len(special_transitions) == 0
                or all(
                    isinstance(k, tuple)
                    and len(k) == 2
                    and isinstance(k[0], tuple)
                    and len(k[0]) == 2
                    and all(isinstance(x, int) for x in k[0])
                    and isinstance(k[1], Action)
                    and isinstance(v, tuple)
                    and len(v) == 2
                    and isinstance(v[0], tuple)
                    and len(v[0]) == 2
                    and all(isinstance(x, int) for x in v[0])
                    and isinstance(v[1], (int, float))
                    for k, v in special_transitions.items()
                )
            )
        ):
            raise ValueError(
                "special_transitions must be a dictionary with keys as tuples where the first element is a tuple of two integers and the second element is an Action, and values as tuples where the first element is a tuple of two integers and the second element is an integer or float, or an empty dictionary"
            )
        if rewards is not None and not (
            isinstance(rewards, dict)
            and (
                len(rewards) == 0
                or all(
                    isinstance(k, str) and isinstance(v, (int, float))
                    for k, v in rewards.items()
                )
            )
        ):
            raise ValueError(
                "rewards must be a dictionary with keys as strings and values as integers or floats, or an empty dictionary"
            )
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer or None")
        if slip_distribution is not None and not (
            isinstance(slip_distribution, dict)
            and (
                len(slip_distribution) == 0
                or all(
                    isinstance(k, Action) and isinstance(v, (int, float))
                    for k, v in slip_distribution.items()
                )
            )
        ):
            raise ValueError(
                "slip_distribution must be a dictionary with keys as Actions and values as integers or floats, or an empty dictionary"
            )
        if not isinstance(p_success, (int, float)) or not (0 <= p_success <= 1):
            raise ValueError("p_success must be a float between 0 and 1")
        if episode_length_limit is not None and not isinstance(
            episode_length_limit, int
        ):
            raise ValueError("episode_length_limit must be an integer or None")

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
        truncated = False
        if not outcomes:
            raise ValueError(
                f"No outcomes defined for state {self.state} and action {action_enum}"
            )

        # Extracting probabilities for random choice
        probabilities = [outcome[3] for outcome in outcomes]
        selected_index = self.np_random.choice(len(outcomes), p=probabilities)
        next_state, reward, done, _ = outcomes[selected_index]

        self.state = next_state
        self.episode_length_counter += 1

        # Check if the episode length limit is reached
        if (
            self.episode_length_limit is not None
            and self.episode_length_counter >= self.episode_length_limit
        ):
            truncated = True

        return self.state, reward, done, truncated, {}

    def reset(self, new_start_state: Optional[Tuple[int, int]] = None):
        """
        Reset the environment to the starting state or a new starting state if provided.

        Args:
            new_start_state (Optional[Tuple[int, int]], optional): A new starting state for the agent. Defaults to None.

        Returns:
            Tuple[int, int]: The initial observation.
        """
        if new_start_state is not None:
            if (
                not isinstance(new_start_state, tuple)
                or len(new_start_state) != 2
                or not all(isinstance(x, int) for x in new_start_state)
            ):
                raise ValueError("new_start_state must be a tuple of two integers")
            self.start_state = new_start_state

        self.state = self.start_state
        self.episode_length_counter = 0
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
