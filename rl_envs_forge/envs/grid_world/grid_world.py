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
            special_transitions (Optional[Dict[Tuple[Tuple[int, int], Action], Tuple[Tuple[int, int], float]]], optional): Intended-action transitions mapped to their destinations and rewards. Defaults to None.
            rewards (Optional[Dict[str, float]], optional): A dictionary mapping reward names to their corresponding values. Defaults to None.
            seed (Optional[int], optional): The seed for random number generation. Defaults to None.
            slip_distribution (Optional[Dict[Action, float]], optional): Global directional probabilities, summing to one. On failure, exclude the intended action and condition the remaining probabilities on an alternative action. Missing directions have probability zero. At least two directions must have positive probability. None gives uniform alternatives.
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
        super().reset(seed=seed)
        self.rows, self.cols = rows, cols
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.walls = walls or set()  # Initialize walls
        self.special_transitions = dict(special_transitions or {})
        self.state = start_state
        self.seed = seed
        self.p_success = p_success
        self.slip_distribution = (
            dict(slip_distribution) if slip_distribution is not None
            else {action: 1.0 / len(Action) for action in Action}
        )

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
        if slip_distribution is not None:
            if not isinstance(slip_distribution, dict) or not all(
                isinstance(action, Action)
                and isinstance(probability, (int, float, np.integer, np.floating))
                and 0 <= probability <= 1
                and np.isfinite(probability)
                for action, probability in slip_distribution.items()
            ):
                raise ValueError(
                    "slip_distribution must map Actions to finite probabilities between 0 and 1"
                )
            if not np.isclose(
                sum(slip_distribution.values()), 1.0, rtol=0, atol=1e-12
            ):
                raise ValueError("slip_distribution probabilities must sum to one")
            if sum(probability > 0 for probability in slip_distribution.values()) < 2:
                raise ValueError(
                    "slip_distribution must allow an alternative to every intended action; "
                    "at least two directions need positive probability"
                )
        if not isinstance(p_success, (int, float)) or not (0 <= p_success <= 1):
            raise ValueError("p_success must be a float between 0 and 1")
        for (from_state, _), (to_state, reward) in (special_transitions or {}).items():
            GridWorld._validate_special_transition(
                from_state, to_state, reward, rows, cols
            )
        if episode_length_limit is not None and not isinstance(
            episode_length_limit, int
        ):
            raise ValueError("episode_length_limit must be an integer or None")

    @staticmethod
    def _validate_special_transition(from_state, to_state, reward, rows, cols):
        for state in (from_state, to_state):
            if not (
                isinstance(state, tuple)
                and len(state) == 2
                and all(isinstance(value, int) for value in state)
                and 0 <= state[0] < rows
                and 0 <= state[1] < cols
            ):
                raise ValueError("special transition states must be grid coordinates")
        if not isinstance(reward, (int, float)) or not np.isfinite(reward):
            raise ValueError("special transition reward must be finite")

    def add_special_transition(
        self,
        from_state: Tuple[int, int],
        action: Action,
        to_state: Optional[Tuple[int, int]] = None,
        reward: Optional[float] = None,
    ):
        """
        Add or replace an intended-action transition and rebuild all models.

        The special reward applies only when the intended action succeeds.
        Slips use ordinary movement. Terminal states remain absorbing.

        Args:
            from_state (Tuple[int, int]): The current state in the MDP.
            action (Action): The action taken from the current state.
            to_state (Optional[Tuple[int, int]]): The next state in the MDP (default is None).
            reward (Optional[float]): The reward for the transition (default is None).

        Returns:
            None
        """
        action = Action(action)
        if to_state is None:
            to_state = self.calculate_next_state(
                from_state, action, check_special=False
            )
        if reward is None:
            reward = self.rewards.get("default", 0.0)

        self._validate_special_transition(
            from_state, to_state, reward, self.rows, self.cols
        )
        self.special_transitions[(from_state, action)] = (to_state, reward)
        self.build_mdp()

    def build_mdp(self):
        """
        Rebuild canonical reward-bearing outcomes and their probability views.

        Every grid cell has a row for each action. Terminal cells have zero-reward
        terminating self-loops; wall cells have zero-reward nonterminating
        self-loops, retaining their existing dense-tensor convention.
        """
        self.mdp = {}

        for state in [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
        ]:
            for action in Action:
                self.mdp[(state, action)] = self.calculate_outcomes(state, action)

        self.transition_probs = self.default_transition_probs()
        self.P = self.create_probability_matrix()

    def create_probability_matrix(self):
        """
        Initialize the probability matrix with zeros
        The matrix size is (rows * cols, rows * cols, len(Action)) to accommodate all state transitions for each action
        """
        P = np.zeros((self.rows * self.cols, self.rows * self.cols, len(Action)))

        for (state, action), outcomes in self.mdp.items():
            state_index = state[0] * self.cols + state[1]
            for next_state, _, _, probability in outcomes:
                next_state_index = next_state[0] * self.cols + next_state[1]
                P[state_index, next_state_index, action] += probability

        return P

    def make_transition(self, state, action):
        """Sample a next state from the MDP without changing the current state."""
        return self._sample_outcome(state, action)[0]

    def _sample_outcome(self, state, action):
        """Sample a reward-bearing outcome from the canonical MDP."""
        action = Action(action)
        outcomes = self.mdp.get((state, action), [])
        if not outcomes:
            raise ValueError(f"No outcomes defined for state {state} and action {action}")
        probabilities = [outcome[3] for outcome in outcomes]
        selected_index = self.np_random.choice(len(outcomes), p=probabilities)
        return outcomes[selected_index]

    def calculate_outcomes(self, state, action):
        """
        Construct the canonical outcomes for an intended action.

        The success branch may be special. Failure branches use ordinary moves,
        with global slip probabilities conditioned on excluding the intended
        action. Distinct rewards are retained even when destinations coincide.

        Args:
            state: The current state of the environment.
            action: The action to be taken in the current state.

        Returns:
            A list of tuples, each containing the next state, reward, and a flag indicating whether the episode is done, along with the probability of the outcome.
        """
        action = Action(action)
        if state in self.terminal_states or state in self.walls:
            return [(*self._ordinary_transition(state, action), 1.0)]

        intended_next_state, intended_reward, intended_done = self.calculate_transition(
            state, action
        )
        outcomes = [
            (intended_next_state, intended_reward, intended_done, self.p_success)
        ]

        if self.p_success < 1.0:
            alternative_mass = sum(
                probability for direction, probability in self.slip_distribution.items()
                if direction != action
            )
            for slip_action in Action:
                if slip_action != action:
                    slip_next_state, slip_reward, slip_done = self._ordinary_transition(
                        state, slip_action
                    )
                    slip_prob = (
                        (1.0 - self.p_success)
                        * (self.slip_distribution.get(slip_action, 0.0) / alternative_mass)
                    )
                    outcomes.append(
                        (slip_next_state, slip_reward, slip_done, slip_prob)
                    )

        return outcomes

    def calculate_transition(self, state, action):
        """Return the deterministic success branch, including special transitions."""
        action = Action(action)
        if state in self.terminal_states or state in self.walls:
            return self._ordinary_transition(state, action)
        if (state, action) in self.special_transitions:
            # Handle special transition
            to_state, reward = self.special_transitions[(state, action)]
            done = to_state in self.terminal_states
            return to_state, reward, done
        return self._ordinary_transition(state, action)

    def _ordinary_transition(self, state, action):
        """Apply ordinary movement and rewards without consulting special jumps."""
        if state in self.terminal_states:
            return state, 0, True
        if state in self.walls:
            return state, 0, False
        to_state = self.default_transition(state, action)
        if to_state in self.walls:
            to_state = state
        reward = self.calculate_reward(state, action, to_state)
        return to_state, reward, to_state in self.terminal_states

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
        Aggregate destination probabilities from the canonical outcomes.

        Returns:
            dict: A dictionary containing the transition probabilities for each state-action pair.
        """
        transition_probs = {}

        for key, outcomes in self.mdp.items():
            probabilities = {}
            for next_state, _, _, probability in outcomes:
                probabilities[next_state] = probabilities.get(next_state, 0) + probability
            transition_probs[key] = probabilities
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
        action = Action(action)
        if check_special:
            return self.calculate_transition(state, action)[0]
        return self._ordinary_transition(state, action)[0]

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
            if self.default_transition(current_state, action) in self.walls:
                return self.rewards.get("wall_collision", -1)
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
        next_state, reward, done, _ = self._sample_outcome(self.state, action)
        truncated = False

        self.state = next_state
        self.episode_length_counter += 1

        # Check if the episode length limit is reached
        if (
            self.episode_length_limit is not None
            and self.episode_length_counter >= self.episode_length_limit
        ):
            truncated = True

        return self.state, reward, done, truncated, {}

    def reset(
        self, new_start_state: Optional[Tuple[int, int]] = None, *, seed=None, options=None
    ):
        """
        Reset the environment to the starting state or a new starting state if provided.

        Args:
            new_start_state (Optional[Tuple[int, int]], optional): A new starting state for the agent. Defaults to None.
            seed (int, optional): Reseed transition sampling.
            options (dict, optional): May contain ``new_start_state``.

        Returns:
            tuple: The initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        if new_start_state is None:
            new_start_state = (options or {}).get("new_start_state")
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
        return self.state, {}

    def render(self, mode="human"):
        """
        Render the environment with matplotlib, with red for the terminal state,
        black gridlines, and an 'A' where the agent is located.
        """
        if mode == "rgb_array":
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

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        else:
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
