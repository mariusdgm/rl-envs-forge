from typing import List, Tuple, Dict
import gymnasium as gym
import numpy as np
from scipy.stats import poisson

from .transition_model import CarRentalTransitionModel, transfer_cars


class JacksCarRental(gym.Env):
    """
    Jack's Car Rental environment
    """
    def __init__(
        self,
        max_cars: int = 20,
        max_move_cars: int = 5,
        rental_credit: int = 10,
        move_car_cost: int = 2,
        request_lambda: List[int] = [3, 4],
        return_lambda: List[int] = [3, 2],
        init_state_option: str = "random",
        poisson_max_value: int = 12,
    ) -> None:
        """
        Initializes the environment with the specified parameters.

        Args:
            max_cars: The maximum number of cars in the environment. Defaults to 20.
            max_move_cars: The maximum number of cars that can be moved in a single action. Defaults to 5.
            rental_credit: The credit earned for renting a car. Defaults to 10.
            move_car_cost: The cost of moving a car. Defaults to 2.
            request_lambda: The lambda values for the Poisson distribution of car rental requests. Defaults to [3, 4].
            return_lambda: The lambda values for the Poisson distribution of car returns. Defaults to [3, 2].
            init_state_option: The option for initializing the initial state. Defaults to "random".
            poisson_max_value: Maximum count in the legacy point-probability table. Planning models aggregate saturated tails independently of this value. Defaults to 12.

        Returns:
            None
        """

        self.max_cars = max_cars
        self.max_move_cars = max_move_cars
        self.rental_credit = rental_credit
        self.move_car_cost = move_car_cost
        self.request_lambda = request_lambda
        self.return_lambda = return_lambda

        self.action_space = gym.spaces.Discrete(2 * max_move_cars + 1)
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(max_cars + 1), gym.spaces.Discrete(max_cars + 1))
        )

        self._precompute_poisson_probabilities(max_value=poisson_max_value)
        self.reset(init_state_option)

    def _move_cars_and_calculate_cost(
        self, state: Tuple[int, int], action: int
    ) -> Tuple[Tuple[int, int], int]:
        """
        Move cars between locations and calculate the cost.

        Args:
            state: A tuple containing the number of cars at location A and location B.
            action: The number of cars to move from A to B or from B to A.

        Returns:
            A tuple containing the new state after moving the cars, with the number of cars at location A and location B.
            The cost of moving the cars.
        """
        return transfer_cars(
            state, action, self.max_cars, self.max_move_cars, self.move_car_cost
        )

    def _simulate_rentals_and_returns(
        self, state: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], int]:
        """
        Simulates the process of rentals and returns of cars at two locations.

        Parameters:
            state (tuple): A tuple containing the number of cars at location A and location B.

        Returns:
            tuple: A tuple containing the updated number of cars at location A and location B after the rentals and returns, and the reward earned from the rentals.
        """
        cars_at_A, cars_at_B = state
        requests_A = poisson.rvs(self.request_lambda[0], random_state=self.np_random)
        requests_B = poisson.rvs(self.request_lambda[1], random_state=self.np_random)
        returns_A = poisson.rvs(self.return_lambda[0], random_state=self.np_random)
        returns_B = poisson.rvs(self.return_lambda[1], random_state=self.np_random)

        rentals_A = min(cars_at_A, requests_A)
        rentals_B = min(cars_at_B, requests_B)

        reward = (rentals_A + rentals_B) * self.rental_credit

        cars_at_A = min(cars_at_A - rentals_A + returns_A, self.max_cars)
        cars_at_B = min(cars_at_B - rentals_B + returns_B, self.max_cars)

        return (cars_at_A, cars_at_B), reward

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, dict]:
        """
        Performs a single step in the environment.

        Args:
            action: Integer index; subtract max_move_cars for the signed transfer.
                Positive transfers move A to B. Requests are clipped by source
                inventory, destination capacity, and the overnight limit. Only
                cars actually moved incur a charge.

        Returns:
            Observation tuple, reward, False, False, and an empty info dict.
            Transfers precede rentals, which precede capacity-capped returns.
        """
        action = action - self.max_move_cars
        moved_state, move_cost = self._move_cars_and_calculate_cost(self.state, action)
        new_state, rental_reward = self._simulate_rentals_and_returns(moved_state)
        reward = rental_reward - move_cost
        self.state = new_state
        return new_state, reward, False, False, {}

    def reset(self, option: str = "random", *, seed=None, options=None):
        """
        Resets the state of the environment.

        Parameters:
            option (str, optional): The reset option. Can be "random" or "equal". Defaults to "random".
            seed (int, optional): Reseed the environment's random stream.
            options (dict, optional): May contain ``option`` to select the reset mode.

        Returns:
            tuple: The new state of the environment after the reset and a dictionary containing the reset option.
        """

        super().reset(seed=seed)
        option = (options or {}).get("option", option)
        if option == "random":
            self.state = (
                int(self.np_random.integers(0, self.max_cars + 1)),
                int(self.np_random.integers(0, self.max_cars + 1)),
            )
        elif option == "equal":
            initial_cars = self.max_cars // 2
            self.state = (initial_cars, initial_cars)
        else:
            raise ValueError("option must be 'random' or 'equal'")
        return self.state, {"Reset Option": option}

    def render(self, mode: str = "human") -> str:
        """
        Render the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment.
                Must be one of "human" or "ansi". Defaults to "human".

        Returns:
            str: The rendered state of the environment in the specified mode.

        Raises:
            NotImplementedError: If the specified mode is not supported.
        """
        if mode == "human":
            print(f"Location A: {self.state[0]} cars, Location B: {self.state[1]} cars")
        elif mode == "ansi":
            return f"Location A: {self.state[0]} cars, Location B: {self.state[1]} cars"
        else:
            raise NotImplementedError("Render mode not supported: " + mode)

    def _precompute_poisson_probabilities(self, max_value: int = 12) -> None:
        """
        Precomputes the Poisson probabilities for the given maximum value.

        Args:
            max_value: The maximum value for the Poisson distribution. Defaults to 12.

        Returns:
            None
        """

        self.poisson_probs: Dict[str, Dict[str, List[float]]] = {
            "request": {
                "A": [
                    poisson.pmf(i, self.request_lambda[0]) for i in range(max_value + 1)
                ],
                "B": [
                    poisson.pmf(i, self.request_lambda[1]) for i in range(max_value + 1)
                ],
            },
            "return": {
                "A": [
                    poisson.pmf(i, self.return_lambda[0]) for i in range(max_value + 1)
                ],
                "B": [
                    poisson.pmf(i, self.return_lambda[1]) for i in range(max_value + 1)
                ],
            },
        }

    def _get_probability_of_rental_requests(
        self, request_A: int, request_B: int
    ) -> float:
        """
        Calculates the probability of rental requests occurring simultaneously.

        Args:
            request_A (int): The number of rental requests from source A.
            request_B (int): The number of rental requests from source B.

        Returns:
            float: The probability of both rental requests occurring.
        """
        prob_A = self.poisson_probs["request"]["A"][request_A]
        prob_B = self.poisson_probs["request"]["B"][request_B]

        return prob_A * prob_B

    def _get_probability_of_car_returns(self, return_A: int, return_B: int) -> float:
        """Calculate the probability of both car returns happening simultaneously.

        Args:
            return_A: The number of car returns at location A.
            return_B: The number of car returns at location B.

        Returns:
            The probability of both car returns happening simultaneously.
        """
        return (
            self.poisson_probs["return"]["A"][return_A]
            * self.poisson_probs["return"]["B"][return_B]
        )

    def build_transition_model(self) -> CarRentalTransitionModel:
        """Build a lazy snapshot of complete state/action transition distributions.

        Keys use the same integer action indices as step(). Each value is an
        immutable tuple of (next_state, conditional_expected_reward, False,
        probability) outcomes. Poisson tails are aggregated at inventory and
        capacity bounds; poisson_max_value does not truncate this model.
        """
        return CarRentalTransitionModel(
            self.max_cars, self.max_move_cars, self.rental_credit,
            self.move_car_cost, self.request_lambda, self.return_lambda,
        )

    def build_mdp(self):
        """Return the legacy representative-state planning dictionary.

        Keys remain (state, signed_transfer), and values remain
        (representative_next_state, expected_reward, False). The representative
        state is the inventory immediately after the feasible transfer, as in
        the original helper's zero-request/zero-return first event. Its reward
        is now the complete expected daily reward, including saturated tails.

        This is not a transition distribution. Use build_transition_model()
        for stochastic planning and Gymnasium-indexed actions. No default-size
        dense transition tensor is constructed by either method.
        """
        model = self.build_transition_model()
        legacy = {}
        for state, action_index in model:
            action = action_index - self.max_move_cars
            legacy[(state, action)] = (
                self._move_cars_and_calculate_cost(state, action)[0],
                model.expected_reward(state, action_index),
                False,
            )
        return legacy
