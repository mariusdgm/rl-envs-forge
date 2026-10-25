"""Factorized Poisson planning for Jack's Car Rental."""

from collections.abc import Mapping

import numpy as np
from scipy.stats import poisson


def transfer_cars(state, requested, max_cars, max_move_cars, move_car_cost):
    """Clip a signed transfer by the source, destination, and overnight limit."""
    cars_a, cars_b = state
    if requested > 0:
        moved = min(requested, max_move_cars, cars_a, max_cars - cars_b)
    else:
        moved = -min(-requested, max_move_cars, cars_b, max_cars - cars_a)
    return (cars_a - moved, cars_b + moved), abs(moved) * move_car_cost


def _capped_poisson(rate, capacity):
    """Probabilities of min(Poisson(rate), capacity), including the entire tail."""
    probabilities = np.empty(capacity + 1, dtype=float)
    probabilities[:-1] = poisson.pmf(np.arange(capacity), rate)
    probabilities[-1] = poisson.sf(capacity - 1, rate)
    return probabilities


class CarRentalTransitionModel(Mapping):
    """A snapshot of lazy state/action transition rows using Gymnasium indices.

    ``model[(state, action_index)]`` returns an immutable tuple of
    ``(next_state, expected_reward, False, probability)`` outcomes. Rewards are
    conditional expectations given the destination, rather than sampled rewards.
    ``expected_reward(state, action_index)`` gives the unconditional row reward.

    Only two per-location probability/reward tables are stored. Rows are computed
    on demand and are not cached. With capacity C and K actions, construction
    takes O(C**3) time and O(C**2) memory; each row costs O(C**2). Iterating all
    keys is lazy, but explicitly materializing every row costs O(K*C**4) space.
    """

    def __init__(
        self, max_cars, max_move_cars, rental_credit, move_car_cost,
        request_lambda, return_lambda,
    ):
        self.max_cars = max_cars
        self.max_move_cars = max_move_cars
        self.rental_credit = rental_credit
        self.move_car_cost = move_car_cost
        self.request_lambda = tuple(request_lambda)
        self.return_lambda = tuple(return_lambda)
        for rates in (self.request_lambda, self.return_lambda):
            if len(rates) != 2 or any(not np.isfinite(rate) or rate < 0 for rate in rates):
                raise ValueError("Poisson rates must contain two finite nonnegative values")

        self._probabilities = []
        self._income_moments = []
        self._expected_income = []
        for request_rate, return_rate in zip(self.request_lambda, self.return_lambda):
            probabilities, income_moments, expected_income = self._location_tables(
                request_rate, return_rate
            )
            for table in (probabilities, income_moments, expected_income):
                table.flags.writeable = False
            self._probabilities.append(probabilities)
            self._income_moments.append(income_moments)
            self._expected_income.append(expected_income)

    def _location_tables(self, request_rate, return_rate):
        capacity = self.max_cars
        probabilities = np.zeros((capacity + 1, capacity + 1))
        income_moments = np.zeros_like(probabilities)
        expected_income = np.zeros(capacity + 1)
        returns = [_capped_poisson(return_rate, free) for free in range(capacity + 1)]
        for inventory in range(capacity + 1):
            requests = _capped_poisson(request_rate, inventory)
            expected_income[inventory] = (
                np.dot(np.arange(inventory + 1), requests) * self.rental_credit
            )
            for rented, request_probability in enumerate(requests):
                remaining = inventory - rented
                branch = request_probability * returns[capacity - remaining]
                probabilities[inventory, remaining:] += branch
                income_moments[inventory, remaining:] += branch * rented * self.rental_credit
        return probabilities, income_moments, expected_income

    def __len__(self):
        return (self.max_cars + 1) ** 2 * (2 * self.max_move_cars + 1)

    def __iter__(self):
        for cars_a in range(self.max_cars + 1):
            for cars_b in range(self.max_cars + 1):
                for action in range(2 * self.max_move_cars + 1):
                    yield (cars_a, cars_b), action

    def _after_transfer(self, state, action):
        if not (
            isinstance(state, tuple) and len(state) == 2
            and all(isinstance(cars, (int, np.integer)) and 0 <= cars <= self.max_cars
                    for cars in state)
            and isinstance(action, (int, np.integer))
            and 0 <= action <= 2 * self.max_move_cars
        ):
            raise KeyError((state, action))
        return transfer_cars(
            state, action - self.max_move_cars, self.max_cars,
            self.max_move_cars, self.move_car_cost,
        )

    def expected_reward(self, state, action):
        """Expected daily rental income minus the actual transfer cost."""
        (cars_a, cars_b), cost = self._after_transfer(state, action)
        return float(self._expected_income[0][cars_a] + self._expected_income[1][cars_b] - cost)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        (cars_a, cars_b), cost = self._after_transfer(*key)
        probabilities_a = self._probabilities[0][cars_a]
        probabilities_b = self._probabilities[1][cars_b]
        outcomes = []
        for next_a in np.flatnonzero(probabilities_a):
            income_a = self._income_moments[0][cars_a, next_a] / probabilities_a[next_a]
            for next_b in np.flatnonzero(probabilities_b):
                probability = probabilities_a[next_a] * probabilities_b[next_b]
                if probability == 0:  # A product can underflow for extremely rare events.
                    continue
                income_b = self._income_moments[1][cars_b, next_b] / probabilities_b[next_b]
                outcomes.append((
                    (int(next_a), int(next_b)), float(income_a + income_b - cost),
                    False, float(probability),
                ))
        return tuple(outcomes)
