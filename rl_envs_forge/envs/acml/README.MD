# ACML environments

This package contains Jack's Car Rental and the Gambler's Problem from
Sutton and Barto's *Reinforcement Learning: An Introduction* (2nd edition),
published in the Adaptive Computation and Machine Learning series.

Both environments support sampling with `step()` and provide `build_mdp()`
helpers for inspecting the problem structure.

## Jack's Car Rental

Move cars between two rental locations before the day's rental requests and
returns arrive. The reward is rental income minus the cost of moving cars.
Requests and returns follow Poisson distributions, with a separate rate for
each location.

```python
from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental

env = JacksCarRental(
    max_cars=10,
    max_move_cars=3,
    request_lambda=[2, 2],
    return_lambda=[2, 2],
)
state, info = env.reset(seed=0, options={"option": "equal"})
assert state == (5, 5)

# Action indices encode a signed number of cars to move.
cars_to_move = 2
action = cars_to_move + env.max_move_cars
state, reward, terminated, truncated, info = env.step(action)
```

The state is `(cars_at_A, cars_at_B)`. Action indices run from zero to
`2 * max_move_cars`. Subtract `max_move_cars` to get the signed transfer:
positive moves cars from A to B, negative from B to A, and zero leaves them
in place. With the settings above, action `3` means no transfer.

`reset(option="equal")` splits the initial inventory evenly using
`max_cars // 2` at each location. `reset(option="random")`, the default,
samples each location's inventory. You can pass the same choice through
`options`, as in the example. A seed controls both inventory sampling and
subsequent requests and returns.

This is a continuing task: `step()` returns both ending flags as `False`.
Choose a horizon in your experiment. `rental_credit` and `move_car_cost`
control the reward. `poisson_max_value` sets how many Poisson probabilities
are precomputed for the planning helper.

## Gambler's Problem

Choose a stake and try to reach `goal_amount` before losing all capital.
The observation is the current capital, and the action is the stake.

```python
from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem

env = GamblersProblem(goal_amount=100, win_probability=0.4, start_capital=10)
capital, info = env.reset(seed=0)

for _ in range(1000):
    stake = min(capital, env.goal_amount - capital)
    capital, reward, terminated, truncated, info = env.step(stake)
    if terminated or truncated:
        break
```

Reaching the goal pays one; all other outcomes pay zero. The episode terminates
at zero capital or the goal. Keep stakes between one and
`min(capital, goal_amount - capital)` for the textbook problem: the declared
action space is broader and does not enforce this state-dependent bound.

`reset()` returns to `start_capital`. Pass a seed to repeat the sequence of
bet outcomes.

## Planning and inspection

The Gambler's Problem provides win and loss outcomes with probabilities for
each valid state and stake. Jack's Car Rental uses signed transfers as keys
and aggregates expected rewards, retaining one representative next state per
pair. Planning methods that need a full transition distribution will need a
fuller car-rental model. Set `poisson_max_value` to at least `max_cars` when
using that helper, which enumerates request and return counts up to `max_cars`.

Use `env.render()` to print the state, or `env.render(mode="ansi")` to get a
string. The examples in the [algorithm repository](https://github.com/mariusdgm/phd-rl-algos)
show how these environments can be used with planning algorithms.

## Diagrams

![Package diagram](../../../docs/diagrams/acml/packages_acml.png)
![Class diagram](../../../docs/diagrams/acml/classes_acml.png)
