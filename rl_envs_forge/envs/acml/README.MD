# ACML environments

The Adaptive Computation and Machine Learning (ACML) environments are toy environments proposed in Reinforcement Learning: An Introduction* (2nd ed.).

## Implemented Environments
- JacksCarRentalEnv: Jack's Car Rental problem
- GamblersProblem: Gambler's problem

## Usage

### Jack's Car Rental
Code example for getting started with the environment:

```python
from rl_envs_forge.envs.acml.car_rental.car_rental_env import JacksCarRental

env = JacksCarRental(
    max_cars=10,
    max_move_cars=3,
    request_lambda=[2, 2],
    return_lambda=[2, 2],
    init_state_option="equal",
)
env.state
```

```output
(5, 5)
```

```python
env.step(2)
```

```output
((5, 3), 48, False, False, {})
```

### Gambler's problem
Code example for getting started with the environment:

```python
from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem

env = GamblersProblem(
    goal_amount=100, 
    win_probability=0.40, 
    start_capital=10
)
env.state
```

```output
10
```

```python
env.step(5)
```

```output
(15, 0, False, False, {})
```

## Using DP algorithms with these environments
Usage with policy iteration: [Algorithm Repository](https://github.com/mariusdgm/phd-rl-algos)

## UML diagrams

### Packages

<img src="../../../docs/diagrams/acml/packages_acml.png" alt="Packages UML" width="300">

### Classes

<img src="../../../docs/diagrams/acml/classes_acml.png" alt="Classes UML" width="300">

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (Adaptive Computation and Machine Learning series).