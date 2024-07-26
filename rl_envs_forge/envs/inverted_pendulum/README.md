# Inverted pendulum environments 

The inverted pendulum problem is a classic challenge in control theory and reinforcement learning, modeling the task of balancing a pendulum in an unstable equilibrium. Imagine a pendulum attached to a pivot point that can move horizontally.

The objective is to apply forces to maintain the pendulum upright despite disturbances and the natural tendency to fall. This problem exemplifies the need for precise control and is fundamental in illustrating various control strategies.

The integration step of the environments is adjustable as presented in [Usage](#usage).

## Usage

# TODO

Code example for getting started with the environment:

```python
from rl_envs_forge.envs.k_armed_bandit.k_armed_bandit import KArmedBandit

bandit = KArmedBandit(k=10, seed=0)
bandit.step(1)
```

```output
(1, 0.5442007795281013, True, False, {})
```

```python
bandit.render()
```

![KArmedBandit render default](../../../docs/figures/k_armed_bandit/default.png)


```python
bandit.render(mode="print")
```

```output
Arm 0:
	Distribution: Normal
	Mean: 1.57
	Std: 1.00
------------------------------
Arm 1:
	Distribution: Normal
	Mean: -0.95
	Std: 1.00
------------------------------
Arm 2:
	Distribution: Normal
	Mean: -1.59
	Std: 1.00

etc...
```



## Customizing an arm
You can customize the distribution of an arm by passing a dictionary that describes the distribution.


```python
custom_bandit = KArmedBandit(k=10, arm_params={1: {"distribution": "normal", "mean": 5, "std": 1}})
custom_bandit.render()
```

![KArmedBandit render single_custom](../../../docs/figures/k_armed_bandit/single_custom.png)


## Shifting parameters
You can define how a parameter changes during sampling for an arm:

```python
def linear_increase(timestep):
    return 0.1 * timestep  # Increase by 0.1 every timestep


def linear_decrease(timestep):
    return - 0.01 * timestep  # Decrease by 0.01 every timestep, starting from 1


arm_params = {
    0: {
        "distribution": "normal",
        "mean": 0,
        "std": 1,
        "param_functions": [
            {"function": linear_increase, "target_param": "mean"},
            {"function": linear_decrease, "target_param": "std"},
        ],
    }
}

custom_bandit = KArmedBandit(k=2, arm_params=arm_params)
custom_bandit.render()

for timestep in range(20):
    custom_bandit.step(1)
    
custom_bandit.render()
```

<table>
<tr>
    <th colspan="1">Initial distributions</th>
    <th colspan="1">After extracting 10 samples</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/k_armed_bandit/changing_init.png" alt="changing_init 1" width="300">
</td>
<td>
<img src="../../../docs/figures/k_armed_bandit/changing_after.png" alt="changing_after 2" width="300">
</td>
</tr>

</table>

```

![KArmedBandit render different_distributions](../../../docs/figures/k_armed_bandit/different_distributions.png)

## UML diagrams

### Packages

<img src="../../../docs/diagrams/k_armed_bandit/packages_k_armed_bandit.png" alt="Packages UML" width="300">

### Classes

<img src="../../../docs/diagrams/k_armed_bandit/classes_k_armed_bandit.png" alt="Classes UML" width="300">
