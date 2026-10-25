# rl-envs-forge

[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github)](https://github.com/mariusdgm/rl-envs-forge)
[![PyPI version](https://img.shields.io/pypi/v/rl-envs-forge.svg)](https://pypi.org/project/rl-envs-forge/)
![License: MIT](https://img.shields.io/github/license/mariusdgm/rl-envs-forge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A collection of small, configurable environments for reinforcement learning.
Use them to try learning and planning algorithms, compare control policies, or
study how changes to an environment affect a problem.

## Table of contents

- [Installation](#installation)
- [Environments](#environments)
  - [Labyrinth](#labyrinth)
  - [KArmedBandit](#karmedbandit)
  - [GridWorld](#gridworld)
  - [ACML](#acml)
  - [Inverted pendulums](#inverted-pendulums)
  - [NetworkGraph](#networkgraph)
- [Usage](#usage)
  - [Repeating a run](#repeating-a-run)
- [Tests](#tests)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

Supports Python 3.10, 3.11, and 3.12. Python 3.13 requires a future dependency
update; the current numerical stack uses NumPy 1.x.

```bash
pip install rl-envs-forge
```

For development from a checkout, use `poetry install` and run commands with
`poetry run`. Alternatively, use pip:

```bash
python -m pip install ".[test]"
```

The `docs` extra installs the Sphinx documentation tools.

## Environments

### Labyrinth

Navigate mazes built from rooms and corridors. Choose the maze generator,
room shapes, and how much of the maze the agent can see.

[Labyrinth usage and settings](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/labyrinth/README.md)

![Labyrinth with a random policy](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/labyrinth/auto_play_demo.gif)

### KArmedBandit

Sample rewards from configurable arm distributions. Parameters can shift over
time to model a nonstationary bandit.

[KArmedBandit usage and settings](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/k_armed_bandit/README.md)

![Bandit reward distributions](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/k_armed_bandit/different_distributions.png)

### GridWorld

Move through a grid with walls, terminal cells, and special transitions.
Movement can be deterministic or include slippage.

[GridWorld usage and settings](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/grid_world/README.md)

![GridWorld](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/grid_world/default.png)

### ACML

Jack's Car Rental and the Gambler's Problem from Sutton and Barto's
*Reinforcement Learning: An Introduction* (2nd edition).

[ACML usage and settings](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/acml/README.MD)

### Inverted pendulums

Balance a pole on a cart or control a weighted disk. Both environments accept
continuous actions and offer configurable starting states and rewards.

[Inverted pendulum usage and settings](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/inverted_pendulum/README.md)

<table>
<tr><th>CartPole</th><th>PendulumDisk</th></tr>
<tr>
<td><img src="https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/inverted_pendulum/cart_pole/render.png" alt="CartPole" width="300"></td>
<td><img src="https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/inverted_pendulum/pendulum_disk/render.png" alt="PendulumDisk" width="300"></td>
</tr>
</table>

### NetworkGraph

Influence opinions in a network using per-node controls. Choose the network
structure and opinion dynamics, then compare progress toward a target against
the cost of control.

[NetworkGraph usage and settings](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/network_graph/README.md)

![NetworkGraph](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/network_graph/example.png)

## Usage

Import an environment class directly, reset it, and pass actions to `step()`:

```python
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth

env = Labyrinth(20, 20, seed=0)
observation, info = env.reset(seed=0)
env.action_space.seed(0)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

The package READMEs describe each environment's actions, observations, episode
endings, and rendering options.

### Repeating a run

Pass a seed to `reset()` when you want to repeat an episode. Calling `reset()`
without a seed continues the environment's random stream. Seed the action
space separately, as above, if your policy uses `action_space.sample()`.

Resets also depend on the task. NetworkGraph keeps its graph and resets the
opinions. Labyrinth generates another maze unless you pass `same_seed=True`.
KArmedBandit rebuilds its arms and starts its parameter schedules from zero.
See each environment's README for fixed starting states and other reset options.

## Tests

From a development checkout:

```bash
poetry run pytest tests
```

Use `poetry run pytest tests -m "not slow"` for a shorter run.
[Testing instructions](tests/README.md) cover running individual groups.

## Documentation

The linked package READMEs contain usage examples and configuration details.
To build the API documentation locally, install `poetry install --extras docs`
and follow the [documentation instructions](docs/README.md).

## Citation

If you use this software in research, the author and repository information are
available in [CITATION.cff](CITATION.cff). No archived release or DOI is currently
specified there.

## License

[MIT](LICENSE).

## Contact

For questions or contributions, open an issue on
[GitHub](https://github.com/mariusdgm/rl-envs-forge) or contact
[marius.dragomir.dgm@gmail.com](mailto:marius.dragomir.dgm@gmail.com).
