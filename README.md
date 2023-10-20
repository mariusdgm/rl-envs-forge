# rl-envs-forge
<!-- Badges -->
[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github)](https://github.com/mariusdgm/rl-envs-forge)
[![PyPI version](https://img.shields.io/pypi/v/rl-envs-forge.svg)](https://pypi.org/project/rl-envs-forge/)
![License: MIT](https://img.shields.io/github/license/mariusdgm/rl-envs-forge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!--  -->

Lightweight environments for reinforcement learning applications.

## Table of Contents

- [rl-envs-forge](#rl-envs-forge)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Environments](#environments)
    - [Labyrinth](#labyrinth)
      - [Labyrinth rendered example](#labyrinth-rendered-example)
    - [KArmedBandit](#karmedbandit)
      - [KArmedBandit rendered example](#karmedbandit-rendered-example)
  - [Usage](#usage)
  - [Tests](#tests)
  - [License](#license)
  - [Contact \& Support](#contact--support)

## Installation

```bash
pip install rl-envs-forge
```

## Environments

### Labyrinth

Labyrinth is a classic maze-solving environment, where the goal is to navigate from a start point to a target. The maze layout is randomly generated based on a set of parametrizable arguments.

<!-- Use github paths for these figures so they will show up in the pypi page -->
📖 **Detailed Documentation**: [Click here to read more about the Labyrinth environment](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/labyrinth/README.md)

#### Labyrinth rendered example

![Labyrinth render GIF](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/labyrinth/auto_play_demo.gif)


### KArmedBandit

KArmedBandit is a bandit environment, which returns a reward from a distribution associated with the chosen arm at each timestep. This implementation includes multiple distributions, and the possibility to shift the distribution parameters during sampling.

📖 **Detailed Documentation**: [Click here to read more about the KArmedBandit environment](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/k_armed_bandit/README.md)

#### KArmedBandit rendered example

![KArmedBandit render](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/docs/figures/k_armed_bandit/different_distributions.png)

## Usage

Example code on setting up and testing the Labyrinth environment.

Note, this code snippet produced the render visible in section [Labyrinth](#labyrinth)

```python
from time import sleep
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth

env = Labyrinth(20, 20, seed=0)

done = False
quit_event = False
while not done and not quit_event:
    action = env.action_space.sample()  
    observation, reward, done, _, info = env.step(action)
    quit_event, _ = env.render()
    sleep(0.1)
```

## Tests

Requirements: pytest and pytest-cov

Run the tests in the root folder with:

```bash
pytest tests
```

## License

This project is licensed under the [MIT License](./LICENSE).

## Contact & Support

For any queries or support, or if you would like to contribute to this project, reach out at [marius.dragomir.dgm@gmail.com](mailto:marius.dragomir.dgm@gmail.com) or raise an issue on our GitHub repository.
