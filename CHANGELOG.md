# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Proposed next version: `6.0.0rc1`, followed by `6.0.0` after validation. The
existing `v5.23.0` tag must not be reused. Version metadata remains `5.23.0`
until the release version is approved. A major version acknowledges the
intentional observation-space, validation, and stochastic-model changes below.

### Changed

- Declare Python 3.10–3.12 support, matching the NumPy 1.x dependency range. Python 3.13 is excluded pending a separately validated numerical-stack update.
- Declare Matplotlib directly and require versions below 3.10, which removed the RGB canvas method used by the existing GridWorld renderer.
- Install Sphinx and its theme through the optional `docs` extra, and expose a `test` extra. Remove unused Python Graphviz, TOML, and external Napoleon dependencies; the diagram regeneration script still needs the external Graphviz application.
- Replace automatic tagging and publishing with validation-only CI for Python 3.10, 3.11, and 3.12 on Linux and Windows, plus documentation, citation, source-archive, and isolated-install checks.
- NetworkGraph FJ observations declare a nonnegative, unbounded space. Propagated states require finite real values rather than an arbitrary endpoint tolerance, while initial/reset opinions stay in `[0,1]`. The raw FJ recurrence and coefficients are preserved; numerical overflow raises instead of returning a nonfinite state.
- GridWorld rejects out-of-grid constructor/reset starts, and GamblersProblem rejects starting capital outside its discrete observation space. Rejected resets preserve episode state and RNG state.
- NetworkGraph validates Friedkin-Johnsen adjacency, lambda, prejudice, and bounded control/initial-state parameters without normalizing matrices or clipping invalid lambda values. Previously accepted configurations outside these assumptions now raise `ValueError`; the FJ formula and valid trajectories are preserved.
- CartPole and PendulumDisk retain `terminated=True` when physical failure coincides with their time limit; both ending flags can now be true.
- Jack's Car Rental clips transfers by source inventory and destination capacity, conserves cars, and charges only for actual movement. Seeded results for fully feasible transfers retain the existing request/return draw order; clipped transfers can change inventory and rewards.
- Jack's Car Rental's legacy `build_mdp()` preserves signed keys and representative-state tuples, with corrected full expected rewards. `poisson_max_value` no longer limits either planning API.
- GridWorld interprets `slip_distribution` as a normalized global directional distribution, conditioned on excluding the intended action after failure. Invalid, empty, and single-direction distributions now raise `ValueError`.
- GridWorld includes terminal and wall self-loops in every planning representation. Custom slips and special transitions can produce different seeded stochastic trajectories; ordinary deterministic movement and default uniform slip ordering are preserved.

### Fixed

- Labyrinth full observations use the declared uint8 dtype and supplied-layout dimensions, and custom reward schemas are assigned correctly.
- KArmedBandit's observation space includes its returned arm indices. NetworkGraph's observation dtype matches its float64 dynamics, and integer adjacency inputs retain sampled fractional weights.
- GridWorld distinguishes configured wall-collision rewards from boundary-collision rewards.
- Strict Sphinx documentation builds resolve inherited physical-step docstrings, the Barabasi-Albert list indentation, and ambiguous Labyrinth action references without changing simulation behavior.
- Supplied Labyrinth layouts retain manual initialization, report missing positions clearly, and avoid applying unset markers to the entire full observation.
- PendulumDisk declares unbounded angular velocity observations, including failure-step overshoot, without clipping or changing physical dynamics.
- NetworkGraph returns independent reset/step observations and accepts float32 FJ row-sum roundoff within four machine epsilons. Float64 adjacency validation and all existing dynamics equations are unchanged.
- Maze and room generation use integer random bounds for Python versions that reject integral floats, preserving the random draws and their order.
- The package-data rule names the actual Labyrinth sprite directory for both wheels and source distributions.
- NetworkGraph diagnostics distinguish graph connectivity criteria, identify the actual Barabasi-Albert retry filter and fallback, and no longer advertise the unimplemented `erdos_renyi` graph alias.
- Jack's Car Rental aggregates saturated Poisson request/return tails, including zero rates, and builds the default legacy model without a support-index error.
- GridWorld uses the same reward-bearing outcomes for simulation, `mdp`, `transition_probs`, and `P`. Special transitions execute only on intended-action success; slips use ordinary movement. Adding or replacing a special transition refreshes every model without consuming random draws.

### Added

- Include tests, test configuration, the dependency lock, documentation sources, citation metadata, and CI configuration in source distributions. Add an isolated artifact validation command and a repository-based `CITATION.cff` without an invented archival identifier.
- NetworkGraph offers opt-in `strict_connectivity` and an additional `connectivity_requirement`, preserving default generation distributions, retry behavior, and random draw order.
- GamblersProblem exposes state-feasible stakes through `feasible_actions()` and `action_mask()`, with optional `strict_actions` validation. Its default action space and permissive behavior remain unchanged.
- Jack's Car Rental `build_transition_model()` returns a lazy snapshot of complete transition probabilities and conditional mean rewards, keyed by Gymnasium action indices, without allocating a full transition tensor.

## [5.23.0] - 2026.10.20

### Added

- Support for `seed` and `options` in every environment's reset method, preserving environment-specific reset arguments.
- Regression tests for reproducible resets, independent random streams, and NetworkGraph behavior across graph types and dynamics models.

### Changed

- KArmedBandit uses NumPy's Generator instead of RandomState, so fixed-seed reward sequences differ from earlier versions. Unseeded resets continue the random stream.
- Reworked the package READMEs with clearer usage examples, reset guidance, and descriptions of actions, rewards, and episode endings.
- Added a NetworkGraph trajectory plot comparing uniform and near-optimal control.

### Fixed

- Pendulum initial states, car-rental requests and returns, and gambler outcomes use each environment's own random generator.
- Labyrinth accepts `seed=0`, seeds predetermined layouts consistently, and copies its random generator independently when deep-copied.
- KArmedBandit resets its timestep along with its arms, restarting parameter schedules.
- GridWorld's transition helper samples from the existing MDP instead of referencing undefined attributes and a matrix.

## [5.22.0] - 2026.04.08

### Added
- Added new propagation methods for opinion dynamics in NetworkGraph

## [5.21.0] - 2026.03.12

### Changed
- More efficient coca and laplacian dynamics

## [5.20.1] - 2026.03.02

### Changed
- Albert-Barabasi graph gen

## [5.20.0] - 2026.02.23

### Added
- New graph generation method


## [5.19.2] - 2026.02.07

### Changed
- Viz params

## [5.19.1] - 2026.01.27

### Changed
- Even better graph visualization

## [5.19.0] - 2026.01.23

### Changed
- Better graph visualization

## [5.18.3] - 2025.10.24

### Changed
- Fixed bug in NetworkGraph related to which step was used for the calcualtion of the reward

## [5.18.2] - 2025.08.21

### Changed

- Axis label for Network Render

## [5.18.1] - 2025.08.14

### Changed

- Constructor documentation for NetworkGraph

## [5.18.0] - 2025.07.05

### Changed

- Changed how the dynamics are being integrated

## [5.17.1] - 2025.07.04

### Fixed

- Changed to using vector for desired opinions

## [5.17.0] - 2025.06.19

### Added

- Added nonlinear dynamics in NetworkGraph

### Fixed

- Arrow direction for graph viz

## [5.16.0] - 2025.06.02

### Changed

- Added parameter to disable terminal state in NetworkGraph

## [5.15.1] - 2025.05.12

### Fixed

- Graph symmetry fix in NetworkGraph

### Changed

- Allow step time 0 in NetworkGraph

### Changed

- Changed graphs in NetworkGraph to probabilistic directed graphs

## [5.15.0] - 2025.05.08

### Changed

- Changed graphs in NetworkGraph to probabilistic directed graphs


## [5.14.0] - 2025.05.06

### Added
- Added seed support to NetworkGraph for reproducible random graph generation
- Added random reset option

## [5.13.0] - 2025.04.29

### Fixed

- Changed reward normalization

## [5.12.1] - 2025.04.28

### Fixed

- Fixed reward in NetworkGraph
- Increased nr of eperiments in test_levy_flight to reduce chance of wrong condition

## [5.12.0] - 2025.04.25

### Added

- Support for node-wise max_u (individual action limits per agent).
- Option to remove budget constraint (allow budget=None).
- New connection distributions for graph generation: "uniform", "normal", and "exponential".
- Automatic connection fix to ensure no isolated nodes (every node has at least one neighbor).

### Changed

- Step behavior: reward now computed on the original action before saturation (clip only before dynamics).
- Improved info dictionary: excludes "remaining_budget" if no budget is defined.

## [5.11.0] - 2025.03.25

### Added

- Terminal reward in network env

## [5.10.1] - 2025.03.24

### Fixed

- Normalized reward for network env

## [5.10.0] - 2025.03.23

### Added

- Normalized reward for network env

## [5.9.0] - 2025.03.03

### Added

- Network Env state attribute

## [5.8.0] - 2025.02.07

### Changed

- Network Env termination based on average opinion

## [5.7.5] - 2025.01.15

### Fixed

- Changed version reading with metadata

## [5.7.2] - 2025.01.15

### Fixed

- Attempt to change build process

## [5.7.1] - 2025.01.15

### Fixed

- Toml file not getting packaged

## [5.7.0] - 2025.01.15

### Added

- Info in opinion network reset

## [5.6.0] - 2024.12.19

### Changed

- Reward function and control resistance in NetworkGraph

## [5.5.0] - 2024.11.04

### Added

- Added compute_dynamics function in NetworkGraph for external state computation.

## [5.4.0] - 2024.09.30

### Changed

- Changed step function of NetworkGraph to have impulse action dynamics

## [5.3.0] - 2024.09.20

### Changed

- Changed step function of NetworkGraph so we can simulate long campaigns

## [5.2.1] - 2024.09.09

### Fixed

- Fixed bug in NetworkGraph step

## [5.2.0] - 2024.09.08

### Changed

- Brought NetworkGraph implementation in line with paper dynamics
- Fixed centrality calculation in NetworkGraph

## [5.1.0] - 2024.08.22

### Added

- Saturation elements in NetworkGraph

## [5.0.0] - 2024.08.18

### Added

- New environment: NetworkGraph
- Matplotlib render mode for inverted pendulum envs

### Changed

- Default value for reward_decay_rate in InvertedPendulum envs, also removed internal attribute

### Fixed

- Main README fig paths for proper display on pypi page

## [4.0.1] - 2024.07.30

### Fixed

- Added needed seed and options to the reset methods in InvertedPendulum envs

## [4.0.0] - 2024.07.28

### Added

- New environment: Inverted Pendulum - CartPole
- New environment: Inverted Pendulum - PendulumDisk

### Fixed

- ACML readme path

## [3.8.0] - 2024.07.09

### Changed

- Added terminal states (and transitions) in the GridWorld and Labyrinth MDPs

## [3.7.0] - 2024.07.01

### Added

- New render mode for returning the render frames

## [3.6.2] - 2024.06.28

### Fixed

- Decopled truncated from done signal in GridWorld

## [3.6.1] - 2024.06.20

### Changed

- Returning the truncated signal when episode reached time limit

## [3.6.0] - 2024.06.20

### Added

- Added option to reset gridworld env to a different starting state

## [3.5.2] - 2024.06.14

### Added

- Allowing empty iterators in GridWorld where the default None could be allowed

## [3.5.1] - 2024.06.14

### Added

- Initializer type checks in GridWorld

## [3.5.0] - 2024.06.13

### Added

- Early episode stop in GridWorld

## [3.4.2] - 2024.03.17

### Fixed

- Bugfixed the terminal state when reaching that transition


## [3.4.1] - 2024.03.16

### Fixed

- Bugfixed the state reset in grid_world when reaching a terminal state

## [3.4.0] - 2024.02.09

### Changed

- Remade grid_world environment to allow for probabilistic transitions, and made the MDP built into it

## [3.3.3] - 2024.02.06

### Fixed

- Fixed random generator seeding in grid_world


## [3.3.2] - 2024.02.04

### Changed

- Updated diagrams for grid_world

## [3.3.1] - 2024.02.03

### Added

- Added a random_move_frequency parameter for grid_world

## [3.3.0] - 2023.12.11

### Added

- Added a new environment: Gambler's Problem

### Fixed

- ACML readme filename, and other info in the README

## [3.2.1] - 2023.11.30

### Fixed

- ACML related documentation

## [3.2.0] - 2023.11.30

### Added

- Added a new environment: JacksCarRental

## [3.1.2] - 2023.11.25

### Fixed

- Removed manifest file as we are using poetry...

## [3.1.1] - 2023.11.25

### Fixed

- Added manifest file to include the pyproject toml file in the package, as it is used for reading the __version__

## [3.1.0] - 2023.11.24

### Added

- added mdp builder to GridWorld

### Changed

- Changed the python version dependency

## [3.0.0] - 2023.11.19

### Added

- 2D GridWorld environment 

## [2.1.0] - 2023.10.24

### Added

- Labyrinth: Vision range feature 

## [2.0.0] - 2023.10.22

### Fixed

- Fixed a behavior of the shifting parameters in the Arms of the Bandit
- Properly incremented the major release digit as we have a new environment now  

## [1.5.0] - 2023.10.20

### Added

- KArmedBandit environment 
- Added 'close' method and property 'state' to Labyrinth

### Changed

- In Labyrinth: made the reset method to return the current state
- Changed the diagram script to dynamically make separate diagrams for each env

### Fixed

- Diagram script fixed to skip folders it should not parse

## [1.4.1] - 2023.10.13

### Fixed

- Packaged the sprites with the module and changed how the paths are read.
  
## [1.4.0] - 2023.10.13

### Changed

- Added the 'done' signal in the mdp transition graph
  
## [1.3.0] - 2023.10.08

### Added

- Added pseudo MDP builder that uses a DFS traversal of the environment
- Deepcopy procedure and state setter for Labyrinth

### Fixed

- Displayer instantiation behavior (relevant when render or human play are stopped)

## [1.2.2] - 2023.09.22

### Fixed

- LFS tracked file moved to normal git so they can be displayed in the readmes

## [1.2.1] - 2023.09.22

### Added

- Added script for UML generation with pyreverse

### Changed

- Improved documentation doctrings
- Added sphinx auto documentation

### Fixed

- Added one more draw call in render when the animation loop ends

## [1.2.0] - 2023.09.18

### Added

- Donut room
- T room
- L room
- Triangle room

### Changed

- Generalized some base Room methods

## [1.1.1] - 2023.09.14

### Added

- Oval room

### Fixed

- PriorityQueue usage in both astar and gbfs. In gbfs the bug was much more noticeable as the corridors were not properly built.
- Bug in cost matrix computation

### Changed

- Workflow badge should now show status on current branch

## [1.0.1] - 2023.09.13

### Added

- Badges in readme
- Diagram with pyreverse

### Fixed

- Readme Github repo link in badge
- Readme image table format

## [1.0.0] - 2023.09.12

### Added

- A* corridor generator algorithm, to generate more direct corridors.
- GBFS, a non optimal A* corridor generator that is faster.

### Changed

- Refactored corridor code from maze into a new CorridorBuilder class.
- Changes in the labyrinth interface.

### Fixed

- Changed readme and img paths to urls so they can get displayed properly on the pypi page too.
- Adjusted default pos close threshold and fixed info print in human_play.

## [0.4.0] - 2023.09.06

### Added

- Refined project readme and several labyrinth module level readmes.
- Reorganized the human_play code a bit, and changed so that the behavior is consistent for normal render too.
- Removed animate logic from displayer
- Added unit test for human_play and for exit events

### Fixed

- Repaired exit behavior when using render.

## [0.3.0] - 2023.09.05

### Added

- Added grid lines.
- Added maze padding.
- Added screen resize functionality.
- Added sprites.
- Added mini animation.

## [0.2.1] - 2023.09.03

### Fixed

- Bugfix of human_play.

## [0.2.0] - 2023.09.03

### Added

- Added new corridor generation setting.
- Added postprocessing step for the maze corridors.

## [0.1.0] - 2023.09.01

### Added

- Initial release of rl_envs_forge - Labyrinth.
- Labyrinth - a 2D gridworld with randomly generated rectangular rooms, maze like paths, a starting position and a target position.
