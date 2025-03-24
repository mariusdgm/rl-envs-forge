# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [5.10.1] - 2025.03.23

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
