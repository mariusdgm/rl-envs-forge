# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
