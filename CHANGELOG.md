# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Bugfix
 - Changed readme and img paths to urls so they can get work properly on the pypi page too 

## [0.4.0] - 2023.09.06
### Added 
 - Refined project readme and several labyrinth module level readmes.
 - Reorganized the human_play code a bit, and changed so that the behavior is consistent for normal render too.
 - Removed animate logic from displayer
 - Added unit test for human_play and for exit events

### Bugfix
 - Repaired exit behavior when using render.


## [0.3.0] - 2023.09.05
### Added
 - Added grid lines.
 - Added maze padding.
 - Added screen resize functionality.
 - Added sprites.
 - Added mini animation.

## [0.2.1] - 2023.09.03
### Bugfix
 - Bugfix of human_play.
 
## [0.2.0] - 2023.09.03
### Added
 - Added new corridor generation setting.
 - Added postprocessing step for the maze corridors.

## [0.1.0] - 2023.09.01
### Added
- Initial release of rl_envs_forge - Labyrinth.
- Labyrinth - a 2D gridworld with randomly generated rectangular rooms, maze like paths, a starting position and a target position.