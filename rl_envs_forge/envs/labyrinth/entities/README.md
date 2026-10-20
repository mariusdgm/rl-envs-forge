# Labyrinth entities

`Player` holds the agent's position and the state used to animate its sprite.
Positions are `(row, column)` pairs. Movement rules and collision checks belong
to the [Labyrinth environment](../README.md).

## Contents

- [Working with a player](#working-with-a-player)

## Working with a player

```python
from rl_envs_forge.envs.labyrinth.constants import Action
from rl_envs_forge.envs.labyrinth.entities.player import Player

player = Player(start_position=(2, 3))
next_position = player.potential_next_position(Action.RIGHT)
assert next_position == (2, 4)

player.set_position(next_position)
```

`potential_next_position()` calculates a neighboring cell without moving the
player or checking walls and bounds. `set_position()` updates both the logical
position and the rendered position.

During animation, `position` is the actual grid cell and `rendered_position`
interpolates toward it through `move_render_position()`. `movement_speed`
controls that interpolation; `moving` and `face_orientation` choose the sprite.

In an environment, use `env.step(action)` for normal movement or
`env.set_state(position)` to place the player on a valid non-target path cell.
Accessing `player.displayer` creates a `PlayerDisplayer` when needed; ordinary
movement does not require opening a display.
