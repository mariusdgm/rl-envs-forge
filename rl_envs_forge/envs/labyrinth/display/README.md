# Labyrinth display

The display package draws Labyrinth with Pygame. The environment creates its
`EnvDisplay` when you first render, so you can run simulations without a window
by leaving out rendering calls.

## Contents

- [Rendering through the environment](#rendering-through-the-environment)
- [Components](#components)

## Rendering through the environment

```python
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth

env = Labyrinth(20, 20, seed=0)
quit_requested, action = env.render(window_size=(800, 800), animate=False)
env.close()
```

`render()` returns a quit flag and an optional arrow-key action. Set
`process_arrow_keys=True` to read arrow keys, or use `env.human_play()` to run
the interactive movement loop. `animate=True` interpolates the player's sprite
between cells; `sleep_time` sets the delay between animation frames.

## Components

`EnvDisplay.draw_state()` draws the maze, target flag, grid, and player, then
updates the window. It also scales cells and sprites when the window is resized.

`EntityDisplayer` defines the `get_sprite()` interface. `PlayerDisplayer` loads
the player sprite sheet, selects an idle or moving frame, and flips it to match
the player's facing direction. The target and player images are stored in
`sprites/`.

To change the appearance, start with `EnvDisplay` for colors and layout or
`PlayerDisplayer` for sprite selection. See the [entities README](../entities/README.md)
for the position and animation state used by the renderer.
