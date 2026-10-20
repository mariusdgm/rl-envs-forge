# GridWorld

GridWorld is a grid navigation environment with walls, terminal cells, and
optional jumps between cells. You can use deterministic movement or add
slippage to make actions uncertain.

## Getting started

```python
from rl_envs_forge.envs.grid_world.grid_world import Action, GridWorld

env = GridWorld(
    rows=6,
    cols=5,
    walls={(5, 0), (5, 1), (5, 3), (5, 4)},
    start_state=(1, 1),
    terminal_states={(0, 0): 1.0, (4, 4): 1.0},
)
state, info = env.reset(seed=0)
state, reward, terminated, truncated, info = env.step(Action.RIGHT)
env.render()
```

States are `(row, column)` tuples, starting at `(0, 0)` in the upper left.
Actions are `Action.UP` (`0`), `RIGHT` (`1`), `DOWN` (`2`), and `LEFT` (`3`).

![GridWorld with walls and terminal cells](../../../docs/figures/grid_world/default.png)

## Movement and rewards

`p_success=1.0` gives deterministic movement. With a smaller value, the intended
action succeeds with probability `p_success`; the remaining probability is
split equally among the other three directions in `step()`.

Moving into a wall or beyond the grid leaves the agent in place. Terminal
cells end the episode and pay the reward in `terminal_states`. Default rewards
are `-0.1` for a regular move and `-1` for hitting a wall or boundary. Pass a
`rewards` dictionary to choose your own values using the keys `valid_move`,
`wall_collision`, `out_of_bounds`, and `default`.

Set `episode_length_limit` to truncate an episode after a fixed number of steps.
Without a limit, an episode ends when the agent reaches a terminal cell.

## Special transitions

A special transition sends the agent to a chosen destination when it takes a
particular action from a particular cell. This example adds a deterministic
jump to the environment above:

```python
env.add_special_transition(
    from_state=(1, 1),
    action=Action.DOWN,
    to_state=(3, 3),
    reward=0.5,
)
env.reset()
state, reward, terminated, truncated, info = env.step(Action.DOWN)
assert state == (3, 3)
```

![GridWorld after a special transition](../../../docs/figures/grid_world/special_transition.png)

## Resets and planning

`reset(seed=0)` repeats the same movement randomness. `reset()` returns to the
start without reseeding. To change the start, use
`reset(new_start_state=(2, 2))` or
`reset(options={"new_start_state": (2, 2)})`; that cell becomes the start for
later resets too.

For planning algorithms, `env.mdp` maps `(state, action)` to possible
`(next_state, reward, terminated, probability)` outcomes. `env.P` is the
transition tensor, indexed by current state, next state, and action. A cell's
flat index is `row * env.cols + column`.

## Rendering

`env.render()` displays the grid with Matplotlib. Use
`env.render(mode="rgb_array")` to obtain an image array.

## Diagrams

![Package diagram](../../../docs/diagrams/grid_world/packages_grid_world.png)
![Class diagram](../../../docs/diagrams/grid_world/classes_grid_world.png)
