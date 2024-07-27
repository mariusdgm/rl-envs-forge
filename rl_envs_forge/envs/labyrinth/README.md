# Labyrinth

The Labyrinth environment consists in a 2D grid-world where an agent is tasked with reaching the target position.

The mazes are randomized using various parameters, presented in section [Usage](#usage).

📖 **Detailed Documentation** [Click here to read more about the Maze and Rooms components](https://github.com/mariusdgm/rl-envs-forge/blob/main/rl_envs_forge/envs/labyrinth/maze/README.md)

## Usage

Code example for getting started with the environment:

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

![Labyrinth render GIF](../../../docs/figures/labyrinth/auto_play_demo.gif)

<br>

There is also a manual mode which allows you can interact using the arrow keys:

```python
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth

env = Labyrinth(20, 20)
env.human_play(print_info=True, animate=True)
```

## Maze examples

<table>
<tr>
    <th colspan="2">Corridor algorithm: prim</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/labyrinth/ss1.png" alt="Screenshot 1" width="300">
</td>
<td>
<img src="../../../docs/figures/labyrinth/ss2.png" alt="Screenshot 2" width="300">
</td>
</tr>

<tr>
<td>
<img src="../../../docs/figures/labyrinth/ss3.png" alt="Screenshot 3" width="300">
</td>
<td>
<img src="../../../docs/figures/labyrinth/ss4.png" alt="Screenshot 4" width="300">
</td>
</tr>

<tr>
    <th colspan="2">Corridor algorithm: astar</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/labyrinth/ss5.png" alt="Screenshot 5" width="300">
</td>
<td>
<img src="../../../docs/figures/labyrinth/ss6.png" alt="Screenshot 6" width="300">
</td>
</tr>

<tr>
    <th colspan="2">Corridor algorithm: gbfs</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/labyrinth/ss7.png" alt="Screenshot 7" width="300">
</td>
<td>
<img src="../../../docs/figures/labyrinth/ss8.png" alt="Screenshot 8" width="300">
</td>
</tr>
</table>

## Settings effect on output

<table>
<tr>
    <th colspan="3">Corridor algorithm: prim</th>
</tr>
<tr>
    <th colspan="1">post_process:False, grid_connect:False</th>
    <th colspan="1">post_process:True, grid_connect:False</th>
    <th colspan="1">post_process:True, grid_connect:True</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/labyrinth/comp_prim_grid_f_postproc_f.png" alt="Comp Screenshot 1" width="200">
</td>
<td>
<img src="../../../docs/figures/labyrinth/comp_prim_grid_f_postproc_t.png" alt="Comp Screenshot 2" width="200">
</td>
<td>
<img src="../../../docs/figures/labyrinth/comp_prim_grid_t_postproc_t.png" alt="Comp Screenshot 3" width="200">
</td>
</tr>

<tr>
    <th colspan="2">Corridor algorithm: astar</th>
</tr>
<tr>
    <th colspan="1">sort_access_points:False</th>
    <th colspan="1">sort_access_points:True</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/labyrinth/comp_astar_acess_sort_f.png" alt="Comp Screenshot 1" width="300">
</td>
<td>
<img src="../../../docs/figures/labyrinth/comp_astar_acess_sort_t.png" alt="Comp Screenshot 2" width="300">
</td>
</tr>

<tr>
    <th colspan="2">Corridor algorithm: gbfs</th>
</tr>
<tr>
    <th colspan="1">sort_access_points:False</th>
    <th colspan="1">sort_access_points:True</th>
</tr>
<tr>
<td>
<img src="../../../docs/figures/labyrinth/comp_gbfs_acess_sort_f.png" alt="Comp Screenshot 1" width="300">
</td>
<td>
<img src="../../../docs/figures/labyrinth/comp_gbfs_acess_sort_t.png" alt="Comp Screenshot 2" width="300">
</td>
</tr>

</table>

## UML diagrams

### Packages

<img src="../../../docs/diagrams/labyrinth/packages_labyrinth.png" alt="Packages UML" width="300">

### Classes

<img src="../../../docs/diagrams/labyrinth/classes_labyrinth.png" alt="Classes UML" width="300">
