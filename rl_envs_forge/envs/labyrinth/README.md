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
![Labyrinth render GIF](https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/assets/labyrinth/doc_imgs/auto_play_demo.gif)


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
<td>
<img src="https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/assets/labyrinth/doc_imgs/ss1.png" alt="Screenshot 1" width="300">
</td>
<td>
<img src="https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/assets/labyrinth/doc_imgs/ss2.png" alt="Screenshot 2" width="300">
</td>
</tr>

<tr>
<td>
<img src="https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/assets/labyrinth/doc_imgs/ss3.png" alt="Screenshot 1" width="300">
</td>
<td>
<img src="https://raw.githubusercontent.com/mariusdgm/rl-envs-forge/main/assets/labyrinth/doc_imgs/ss4.png" alt="Screenshot 2" width="300">
</td>
</tr>
</table>


## Configuration

The following parameters can be tuned:

- **Rows**: Number of rows in the labyrinth.
    - Type: int

    <br>

- **Columns**: Number of columns in the labyrinth.
    - Type: int

    <br>

- **Maze Number of Desired Rooms**:
    - Description: Desired number of rooms in the maze. If not set, a value will be randomly selected from the provided range.
    - Type: int (optional)
    - Default: (1, 8)

    <br>

- **Maze Global Room Ratio**:
    - Description: Defines the overall room ratio in the maze. If not set, a value is chosen from the provided range.
    - Type: float (optional)
    - Default: (0.1, 0.8)

    <br>

- **Maze Grid Connect Corridors Option**:
    - Description: Determines the nature of corridor path connectivity.
        - `True`: Corridors will be grid-connected.
        - `False`: Corridors will not be grid-connected.
        - `"random"`: The choice will be made randomly.
    - Type: Union[bool, str] (optional)
    - Default: False

    <br>

- **Room Access Points**:
    - Description: Number of access points in each room. If unset, the value is drawn from the given range.
    - Type: int (optional)
    - Default: (1, 4)

    <br>

- **Room Types**:
    - Description: Types of rooms to be added to the maze. If unset, a random selection from all implemented room types is made.
    - Type: list (optional)
    - Default: None

    <br>

- **Room Ratio**:
    - Description: Defines the individual width to height room ratio. If not specified, a value is drawn from the provided range.
    - Type: float (optional)
    - Default: (0.5, 1.5)

    <br>

- **Reward Schema**:
    - Description: A dictionary defining the reward schema for the labyrinth.
    - Type: dict (optional)
    - Default: None

    <br>

- **Seed**:
    - Description: The seed used for generating random numbers to ensure reproducibility.
    - Type: int (optional)
    - Default: None

    <br>

