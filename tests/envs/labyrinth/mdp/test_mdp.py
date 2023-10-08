import pytest
import numpy as np
import random

from rl_envs_forge.envs.labyrinth.mdp.mdp import LabyrinthMDP
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.labyrinth.constants import WALL, PATH, CorridorMoveStatus


class TestLabirinthMDP:
    def test_build_mdp(self):
        env = Labyrinth(rows=10, cols=10)
        mdp = LabyrinthMDP()
        
        t_r_dict, explored_states = mdp.build_mdp(env)

        nr_path_cells = np.sum(env.maze.grid == PATH)
        assert len(explored_states) == nr_path_cells
        assert len(t_r_dict) == (nr_path_cells - 1) * env.action_space.n
