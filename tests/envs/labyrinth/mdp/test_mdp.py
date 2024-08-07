import pytest
import numpy as np
import random

from rl_envs_forge.envs.labyrinth.mdp.mdp import LabyrinthMDP
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.labyrinth.constants import WALL, PATH, CorridorMoveStatus


class TestLabirinthMDP:
    def test_build_mdp_dfs(self):
        """
        WARNING: there are some labyrinth layouts that will not produce valid MDPs (Most likely due to DonutRoom),
        because not all states are reachable by starting from the initial state.

        It is better to use the other build_mdp function...
        """
        env = Labyrinth(rows=10, cols=10, seed=0, room_types=["rectangle"])
        mdp = LabyrinthMDP()

        t_r_dict, explored_states = mdp.build_mdp_dfs(env)

        nr_path_cells = np.sum(env.maze.grid == PATH)
        nr_explored_states = nr_path_cells  # Including terminal state
        expected_transitions = nr_path_cells * env.action_space.n 

        assert len(explored_states) == nr_explored_states, f"Expected {nr_explored_states} explored states, but got {len(explored_states)}"
        assert len(t_r_dict) == expected_transitions, f"Expected {expected_transitions} transitions, but got {len(t_r_dict)}"

    def test_build_mdp(self):
        """
        WARNING: there are some labyrinth layouts that will not produce valid MDPs, because not all states are
        reachable by starting from the initial state.

        It is better to use the other build_mdp function...
        """
        for i in [10, 15, 20]:
            env = Labyrinth(rows=i, cols=i, seed=0)
            mdp = LabyrinthMDP()

            t_r_dict, explored_states = mdp.build_mdp(env)

            nr_path_cells = np.sum(env.maze.grid == PATH)
            nr_explored_states = nr_path_cells  # Including terminal state
            expected_transitions = (nr_path_cells - 1) * env.action_space.n + env.action_space.n

            assert len(explored_states) == nr_explored_states, f"Expected {nr_explored_states} explored states, but got {len(explored_states)}"
            assert len(t_r_dict) == expected_transitions, f"Expected {expected_transitions} transitions, but got {len(t_r_dict)}"