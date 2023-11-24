import pytest
import numpy as np
import random

from rl_envs_forge.envs.grid_world.mdp.mdp import GridWorldMDP
from rl_envs_forge.envs.grid_world.grid_world import GridWorld
from rl_envs_forge.envs.grid_world.grid_world import Action


class TestGridWorldMDP:
    def test_build_mdp(self):
        walls = [(5, 0), (5, 1), (5, 3), (5, 4)]
        env = GridWorld(
            rows=6,
            cols=5,
            walls=walls,
            start_state=(1, 1),
            terminal_states={(0, 0): 1.0, (4, 4): 1.0},
        )

        special_state = env.state
        special_action = Action.DOWN
        jump_state = (3, 3)
        special_reward = 0.5
        env.add_special_transition(
            from_state=special_state,
            action=special_action,
            to_state=jump_state,
            reward=special_reward,
        )

        mdp_builder = GridWorldMDP()
        tr_dict = mdp_builder.build_mdp(env)

        nr_viable_states = (env.rows * env.cols - len(walls) - len(env.terminal_states))
        nr_expected_keys = nr_viable_states * env.action_space.n
        assert len(tr_dict) == nr_expected_keys
