import pytest
import numpy as np
from rl_envs.envs.labyrinth.labyrinth import Labyrinth
from rl_envs.envs.labyrinth.constants import *

class TestLabyrinth:
    @pytest.fixture
    def labyrinth(self):
        return Labyrinth(31, 31)

    @pytest.fixture
    def movement_labyrinth(self):
        lab = Labyrinth(31, 31)
        
        # Set up a specific corner scenario for testing
        grid = np.ones((31, 31), dtype=np.uint8) * WALL
        grid[0, 0:3] = PATH
        grid[0, 1] = PLAYER
        grid[0, 2] = START
        lab.maze.grid = grid
        lab.player.position = (0, 1)
        
        return lab

    def test_initialization(self, labyrinth):
        assert labyrinth.state is not None
        assert labyrinth.player is not None
        assert labyrinth.maze is not None

    def test_valid_move(self, movement_labyrinth):
        initial_position = movement_labyrinth.player.position
    
        # Perform a valid move (assuming moving right is a valid move)
        state, reward, done, _, info = movement_labyrinth.step(Action.RIGHT)
        
        # Test that the position changed as expected
        assert movement_labyrinth.player.position == (initial_position[0], initial_position[1] + 1)
        
        # Test that the neutral reward is returned
        assert reward == movement_labyrinth.reward_schema["neutral_reward"]

    def test_invalid_move(self, movement_labyrinth):
        initial_position = movement_labyrinth.player.position
        
        # Perform the action
        state, reward, done, _, info = movement_labyrinth.step(Action.UP)
        
        # Test that the position didn't change
        assert movement_labyrinth.player.position == initial_position
        
        # Test that the wall collision reward is returned
        assert reward == movement_labyrinth.reward_schema["wall_collision_reward"]

    def test_reaching_target(self, labyrinth):
        # Set the player's position near the target for simplicity
        labyrinth.player.position = (labyrinth.maze.target_position[0] - 1, labyrinth.maze.target_position[1])
        _, reward, done, _, _ = labyrinth.step(Action.DOWN)  # Assuming down leads to target
        assert reward == labyrinth.reward_schema["target_reached_reward"]
        assert done

    def test_reset(self, labyrinth):
        initial_state = labyrinth.state.copy()
        labyrinth.reset()
        assert not np.array_equal(labyrinth.state, initial_state)

    def test_seeding(self):
        env1 = Labyrinth(31, 31, seed=42)
        env2 = Labyrinth(31, 31, seed=42)
        assert np.array_equal(env1.state, env2.state)
        assert np.array_equal(env1.maze.start_position, env2.maze.start_position)
        assert np.array_equal(env1.maze.target_position, env2.maze.target_position)

    def test_labyrinth_reset_with_same_seed(self):
        # Given
        seed = 42  # You can pick any seed value you prefer
        labyrinth = Labyrinth(20, 20, seed=seed)
        initial_state = labyrinth.state  

        # When
        labyrinth.reset(seed=seed)  # Assuming the reset method allows passing the seed
        post_reset_state = labyrinth.state

        # Then
        assert np.array_equal(initial_state, post_reset_state)