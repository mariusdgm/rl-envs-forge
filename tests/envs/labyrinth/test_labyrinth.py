import pytest
from unittest.mock import Mock, patch
import pygame
import numpy as np
import os
import random
import copy

# So we avoid the screen appearing during testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.labyrinth.constants import *


class TestLabyrinth:
    # Counter as a class attribute
    iteration_counter = 0

    @staticmethod
    def mock_render(window_size, animate, process_arrow_keys):
        # Increase the counter
        TestLabyrinth.iteration_counter += 1

        # If it's the 100th iteration, simulate a quit event
        if TestLabyrinth.iteration_counter >= 100:
            return True, None  # True indicates a quit event

        # Otherwise, continue with the rest of the mock logic:
        mock_event = Mock()
        mock_event.type = pygame.KEYDOWN

        # Randomly select a direction
        directions = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
        mock_event.key = random.choice(directions)

        # Map pygame keys to the Action enum
        action_mapping = {
            pygame.K_UP: Action.UP,
            pygame.K_DOWN: Action.DOWN,
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
        }

        action = action_mapping.get(mock_event.key)

        # Return the simulated event in the expected format
        return False, action

    def test_deepcopy(self):
        env_original = Labyrinth(rows=20, cols=20)

        # Make a deep copy of the original environment
        env_copy = copy.deepcopy(env_original)

        assert np.array_equal(
            env_original.state, env_copy.state
        ), "States are the same"
        
        # Perform valid moves on the original environment until a new state is reached
        original_state_before_move = env_original.state.copy()
        new_state_reached = False
        actions = list(Action)

        while not new_state_reached:
            for action in actions:
                env_original.step(action)
                if not np.array_equal(original_state_before_move, env_original.state):
                    new_state_reached = True
                    break

        # Verify that the states of the original and copied environments are different
        assert not np.array_equal(
            env_original.state, env_copy.state
        ), "States are still equal after a valid move"

    def test_human_play(self):
        env = Labyrinth(rows=20, cols=20)

        # Reset the counter at the beginning of each test
        TestLabyrinth.iteration_counter = 0

        # Patch the `render` method to use our mock instead
        with patch.object(env, "render", side_effect=self.mock_render):
            try:
                env.human_play()
            except Exception as e:
                # If there's any exception, the test should fail
                assert False, f"An exception occurred: {e}"

            # If no exception, the test passes
            assert True

    def test_human_play_prints(self, capsys):
        env = Labyrinth(rows=20, cols=20)

        # Reset the counter at the beginning of each test
        TestLabyrinth.iteration_counter = 0

        # Patch the render method to use our mock instead
        with patch.object(env, "render", side_effect=self.mock_render):
            env.human_play(print_info=True)

            # Capture the printed output
            captured = capsys.readouterr()
            printed_output = captured.out

            # Check for specific patterns or substrings
            assert "Initialized environment with seed:" in printed_output
            assert "Reward: " in printed_output
            assert "Done: " in printed_output
            assert "Info: " in printed_output

    @pytest.fixture
    def labyrinth(self):
        return Labyrinth(30, 30)

    @pytest.fixture
    def movement_labyrinth(self):
        lab = Labyrinth(30, 30)

        # Set up a specific corner scenario for testing
        grid = np.ones((30, 30), dtype=np.uint8) * WALL
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
        assert movement_labyrinth.player.position == (
            initial_position[0],
            initial_position[1] + 1,
        )

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
        labyrinth.player.position = (
            labyrinth.maze.target_position[0] - 1,
            labyrinth.maze.target_position[1],
        )
        _, reward, done, _, _ = labyrinth.step(
            Action.DOWN
        )  # Assuming down leads to target
        assert reward == labyrinth.reward_schema["target_reached_reward"]
        assert done

    def test_reset(self, labyrinth):
        initial_state = labyrinth.state.copy()
        labyrinth.reset()
        assert not np.array_equal(labyrinth.state, initial_state)

    def test_reset_same_seed(self, labyrinth):
        initial_state = labyrinth.state.copy()
        labyrinth.reset(same_seed=True)
        assert np.array_equal(labyrinth.state, initial_state)

    def test_seeding(self):
        env1 = Labyrinth(30, 30, seed=42)
        env2 = Labyrinth(30, 30, seed=42)
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

    def test_render(self):
        env = Labyrinth(20, 20)

        try:
            # Render the environment for a few frames
            for _ in range(10):
                # Choose a random action
                action = random.choice(list(Action))

                # Apply the action
                env.step(action)

                env.render(window_size=(800, 600), animate=True)

            # If we reach here, it means there were no exceptions while rendering
            assert True

        except Exception as e:
            assert False, e

    def test_quit_event(self):
        env = Labyrinth(rows=20, cols=20)

        # Mock render to simulate QUIT event
        @staticmethod
        def mock_render_quit(window_size, animate, process_arrow_keys):
            mock_event = Mock()
            mock_event.type = pygame.QUIT
            return True, None  # True indicates a quit event

        with patch.object(env, "render", side_effect=mock_render_quit):
            try:
                env.human_play()
            except Exception as e:
                # If there's any exception, the test should fail
                assert False, f"An exception occurred: {e}"

            # If no exception, the test passes
            assert True

    def test_escape_event(self):
        env = Labyrinth(rows=20, cols=20)

        # Mock render to simulate K_ESCAPE event
        @staticmethod
        def mock_render_escape(window_size, animate, process_arrow_keys):
            mock_event = Mock()
            mock_event.type = pygame.KEYDOWN
            mock_event.key = pygame.K_ESCAPE
            return True, None  # True indicates a quit event

        with patch.object(env, "render", side_effect=mock_render_escape):
            try:
                env.human_play()
            except Exception as e:
                # If there's any exception, the test should fail
                assert False, f"An exception occurred: {e}"

            # If no exception, the test passes
            assert True
