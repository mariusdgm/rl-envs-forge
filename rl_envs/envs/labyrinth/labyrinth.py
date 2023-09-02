from typing import List, Tuple, Union, Optional

import gymnasium as gym
import numpy as np
import random
import pygame

from .maze import MazeFactory
from .constants import WALL, PATH, TARGET, START, PLAYER, COLORS, CELL_SIZE, Action
from .display import Display
from .player import Player


class Labyrinth(gym.Env):
    def __init__(
        self,
        rows: int,
        cols: int,
        maze_nr_desired_rooms: Optional[int] = None,
        maze_nr_desired_rooms_range: Tuple[int, int] = (1, 8),
        maze_global_room_ratio: Optional[float] = None,
        maze_global_room_ratio_range: Tuple[float, float] = (0.1, 0.8),
        room_access_points: Optional[int] = None,
        room_access_points_range: Tuple[int, int] = (1, 4),
        room_types: Optional[List[str]] = None,
        room_ratio: Optional[Union[int, float]] = None,
        room_ratio_range: Tuple[Union[int, float], Union[int, float]] = (0.5, 1.5),
        reward_schema: Optional[dict] = None,  # Assuming it's a dictionary; adjust if not
        seed: Optional[int] = None,
    ):
        """
        Labyrinth environment for reinforcement learning.

        Arguments maze_num_rooms, maze_global_room_ratio, room_access_points
        are used to fix specific values, otherwise the values are drawn from the minimum and maximum distibutions.

        room_types is used to determine what types of rooms are to be added in the maze, if None, the random seletion
        considers all the implemented room types

        Args:
            rows (int): The number of rows in the labyrinth.
            cols (int): The number of columns in the labyrinth.
            maze_nr_desired_rooms (int, optional): The desired number of rooms in the maze. Defaults to None.
            maze_nr_desired_rooms_range (tuple, optional): The range of desired number of rooms. Defaults to (1, 8).
            maze_global_room_ratio (float, optional): The global room ratio in the maze. Defaults to None.
            maze_global_room_ratio_range (tuple, optional): The range of global room ratio. Defaults to (0.1, 0.8).
            room_access_points (int, optional): The number of access points in each room. Defaults to None.
            room_access_points_range (tuple, optional): The range of access points per room. Defaults to (1, 4).
            room_types (list, optional): The types of rooms to be added in the maze. Defaults to None.
            room_ratio (float, optional): The room ratio. Defaults to None.
            room_ratio_range (tuple, optional): The range of room ratio. Defaults to (0.5, 1.5).
            reward_schema (dict, optional): A dictionary defining the reward schema for the labyrinth. Defaults to None.
            seed (int, optional): The seed to use for generating random numbers. Defaults to None.


        """

        super().__init__()

        self.rows, self.cols = rows, cols
        self.state = np.ones((rows, cols), dtype=np.uint8) * WALL

        self.seed = seed
        if self.seed is None:
            self.seed = random.randint(0, 1e6)

        self.py_random = random.Random(self.seed)
        self.np_random = np.random.RandomState(self.seed)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(rows, cols), dtype=np.uint8
        )

        if not reward_schema:
            self.reward_schema = {
                "neutral_reward": -0.01,
                "wall_collision_reward": -1,
                "target_reached_reward": 10,
            }

        #### Setup maze factory settings ####
        self.maze_nr_desired_rooms = maze_nr_desired_rooms
        self.maze_nr_desired_rooms_range = maze_nr_desired_rooms_range
        self.maze_global_room_ratio = maze_global_room_ratio
        self.maze_global_room_ratio_range = maze_global_room_ratio_range
        self.room_access_points = room_access_points
        self.room_access_points_range = room_access_points_range
        self.room_types = room_types
        self.room_ratio = room_ratio
        self.room_ratio_range = room_ratio_range

        self.player = Player()

        self.setup_labyrinth()

        self.env_displayer = None # only initialize if render is needed

    def setup_labyrinth(self):
        self.make_maze_factory()
        self.maze = self.maze_factory.create_maze()
        self.player.position = self.maze.start_position
        self.build_state_matrix()

    def make_maze_factory(self):
        maze_factory_seed = self.seed + 1
        self.maze_factory = MazeFactory(
            rows=self.rows,
            cols=self.cols,
            nr_desired_rooms=self.maze_nr_desired_rooms,
            nr_desired_rooms_range=self.maze_nr_desired_rooms_range,
            global_room_ratio=self.maze_global_room_ratio,
            global_room_ratio_range=self.maze_global_room_ratio_range,
            access_points_per_room=self.room_access_points,
            access_points_per_room_range=self.room_access_points_range,
            room_types=self.room_types,
            room_ratio=self.room_ratio,
            room_ratio_range=self.room_ratio_range,
            seed=maze_factory_seed,
        )
        return self.maze_factory

    def build_state_matrix(self):
        """Sequentially build the state matrix."""
        self.state = self.maze.grid.copy()
        self.state[self.maze.start_position] = START
        self.state[self.maze.target_position] = TARGET
        self.state[self.player.position] = PLAYER
        return self.state

    def step(self, action):
        # Initial reward
        reward = self.reward_schema["neutral_reward"]
        done = False
        truncated = False

        # Check if the action is valid
        if not self.is_valid_move(self.player, action):
            reward = self.reward_schema["wall_collision_reward"]
            return self.state, reward, done, truncated, {"info": "Invalid move!"}

        # Move the agent
        self.agent_move(action)

        # Check if the agent reached the target
        if self.player.position == self.maze.target_position:
            reward = self.reward_schema["target_reached_reward"]
            done = True
            return self.state, reward, done, truncated, {"info": "Reached the target!"}

        # Flush all info to state matrix
        self.state = self.build_state_matrix()

        return self.state, reward, done, truncated, {}

    def is_valid_move(self, player, action):
        potential_position = player.potential_next_position(action)
        is_inside_bounds = (0 <= potential_position[0] < self.rows) and (
            0 <= potential_position[1] < self.cols
        )
        if not is_inside_bounds:
            return False
        if self.maze.grid[potential_position[0], potential_position[1]] == WALL:
            return False
        return True

    def agent_move(self, action):
        new_position = self.player.potential_next_position(action)
        self.player.position = new_position

    def reset(self, seed=None):
        """Reset and regenerate another labyrinth. If the same seed as the one at the initialization is provided,
        then the same labyrinth should be regenerated.

        Args:
            seed (int, optional): External seed if a user wants to provide one. Defaults to None.

        """
        if seed:
            self.seed = seed
        else:
            self.seed += 1

        self.setup_labyrinth()

    def seed(self, seed=None):
        self.seed = seed

    def render(self, mode=None, sleep_time=100):
        if self.env_displayer is None:
            # Initialize only if render is needed
            self.env_displayer = Display(self.rows, self.cols)

        reward, done, info = None, None, None
        key_press = False

        self.env_displayer.draw_state(self.state)

        if mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    key_press = True

                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()

                    if event.key == pygame.K_UP:
                        _, reward, done, _, info = self.step(Action.UP)
                    elif event.key == pygame.K_RIGHT:
                        _, reward, done, _, info = self.step(Action.RIGHT)
                    elif event.key == pygame.K_DOWN:
                        _, reward, done, _, info = self.step(Action.DOWN)
                    elif event.key == pygame.K_LEFT:
                        _, reward, done, _, info = self.step(Action.LEFT)

        else:  # mode is not human, so model will play
            # Sleep for a bit so you can see the change
            pygame.time.wait(sleep_time)

        return self.state, reward, done, {}, info, key_press

    def human_play(self, print_info=False):
        """Continously display environment and allow user to play.
        Exit by closing the window or pressing ESC.
        """
        done = False
        while not done:  # Play one episode
            state, reward, done, _, info, key_pressed = env.render(mode="human")

            if print_info and key_pressed:

                print(f"Reward: {reward}, Done: {done}, Info: {info}, Seed: {self.seed}")

            if done:
                env.reset()


if __name__ == "__main__":
    env = Labyrinth(31, 31)
    env.human_play(print_info=True)
