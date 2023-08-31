import gymnasium as gym
import numpy as np
import random
import pygame

from .maze import Maze
from .constants import WALL, PATH, TARGET, START, PLAYER, COLORS, CELL_SIZE, Action
from .display import Display
from .player import Player


class Labyrinth(gym.Env):
    def __init__(
        self,
        rows,
        cols,
        seed=None,
        maze_num_rooms=None,
        maze_min_num_rooms=1,
        maze_max_num_rooms=8,
        maze_global_room_ratio=None,
        maze_min_global_room_ratio=0.1,
        maze_max_global_room_ratio=0.7,
    ):
        super().__init__()

        self.rows, self.cols = rows, cols
        self.state = np.ones((rows, cols), dtype=np.uint8) * WALL

        if seed is None:
            seed = random.randint(0, 1e6)
        self.seed = seed  # this will change during every reset

        self.py_random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(rows, cols), dtype=np.uint8
        )

        self.maze_num_rooms = maze_num_rooms
        if self.maze_num_rooms is None:
            self.maze_num_rooms = self.py_random.randint(maze_min_num_rooms, maze_max_num_rooms)

        self.maze_global_room_ratio = maze_global_room_ratio
        if self.maze_global_room_ratio is None:
            self.maze_global_room_ratio = self.py_random.uniform(
                maze_min_global_room_ratio, maze_max_global_room_ratio
            )

        self.player = Player()

        self.setup_labyrinth()

        self.env_displayer = Display(self.rows, self.cols)

    def setup_labyrinth(self):
        maze_seed = self.np_random.randint(0, 1e6)
        self.maze = Maze(
            rows=self.rows,
            cols=self.cols,
            nr_desired_rooms=self.maze_num_rooms,
            global_room_ratio=self.maze_global_room_ratio,
            seed=maze_seed,
        )
        self.player.position = self.maze.start_position
        self.build_state_matrix()
        
    def build_state_matrix(self):
        """Sequentially build the state matrix."""
        self.state = self.maze.grid.copy()
        self.state[self.maze.start_position] = START
        self.state[self.maze.target_position] = TARGET
        self.state[self.player.position] = PLAYER
        return self.state
    
    def step(self, action):
        # Initial reward
        reward = -0.01
        done = False
        truncated = False

        # Check if the action is valid
        if not self.is_valid_move(self.player, action):
            reward = -1  # Penalize invalid moves
            return self.state, reward, done, truncated, {"info": "Invalid move!"}

        # Move the agent
        self.agent_move(action)

        # Check if the agent reached the target
        if self.player.position == self.maze.target_position:
            reward = 10  # Reward for reaching the target
            done = True
            return self.state, reward, done, truncated, {"info": "Reached the target!"}

        # Flush all info to state matrix
        self.state = self.build_state_matrix()

        return self.state, reward, done, truncated, {}
    
    def is_valid_move(self, player, action):
        potential_position = player.potential_next_position(action)
        is_inside_bounds = (0 <= potential_position[0] < self.rows) and \
                           (0 <= potential_position[1] < self.cols)
        if not is_inside_bounds:
            return False
        if self.maze.grid[potential_position[0], potential_position[1]] == WALL:
            return False
        return True
    
    def agent_move(self, action):
        new_position = self.player.potential_next_position(action)
        self.player.position = new_position

    def reset(self, seed=None):
        """Reset end and return a differenly seeded result

        Args:
            seed (int, optional): External seed if a user wants to provide one. Defaults to None.

        """
        # Increment the seed value
        if self.seed is None:
            self.seed = self.py_random.randint(0, 1e6)
        else:
            self.seed += 1

        self.setup_labyrinth()

    def seed(self, seed=None):
        self.seed = seed
        
    def render(self, mode=None, sleep_time=100):
        self.env_displayer.draw_state(self.state)

        if mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        _, reward, done, _, info = self.step(Action.UP)
                    elif event.key == pygame.K_RIGHT:
                        _, reward, done, _, info = self.step(Action.RIGHT)
                    elif event.key == pygame.K_DOWN:
                        _, reward, done, _, info = self.step(Action.DOWN)
                    elif event.key == pygame.K_LEFT:
                        _, reward, done, _, info = self.step(Action.LEFT)
                        
                    return self.state, reward, done, {}, info

        else:  # mode is not human, so model will play
            # Sleep for a bit so you can see the change
            pygame.time.wait(sleep_time)
            return (None, None, None, None, None)
            


if __name__ == "__main__":
    env = Labyrinth(31, 31)
    print_info = True

    done = False
    while not done:  # Play one episode
        state, reward, done, _, info = env.render(mode="human")

        if print_info:
            print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if done:
            env.reset()
