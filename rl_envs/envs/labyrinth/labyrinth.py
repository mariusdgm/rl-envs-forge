import gymnasium as gym

from .maze import Maze
from .constants import WALL, PATH, GOAL, START, AGENT, COLORS, CELL_SIZE

import numpy as np
import random
import pygame




class Labyrinth(gym.Env):
    def __init__(
        self,
        rows,
        cols,
        seed=None,
        maze_num_rooms=None,
        maze_min_num_rooms=1,
        maze_max_num_rooms=8,
        maze_room_padding=1,
        maze_global_room_ratio=None,
        maze_min_global_room_ratio=0.1,
        maze_max_global_room_ratio=0.7,
        
    ):
        super().__init__()

        self.rows, self.cols = rows, cols
        self.agent_position = None
        self.start_position = None
        self.goal_position = None
        self.seed_value = seed  # seed passed during initialization
        self.current_seed = seed  # this will change during every reset

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(rows, cols), dtype=np.uint8
        )

        if maze_num_rooms is None:
            maze_num_rooms = random.randint(maze_min_num_rooms, maze_max_num_rooms)

        if maze_global_room_ratio is None:
            maze_global_room_ratio = random.uniform(
                maze_min_global_room_ratio, maze_max_global_room_ratio
            )

        self.maze = Maze(
            rows = self.rows,
            cols = self.cols,
            num_rooms=maze_num_rooms,
            padding=maze_room_padding,
            global_room_ratio=maze_global_room_ratio,
            seed=self.current_seed,
        )

        self.screen = pygame.display.set_mode((CELL_SIZE * cols, CELL_SIZE * rows))
        pygame.display.set_caption("Labyrinth")

        # Use the initial seed for the first maze generation
        self._set_random_seed(self.current_seed)

    def _set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def state(self):
        return self.maze.grid

    def _random_start(self, room_cells):
        """Randomly select a start position from anywhere in the maze."""
        start_cell = np.random.choice(room_cells)
        return start_cell

    def _random_goal(self, path, room_cells):
        """Randomly select a goal position from anywhere in the path except the start position."""
        while True:
            goal_position = np.random.choice(path)
            if not np.array_equal(goal_position, self.start_position):
                return goal_position

    def _inside_bounds(self, cell):
        return (0 <= cell).all() and (cell < np.array([self.rows, self.cols])).all()

    def step(self, action):
        # Implement this to handle an action and return the next state, reward, done, and optional info dict
        pass

    def reset(self):
        # Increment the seed value
        if self.current_seed is None:
            self.current_seed = random.randint(0, 1e6)
        else:
            self.current_seed += 1

        # Set new random seed
        self._set_random_seed(self.current_seed)

        return self.state()

    def _draw_maze(self):
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.maze.grid[row, col]
                color = COLORS.get(cell_value)

                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )

                # Draw pictograms for special states
                center_x, center_y = (
                    col * CELL_SIZE + CELL_SIZE // 2,
                    row * CELL_SIZE + CELL_SIZE // 2,
                )
                radius = CELL_SIZE // 2

                if cell_value == AGENT:
                    pygame.draw.circle(self.screen, COLORS['AGENT'], (center_x, center_y), radius)
                elif cell_value == GOAL:
                    pygame.draw.circle(self.screen, COLORS['GOAL'], (center_x, center_y), radius)

        pygame.display.flip()

    def agent_move(self, action):
        # Determine the new position based on the action
        if action == 0:  # Up
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)
        elif action == 2:  # Down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 3:  # Left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        else:
            raise ValueError("Invalid action!")

        # Check if the new position is the goal BEFORE updating the maze
        if new_position == self.goal_position:
            print("Reached the goal!")
            self.reset()
            return

        # Check if the new position is a valid move
        if (
            0 <= new_position[0] < self.rows
            and 0 <= new_position[1] < self.cols
            and self.maze.grid[new_position[0], new_position[1]] != WALL
        ):
            # Update the maze to reflect the agent's old and new positions
            self.maze.grid[self.agent_position[0], self.agent_position[1]] = PATH
            self.maze.grid[new_position[0], new_position[1]] = AGENT

            # Update the agent's position
            self.agent_position = new_position

    def render(self, mode="human"):
        self._draw_maze()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.agent_move(0)
                elif event.key == pygame.K_RIGHT:
                    self.agent_move(1)
                elif event.key == pygame.K_DOWN:
                    self.agent_move(2)
                elif event.key == pygame.K_LEFT:
                    self.agent_move(3)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = Labyrinth(31, 31)
    while True:
        env.render()
