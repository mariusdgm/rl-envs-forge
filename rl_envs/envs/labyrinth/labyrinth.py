import gymnasium as gym

import numpy as np
import random 
import pygame

CELL_SIZE = 40
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PATH = 0
WALL = 1
GOAL = 2
START = 3
AGENT = 4

class Labyrinth(gym.Env):
    def __init__(self, rows=10, cols=10, seed=None):
        super().__init__()

        self.rows, self.cols = rows, cols
        self.agent_position = None
        self.start_position = None
        self.goal_position = None
        self.maze = None
        self.seed_value = seed  # seed passed during initialization
        self.current_seed = seed  # this will change during every reset

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(rows, cols), dtype=np.uint8
        )

        self.screen = pygame.display.set_mode((CELL_SIZE * cols, CELL_SIZE * rows))
        pygame.display.set_caption("Labyrinth")

        # Use the initial seed for the first maze generation
        self._set_random_seed(self.current_seed)
        self.generate_maze()

    def _set_random_seed(self, seed):
        """
        Set the random seed for both Python's `random` library and numpy's randomness.
        """
        random.seed(seed)
        np.random.seed(seed)

    def state(self):
        return self.maze

    def generate_maze(self):
        # Initialize grid as all walls
        self.maze = np.ones((self.rows, self.cols), dtype=np.uint8) * WALL

        # Start in the middle for simplicity
        start = self._random_start()
        self.maze[start] = START
        self.start_position = start

        # List to store cells which will be processed
        cell_list = [start]

        # Possible moves from current cell
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        path = []
        while cell_list:
            cell = random.choice(cell_list)

            random.shuffle(directions)

            extended = False  # Flag to see if we have extended from the current cell
            for dx, dy in directions:

                # Now scale the direction by move factor
                dx *= 2
                dy *= 2

                # Calculate the next cell position
                next_cell = (cell[0] + dx, cell[1] + dy)

                # Calculate the 'in-between' cell position
                between_cell = (cell[0] + dx // 2, cell[1] + dy // 2)

                # Check if the next cell is valid and is a wall 
                if (1 <= next_cell[0] < self.rows-1 and 1 <= next_cell[1] < self.cols-1) and self.maze[next_cell] == WALL:
                    # Make the next cell and the in-between cell part of the path
                    self.maze[next_cell] = PATH
                    self.maze[between_cell] = PATH
                    path.extend([next_cell, between_cell])
                    cell_list.append(next_cell)
                    extended = True
                    break

            if not extended:
                cell_list.remove(cell)

        # Randomly select a goal position
        self.goal_position = self._random_goal(path)
        self.maze[self.goal_position] = GOAL

        # Set the initial agent position
        self.agent_position = self.start_position
        self.maze[self.agent_position] = AGENT


    def _random_start(self):
        """Randomly select a start position from anywhere in the maze."""
        # Randomly select a start position, ensuring it's not on the perimeter
        start_row = random.randint(1, self.rows - 2)
        start_col = random.randint(1, self.cols - 2)
        return (start_row, start_col)
        
    def _random_goal(self, path):
        """Randomly select a goal position from anywhere in the path except the start position."""
        while True:
            goal_position = random.choice(path)
            if goal_position != self.start_position:
                return goal_position

    def _inside_bounds(self, cell):
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def _opposite_cell(self, wall):
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        viable_opposites = []

        for dx, dy in neighbors:
            if self._inside_bounds((wall[0] + dx, wall[1] + dy)) and self.maze[wall[0] + dx, wall[1] + dy] == PATH:
                opposite = wall[0] + 2*dx, wall[1] + 2*dy
                if self._inside_bounds(opposite) and self.maze[opposite] == WALL:
                    viable_opposites.append(opposite)

        if viable_opposites:
            return random.choice(viable_opposites)

        return None

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

        # Generate a new maze
        self.generate_maze()

        return self.state()
    
    def _draw_maze(self):
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.maze[row, col]
                # Set the base color
                if cell_value == WALL:
                    color = BLACK
                else:
                    color = WHITE
                    
                pygame.draw.rect(self.screen, color, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
                # Draw pictograms for special states
                center_x, center_y = col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2
                radius = CELL_SIZE // 2

                if cell_value == AGENT:
                    pygame.draw.circle(self.screen, RED, (center_x, center_y), radius)
                elif cell_value == GOAL:
                    pygame.draw.circle(self.screen, GREEN, (center_x, center_y), radius)
                    
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
        if (0 <= new_position[0] < self.rows and
                0 <= new_position[1] < self.cols and
                self.maze[new_position[0], new_position[1]] != WALL):

            # Update the maze to reflect the agent's old and new positions
            self.maze[self.agent_position[0], self.agent_position[1]] = PATH
            self.maze[new_position[0], new_position[1]] = AGENT

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
    env = Labyrinth(21, 21)
    while True:
        env.render()