import os
import pygame
from abc import ABC, abstractmethod

from ..constants import *

class EntityDisplayer(ABC):
    def __init__(self, entity):
        self.entity = entity

    @abstractmethod
    def get_sprite(self) -> pygame.Surface:
        """
        Return the pygame surface representing the current state of the entity.
        """
        pass

class EnvDisplay:
    STATE_COLORS = {
        WALL: (0, 0, 0),
        PATH: (255, 255, 255),
        PLAYER: (0, 255, 0),
        TARGET: (255, 0, 0),
        START: (0, 0, 255),
    }
    GRID_COLOR = (200, 200, 200)  # A light gray color for grid lines
    BORDER_COLOR = (50, 50, 50)  # A darker gray for the maze border
    BORDER_PADDING = 5

    def __init__(
        self,
        rows: int,
        cols: int,
        window_width: int = 800,
        window_height: int = 800,
        labyrinth=None,
    ):
        """
        Args:
            rows (int): The number of rows in the display.
            cols (int): The number of columns in the display.
            window_width (int): Width of the window.
            window_height (int): Height of the window.
            labyrinth (Laabyrinth): The labyrinth to display.
        """
        self.rows = rows
        self.cols = cols
        self.window_width = window_width
        self.window_height = window_height
        self.cell_width = None
        self.cell_height = None
        self.additional_padding_x = None
        self.additional_padding_y = None
        self.labyrinth = labyrinth

        self.target_sprite = pygame.image.load(
            os.path.join("assets", "labyrinth", "sprites", "flag.png")
        )

        self.player_displayer = self.labyrinth.player.displayer
        
        self._compute_sizes_and_paddings()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )

        pygame.display.set_caption("Labyrinth")

    def _compute_sizes_and_paddings(self):
        self.cell_width = (self.window_width - 2 * EnvDisplay.BORDER_PADDING) // self.cols
        self.cell_height = (
            self.window_height - 2 * EnvDisplay.BORDER_PADDING
        ) // self.rows

        total_cell_width = round(self.cell_width * self.cols)
        total_cell_height = round(self.cell_height * self.rows)

        self.additional_padding_x = (
            self.window_width - round(total_cell_width) - 2 * EnvDisplay.BORDER_PADDING
        ) 
        self.additional_padding_y = (
            self.window_height - round(total_cell_height) - 2 * EnvDisplay.BORDER_PADDING
        ) 

    def draw_state(self):
        """Display the state of the env

        Args:
            state (2D Numpy array): The discrete state of the environment
            animate (bool, optional): Whether to animate movement of agents. Defaults to True.
        """
        # 1. Fill background with border color
        self.screen.fill(EnvDisplay.BORDER_COLOR)

        # 2. Draw maze
        self.draw_maze(self.labyrinth.maze.grid)

        # 3. Draw target #TODO
        self.draw_target_at_position(self.labyrinth.maze.target_position)

        # 4. Draw the grid
        self.draw_grid(self.labyrinth.maze.grid.shape[0], self.labyrinth.maze.grid.shape[1])

        #### Draw special features
        # 5. Draw player
        self.draw_player() #TODO

        pygame.display.flip()

    def draw_player(self):
        draw_pos = self.labyrinth.player.rendered_position
            
        x, y = self._adjust_coords_for_padding(*draw_pos)

        sprite = self.player_displayer.get_sprite()

        # Scale the sprite
        desired_width = self.cell_width * 0.8
        desired_height = self.cell_height * 0.8
        sprite = pygame.transform.scale(sprite, (desired_width, desired_height))

        # micro-adjustement
        x = x + (self.BORDER_PADDING / 2)
        y = y + (self.BORDER_PADDING / 2)
        self.screen.blit(sprite, (x, y))

    
    def draw_target_at_position(self, position):
        x, y = self._adjust_coords_for_padding(*position)
        sprite = self.get_target_sprite()
        self.screen.blit(sprite, (x, y))

    def get_target_sprite(self):
        desired_width = self.cell_width * 0.9
        desired_height = self.cell_height * 0.9
        sprite = pygame.transform.scale(
            self.target_sprite, (desired_width, desired_height)
        )
        return sprite

    def _adjust_coords_for_padding(self, row, col):
        """
        Translate a logical grid position (row, col) to pixel coordinates.
        """
        x = self.additional_padding_x + col * self.cell_width
        y = self.additional_padding_y + row * self.cell_height
        return x, y

    def draw_maze(self, state):
        for row in range(len(state)):
            for col in range(len(state[row])):
                cell_value = state[row][col]
                if cell_value == WALL:
                    self.draw_cell(row, col, WALL)
                else:
                    self.draw_cell(row, col, PATH)

    def draw_cell(self, row, col, cell_value):
        color = self.STATE_COLORS.get(cell_value)

        # Adjust for padding
        x = round(col * self.cell_width) + self.additional_padding_x
        y = round(row * self.cell_height) + self.additional_padding_y

        pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(x, y, round(self.cell_width), round(self.cell_height)),
        )

    def draw_grid(self, rows, cols):
        for col in range(1, cols):
            x = col * self.cell_width + self.additional_padding_x
            pygame.draw.line(
                self.screen,
                EnvDisplay.GRID_COLOR,
                (x, self.additional_padding_y),
                (x, (rows * self.cell_height + self.additional_padding_y)),
            )

        for row in range(1, rows):
            y = row * self.cell_height + self.additional_padding_y
            pygame.draw.line(
                self.screen,
                EnvDisplay.GRID_COLOR,
                (self.additional_padding_x, y),
                ((cols * self.cell_width + self.additional_padding_x), y),
            )

    def resize(self, new_width, new_height):
        self.window_width = new_width
        self.window_height = new_height
        self._compute_sizes_and_paddings()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
