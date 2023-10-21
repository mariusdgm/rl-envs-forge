import os
import pygame
import pkg_resources
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
            pkg_resources.resource_filename(
                "rl_envs_forge",
                os.path.join("envs", "labyrinth", "display", "sprites", "flag.png"),
            )
        )

        self.player_displayer = self.labyrinth.player.displayer

        self._compute_sizes_and_paddings()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )

        pygame.display.set_caption("Labyrinth")

    def _compute_sizes_and_paddings(self) -> None:
        """
        Computes the sizes and paddings for the display of the environment.

        This function calculates the width and height of each cell in the display based on the
        window width, the number of columns, the window height, and the number of rows. It then
        calculates the total width and height of the display by multiplying the cell width and
        height by the number of columns and rows, respectively. Finally, it calculates the
        additional padding in the x and y directions by subtracting the total width and height
        from the window width and height, respectively.

        Parameters:
            None

        Returns:
            None
        """
        self.cell_width = (
            self.window_width - 2 * EnvDisplay.BORDER_PADDING
        ) // self.cols
        self.cell_height = (
            self.window_height - 2 * EnvDisplay.BORDER_PADDING
        ) // self.rows

        total_cell_width = round(self.cell_width * self.cols)
        total_cell_height = round(self.cell_height * self.rows)

        self.additional_padding_x = (
            self.window_width - round(total_cell_width) - 2 * EnvDisplay.BORDER_PADDING
        )
        self.additional_padding_y = (
            self.window_height
            - round(total_cell_height)
            - 2 * EnvDisplay.BORDER_PADDING
        )

    def draw_state(self) -> None:
        """
        Draws the current state of the game on the screen.

        Parameters:
            None

        Returns:
            None
        """

        # 1. Fill background with border color
        self.screen.fill(EnvDisplay.BORDER_COLOR)

        # 2. Draw maze
        self.draw_maze(self.labyrinth.maze.grid)

        # 3. Draw target 
        if self.labyrinth.maze.target_position is not None:
            self.draw_target_at_position(self.labyrinth.maze.target_position)

        # 4. Draw the grid
        self.draw_grid(
            self.labyrinth.maze.grid.shape[0], self.labyrinth.maze.grid.shape[1]
        )

        #### Draw special features
        # 5. Draw player
        self.draw_player()  

        pygame.display.flip()

    def draw_player(self) -> None:
        """
        Draws the player on the screen.

        This function retrieves the rendered position of the player from the labyrinth and adjusts it for padding. It then gets the sprite of the player using the player_displayer object. The sprite is scaled to the desired width and height based on the cell width and height. After micro-adjustment, the sprite is blitted onto the screen at the adjusted coordinates.

        Parameters:
            None

        Returns:
            None
        """
        if self.labyrinth.player.rendered_position is None:
            return
        
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

    def draw_target_at_position(self, position: tuple) -> None:
        """
        Draws a target sprite at the specified position on the screen.

        Args:
            position (tuple): The x and y coordinates of the position to draw the target sprite at.

        Returns:
            None
        """
        x, y = self._adjust_coords_for_padding(*position)
        sprite = self.get_target_sprite()
        self.screen.blit(sprite, (x, y))

    def get_target_sprite(self) -> pygame.Surface:
        """
        Return the target sprite after scaling it to the desired width and height.

        :return: The scaled target sprite as a pygame.Surface object.
        :rtype: pygame.Surface
        """
        desired_width = self.cell_width * 0.9
        desired_height = self.cell_height * 0.9
        sprite = pygame.transform.scale(
            self.target_sprite, (desired_width, desired_height)
        )
        return sprite

    def _adjust_coords_for_padding(self, row: int, col: int) -> tuple:
        """
        Adjusts the coordinates for padding.

        Args:
            row (int): The row index.
            col (int): The column index.

        Returns:
            tuple: A tuple containing the adjusted x and y coordinates.
        """
        x = self.additional_padding_x + col * self.cell_width
        y = self.additional_padding_y + row * self.cell_height
        return x, y

    def draw_maze(self, state: list) -> None:
        """
        Draws the maze on the screen using the provided state.

        Args:
            state (list): A 2D list representing the maze state.

        Returns:
            None
        """
        for row in range(len(state)):
            for col in range(len(state[row])):
                cell_value = state[row][col]
                if cell_value == WALL:
                    self.draw_cell(row, col, WALL)
                else:
                    self.draw_cell(row, col, PATH)

    def draw_cell(self, row: int, col: int, cell_value: int) -> None:
        """
        Draws a cell on the screen at the specified row and column with the given cell value.

        Parameters:
            row (int): The row index of the cell.
            col (int): The column index of the cell.
            cell_value (int): The value of the cell.

        Returns:
            None
        """
        color = self.STATE_COLORS.get(cell_value)

        # Adjust for padding
        x = round(col * self.cell_width) + self.additional_padding_x
        y = round(row * self.cell_height) + self.additional_padding_y

        pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(x, y, round(self.cell_width), round(self.cell_height)),
        )

    def draw_grid(self, rows: int, cols: int) -> None:
        """
        Draw a grid on the screen.

        Parameters:
            rows (int): The number of rows in the grid.
            cols (int): The number of columns in the grid.

        Returns:
            None
        """
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

    def resize(self, new_width: int, new_height: int) -> None:
        """
        Resizes the window to the specified width and height.

        Args:
            new_width (int): The new width of the window.
            new_height (int): The new height of the window.

        Returns:
            None
        """
        self.window_width = new_width
        self.window_height = new_height
        self._compute_sizes_and_paddings()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
