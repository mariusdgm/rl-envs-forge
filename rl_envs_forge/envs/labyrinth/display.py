import pygame

from rl_envs_forge.envs.labyrinth.constants import *


class Display:
    STATE_COLORS = {
        WALL: (0, 0, 0),
        PATH: (255, 255, 255),
        PLAYER: (255, 0, 0),
        TARGET: (0, 255, 0),
        START: (0, 0, 255),
    }
    GRID_COLOR = (200, 200, 200)  # A light gray color for grid lines
    BORDER_COLOR = (50, 50, 50)  # A darker gray for the maze border
    BORDER_PADDING = 5

    def __init__(
        self, rows: int, cols: int, window_width: int = 800, window_height: int = 800
    ):
        """
        Args:
            rows (int): The number of rows in the display.
            cols (int): The number of columns in the display.
            window_width (int): Width of the window.
            window_height (int): Height of the window.
        """
        self.rows = rows
        self.cols = cols
        self.window_width = window_width
        self.window_height = window_height
        self.cell_width = None
        self.cell_height = None
        self.additional_padding_x = None
        self.additional_padding_y = None

        self._compute_sizes_and_paddings()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )

        pygame.display.set_caption("Labyrinth")

    def _compute_sizes_and_paddings(self):
        self.cell_width = (self.window_width - 2 * Display.BORDER_PADDING) // self.cols
        self.cell_height = (
            self.window_height - 2 * Display.BORDER_PADDING
        ) // self.rows

        total_cell_width = round(self.cell_width * self.cols)
        total_cell_height = round(self.cell_height * self.rows)

        self.additional_padding_x = (
            self.window_width - round(total_cell_width) - 2 * Display.BORDER_PADDING
        ) // 2
        self.additional_padding_y = (
            self.window_height - round(total_cell_height) - 2 * Display.BORDER_PADDING
        ) // 2

    def draw_state(self, state):
        # 1. Fill background with border color
        self.screen.fill(Display.BORDER_COLOR)

        # Calculate additional paddings (to center the maze area)
        total_cell_width = self.cell_width * state.shape[1]
        total_cell_height = self.cell_height * state.shape[0]
        additional_padding_x = (self.screen.get_width() - total_cell_width) // 2
        additional_padding_y = (self.screen.get_height() - total_cell_height) // 2

        # 2. Draw the cells
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                cell_value = state[row, col]
                self.draw_cell(
                    row, col, cell_value, additional_padding_x, additional_padding_y
                )

        # 3. Draw the grid
        self.draw_grid(
            state.shape[0], state.shape[1], additional_padding_x, additional_padding_y
        )

        boundary_rect = pygame.Rect(
            self.additional_padding_x + Display.BORDER_PADDING,
            self.additional_padding_y + Display.BORDER_PADDING,
            self.cols * self.cell_width,
            self.rows * self.cell_height,
        )
    
        pygame.display.flip()

    def draw_cell(
        self, row, col, cell_value, additional_padding_x, additional_padding_y
    ):
        color = self.STATE_COLORS.get(cell_value)

        # Adjust for padding
        x = round(col * self.cell_width) + additional_padding_x
        y = round(row * self.cell_height) + additional_padding_y

        pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(x, y, round(self.cell_width), round(self.cell_height)),
        )

        # Draw pictograms for special states
        center_x = x + self.cell_width // 2
        center_y = y + self.cell_height // 2

        radius_x = self.cell_width // 2
        radius_y = self.cell_height // 2

        if cell_value == "PLAYER":
            pygame.draw.ellipse(
                self.screen,
                self.STATE_COLORS["PLAYER"],
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    self.cell_width,
                    self.cell_height,
                ),
            )
        elif cell_value == "TARGET":
            pygame.draw.ellipse(
                self.screen,
                self.STATE_COLORS["TARGET"],
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    self.cell_width,
                    self.cell_height,
                ),
            )

    def draw_grid(self, rows, cols, additional_padding_x, additional_padding_y):
        for col in range(1, cols):
            x = col * self.cell_width + additional_padding_x
            pygame.draw.line(
                self.screen,
                Display.GRID_COLOR,
                (x, additional_padding_y),
                (x, (rows * self.cell_height + additional_padding_y)),
            )

        for row in range(1, rows):
            y = row * self.cell_height + additional_padding_y
            pygame.draw.line(
                self.screen,
                Display.GRID_COLOR,
                (additional_padding_x, y),
                ((cols * self.cell_width + additional_padding_x), y),
            )

    def resize(self, new_width, new_height):
        self.window_width = new_width
        self.window_height = new_height
        self._compute_sizes_and_paddings()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
