import pygame

from rl_envs_forge.envs.labyrinth.constants import *


class Display:
    def __init__(self, rows: int, cols: int):
        """
        Args:
            rows (int): The number of rows in the display.
            cols (int): The number of columns in the display.
        """
        self.screen = pygame.display.set_mode((CELL_SIZE * cols, CELL_SIZE * rows))
        pygame.display.set_caption("Labyrinth")

    def draw_state(self, state):
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                cell_value = state[row, col]
                self.draw_cell(row, col, cell_value)
        pygame.display.flip()

    def draw_cell(self, row, col, cell_value):
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

        if cell_value == PLAYER:
            pygame.draw.circle(
                self.screen, COLORS[PLAYER], (center_x, center_y), radius
            )
        elif cell_value == TARGET:
            pygame.draw.circle(
                self.screen, COLORS[TARGET], (center_x, center_y), radius
            )
