import os
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
        self, rows: int, cols: int, window_width: int = 800, window_height: int = 800, labyrinth = None
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

        self.target_sprite = pygame.image.load(os.path.join('assets', 'labyrinth', 'sprites', 'flag.png'))
        sprite_sheet = pygame.image.load(os.path.join('assets', 'labyrinth', 'sprites', 'player.png'))
        frame_height = sprite_sheet.get_height() // 2  
        self.player_frames = [
            sprite_sheet.subsurface(pygame.Rect(0, 0, sprite_sheet.get_width(), frame_height)),
            sprite_sheet.subsurface(pygame.Rect(0, frame_height, sprite_sheet.get_width(), frame_height))
        ]

        self.player_idle_image = self.player_frames[0]
        self.player_animation_image = self.player_frames[1]
        

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

    def draw_state(self, state, animate=True):
        """Display the state of the env

        Args:
            state (2D Numpy array): The discrete state of the environment
            animate (bool, optional): Wether to animate movement of agents. Defaults to True.
        """
        # 1. Fill background with border color
        self.screen.fill(Display.BORDER_COLOR)

        # Calculate additional paddings (to center the maze area)
        total_cell_width = self.cell_width * state.shape[1]
        total_cell_height = self.cell_height * state.shape[0]
        additional_padding_x = (self.screen.get_width() - total_cell_width) // 2
        additional_padding_y = (self.screen.get_height() - total_cell_height) // 2

        # 2. Draw maze
        self.draw_maze(state, additional_padding_x, additional_padding_y)

        # 3. Draw target
        self.draw_target_at_position(self.labyrinth.maze.target_position) 

        # 4. Draw the grid
        self.draw_grid(
            state.shape[0], state.shape[1], additional_padding_x, additional_padding_y
        )
        
        #### Draw special features
        # 5. Draw player
        # TODO: Fix player rendering
        if animate:
            # Draw player based on its rendered position
            self.draw_player(self.labyrinth.player.rendered_position)
        else:
            # Draw player based on its discrete state position
            self.draw_player(self.labyrinth.player.position)

        pygame.display.flip()

    def draw_player(self, position):
        x, y = self._calculate_position_coords(*position)
        image = self.get_player_sprite()
        centered_x = x + (self.BORDER_PADDING / 2)
        centered_y = y + (self.BORDER_PADDING / 2)        
        self.screen.blit(image, (centered_x, centered_y))

    def get_player_sprite(self):
        sprite = self.player_animation_image if self.labyrinth.player.moving else self.player_idle_image
        desired_width = self.cell_width * 0.9
        desired_height = self.cell_height * 0.9
        sprite = pygame.transform.scale(sprite, (desired_width, desired_height))
        if self.labyrinth.player.heading_direction == Action.RIGHT or self.labyrinth.player.heading_direction == Action.DOWN:
            sprite = pygame.transform.flip(sprite, True, False)
        return sprite

    def draw_target_at_position(self, position):
        x, y = self._calculate_position_coords(*position)
        sprite = self.get_target_sprite()
        self.screen.blit(sprite, (x, y))

    def get_target_sprite(self):
        desired_width = self.cell_width * 0.9
        desired_height = self.cell_height * 0.9
        sprite = pygame.transform.scale(self.target_sprite, (desired_width, desired_height))
        return sprite

    def _calculate_position_coords(self, row, col):
        """
        Translate a logical grid position (row, col) to pixel coordinates.
        """
        x = self.additional_padding_x*2 + col * self.cell_width
        y = self.additional_padding_y*2 + row * self.cell_height
        return x, y

    def draw_maze(self, state, additional_padding_x, additional_padding_y):
        for row in range(len(state)):
            for col in range(len(state[row])):
                cell_value = state[row][col]
                if cell_value == WALL:
                    self.draw_cell(row, col, WALL, additional_padding_x, additional_padding_y)
                else:
                    self.draw_cell(row, col, PATH, additional_padding_x, additional_padding_y)

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
