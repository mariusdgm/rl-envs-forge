import os
import pygame
import pkg_resources

from .display import EntityDisplayer
from ..constants import Action


class PlayerDisplayer(EntityDisplayer):
    def __init__(self, entity):
        super().__init__(entity)

        sprite_sheet = pygame.image.load(
            pkg_resources.resource_filename(
                "rl_envs_forge",
                os.path.join("envs", "labyrinth", "display", "sprites", "player.png"),
            )
        )
        frame_height = sprite_sheet.get_height() // 2
        self.player_frames = [
            sprite_sheet.subsurface(
                pygame.Rect(0, 0, sprite_sheet.get_width(), frame_height)
            ),
            sprite_sheet.subsurface(
                pygame.Rect(0, frame_height, sprite_sheet.get_width(), frame_height)
            ),
        ]

        self.idle_sprite = self.player_frames[0]
        self.moving_sprite = self.player_frames[1]

    def get_sprite(self) -> pygame.Surface:
        """
        Returns the sprite for the entity.

        Returns:
            pygame.Surface: The sprite for the entity.
        """
        sprite = self.moving_sprite if self.entity.moving else self.idle_sprite

        if self.entity.face_orientation == Action.RIGHT:
            sprite = pygame.transform.flip(sprite, True, False)

        return sprite
