import os
import pygame

from .display import EntityDisplayer
from ..constants import Action


class PlayerDisplayer(EntityDisplayer):
    def __init__(self, entity):
        super().__init__(entity)

        sprite_sheet = pygame.image.load(
            os.path.join("assets", "labyrinth", "sprites", "player.png")
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
        sprite = self.moving_sprite if self.entity.moving else self.idle_sprite

        if self.entity.face_orientation == Action.RIGHT:
            sprite = pygame.transform.flip(sprite, True, False)

        return sprite
