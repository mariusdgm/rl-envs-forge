from typing import Tuple

from ..constants import Action
from ..display.player import PlayerDisplayer


class Player:
    def __init__(self, start_position: Tuple[int, int] = None):
        self._position = list(start_position) if start_position else None
        self._rendered_position = list(start_position) if start_position else None

        self.movement_speed = 0.25  # Adjust for desired speed
        self.moving = False
        self.face_orientation = Action.LEFT  # default sprite looks left

        self.displayer = PlayerDisplayer(self)

    @property
    def position(self):
        return tuple(self._position)

    @position.setter
    def position(self, value):
        if isinstance(value, tuple):
            self._position = list(value)
        else:
            self._position = value

    @property
    def rendered_position(self):
        return tuple(self._rendered_position)

    @rendered_position.setter
    def rendered_position(self, value):
        if isinstance(value, tuple):
            self._rendered_position = list(value)
        else:
            self._rendered_position = value

    def potential_next_position(self, action):
        """Returns the potential next position based on the action, without checking its validity."""
        potential_position = list(self.position)  

        if action == Action.UP:  # Up
            potential_position[0] -= 1
        elif action == Action.RIGHT:  # Right
            potential_position[1] += 1
        elif action == Action.DOWN:  # Down
            potential_position[0] += 1
        elif action == Action.LEFT:  # Left
            potential_position[1] -= 1

        return tuple(potential_position)

    def move_render_position(self):
        """Move the rendered_position towards position."""
        new_rendered_position = list(self._rendered_position)

        for i in range(2):  # For x and y coordinates
            diff = self._position[i] - new_rendered_position[i]
            if abs(diff) > 0.01:  # Adjust for desired precision
                new_rendered_position[i] += diff * self.movement_speed

        self._rendered_position = new_rendered_position

    def _positions_are_close(self, pos1, pos2, threshold=0.1):
        return abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold
