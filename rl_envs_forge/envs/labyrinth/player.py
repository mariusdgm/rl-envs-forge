from typing import Tuple

from rl_envs_forge.envs.labyrinth.constants import Action

class Player:
    def __init__(self, start_position: Tuple[int, int]=None):
        self.position = start_position
        self.rendered_position = None
        self.target_position = None
        self.movement_speed = 0.1  # Adjust for desired speed
        self.moving = False
        self.heading_direction = Action.LEFT
        
    def potential_next_position(self, action):
        """Returns the potential next position based on the action, without checking its validity."""
        if action == Action.UP:  # Up
            return (self.position[0] - 1, self.position[1])
        elif action == Action.RIGHT:  # Right
            return (self.position[0], self.position[1] + 1)
        elif action == Action.DOWN:  # Down
            return (self.position[0] + 1, self.position[1])
        elif action == Action.LEFT:  # Left
            return (self.position[0], self.position[1] - 1)
        else:
            # Do nothing
            return (self.position[0], self.position[1])
        
    def move_towards_target(self):
        """Move the rendered_position towards target_position."""
        for i in range(2):  # For x and y coordinates
            diff = self.target_position[i] - self.rendered_position[i]
            if abs(diff) > 0.01:  # Adjust for desired precision
                self.rendered_position[i] += diff * self.movement_speed

    def _positions_are_close(self, pos1, pos2, threshold=0.1):
        return abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold