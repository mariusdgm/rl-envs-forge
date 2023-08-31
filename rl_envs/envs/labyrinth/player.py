from rl_envs.envs.labyrinth.constants import Action

class Player:
    def __init__(self, start_position=None):
        self.position = start_position
        
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
            raise ValueError("Invalid action!")