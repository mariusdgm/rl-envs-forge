import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class PendulumDisk(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        tau=0.005,
        continuous_reward=False,
        episode_length_limit=1000,
        angle_termination=None,
        initial_state=None,
        nonlinear_reward=False,
        reward_decay_rate=30.0,
        reward_angle_range=(-0.2, 0.2),  # New parameter for angle range
    ):
        """
        Initialize the Inverted Pendulum environment.

        Args:
            tau (float, optional): The time step between state updates. Defaults to 0.005.
            continuous_reward (bool, optional): Whether to use a continuous reward. Defaults to False.
            episode_length_limit (int, optional): The maximum length of an episode. Defaults to 1000.
            angle_termination (float, optional): The angle at which the episode terminates. Defaults to None.
            initial_state (numpy.ndarray, optional): The initial state of the environment. Defaults to None.
            nonlinear_reward (bool, optional): Whether to use a nonlinear reward. Defaults to False.
            reward_decay_rate (float, optional): The curve parameter for the nonlinear reward. Defaults to 30.0.
            reward_angle_range (tuple, optional): The angle range for discrete reward. Defaults to (-0.2, 0.2).

        Returns:
            None
        """
        super().__init__()

        self.tau = tau  # seconds between state updates
        self.continuous_reward = continuous_reward
        self.nonlinear_reward = nonlinear_reward
        self.curve_param = reward_decay_rate
        self.episode_length_limit = episode_length_limit
        self.angle_termination = angle_termination
        self.initial_state = initial_state
        self.reward_angle_range = reward_angle_range

        # Define action and observation space
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -15 * np.pi]),
            high=np.array([np.pi, 15 * np.pi]),
            shape=(2,),
            dtype=np.float64,
        )

        # Environment parameters
        self.J = 1.91e-4  # kg.m^2
        self.m = 0.055  # kg
        self.g = 9.81  # m/s^2
        self.l = 0.042  # m
        self.b = 3e-6  # kg/s
        self.K = 0.0536  # Nm/A
        self.R = 9.5  # ohm

        # Reward parameters
        self.Qrew = np.array([[5, 0], [0, 0.1]])
        self.Rrew = 1.0
        self.gamma = 0.98

        # Initialize state
        self.state = self._initialize_state()
        self.steps_beyond_done = None
        self.current_step = 0
        self.current_reward = 0.0  # Initialize current reward
        self.total_reward = 0.0  # Initialize total reward

        # Initialize viewer
        self.viewer = None

    def _initialize_state(self):
        if self.initial_state is not None:
            return np.array(self.initial_state, dtype=np.float64)
        else:
            return np.random.uniform(low=-0.01, high=0.01, size=(2,))

    def normalize_angle(self, angle):
        """Normalize the angle to be within the range [-pi, pi]."""
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        if angle == -np.pi:
            angle = np.pi
        return angle

    def reset(self, initial_state=None):
        # Start state
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            self.state = self._initialize_state()
        self.steps_beyond_done = None
        self.current_step = 0
        self.current_reward = 0.0  # Reset current reward
        self.total_reward = 0.0  # Reset total reward
        return np.array(self.state, dtype=np.float64)

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        state = self.state
        alpha, alpha_dot = state
        u = action[0]

        alpha_dot_dot = (
            self.m * self.g * self.l * np.sin(alpha)
            - self.b * alpha_dot
            - (self.K**2 / self.R) * alpha_dot
            + self.K / self.R * u
        ) / self.J

        alpha = self.normalize_angle(alpha + self.tau * alpha_dot)
        alpha_dot = alpha_dot + self.tau * alpha_dot_dot

        self.state = (alpha, alpha_dot)

        # Check for termination conditions
        done = bool(
            alpha < -np.pi
            or alpha > np.pi
            or alpha_dot < -15 * np.pi
            or alpha_dot > 15 * np.pi
        )

        if self.angle_termination is not None:
            done = done or (abs(alpha) > self.angle_termination)

        if self.continuous_reward:
            if self.nonlinear_reward:
                self.current_reward = 1.0 - (1.0 - np.exp(-self.curve_param * abs(alpha))) / (
                    1.0 - np.exp(-self.curve_param * np.pi)
                )
            else:
                self.current_reward = 1.0 - abs(alpha) / np.pi
        else:
            if self.reward_angle_range[0] <= alpha <= self.reward_angle_range[1]:
                self.current_reward = 1.0
            else:
                self.current_reward = 0.0

        self.total_reward += self.current_reward  # Update total reward

        self.current_step += 1
        truncated = self.current_step >= self.episode_length_limit

        if truncated:
            done = False

        info = {
            "truncated": truncated,
            "alpha_dot_dot": alpha_dot_dot,
            "force": u,
            "steps": self.current_step,
        }

        return (
            np.array(self.state, dtype=np.float64),
            self.current_reward,
            done,
            truncated,
            info,
        )

    def render(self, mode="human"):
        screen_width = 800
        screen_height = 800

        world_width = np.pi * 2
        scale = screen_width / world_width
        pivot_x = screen_width / 2  # MIDDLE OF SCREEN WIDTH
        pivot_y = screen_height / 2  # MIDDLE OF SCREEN HEIGHT
        disk_radius = scale * 1.8  # Hardcoded size for the disk
        weight_radius = 10.0  # Radius of the weight

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Inverted Pendulum")
            self.font = pygame.font.Font(None, 36)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.viewer = None
                return

        if self.state is None:
            return None

        self.viewer.fill((255, 255, 255))

        # Draw disk
        pygame.draw.circle(self.viewer, (0, 0, 0), (int(pivot_x), int(pivot_y)), int(disk_radius), 2)

        # Draw weight
        alpha = self.state[0]
        weight_x = pivot_x + disk_radius * np.sin(alpha)
        weight_y = pivot_y - disk_radius * np.cos(alpha)
        pygame.draw.circle(self.viewer, (255, 0, 0), (int(weight_x), int(weight_y)), int(weight_radius))

        # Draw line from center of the disk to the weight
        pygame.draw.line(self.viewer, (0, 0, 0), (pivot_x, pivot_y), (weight_x, weight_y), 2)

        # Draw state information
        state_info = [
            f"alpha: {alpha:.2f}, alpha_dot: {self.state[1]:.2f}",
            f"reward: {self.current_reward:.2f}",
            f"total_reward: {self.total_reward:.2f}"
        ]
        for i, line in enumerate(state_info):
            text_surface = self.font.render(line, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(screen_width / 2, screen_height - 120 + i * 20))
            self.viewer.blit(text_surface, text_rect)

        pygame.display.flip()

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None
