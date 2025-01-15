import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib.pyplot as plt


class CartPole(gym.Env):
    metadata = {"render.modes": ["human", "matplotlib"]}

    def __init__(
        self,
        tau=0.02,
        continuous_reward=False,
        episode_length_limit=1000,
        angle_termination=None,
        initial_state=None,
        nonlinear_reward=False,
        reward_decay_rate=3.0,
        reward_angle_range=(-0.1, 0.1),
    ):
        """
        Initialize the CartPole environment.

        Args:
            tau (float, optional): The time step between state updates. Defaults to 0.02.
            continuous_reward (bool, optional): Whether to use a continuous reward. Defaults to False.
            episode_length_limit (int, optional): The maximum length of an episode. Defaults to 1000.
            angle_termination (float, optional): The angle at which the episode terminates. Defaults to None.
            initial_state (numpy.ndarray, optional): The initial state of the environment. Defaults to None.
            nonlinear_reward (bool, optional): Whether to use a nonlinear reward. Defaults to False.
            reward_decay_rate (float, optional): The curve parameter for the nonlinear reward. Defaults to 30.0.
            reward_angle_range (tuple, optional): The angle range for discrete reward. Defaults to (-0.1, 0.1).

        Returns:
            None
        """
        super().__init__()

        self.tau = tau  # seconds between state updates
        self.continuous_reward = continuous_reward
        self.nonlinear_reward = nonlinear_reward
        self.reward_decay_rate = reward_decay_rate
        self.episode_length_limit = episode_length_limit
        self.angle_termination = angle_termination
        self.initial_state = initial_state
        self.reward_angle_range = reward_angle_range

        # Define action and observation space
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )

        # Environment parameters
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # actually half the pole's length
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.theta_threshold_radians = 0.2
        self.x_threshold = 2.4

        # Initialize state
        self.state = self._initialize_state()
        self.steps_beyond_done = None
        self.current_step = 0
        self.current_reward = 0.0  # Initialize current reward
        self.total_reward = 0.0  # Initialize total reward

        # Initialize viewer
        self.viewer = None
        self.fig = None
        self.ax = None

    def _initialize_state(self):
        if self.initial_state is not None:
            return np.array(self.initial_state, dtype=np.float64)
        else:
            return np.random.uniform(low=-0.01, high=0.01, size=(4,))

    def normalize_angle(self, angle):
        """Normalize the angle to be within the range [-pi, pi]."""
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        if angle == -np.pi:
            angle = np.pi
        return angle

    def reset(self, seed=None, options=None, initial_state=None):

        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Start state
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            self.state = self._initialize_state()
        self.steps_beyond_done = None
        self.current_step = 0
        self.current_reward = 0.0  # Reset current reward
        self.total_reward = 0.0  # Reset total reward
        return np.array(self.state, dtype=np.float64), {}

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        state = self.state
        x, x_dot, theta, theta_dot = state
        force = action[0]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (
            force + self.pole_mass_length * theta_dot**2 * sin_theta
        ) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = self.normalize_angle(
            theta + self.tau * theta_dot
        )  # Normalize the angle
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = (x, x_dot, theta, theta_dot)

        # Check for termination conditions
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if self.angle_termination is not None:
            done = done or (abs(theta) > self.angle_termination)

        if self.continuous_reward:
            if self.nonlinear_reward:
                self.current_reward = 1.0 - (
                    1.0 - np.exp(-self.reward_decay_rate * abs(theta))
                ) / (1.0 - np.exp(-self.reward_decay_rate * self.theta_threshold_radians))
            else:
                self.current_reward = 1.0 - abs(theta) / self.theta_threshold_radians
        else:
            if self.reward_angle_range[0] <= theta <= self.reward_angle_range[1]:
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
            "x_acc": x_acc,
            "theta_acc": theta_acc,
            "force": force,
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
        if mode == "human":
            self._render_pygame()
        elif mode == "matplotlib":
            self._render_matplotlib()
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented.")

    def _render_pygame(self):
        screen_width = 800
        screen_height = 800

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("CartPole")
            self.font = pygame.font.Font(None, 36)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.viewer = None
                return

        if self.state is None:
            return None

        self.viewer.fill((255, 255, 255))

        pygame.draw.line(self.viewer, (0, 0, 0), (0, carty), (screen_width, carty), 1)

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        cart_y = carty
        cart_rect = pygame.Rect(
            cartx - cartwidth / 2, cart_y - cartheight / 2, cartwidth, cartheight
        )
        pygame.draw.rect(self.viewer, (0, 0, 0), cart_rect)

        pole_x = cartx
        pole_y = cart_y
        pole_end_x = pole_x + polelen * np.sin(x[2])
        pole_end_y = pole_y - polelen * np.cos(x[2])
        pygame.draw.line(
            self.viewer,
            (255, 0, 0),
            (pole_x, pole_y),
            (pole_end_x, pole_end_y),
            int(polewidth),
        )

        pygame.draw.circle(
            self.viewer, (0, 0, 255), (int(pole_x), int(pole_y)), int(polewidth / 2)
        )

        state_info = [
            f"x: {x[0]:.2f}, theta: {x[2]:.2f}",
            f"x_dot: {x[1]:.2f}, theta_dot: {x[3]:.2f}",
            f"reward: {self.current_reward:.2f}",
            f"total_reward: {self.total_reward:.2f}",
        ]
        for i, line in enumerate(state_info):
            text_surface = self.font.render(line, True, (0, 0, 0))
            text_rect = text_surface.get_rect(
                center=(screen_width / 2, screen_height - 120 + i * 20)
            )
            self.viewer.blit(text_surface, text_rect)

        pygame.display.flip()

    def _render_matplotlib(self):
        if self.fig is None and self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(-self.x_threshold * 2, self.x_threshold * 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_aspect("equal")
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(-self.x_threshold * 2, self.x_threshold * 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect("equal")

        x = self.state

        cartx = x[0]
        cart_y = 0  # Since this is 2D
        pole_end_x = cartx + self.length * np.sin(x[2])
        pole_end_y = self.length * np.cos(x[2])

        # Draw cart
        cart_width = 0.4
        cart_height = 0.2
        cart = plt.Rectangle(
            (cartx - cart_width / 2, cart_y - cart_height / 2),
            cart_width,
            cart_height,
            color="black",
        )
        self.ax.add_patch(cart)

        # Draw pole
        self.ax.plot(
            [cartx, pole_end_x], [cart_y, pole_end_y], color="red", linewidth=4
        )

        # Draw state information
        state_info = f"x: {x[0]:.2f}, theta: {x[2]:.2f}, x_dot: {x[1]:.2f}, theta_dot: {x[3]:.2f}\nreward: {self.current_reward:.2f}, total_reward: {self.total_reward:.2f}"
        self.ax.text(
            0.5,
            -1.8,
            state_info,
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
        )

        plt.draw()
        plt.pause(0.01)

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None
