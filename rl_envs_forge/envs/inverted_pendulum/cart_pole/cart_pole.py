import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class CartPole(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, tau=0.02, continuous_reward=False, episode_length_limit=1000, angle_termination=None, initial_state=None):
        """
        Initializes a new instance of the CartPole class.

        Args:
            tau (float, optional): The time interval between state updates. Defaults to 0.02.
            continuous_reward (bool, optional): Whether to use a continuous reward function. Defaults to False.
            episode_length_limit (int, optional): The maximum number of steps per episode. Defaults to 1000.
            angle_termination (float, optional): The angle at which the episode terminates. Defaults to None.
            initial_state (numpy.ndarray, optional): The initial state of the environment. Defaults to None.

        Returns:
            None
        """
        super(CartPole, self).__init__()

        self.tau = tau  # seconds between state updates
        self.continuous_reward = continuous_reward
        self.episode_length_limit = episode_length_limit
        self.angle_termination = angle_termination
        self.initial_state = initial_state

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
        self.state = None
        self.steps_beyond_done = None
        self.current_step = 0

        # Initialize viewer
        self.viewer = None

    def reset(self, initial_state=None):
        # Start state
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        elif self.initial_state is not None:
            self.state = np.array(self.initial_state, dtype=np.float64)
        else:
            self.state = np.random.uniform(low=-0.01, high=0.01, size=(4,))
        self.steps_beyond_done = None
        self.current_step = 0
        return np.array(self.state, dtype=np.float64)

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
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = (x, x_dot, theta, theta_dot)
        self.current_step += 1

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        truncated = self.current_step >= self.episode_length_limit

        if self.angle_termination is not None:
            if abs(theta) < self.angle_termination:
                done = True
                truncated = False

        if not done and not truncated:
            if self.continuous_reward:
                reward = max(0.0, (1.0 - abs(theta) / self.theta_threshold_radians))
            else:
                reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0 if not self.continuous_reward else 0.0
        else:
            if self.steps_beyond_done == 0:
                warnings.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.",
                    category=RuntimeWarning,
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float64), reward, done, {"truncated": truncated}

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # MIDDLE OF SCREEN HEIGHT
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

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

        # Draw cart
        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # CENTER OF CART ON SCREEN WIDTH
        cart_y = carty
        cart_rect = pygame.Rect(
            cartx - cartwidth / 2, cart_y - cartheight / 2, cartwidth, cartheight
        )
        pygame.draw.rect(self.viewer, (0, 0, 0), cart_rect)

        # Draw pole
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

        # Draw axle
        pygame.draw.circle(
            self.viewer, (0, 0, 255), (int(pole_x), int(pole_y)), int(polewidth / 2)
        )

        # Draw state information
        state_info = f"x: {x[0]:.2f}, theta: {x[2]:.2f}, x_dot: {x[1]:.2f}, theta_dot: {x[3]:.2f}"
        text_surface = self.font.render(state_info, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(screen_width / 2, screen_height - 20))
        self.viewer.blit(text_surface, text_rect)

        pygame.display.flip()

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None
