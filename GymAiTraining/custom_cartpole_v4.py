import math
from typing import Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled


class CustomCartPoleEnv(gym.Env):
    """
    A cart-pole environment with:
    - Uniform rod pole (mass pole_mass, length pole_length).
    - Discrete velocity commands for the cart.
    - Multi-step (sub-step) integration per RL step for smoother physics.

    This revised version returns the full state [x, v, theta, theta_dot],
    uses more forgiving termination thresholds, and applies a shaped reward.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            pivot_friction: float = 0.0,  # Damping at the pivot
            gravity: float = 9.81,  # Gravitational acceleration
            tau: float = 0.02,  # Timestep (seconds) for *one* RL step
            substeps: int = 5,  # Number of internal physics sub-steps per RL step
    ):
        super().__init__()

        # Physical parameters
        self.pole_mass = 0.1  # kg
        self.pole_length = 0.3  # meters (uniform rod)
        self.pivot_friction = pivot_friction
        self.gravity = gravity

        # Time parameters
        self.tau = tau
        self.substeps = substeps  # finer integration steps per RL step
        self.dt = self.tau / self.substeps

        # Moment of inertia: I = (1/3)*m*L^2 for a uniform rod pivoted at one end
        self.I = (1.0 / 3.0) * self.pole_mass * (self.pole_length ** 2)

        # Termination boundaries -- MODIFIED to be more forgiving:
        self.x_threshold = 2.4  # increased from 0.1 to 2.4 (typical cart-pole value)
        self.theta_threshold_radians = 24.0 * math.pi / 180.0  # increased from 12° to 24° in radians

        # Discrete velocity commands for the cart
        self.vel_candidates = [-0.17, -0.085, 0.0, 0.085, 0.17]
        self.action_space = spaces.Discrete(len(self.vel_candidates))

        # Observations: now [x, v, theta, theta_dot] are exposed to the agent -- MODIFIED!
        high = np.array([
            2.4,  # x: using the same as x_threshold (or could be scaled higher)
            np.finfo(np.float32).max,  # v: no specific bound
            2 * math.pi,  # theta: allow a full rotation range (or adjust as needed)
            np.finfo(np.float32).max  # theta_dot: unbounded
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Internal state: now stored as [x, v, theta, theta_dot]
        self.state = None

        # Rendering parameters
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 400
        self.isopen = True

    def _get_obs(self) -> np.ndarray:
        """Return [x, v, theta, theta_dot] as the agent's observation. MODIFIED!"""
        x, v, theta, theta_dot = self.state
        return np.array([x, v, theta, theta_dot], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Initialize state near upright, small random values
        x = self.np_random.uniform(-0.01, 0.01)
        v = self.np_random.uniform(-0.01, 0.01)
        theta = self.np_random.uniform(-0.05, 0.05)  # ~±3° initially
        theta_dot = self.np_random.uniform(-0.05, 0.05)

        self.state = np.array([x, v, theta, theta_dot], dtype=np.float64)
        return self._get_obs(), {}

    def step(self, action: int):
        # Unpack current state: x = cart position, v = cart velocity,
        # theta = pendulum angle, theta_dot = angular velocity
        x, v, theta, theta_dot = self.state

        # The target cart velocity selected by the agent.
        v_target = self.vel_candidates[action]
        # Compute the required cart acceleration to reach v_target over one RL step.
        a_cart = (v_target - v) / self.tau

        # Instead of using the final v_target in every substep, update v gradually.
        for _ in range(self.substeps):
            # Update cart velocity and position gradually.
            v = v + a_cart * self.dt
            x = x + v * self.dt

            # --- Pendulum Dynamics ---
            m = self.pole_mass
            L = self.pole_length
            b = self.pivot_friction

            sin_th = math.sin(theta)
            cos_th = math.cos(theta)

            # Gravity torque: m * g * (L/2) * sin(theta)
            torque_gravity = m * self.gravity * (L / 2.0) * sin_th
            # Inertial torque: -m * a_cart * (L/2) * cos(theta)
            torque_inertial = -m * a_cart * (L / 2.0) * cos_th
            # Friction torque at the pivot
            torque_friction = -b * theta_dot

            net_torque = torque_gravity + torque_inertial + torque_friction
            theta_acc = net_torque / self.I

            # Update pendulum angular velocity and angle.
            theta_dot = theta_dot + theta_acc * self.dt
            theta = theta + theta_dot * self.dt

        # Save updated state.
        self.state = np.array([x, v, theta, theta_dot], dtype=np.float64)

        # Termination condition: if the cart goes out of bounds or the pole falls too far.
        terminated = (
                x < -self.x_threshold or x > self.x_threshold or
                theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )
        truncated = False

        # Shaped reward function -- MODIFIED!
        # (Here, if not terminated, the agent receives a higher reward when the cart is near center
        #  and the pole is closer to vertical.)
        if terminated:
            reward = 0.0
        else:
            reward = 20.0 - 6 * abs(x / self.x_threshold) - 5 * abs(theta / self.theta_threshold_radians)
            # Alternatively, you might simply use: reward = 1.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled("pygame not installed. `pip install pygame`")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Convert x in [-0.1, 0.1] to screen coords
        world_width = self.x_threshold * 2.0  # 0.2
        scale = self.screen_width / world_width

        # We'll draw the cart around mid-height
        carty = self.screen_height // 2

        x, v, theta, theta_dot = self.state
        cartx = x * scale + (self.screen_width // 2)

        import pygame
        cart_width = 40
        cart_height = 20
        cart_rect = pygame.Rect(
            cartx - cart_width / 2,
            carty - cart_height / 2,
            cart_width,
            cart_height
        )
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)

        # Draw the rod (stick)
        L_pix = scale * self.pole_length
        tip_x = cartx + L_pix * math.sin(theta)  # theta>0 => tip to the right
        tip_y = carty - L_pix * math.cos(theta)  # subtract because screen coords go down
        pygame.draw.line(self.screen, (255, 0, 0), (cartx, carty), (tip_x, tip_y), 6)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
