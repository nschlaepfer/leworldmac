"""
TwoRoom: Simple 2D navigation environment.

Two rooms separated by a wall with a single door connecting them.
Agent (red dot) must navigate to a target location (green dot) in the other room.
Continuous action space (2D velocity).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TwoRoomEnv(gym.Env):
    """TwoRoom 2D navigation environment.

    The environment is a 2D plane split into two rooms by a vertical wall
    with a door opening. The agent must navigate from one room to the other.

    Observation: RGB image (render_size x render_size x 3)
    Action: 2D continuous velocity [-1, 1]^2
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_size: int = 224,
        max_steps: int = 200,
        room_size: float = 400.0,
        door_width: float = 60.0,
        agent_radius: float = 8.0,
        target_radius: float = 8.0,
        speed: float = 5.0,
        success_threshold: float = 20.0,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.render_size = render_size
        self.max_steps = max_steps
        self.room_size = room_size
        self.door_width = door_width
        self.agent_radius = agent_radius
        self.target_radius = target_radius
        self.speed = speed
        self.success_threshold = success_threshold
        self.render_mode = render_mode

        # Wall at x = room_size/2
        self.wall_x = room_size / 2
        self.door_y_min = (room_size - door_width) / 2
        self.door_y_max = (room_size + door_width) / 2

        self.observation_space = spaces.Box(0, 255, (render_size, render_size, 3), dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.agent_pos = None
        self.target_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np_rng = self.np_random

        # Place agent in left room, target in right room (or vice versa)
        margin = 30.0
        if np_rng.random() < 0.5:
            # Agent in left room
            ax = np_rng.uniform(margin, self.wall_x - margin)
            ay = np_rng.uniform(margin, self.room_size - margin)
            tx = np_rng.uniform(self.wall_x + margin, self.room_size - margin)
            ty = np_rng.uniform(margin, self.room_size - margin)
        else:
            # Agent in right room
            ax = np_rng.uniform(self.wall_x + margin, self.room_size - margin)
            ay = np_rng.uniform(margin, self.room_size - margin)
            tx = np_rng.uniform(margin, self.wall_x - margin)
            ty = np_rng.uniform(margin, self.room_size - margin)

        self.agent_pos = np.array([ax, ay], dtype=np.float32)
        self.target_pos = np.array([tx, ty], dtype=np.float32)
        self.steps = 0

        obs = self._render()
        return obs, {"agent_pos": self.agent_pos.copy(), "target_pos": self.target_pos.copy()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        new_pos = self.agent_pos + action * self.speed

        # Clamp to room boundaries
        new_pos = np.clip(new_pos, 0.0, self.room_size)

        # Wall collision
        if not self._can_move(self.agent_pos, new_pos):
            # Try to slide along wall
            new_pos_x_only = np.array([new_pos[0], self.agent_pos[1]])
            new_pos_y_only = np.array([self.agent_pos[0], new_pos[1]])
            if self._can_move(self.agent_pos, new_pos_x_only):
                new_pos = new_pos_x_only
            elif self._can_move(self.agent_pos, new_pos_y_only):
                new_pos = new_pos_y_only
            else:
                new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos
        self.steps += 1

        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        success = dist < self.success_threshold
        terminated = success
        truncated = self.steps >= self.max_steps

        obs = self._render()
        info = {
            "agent_pos": self.agent_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "distance": dist,
            "success": success,
        }

        return obs, 0.0, terminated, truncated, info

    def _can_move(self, old_pos, new_pos):
        """Check if movement crosses the wall illegally."""
        old_x, old_y = old_pos
        new_x, new_y = new_pos

        # Check if crossing wall
        if (old_x < self.wall_x) != (new_x < self.wall_x):
            # Crossing wall - check if through door
            # Interpolate y position at wall
            if abs(new_x - old_x) < 1e-8:
                y_at_wall = old_y
            else:
                t = (self.wall_x - old_x) / (new_x - old_x)
                y_at_wall = old_y + t * (new_y - old_y)
            return self.door_y_min <= y_at_wall <= self.door_y_max
        return True

    def _render(self) -> np.ndarray:
        """Render the environment as an RGB image."""
        img = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 220

        scale = self.render_size / self.room_size

        # Draw wall
        wall_px = int(self.wall_x * scale)
        door_y_min_px = int(self.door_y_min * scale)
        door_y_max_px = int(self.door_y_max * scale)
        img[:door_y_min_px, wall_px - 1 : wall_px + 2, :] = [60, 60, 60]
        img[door_y_max_px:, wall_px - 1 : wall_px + 2, :] = [60, 60, 60]

        # Draw target (green dot)
        tx, ty = int(self.target_pos[0] * scale), int(self.target_pos[1] * scale)
        r = max(2, int(self.target_radius * scale))
        self._draw_circle(img, tx, ty, r, [0, 200, 0])

        # Draw agent (red dot)
        ax, ay = int(self.agent_pos[0] * scale), int(self.agent_pos[1] * scale)
        r = max(2, int(self.agent_radius * scale))
        self._draw_circle(img, ax, ay, r, [220, 30, 30])

        return img

    def _draw_circle(self, img, cx, cy, r, color):
        """Draw a filled circle on the image."""
        H, W = img.shape[:2]
        y, x = np.ogrid[:H, :W]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        img[mask] = color

    def render(self):
        return self._render()

    def get_goal_obs(self) -> np.ndarray:
        """Render observation showing just the target (for planning)."""
        saved_agent = self.agent_pos.copy()
        self.agent_pos = self.target_pos.copy()
        obs = self._render()
        self.agent_pos = saved_agent
        return obs
