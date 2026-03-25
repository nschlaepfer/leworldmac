"""
PushT: 2D manipulation environment.

An agent (blue dot) pushes a T-shaped block to match a target configuration.
Continuous action space (2D velocity of the agent).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PushTEnv(gym.Env):
    """PushT 2D manipulation environment.

    A blue dot agent pushes a T-shaped block to a target pose.
    The T-block is rendered as two rectangles forming a T shape.

    Observation: RGB image (render_size x render_size x 3)
    Action: 2D continuous velocity [-1, 1]^2
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_size: int = 224,
        max_steps: int = 300,
        area_size: float = 512.0,
        agent_radius: float = 15.0,
        block_size: float = 60.0,
        speed: float = 8.0,
        push_force: float = 0.4,
        success_threshold: float = 30.0,
        angle_threshold: float = 0.3,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.render_size = render_size
        self.max_steps = max_steps
        self.area_size = area_size
        self.agent_radius = agent_radius
        self.block_size = block_size
        self.speed = speed
        self.push_force = push_force
        self.success_threshold = success_threshold
        self.angle_threshold = angle_threshold
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0, 255, (render_size, render_size, 3), dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.agent_pos = None
        self.block_pos = None
        self.block_angle = None
        self.target_pos = None
        self.target_angle = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np_rng = self.np_random

        margin = 80.0
        # Random agent position
        self.agent_pos = np_rng.uniform(margin, self.area_size - margin, size=2).astype(np.float32)

        # Random block position and angle
        self.block_pos = np_rng.uniform(margin + 50, self.area_size - margin - 50, size=2).astype(
            np.float32
        )
        self.block_angle = float(np_rng.uniform(-np.pi, np.pi))

        # Target: different position and angle
        self.target_pos = np.array(
            [self.area_size / 2, self.area_size / 2], dtype=np.float32
        )
        self.target_angle = np.float32(0.0)

        self.steps = 0
        obs = self._render()
        return obs, self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Move agent
        new_agent = self.agent_pos + action * self.speed
        new_agent = np.clip(new_agent, 0.0, self.area_size)

        # Check if agent pushes block
        dist_to_block = np.linalg.norm(new_agent - self.block_pos)
        push_radius = self.agent_radius + self.block_size * 0.5

        if dist_to_block < push_radius:
            # Push direction
            push_dir = self.block_pos - new_agent
            push_dist = np.linalg.norm(push_dir)
            if push_dist > 1e-6:
                push_dir = push_dir / push_dist
                overlap = push_radius - dist_to_block
                self.block_pos = self.block_pos + push_dir * overlap * self.push_force
                # Add slight rotation from off-center push
                cross = (new_agent[0] - self.block_pos[0]) * push_dir[1] - (
                    new_agent[1] - self.block_pos[1]
                ) * push_dir[0]
                self.block_angle += cross * 0.002

            self.block_pos = np.clip(self.block_pos, self.block_size, self.area_size - self.block_size)

        self.agent_pos = new_agent
        self.steps += 1

        # Check success
        pos_dist = np.linalg.norm(self.block_pos - self.target_pos)
        angle_dist = abs(self._angle_diff(self.block_angle, self.target_angle))
        success = pos_dist < self.success_threshold and angle_dist < self.angle_threshold

        terminated = success
        truncated = self.steps >= self.max_steps

        obs = self._render()
        return obs, 0.0, terminated, truncated, self._get_info()

    def _angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    def _get_info(self):
        return {
            "agent_pos": self.agent_pos.copy(),
            "block_pos": self.block_pos.copy(),
            "block_angle": float(self.block_angle),
            "target_pos": self.target_pos.copy(),
            "target_angle": float(self.target_angle),
            "success": False,
        }

    def _render(self) -> np.ndarray:
        """Render environment as RGB image."""
        img = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 255
        scale = self.render_size / self.area_size

        # Draw target T (light gray)
        self._draw_t(img, self.target_pos, self.target_angle, scale, [200, 200, 200])

        # Draw block T (orange)
        self._draw_t(img, self.block_pos, self.block_angle, scale, [230, 130, 50])

        # Draw agent (blue dot)
        ax, ay = int(self.agent_pos[0] * scale), int(self.agent_pos[1] * scale)
        r = max(2, int(self.agent_radius * scale))
        self._draw_circle(img, ax, ay, r, [50, 100, 230])

        return img

    def _draw_t(self, img, pos, angle, scale, color):
        """Draw a T-shaped block."""
        H, W = img.shape[:2]
        s = self.block_size * scale

        # T-shape: horizontal top bar + vertical stem
        # Top bar: width=s, height=s/3
        # Stem: width=s/3, height=s*2/3
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Define T in local coords (centered at origin)
        # Top bar
        bar_pts = np.array([
            [-s / 2, -s / 2],
            [s / 2, -s / 2],
            [s / 2, -s / 2 + s / 3],
            [-s / 2, -s / 2 + s / 3],
        ])
        # Stem
        stem_pts = np.array([
            [-s / 6, -s / 2 + s / 3],
            [s / 6, -s / 2 + s / 3],
            [s / 6, s / 2],
            [-s / 6, s / 2],
        ])

        for pts in [bar_pts, stem_pts]:
            # Rotate
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = pts @ rot.T
            # Translate
            translated = rotated + pos[None, :] * scale
            self._fill_polygon(img, translated, color)

    def _fill_polygon(self, img, pts, color):
        """Fill a polygon using scanline."""
        H, W = img.shape[:2]
        pts = pts.astype(int)
        min_y = max(0, pts[:, 1].min())
        max_y = min(H - 1, pts[:, 1].max())

        for y in range(min_y, max_y + 1):
            intersections = []
            n = len(pts)
            for i in range(n):
                j = (i + 1) % n
                y1, y2 = pts[i, 1], pts[j, 1]
                if y1 == y2:
                    continue
                if min(y1, y2) <= y < max(y1, y2):
                    x = pts[i, 0] + (y - y1) * (pts[j, 0] - pts[i, 0]) / (y2 - y1)
                    intersections.append(int(x))
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x1 = max(0, intersections[k])
                x2 = min(W - 1, intersections[k + 1])
                img[y, x1 : x2 + 1] = color

    def _draw_circle(self, img, cx, cy, r, color):
        """Draw a filled circle."""
        H, W = img.shape[:2]
        y, x = np.ogrid[:H, :W]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        img[mask] = color

    def render(self):
        return self._render()

    def get_goal_obs(self) -> np.ndarray:
        """Render goal state (block at target position/angle, agent at target)."""
        saved_agent = self.agent_pos.copy()
        saved_block = self.block_pos.copy()
        saved_angle = self.block_angle

        self.block_pos = self.target_pos.copy()
        self.block_angle = self.target_angle
        self.agent_pos = self.target_pos.copy()

        obs = self._render()

        self.agent_pos = saved_agent
        self.block_pos = saved_block
        self.block_angle = saved_angle
        return obs
