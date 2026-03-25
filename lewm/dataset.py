"""
Trajectory dataset for LeWorldModel training.

Handles data collection from environments and loading trajectories
with frame skipping and sub-trajectory sampling.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TrajectoryDataset(Dataset):
    """Dataset of observation-action trajectories.

    Each item is a sub-trajectory of T frames and T action blocks,
    with frame skip applied (grouping consecutive actions).

    Args:
        observations: list of (episode_len, H, W, C) arrays.
        actions: list of (episode_len, A) arrays.
        sub_traj_len: number of frames per sub-trajectory.
        frame_skip: number of raw frames per sub-trajectory step.
        img_size: resize observations to (img_size, img_size).
    """

    def __init__(
        self,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        sub_traj_len: int = 4,
        frame_skip: int = 5,
        img_size: int = 224,
    ):
        self.sub_traj_len = sub_traj_len
        self.frame_skip = frame_skip
        self.img_size = img_size

        # Build index of valid sub-trajectory starting points
        self.indices = []  # (episode_idx, start_step)
        self.observations = observations
        self.actions = actions

        for ep_idx, obs in enumerate(observations):
            ep_len = len(obs)
            # Need sub_traj_len * frame_skip raw steps
            required = sub_traj_len * frame_skip
            for start in range(0, ep_len - required + 1, frame_skip):
                self.indices.append((ep_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]

        obs_frames = []
        action_blocks = []

        for t in range(self.sub_traj_len):
            frame_idx = start + t * self.frame_skip
            # Get observation frame
            obs = self.observations[ep_idx][frame_idx]  # (H, W, C)
            obs_frames.append(obs)

            # Get action block (average of frame_skip actions)
            act_start = frame_idx
            act_end = min(frame_idx + self.frame_skip, len(self.actions[ep_idx]))
            action_block = self.actions[ep_idx][act_start:act_end].mean(axis=0)
            action_blocks.append(action_block)

        # Stack
        obs_tensor = np.stack(obs_frames)  # (T, H, W, C)
        act_tensor = np.stack(action_blocks)  # (T, A)

        # Convert to torch: (T, C, H, W) float [0, 1]
        obs_tensor = torch.from_numpy(obs_tensor).float().permute(0, 3, 1, 2) / 255.0
        act_tensor = torch.from_numpy(act_tensor).float()

        # Resize if needed
        if obs_tensor.shape[-1] != self.img_size:
            obs_tensor = torch.nn.functional.interpolate(
                obs_tensor, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        return obs_tensor, act_tensor


def collect_trajectories(
    env_name: str,
    n_episodes: int = 1000,
    save_dir: str | None = None,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect trajectories using a heuristic policy.

    Args:
        env_name: 'two_room' or 'push_t'.
        n_episodes: number of episodes to collect.
        save_dir: directory to save collected data.
        seed: random seed.

    Returns:
        observations: list of (T, H, W, C) arrays.
        actions: list of (T, A) arrays.
    """
    from lewm.envs import TwoRoomEnv, PushTEnv

    if env_name == "two_room":
        env = TwoRoomEnv()
    elif env_name == "push_t":
        env = PushTEnv()
    else:
        raise ValueError(f"Unknown env: {env_name}")

    all_obs = []
    all_actions = []

    for ep in tqdm(range(n_episodes), desc=f"Collecting {env_name} data"):
        obs, info = env.reset(seed=seed + ep)
        episode_obs = [obs]
        episode_actions = []

        done = False
        while not done:
            if env_name == "two_room":
                action = _two_room_heuristic(info, env)
            else:
                action = _push_t_heuristic(info, env)

            action = action + np.random.randn(*action.shape) * 0.2
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_obs.append(obs)
            episode_actions.append(action)

        all_obs.append(np.stack(episode_obs))
        all_actions.append(np.stack(episode_actions))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{env_name}_data.npz")
        # Use object arrays for ragged episode lengths
        obs_arr = np.empty(len(all_obs), dtype=object)
        act_arr = np.empty(len(all_actions), dtype=object)
        for i in range(len(all_obs)):
            obs_arr[i] = all_obs[i]
            act_arr[i] = all_actions[i]
        np.savez_compressed(save_path, observations=obs_arr, actions=act_arr)
        print(f"Saved {len(all_obs)} episodes to {save_dir}")

    return all_obs, all_actions


def _two_room_heuristic(info, env):
    """Simple heuristic: move toward door, then toward target."""
    agent = info["agent_pos"]
    target = info["target_pos"]
    door_y = (env.door_y_min + env.door_y_max) / 2
    door_pos = np.array([env.wall_x, door_y])

    # Check if agent and target are in same room
    same_side = (agent[0] < env.wall_x) == (target[0] < env.wall_x)

    if same_side:
        # Go directly to target
        direction = target - agent
    else:
        # Go to door first
        dist_to_door = np.linalg.norm(agent - door_pos)
        if dist_to_door > 20:
            direction = door_pos - agent
        else:
            direction = target - agent

    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    return direction.astype(np.float32)


def _push_t_heuristic(info, env):
    """Simple heuristic: move agent toward block, push toward target."""
    agent = info["agent_pos"]
    block = info["block_pos"]
    target = info["target_pos"]

    # Direction from block to target
    block_to_target = target - block
    dist_bt = np.linalg.norm(block_to_target)

    if dist_bt > 1e-6:
        push_dir = block_to_target / dist_bt
    else:
        push_dir = np.zeros(2, dtype=np.float32)

    # Position behind block (opposite to push direction)
    behind_block = block - push_dir * (env.block_size * 0.8)

    # Move toward behind-block position, then push
    agent_to_behind = behind_block - agent
    dist_ab = np.linalg.norm(agent_to_behind)

    if dist_ab > env.agent_radius * 2:
        direction = agent_to_behind / dist_ab
    else:
        direction = push_dir

    return direction.astype(np.float32)


def load_trajectories(data_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load saved trajectory data."""
    data = np.load(data_path, allow_pickle=True)
    observations = list(data["observations"])
    actions = list(data["actions"])
    return observations, actions
