#!/usr/bin/env python3
"""
Evaluate LeWorldModel via latent planning with CEM + MPC.

Usage:
    python plan.py --env two_room --checkpoint checkpoints/lewm_two_room.pt --episodes 10
"""

import argparse
import os
import yaml
import torch
import numpy as np

from lewm.utils import get_device
from lewm.world_model import LeWorldModel
from lewm.planner import CEMPlanner, MPCController
from lewm.envs import TwoRoomEnv, PushTEnv


def make_env(env_name):
    if env_name == "two_room":
        return TwoRoomEnv()
    elif env_name == "push_t":
        return PushTEnv()
    raise ValueError(f"Unknown env: {env_name}")


def obs_to_tensor(obs, device):
    """Convert (H, W, C) uint8 observation to (C, H, W) float tensor."""
    t = torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0
    # Resize to 224x224 if needed
    if t.shape[-1] != 224:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
    return t.to(device)


def evaluate(model, env_name, config, device, n_episodes=10, max_steps=200, seed=42):
    """Evaluate the model with CEM planning."""
    env = make_env(env_name)

    planner = CEMPlanner(
        world_model=model,
        action_dim=config["action_dim"],
        horizon=config.get("cem_horizon", 5),
        n_samples=config.get("cem_samples", 300),
        n_elites=config.get("cem_elites", 30),
        n_iterations=config.get("cem_iterations", 10),
        action_low=-1.0,
        action_high=1.0,
        device=device,
    )

    controller = MPCController(
        planner=planner,
        world_model=model,
        replan_horizon=config.get("mpc_replan_horizon", 5),
        frame_skip=config.get("frame_skip", 5),
    )

    successes = 0
    total_steps_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        goal_obs = env.get_goal_obs()

        obs_tensor = obs_to_tensor(obs, device)
        goal_tensor = obs_to_tensor(goal_obs, device)

        controller.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = controller.get_action(obs_tensor, goal_tensor)
            action_np = action.cpu().numpy()

            # Execute action with frame_skip
            for _ in range(config.get("frame_skip", 5)):
                obs, _, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                steps += 1
                if done:
                    break

            obs_tensor = obs_to_tensor(obs, device)

        success = info.get("success", False)
        if success:
            successes += 1
        total_steps_list.append(steps)

        print(f"Episode {ep + 1}/{n_episodes}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)")

    success_rate = successes / n_episodes * 100
    avg_steps = np.mean(total_steps_list)
    print(f"\nResults: {success_rate:.1f}% success rate, {avg_steps:.1f} avg steps")
    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeWorldModel with planning")
    parser.add_argument("--env", type=str, default="two_room", choices=["two_room", "push_t"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    config["env"] = args.env

    # Create model
    model = LeWorldModel(
        embed_dim=config["embed_dim"],
        action_dim=config["action_dim"],
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        pred_n_layers=config["pred_n_layers"],
        pred_n_heads=config["pred_n_heads"],
        pred_dropout=config["pred_dropout"],
        sigreg_lambda=config["sigreg_lambda"],
        sigreg_projections=config["sigreg_projections"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    if "history" in ckpt:
        h = ckpt["history"][-1]
        print(f"Final training loss: {h['loss']:.4f} (pred={h['pred_loss']:.4f}, sig={h['sigreg_loss']:.4f})")

    evaluate(model, args.env, config, device, n_episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)


if __name__ == "__main__":
    main()
