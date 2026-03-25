#!/usr/bin/env python3
"""
Train LeWorldModel from collected trajectory data.

Usage:
    python train.py --env two_room --epochs 10
    python train.py --env push_t --epochs 10 --batch_size 32
"""

import argparse
import os
import yaml
import torch

from lewm.utils import get_device
from lewm.world_model import LeWorldModel
from lewm.dataset import TrajectoryDataset, collect_trajectories, load_trajectories
from lewm.train import train


def main():
    parser = argparse.ArgumentParser(description="Train LeWorldModel")
    parser.add_argument("--env", type=str, default="two_room", choices=["two_room", "push_t"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sigreg_lambda", type=float, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    config["env"] = args.env
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.n_episodes is not None:
        config["n_episodes"] = args.n_episodes
    if args.lr is not None:
        config["lr"] = args.lr
    if args.sigreg_lambda is not None:
        config["sigreg_lambda"] = args.sigreg_lambda

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: {config}")

    # Collect or load data
    data_path = os.path.join(args.data_dir, f"{args.env}_data.npz")
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        observations, actions = load_trajectories(data_path)
    else:
        print(f"Collecting {config['n_episodes']} episodes...")
        observations, actions = collect_trajectories(
            args.env,
            n_episodes=config["n_episodes"],
            save_dir=args.data_dir,
        )

    print(f"Dataset: {len(observations)} episodes")

    # Create dataset
    dataset = TrajectoryDataset(
        observations,
        actions,
        sub_traj_len=config["sub_traj_len"],
        frame_skip=config["frame_skip"],
        img_size=config["img_size"],
    )
    print(f"Training samples: {len(dataset)}")

    # Create model
    action_dim = config["action_dim"]
    model = LeWorldModel(
        embed_dim=config["embed_dim"],
        pred_hidden_dim=config.get("pred_hidden_dim", 384),
        action_dim=action_dim,
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        pred_n_layers=config["pred_n_layers"],
        pred_n_heads=config["pred_n_heads"],
        pred_dropout=config["pred_dropout"],
        sigreg_lambda=config["sigreg_lambda"],
        sigreg_projections=config["sigreg_projections"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Train
    history = train(model, dataset, config, device)

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"lewm_{args.env}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
