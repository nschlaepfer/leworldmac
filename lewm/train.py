"""
Training loop for LeWorldModel.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pred = 0.0
    total_sigreg = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for obs, actions in pbar:
        obs = obs.to(device)  # (B, T, C, H, W)
        actions = actions.to(device)  # (B, T, A)

        optimizer.zero_grad()
        out = model(obs, actions)
        loss = out["loss"]
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_pred += out["pred_loss"].item()
        total_sigreg += out["sigreg_loss"].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss / n_batches:.4f}",
            "pred": f"{total_pred / n_batches:.4f}",
            "sig": f"{total_sigreg / n_batches:.4f}",
        })

    return {
        "loss": total_loss / n_batches,
        "pred_loss": total_pred / n_batches,
        "sigreg_loss": total_sigreg / n_batches,
    }


def train(model, dataset, config, device):
    """Full training procedure."""
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 128),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("epochs", 10) * len(dataloader),
    )

    history = []
    for epoch in range(1, config.get("epochs", 10) + 1):
        metrics = train_epoch(model, dataloader, optimizer, scheduler, device, epoch)
        history.append(metrics)
        print(
            f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
            f"pred={metrics['pred_loss']:.4f}, sigreg={metrics['sigreg_loss']:.4f}"
        )

    return history
