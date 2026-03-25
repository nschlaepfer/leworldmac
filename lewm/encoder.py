"""
ViT-Tiny Encoder for LeWorldModel.

Maps raw pixel observations to compact latent representations.
Uses a Vision Transformer (ViT-Tiny) backbone followed by a
projection head (Linear + BatchNorm).
"""

import torch
import torch.nn as nn
import timm


class ProjectionHead(nn.Module):
    """1-layer MLP projection: Linear -> BatchNorm."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, in_dim)
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.bn(self.linear(x))
        return x.reshape(*shape, -1)


class ViTEncoder(nn.Module):
    """ViT-Tiny encoder with projection head.

    Architecture (from paper):
    - ViT-Tiny: patch_size=14, 12 layers, 3 attention heads, hidden_dim=192
    - [CLS] token embedding from last layer -> 192-dim
    - Projection head: Linear + BatchNorm -> 192-dim output
    """

    def __init__(self, embed_dim: int = 192, img_size: int = 224, patch_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim

        # ViT-Tiny from timm (use patch16 base, override patch_size to 14)
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,  # Remove classification head, get [CLS] token
        )

        vit_dim = self.vit.embed_dim  # Should be 192 for ViT-Tiny
        self.projection = ProjectionHead(vit_dim, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent embeddings.

        Args:
            obs: (B, T, C, H, W) or (B, C, H, W) pixel observations in [0, 1].

        Returns:
            z: (B, T, D) or (B, D) latent embeddings.
        """
        has_time = obs.dim() == 5
        if has_time:
            B, T, C, H, W = obs.shape
            obs = obs.reshape(B * T, C, H, W)

        # ViT forward -> [CLS] token
        features = self.vit(obs)  # (B*T, vit_dim)

        # Project
        z = self.projection(features)  # (B*T, embed_dim)

        if has_time:
            z = z.reshape(B, T, self.embed_dim)

        return z
