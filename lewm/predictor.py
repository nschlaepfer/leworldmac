"""
Transformer Predictor with Adaptive Layer Normalization (AdaLN).

Predicts next latent state z_{t+1} given current state z_t and action a_t.
Uses causal masking for autoregressive prediction over a history of frames.

The predictor uses a ViT-S-scale internal dimension (384) while accepting
and producing embeddings at the encoder's dimension (192), matching the
paper's ~10M parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on action embeddings.

    Modulates the normalized features with scale (gamma) and shift (beta)
    predicted from the action embedding. Initialized to zero for stable training.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        # Zero-initialize so AdaLN starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) features to normalize
            cond: (B, T, cond_dim) conditioning signal (action embeddings)
        """
        x = self.norm(x)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class PredictorBlock(nn.Module):
    """Transformer block with AdaLN for action conditioning."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.adaln1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention with AdaLN
        normed = self.adaln1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # MLP with AdaLN
        normed = self.adaln2(x, cond)
        x = x + self.mlp(normed)
        return x


class ProjectionHead(nn.Module):
    """Same projection head as encoder: Linear + BatchNorm."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.bn(self.linear(x))
        return x.reshape(*shape, -1)


class Predictor(nn.Module):
    """Transformer predictor for latent dynamics.

    Architecture (from paper):
    - ViT-S backbone: 6 layers, 16 attention heads, 10% dropout (~10M parameters)
    - Internal dimension 384 (ViT-S scale), with input/output projections
      to/from the encoder's embed_dim (192)
    - AdaLN for action conditioning (zero-initialized)
    - Learned positional embeddings
    - Temporal causal masking
    - Projection head (same as encoder)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        hidden_dim: int = 384,
        action_dim: int = 2,
        n_layers: int = 6,
        n_heads: int = 16,
        dropout: float = 0.1,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Input projection: encoder dim -> predictor internal dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Action embedding to predictor internal dim
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Learned positional embeddings at internal dim
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer blocks with AdaLN at internal dim
        self.blocks = nn.ModuleList([
            PredictorBlock(hidden_dim, n_heads, cond_dim=hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection: predictor internal dim -> encoder dim
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

        # Projection head (same as encoder: Linear + BatchNorm)
        self.projection = ProjectionHead(embed_dim, embed_dim)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next latent states autoregressively.

        Args:
            z: (B, T, D) encoded frame representations (embed_dim).
            actions: (B, T, A) actions taken at each timestep.

        Returns:
            z_hat: (B, T, D) predicted next-step embeddings (embed_dim).
                   z_hat[:, t] is the prediction for z[:, t+1].
        """
        B, T, D = z.shape

        # Project to internal dimension
        x = self.input_proj(z)  # (B, T, hidden_dim)

        # Embed actions at internal dim
        action_cond = self.action_embed(actions)  # (B, T, hidden_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :T, :]

        # Causal mask
        mask = self._causal_mask(T, z.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, action_cond, attn_mask=mask)

        x = self.final_norm(x)

        # Project back to encoder dimension
        x = self.output_proj(x)  # (B, T, embed_dim)

        # Projection head
        z_hat = self.projection(x)  # (B, T, embed_dim)

        return z_hat

    def predict_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-step prediction for planning (no teacher forcing).

        Args:
            z: (B, D) current latent state.
            action: (B, A) action to take.
            history: (B, H, D) optional history of previous states.
            history_actions: (B, H, A) optional history of previous actions.

        Returns:
            z_next: (B, D) predicted next latent state.
        """
        if history is not None and history_actions is not None:
            all_z = torch.cat([history, z.unsqueeze(1)], dim=1)
            all_a = torch.cat([history_actions, action.unsqueeze(1)], dim=1)
        else:
            all_z = z.unsqueeze(1)
            all_a = action.unsqueeze(1)

        z_hat = self.forward(all_z, all_a)
        return z_hat[:, -1, :]
