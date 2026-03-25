"""
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture.

Combines encoder and predictor, trained with two loss terms:
1. MSE prediction loss (teacher-forcing)
2. SIGReg regularization (anti-collapse)

L_LeWM = L_pred + lambda * SIGReg(Z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.encoder import ViTEncoder
from lewm.predictor import Predictor
from lewm.sigreg import sigreg


class LeWorldModel(nn.Module):
    """LeWorldModel: end-to-end JEPA world model.

    Training procedure (Algorithm 1 from paper):
        emb = encoder(obs)                    # (B, T, D)
        next_emb = predictor(emb, actions)    # (B, T, D)
        pred_loss = MSE(next_emb[:, 1:], emb[:, :-1])  -- wait, let me re-read

    Actually from the paper pseudocode:
        emb = encoder(obs)                    # (B, T, D)
        next_emb = predictor(emb, actions)    # (B, T, D)
        pred_loss = MSE(emb[:, 1:], next_emb[:, :-1])
        sigreg_loss = mean(SIGReg(emb.transpose(0,1)))
        loss = pred_loss + lambd * sigreg_loss

    The predictor at position t predicts the embedding at t+1.
    So next_emb[:, t] should match emb[:, t+1].
    Therefore: pred_loss = MSE(next_emb[:, :-1], emb[:, 1:])
    """

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 2,
        img_size: int = 224,
        patch_size: int = 14,
        pred_n_layers: int = 6,
        pred_n_heads: int = 16,
        pred_dropout: float = 0.1,
        sigreg_lambda: float = 0.1,
        sigreg_projections: int = 1024,
    ):
        super().__init__()
        self.encoder = ViTEncoder(embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
        self.predictor = Predictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            n_layers=pred_n_layers,
            n_heads=pred_n_heads,
            dropout=pred_dropout,
        )
        self.sigreg_lambda = sigreg_lambda
        self.sigreg_projections = sigreg_projections
        self.embed_dim = embed_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Forward pass computing all losses.

        Args:
            obs: (B, T, C, H, W) raw pixel observations in [0, 1].
            actions: (B, T, A) action sequences.

        Returns:
            dict with 'loss', 'pred_loss', 'sigreg_loss', 'emb'.
        """
        B, T, C, H, W = obs.shape

        # Encode all observations
        emb = self.encoder(obs)  # (B, T, D)

        # Predict next embeddings
        next_emb = self.predictor(emb, actions)  # (B, T, D)

        # Prediction loss: MSE between predicted and actual next embedding
        # next_emb[:, t] predicts emb[:, t+1]
        pred_loss = F.mse_loss(next_emb[:, :-1], emb[:, 1:])

        # SIGReg: step-wise regularization
        # Apply SIGReg per timestep and average (as in paper pseudocode)
        sigreg_losses = []
        for t in range(T):
            sigreg_t = sigreg(emb[:, t], n_projections=self.sigreg_projections)
            sigreg_losses.append(sigreg_t)
        sigreg_loss = torch.stack(sigreg_losses).mean()

        loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "emb": emb.detach(),
        }

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations (for planning).

        Args:
            obs: (B, C, H, W) or (B, T, C, H, W)

        Returns:
            z: (B, D) or (B, T, D) latent embeddings.
        """
        return self.encoder(obs)

    def predict_next(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next latent state (for planning).

        Args:
            z: (B, D) current latent.
            action: (B, A) action.
            history: optional (B, H, D) past states.
            history_actions: optional (B, H, A) past actions.

        Returns:
            z_next: (B, D) predicted next state.
        """
        return self.predictor.predict_step(z, action, history, history_actions)
