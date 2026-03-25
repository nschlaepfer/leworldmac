"""
SIGReg: Sketched-Isotropic-Gaussian Regularizer.

Encourages latent embeddings to match an isotropic Gaussian distribution
by projecting onto random directions and testing normality via the
Epps-Pulley test statistic.
"""

import torch
import torch.nn.functional as F
import math


def _epps_pulley(h: torch.Tensor, n_quadrature: int = 50) -> torch.Tensor:
    """Compute the Epps-Pulley test statistic for a batch of 1D projections.

    Tests whether the distribution of h matches N(0,1).

    Args:
        h: (N,) or (N, M) tensor of 1D projected values.
           N = number of samples, M = number of projections.
        n_quadrature: number of quadrature nodes for numerical integration.

    Returns:
        Scalar or (M,) tensor of test statistics.
    """
    if h.dim() == 1:
        h = h.unsqueeze(-1)  # (N, 1)

    N = h.shape[0]

    # Standardize: zero mean, unit variance (makes the test more robust)
    h = h - h.mean(dim=0, keepdim=True)
    std = h.std(dim=0, keepdim=True).clamp(min=1e-8)
    h = h / std

    # Quadrature nodes uniformly in [0.2, 4] as specified in paper
    t = torch.linspace(0.2, 4.0, n_quadrature, device=h.device, dtype=h.dtype)  # (Q,)

    # Weighting function: w(t) = exp(-t^2 / (2 * lambda^2)), lambda=1
    w = torch.exp(-t ** 2 / 2.0)  # (Q,)

    # Empirical characteristic function: phi_N(t; h) = (1/N) sum_n exp(i*t*h_n)
    # For real computation: split into cos and sin parts
    # t: (Q,), h: (N, M) -> th: (Q, N, M)
    th = t[:, None, None] * h[None, :, :]  # (Q, N, M)

    ecf_real = torch.cos(th).mean(dim=1)  # (Q, M)
    ecf_imag = torch.sin(th).mean(dim=1)  # (Q, M)

    # Target characteristic function of N(0,1): phi_0(t) = exp(-t^2/2)
    phi0 = torch.exp(-t ** 2 / 2.0)  # (Q,)

    # |phi_N(t) - phi_0(t)|^2
    diff_real = ecf_real - phi0[:, None]
    diff_sq = diff_real ** 2 + ecf_imag ** 2  # (Q, M)

    # Integrate: sum w(t) * |phi_N - phi_0|^2 * dt using trapezoid rule
    dt = (4.0 - 0.2) / (n_quadrature - 1)
    integrand = w[:, None] * diff_sq  # (Q, M)
    # Trapezoid rule
    result = (integrand[:-1] + integrand[1:]).sum(dim=0) * dt / 2.0  # (M,)

    return result


def sigreg(Z: torch.Tensor, n_projections: int = 1024, n_quadrature: int = 50) -> torch.Tensor:
    """Compute the SIGReg regularization term.

    Args:
        Z: (N, D) tensor of latent embeddings.
           N = total number of samples (e.g., batch * time), D = embedding dim.
        n_projections: M, number of random projection directions.
        n_quadrature: number of nodes for Epps-Pulley quadrature.

    Returns:
        Scalar SIGReg loss.
    """
    N, D = Z.shape

    # Sample M random unit-norm directions on S^{D-1}
    U = torch.randn(D, n_projections, device=Z.device, dtype=Z.dtype)
    U = F.normalize(U, dim=0)  # (D, M)

    # Project: h^(m) = Z @ u^(m), shape (N, M)
    H = Z @ U  # (N, M)

    # Compute EP test statistic for each projection
    T = _epps_pulley(H, n_quadrature=n_quadrature)  # (M,)

    return T.mean()
