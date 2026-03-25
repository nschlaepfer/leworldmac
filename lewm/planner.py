"""
CEM (Cross-Entropy Method) Planner for latent planning with MPC.

Given a world model, initial observation, and goal observation,
optimizes an action sequence to reach the goal in latent space.
"""

import torch
import torch.nn.functional as F


class CEMPlanner:
    """Cross-Entropy Method planner for LeWorldModel.

    At each planning step:
    1. Sample N candidate action sequences from N(mu, sigma)
    2. Roll out each sequence in the world model latent space
    3. Compute cost = ||z_H - z_goal||^2
    4. Select top K elites
    5. Update mu, sigma from elites
    6. Repeat for T iterations
    """

    def __init__(
        self,
        world_model,
        action_dim: int = 2,
        horizon: int = 5,
        n_samples: int = 300,
        n_elites: int = 30,
        n_iterations: int = 30,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: torch.device | None = None,
    ):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.action_low = action_low
        self.action_high = action_high
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def plan(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Plan an action sequence to reach z_goal from z_init.

        Args:
            z_init: (D,) or (1, D) initial latent state.
            z_goal: (D,) or (1, D) goal latent state.

        Returns:
            best_actions: (H, A) optimized action sequence.
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)

        H, A = self.horizon, self.action_dim
        N = self.n_samples

        # Initialize sampling distribution
        mu = torch.zeros(H, A, device=self.device)
        sigma = torch.ones(H, A, device=self.device)

        for iteration in range(self.n_iterations):
            # Sample candidate action sequences: (N, H, A)
            noise = torch.randn(N, H, A, device=self.device)
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            actions = actions.clamp(self.action_low, self.action_high)

            # Roll out each sequence in the world model
            costs = self._rollout_costs(z_init, z_goal, actions)  # (N,)

            # Select elites
            _, elite_idx = costs.topk(self.n_elites, largest=False)
            elite_actions = actions[elite_idx]  # (K, H, A)

            # Update distribution
            mu = elite_actions.mean(dim=0)  # (H, A)
            sigma = elite_actions.std(dim=0).clamp(min=1e-6)  # (H, A)

        return mu  # (H, A) best action sequence

    def _rollout_costs(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out action sequences and compute costs.

        Maintains a rolling history window during the rollout so the
        predictor can use its full context (matching paper's history
        length of 3 for PushT/OGBench, 1 for TwoRoom).

        Args:
            z_init: (1, D) initial state.
            z_goal: (1, D) goal state.
            actions: (N, H, A) candidate action sequences.

        Returns:
            costs: (N,) cost for each sequence.
        """
        N, H, A = actions.shape
        D = z_init.shape[-1]

        # Expand z_init for all candidates
        z = z_init.expand(N, -1)  # (N, D)

        # Accumulate history for predictor context
        z_history = []  # list of (N, D) tensors
        a_history = []  # list of (N, A) tensors

        for t in range(H):
            a_t = actions[:, t, :]  # (N, A)

            if len(z_history) > 0:
                hist_z = torch.stack(z_history, dim=1)  # (N, t, D)
                hist_a = torch.stack(a_history, dim=1)  # (N, t, A)
                z_next = self.world_model.predict_next(z, a_t, history=hist_z, history_actions=hist_a)
            else:
                z_next = self.world_model.predict_next(z, a_t)

            z_history.append(z)
            a_history.append(a_t)
            z = z_next

        # Cost: MSE to goal
        z_goal_expanded = z_goal.expand(N, -1)  # (N, D)
        costs = ((z - z_goal_expanded) ** 2).sum(dim=-1)  # (N,)

        return costs


class MPCController:
    """Model Predictive Control wrapper around CEM planner.

    Executes the first K actions from the planned sequence,
    then replans from the new observation.
    """

    def __init__(
        self,
        planner: CEMPlanner,
        world_model,
        replan_horizon: int = 5,
        frame_skip: int = 5,
    ):
        self.planner = planner
        self.world_model = world_model
        self.replan_horizon = replan_horizon
        self.frame_skip = frame_skip
        self._current_plan = None
        self._plan_step = 0

    @torch.no_grad()
    def get_action(
        self,
        obs: torch.Tensor,
        goal_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Get next action(s) using MPC.

        Args:
            obs: (C, H, W) current observation.
            goal_obs: (C, H, W) goal observation.

        Returns:
            action: (A,) action to execute.
        """
        self.world_model.eval()

        # Re-plan if needed
        if self._current_plan is None or self._plan_step >= self.replan_horizon:
            z_init = self.world_model.encode(obs.unsqueeze(0))  # (1, D)
            z_goal = self.world_model.encode(goal_obs.unsqueeze(0))  # (1, D)
            self._current_plan = self.planner.plan(z_init.squeeze(0), z_goal.squeeze(0))
            self._plan_step = 0

        action = self._current_plan[self._plan_step]
        self._plan_step += 1

        return action

    def reset(self):
        """Reset planner state for new episode."""
        self._current_plan = None
        self._plan_step = 0
