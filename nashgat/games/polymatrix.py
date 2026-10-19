"""Graphical (polymatrix) games with exact exploitability and a differentiable
NashGAT-style objective.

Agent i chooses a_i in {0..m-1}; its payoff is
    u_i(a) = b_i[a_i] + sum_{j in N(i)} A_ij[a_i, a_j].
For independent mixed strategies x = (x_1..x_n) the action values are
    q_i(a_i) = b_i[a_i] + sum_j (A_ij x_j)[a_i]
so every quantity used by the method (J_i, best responses, NashConv, local gap)
is available in closed form. In this stateless setting the local gap equals
exploitability exactly (Proposition 1 in MATHEMATICAL_MODEL.md with C_i = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import torch


@dataclass
class PolymatrixGame:
    n_agents: int
    n_actions: int
    edges: list[tuple[int, int]]  # undirected
    pair_payoff: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)  # (i,j) -> (m,m) payoff of i
    bias: np.ndarray | None = None  # (n, m)

    def neighbours(self, i: int) -> list[int]:
        return [j for (a, b) in self.edges for (u, j) in ((a, b), (b, a)) if u == i]

    def action_values(self, x: np.ndarray) -> np.ndarray:
        """q[i, a] for mixed profile x of shape (n, m)."""
        q = np.zeros((self.n_agents, self.n_actions)) if self.bias is None else self.bias.copy()
        for (i, j), M in self.pair_payoff.items():
            q[i] += M @ x[j]
        return q

    def payoffs(self, x: np.ndarray) -> np.ndarray:
        return (self.action_values(x) * x).sum(1)

    def nash_conv(self, x: np.ndarray) -> float:
        q = self.action_values(x)
        return float((q.max(1) - (q * x).sum(1)).sum())

    def welfare(self, x: np.ndarray) -> float:
        return float(self.payoffs(x).sum())

    def pure_nash_equilibria(self, max_profiles: int = 2**16) -> list[tuple[int, ...]]:
        """Brute-force enumeration (only for small games)."""
        assert self.n_actions**self.n_agents <= max_profiles, "game too large for enumeration"
        eq = []
        eye = np.eye(self.n_actions)
        for prof in product(range(self.n_actions), repeat=self.n_agents):
            x = eye[list(prof)]
            if self.nash_conv(x) < 1e-9:
                eq.append(prof)
        return eq

    # ---- torch versions used by the gradient-based learners ----
    def dense_payoff(self) -> np.ndarray:
        """(n, n, m, m) tensor with M[i, j] = A_ij for neighbours and 0 otherwise."""
        M = np.zeros((self.n_agents, self.n_agents, self.n_actions, self.n_actions))
        for (i, j), P in self.pair_payoff.items():
            M[i, j] = P
        return M

    def torch_action_values(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_dense"):
            self._dense = torch.as_tensor(self.dense_payoff())
        q = torch.einsum("ijab,jb->ia", self._dense.to(x.dtype), x)
        if self.bias is not None:
            q = q + torch.as_tensor(self.bias, dtype=x.dtype)
        return q


def networked_prisoners_dilemma(rows: int, cols: int, T: float = 5.0, R: float = 3.0,
                                P: float = 1.0, S: float = 0.0) -> PolymatrixGame:
    """Pairwise PD on a grid. Defect (action 1) is strictly dominant, so the
    unique Nash equilibrium is all-defect, while all-cooperate maximises welfare."""
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    M = np.array([[R, S], [T, P]], dtype=float)
    pp = {}
    for i, j in edges:
        pp[(i, j)] = M.copy()
        pp[(j, i)] = M.copy()
    return PolymatrixGame(n, 2, edges, pp)


def grid_anti_coordination(rows: int, cols: int, gain: float = 1.0) -> PolymatrixGame:
    """Agents gain when they pick a different action from each neighbour
    (two-colouring). On a bipartite grid the two checkerboard profiles are pure Nash
    equilibria and also welfare-optimal; used to check that the regulariser does not
    move learners away from an equilibrium that is already efficient."""
    base = networked_prisoners_dilemma(rows, cols)
    M = np.array([[0.0, gain], [gain, 0.0]])
    pp = {k: M.copy() for k in base.pair_payoff}
    return PolymatrixGame(base.n_agents, 2, base.edges, pp)


def train_softmax_profile(game: PolymatrixGame, objective: str, lam: float = 0.0, steps: int = 3000,
                          lr: float = 0.1, seed: int = 0, init_scale: float = 0.1) -> dict:
    """Exact-gradient learning of independent softmax policies.

    objective:
      'independent' : each agent ascends its own payoff (simultaneous gradient play)
      'welfare'     : ascend sum_i J_i
      'welfare_gap' : ascend sum_i J_i - lam * sum_i gap_i, where
                      gap_i = max_a q_i(a) - x_i . q_i with q_i treated as fixed
                      (stop-gradient through the other agents), mirroring the
                      critic-based regulariser used in the traffic experiments.
      'welfare_mix' : ascend sum_i J_i + lam * sum_i x_i . q_i(fixed); its gradient is
                      identical to 'welfare_gap' (Proposition 4).
      'welfare_fullgap': ascend sum_i J_i - lam * NashConv(x) with the exact gradient.
    """
    g = torch.Generator().manual_seed(seed)
    logits = (init_scale * torch.randn(game.n_agents, game.n_actions, generator=g, dtype=torch.float64)).requires_grad_()
    opt = torch.optim.Adam([logits], lr=lr)
    hist = []
    for t in range(steps):
        x = torch.softmax(logits, -1)
        if objective == "independent":
            # each agent differentiates only its own payoff w.r.t. its own policy
            q = game.torch_action_values(x.detach())
            loss = -(x * q).sum()
        elif objective in ("welfare", "welfare_gap", "welfare_mix", "welfare_fullgap"):
            q = game.torch_action_values(x)
            loss = -(x * q).sum()
            if objective == "welfare_gap":
                # semi-gradient: Q_i treated as fixed, as with a detached critic
                qd = game.torch_action_values(x.detach()).detach()
                gap = qd.max(1).values - (x * qd).sum(1)
                loss = loss + lam * gap.sum()
            elif objective == "welfare_mix":
                # reward mixing: welfare + lam * each agent's own payoff (independent gradient)
                qd = game.torch_action_values(x.detach()).detach()
                loss = loss - lam * (x * qd).sum()
            elif objective == "welfare_fullgap":
                # full gradient of NashConv: also moves the *other* agents so as to remove
                # agent i's incentive to deviate (not available with a detached critic)
                gap = q.max(1).values - (x * q).sum(1)
                loss = loss + lam * gap.sum()
        else:
            raise ValueError(objective)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % max(1, steps // 100) == 0 or t == steps - 1:
            xn = x.detach().numpy()
            hist.append((t, game.welfare(xn), game.nash_conv(xn)))
    xn = torch.softmax(logits, -1).detach().numpy()
    return {"x": xn, "welfare": game.welfare(xn), "nash_conv": game.nash_conv(xn), "history": hist}
