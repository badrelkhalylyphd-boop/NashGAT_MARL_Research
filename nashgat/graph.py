"""Dynamic communication graph G_t = (V, E_t) and graph batching utilities."""
from __future__ import annotations

import numpy as np
import torch


class DynamicGraphBuilder:
    """Maintains the active edge set E_t over the static road topology E_0.

    An edge j->i of E_0 is active at step t iff sender j chose to transmit at t-1
    (communication gate g_j = 1) and the link did not fail. With gating disabled all
    edges are active (static graph). Self-loops are added inside the GNN layers and
    never count as communication.
    """

    def __init__(self, edge_index: np.ndarray, n_agents: int, gated: bool, fail_prob: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.edge_index = edge_index
        self.n_agents = n_agents
        self.gated = gated
        self.fail_prob = fail_prob
        self.rng = rng or np.random.default_rng(0)
        self.out_degree = np.bincount(edge_index[0], minlength=n_agents) if edge_index.size else np.zeros(n_agents, int)
        self.reset()

    def reset(self) -> None:
        self.gates = np.ones(self.n_agents, dtype=np.int64)

    def active_mask(self) -> np.ndarray:
        mask = self.gates[self.edge_index[0]].astype(bool) if self.gated else np.ones(self.edge_index.shape[1], bool)
        if self.fail_prob > 0:
            mask &= self.rng.random(mask.shape[0]) >= self.fail_prob
        return mask

    def update(self, gates: np.ndarray | None) -> None:
        if self.gated and gates is not None:
            self.gates = np.asarray(gates, dtype=np.int64)

    def comm_cost(self, mask: np.ndarray) -> np.ndarray:
        """Messages sent by each agent at this step (number of active out-edges)."""
        return np.bincount(self.edge_index[0][mask], minlength=self.n_agents).astype(np.float64)


def batch_graphs(edge_index: torch.Tensor, n_nodes: int, masks: torch.Tensor,
                 edge_attr: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Disjoint union of B copies of the same node set with per-copy edge masks.

    edge_index: (2, E) base edges; masks: (B, E) bool; edge_attr: (B, E, F) or None.
    Returns edge_index (2, E_active) over B*n_nodes nodes and matching edge_attr.
    """
    B, E = masks.shape
    offsets = (torch.arange(B, device=edge_index.device) * n_nodes).view(1, B, 1)
    ei = (edge_index.view(2, 1, E) + offsets).reshape(2, B * E)
    m = masks.reshape(B * E)
    ei = ei[:, m]
    ea = edge_attr.reshape(B * E, -1)[m] if edge_attr is not None else None
    return ei, ea
