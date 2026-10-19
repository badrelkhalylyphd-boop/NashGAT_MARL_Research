"""Encoders, decentralised actors and centralised / graph critics."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter


def mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class MLPEncoder(nn.Module):
    """No communication: h_i = f(o_i)."""

    def __init__(self, obs_dim: int, hidden: int, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x, edge_index=None, edge_attr=None, return_attention=False):
        h = self.net(x)
        return (h, None) if return_attention else h


class MeanAggEncoder(nn.Module):
    """Uniform neighbour averaging (the 'no attention' ablation): h_i = f([z_i, mean_{j in N_t(i)} z_j])."""

    def __init__(self, obs_dim: int, hidden: int, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        z = self.inp(x)
        if edge_index.numel():
            agg = scatter(z[edge_index[0]], edge_index[1], dim=0, dim_size=z.size(0), reduce="mean")
        else:
            agg = torch.zeros_like(z)
        h = self.out(torch.cat([z, agg], -1))
        return (h, None) if return_attention else h


class GATEncoder(nn.Module):
    """One round of GATv2 message passing (one-hop communication).

    With edge features the attention logit of edge j->i is
        e_ij = a^T LeakyReLU(W_s z_j + W_t z_i + W_e x_ij),
    so the weight an agent places on a neighbour depends on both agents' states and on
    the measured traffic coupling of the road joining them (strategic-influence proxy).
    """

    def __init__(self, obs_dim: int, hidden: int, heads: int = 4, edge_dim: int | None = None, layers: int = 1, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.convs = nn.ModuleList(
            [GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim, add_self_loops=True,
                       fill_value="mean" if edge_dim else 0.0) for _ in range(layers)])
        self.edge_dim = edge_dim
        self.out_dim = 2 * hidden

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        z = self.inp(x)
        h = z
        att = None
        ea = edge_attr if self.edge_dim else None
        for conv in self.convs:
            if return_attention:
                h2, att = conv(h, edge_index, edge_attr=ea, return_attention_weights=True)
            else:
                h2 = conv(h, edge_index, edge_attr=ea)
            h = torch.relu(h2) + h
        out = torch.cat([z, h], -1)  # keep the agent's own features explicitly
        return (out, att) if return_attention else out


ENCODERS = {"mlp": MLPEncoder, "mean": MeanAggEncoder, "gat": GATEncoder}


class Actor(nn.Module):
    """Decentralised policy pi_theta(a_i, g_i | h_i). Parameters are shared across agents."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int, encoder: str, edge_dim: int | None,
                 heads: int = 4, comm_gate: bool = False):
        super().__init__()
        self.encoder = ENCODERS[encoder](obs_dim=obs_dim, hidden=hidden, heads=heads, edge_dim=edge_dim)
        self.pi = mlp(self.encoder.out_dim, hidden, n_actions)
        self.comm_gate = comm_gate
        self.gate = mlp(self.encoder.out_dim, hidden, 1) if comm_gate else None
        # bias the gate towards transmitting at initialisation so early learning sees messages
        if self.gate is not None:
            nn.init.constant_(self.gate[-1].bias, 2.0)

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        h, att = self.encoder(x, edge_index, edge_attr, return_attention=True)
        logits = self.pi(h)
        gate_logit = self.gate(h).squeeze(-1) if self.gate is not None else None
        return (logits, gate_logit, att) if return_attention else (logits, gate_logit)


class LocalCritic(nn.Module):
    """IPPO critic: V_i(o_i, t) and Q_i(o_i, t, .)."""

    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: int, **_):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, n_actions)

    def forward(self, obs, tfeat, edge_index=None, edge_attr=None):
        # obs (B, N, D), tfeat (B,)
        B, N, _ = obs.shape
        h = self.trunk(torch.cat([obs, tfeat.view(B, 1, 1).expand(B, N, 1)], -1))
        return self.v(h).squeeze(-1), self.q(h)


class CentralCritic(nn.Module):
    """MAPPO critic on the concatenated global observation plus an agent one-hot."""

    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: int, **_):
        super().__init__()
        self.n = n_agents
        self.trunk = nn.Sequential(nn.Linear(obs_dim * n_agents + n_agents + 1, 2 * hidden), nn.ReLU(),
                                   nn.Linear(2 * hidden, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, n_actions)

    def forward(self, obs, tfeat, edge_index=None, edge_attr=None):
        B, N, D = obs.shape
        g = obs.reshape(B, 1, N * D).expand(B, N, N * D)
        eye = torch.eye(N, device=obs.device).unsqueeze(0).expand(B, N, N)
        h = self.trunk(torch.cat([g, eye, tfeat.view(B, 1, 1).expand(B, N, 1)], -1))
        return self.v(h).squeeze(-1), self.q(h)


class GraphCritic(nn.Module):
    """Graph-conditioned centralised critic: two GATv2 layers over the full static
    graph (training-time information only) plus a global mean-pooled context."""

    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: int, edge_dim: int | None = None,
                 heads: int = 4, **_):
        super().__init__()
        self.enc = GATEncoder(obs_dim + 1, hidden, heads=heads, edge_dim=edge_dim, layers=2)
        d = self.enc.out_dim
        self.trunk = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, n_actions)

    def forward(self, obs, tfeat, edge_index, edge_attr=None):
        B, N, D = obs.shape
        x = torch.cat([obs, tfeat.view(B, 1, 1).expand(B, N, 1)], -1).reshape(B * N, D + 1)
        h = self.enc(x, edge_index, edge_attr).view(B, N, -1)
        g = h.mean(1, keepdim=True).expand_as(h)
        z = self.trunk(torch.cat([h, g], -1))
        return self.v(z).squeeze(-1), self.q(z)


CRITICS = {"local": LocalCritic, "central": CentralCritic, "graph": GraphCritic}
