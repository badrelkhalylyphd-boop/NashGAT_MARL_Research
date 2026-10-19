from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from nashgat.algos.mappo import PRESETS, MAPPOTrainer, make_config
from nashgat.envs.roadnet import load_flows, load_roadnet, validate
from nashgat.envs.traffic_env import TrafficMAEnv, default_env_config
from nashgat.graph import DynamicGraphBuilder, batch_graphs
from nashgat.stats import holm, mean_ci

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/colight/data/Jinan/3_4"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="benchmark data not downloaded")


@needs_data
def test_jinan_dataset_is_consistent():
    net = load_roadnet(DATA / "roadnet_3_4.json")
    rep = validate(net, load_flows(DATA / "anon_3_4_jinan_real.json"))
    assert rep["n_signalised"] == 12
    assert rep["n_vehicles"] == 6295
    assert rep["invalid_routes"] == 0 and rep["issues"] == []
    assert rep["n_phases"] == [9] and rep["n_movements"] == [12]


@needs_data
def test_env_conservation_and_shapes():
    env = TrafficMAEnv(default_env_config(ROOT, "jinan", horizon=600))
    obs, _ = env.reset()
    assert obs.shape == (12, env.obs_dim) and env.n_actions == 8
    rng = np.random.default_rng(0)
    while True:
        obs, r, _, trunc, info = env.step(rng.integers(0, env.n_actions, env.n_agents))
        assert (r <= 0).all()
        sim = env.sim
        # vehicle conservation: departed = finished + on roads + waiting at entry
        on_roads = int(sim.count.sum())
        assert sim.next_dep == int((sim.finish[: sim.next_dep] >= 0).sum()) + on_roads + info.get("metrics", sim.metrics())["entry_backlog"]
        assert (sim.count <= sim.cap).all()
        if trunc:
            break
    assert env.sim.t == 600


@needs_data
def test_simulator_is_deterministic():
    def run():
        env = TrafficMAEnv(default_env_config(ROOT, "jinan", horizon=300))
        env.reset()
        for k in range(30):
            env.step(np.full(12, k % 8))
        return env.metrics()
    assert run() == run()


def test_dynamic_graph_gating_and_cost():
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    g = DynamicGraphBuilder(ei, 3, gated=True)
    assert g.active_mask().all()
    g.update(np.array([1, 0, 1]))
    m = g.active_mask()
    assert m.tolist() == [True, False, False, True]
    assert g.comm_cost(m).tolist() == [1.0, 0.0, 1.0]


def test_batch_graphs_offsets():
    ei = torch.tensor([[0, 1], [1, 0]])
    masks = torch.tensor([[True, False], [True, True]])
    out, ea = batch_graphs(ei, 2, masks, torch.ones(2, 2, 3))
    assert out.tolist() == [[0, 2, 3], [1, 3, 2]]
    assert ea.shape == (3, 3)


@needs_data
@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_method_runs_one_update(preset):
    torch.manual_seed(0)
    env = TrafficMAEnv(default_env_config(ROOT, "jinan", horizon=200))
    tr = MAPPOTrainer(env, make_config(preset), seed=0)
    hist = tr.train(1)
    assert np.isfinite(hist[-1]["loss_v"]) and hist[-1]["loss_gap"] >= 0
    ev = tr.evaluate(1)
    assert len(ev["J_per_agent"]) == 12


@needs_data
def test_gap_regulariser_gradient_points_to_greedy_action():
    """With lambda only (no PG), the actor should move probability mass towards argmax Q."""
    torch.manual_seed(0)
    env = TrafficMAEnv(default_env_config(ROOT, "jinan", horizon=100))
    tr = MAPPOTrainer(env, make_config("nash_mappo"), seed=0)
    obs = torch.rand(1, 12, env.obs_dim)  # distinct inputs: at reset all agents observe the same vector
    E = tr.edge_index.shape[1]
    m = torch.zeros(1, E, dtype=torch.bool)
    ea = torch.zeros(1, E, 2)
    q = torch.randn(1, 12, 8)
    opt = torch.optim.Adam(tr.actor.parameters(), lr=0.01)
    best = q.argmax(-1, keepdim=True)

    def gap_and_pbest():
        logits, _ = tr._actor_forward(tr.actor, obs, m, ea)
        pi = torch.softmax(logits, -1)
        return (q.max(-1).values - (pi * q).sum(-1)).mean(), pi.gather(-1, best).mean()

    g0, p0 = gap_and_pbest()
    for _ in range(300):
        gap, _ = gap_and_pbest()
        opt.zero_grad()
        gap.backward()
        opt.step()
    g1, p1 = gap_and_pbest()
    # the surrogate is linear in pi, so its softmax gradient is a policy-improvement step towards
    # argmax_a Q; it can stall near deterministic suboptimal actions (vanishing softmax gradients),
    # hence we test a large reduction rather than exact greedy convergence.
    assert g1 < 0.8 * g0
    assert p1 > p0


def test_stats_helpers():
    m, s, lo, hi = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert m == 2.0 and lo < 2.0 < hi
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
