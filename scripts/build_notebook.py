"""Build notebooks/NashGAT_MARL_Research.ipynb from the package sources.

The notebook is self-contained: it writes the `nashgat` package from %%writefile
cells, so it runs in Colab without cloning this repository. Re-run this script
after changing any module so the notebook stays in sync with the code.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/NashGAT_MARL_Research.ipynb"

MODULES = [
    ("nashgat/__init__.py", None),
    ("nashgat/utils.py", "Seeding, device selection and JSON helpers."),
    ("nashgat/games/__init__.py", None),
    ("nashgat/games/matrix_games.py", "Bimatrix games with closed-form Nash equilibria and NashConv (unit-test oracles)."),
    ("nashgat/games/polymatrix.py", "Graphical (polymatrix) games: exact exploitability and the gap / mixing / full-gradient learners."),
    ("nashgat/envs/__init__.py", None),
    ("nashgat/envs/roadnet.py", "CityFlow roadnet/flow parser and validator."),
    ("nashgat/envs/queue_sim.py", "**QueueSim**: the lightweight, *separate* mesoscopic simulator used when CityFlow is unavailable."),
    ("nashgat/envs/cityflow_backend.py", "CityFlow backend with the same interface (used when `import cityflow` succeeds)."),
    ("nashgat/envs/traffic_env.py", "Multi-agent environment: observations, individual rewards, edge features, robustness options."),
    ("nashgat/graph.py", "Dynamic graph $G_t=(V,E_t)$ with communication gates and link failures; graph batching."),
    ("nashgat/models.py", "Encoders (MLP / mean aggregation / GATv2 with edge features), decentralised actor, critics."),
    ("nashgat/algos/__init__.py", None),
    ("nashgat/algos/mappo.py", "Unified PPO/MAPPO trainer: IPPO, MAPPO, GNN-MAPPO, Nash-MAPPO, NashGAT-MARL and ablations."),
    ("nashgat/algos/best_response.py", "Approximate best-response learning and sampled NashConv (exploitability lower-bound estimate)."),
    ("nashgat/rule_based.py", "FixedTime and MaxPressure reference controllers."),
    ("nashgat/stats.py", "Mean, std, t-intervals, Welch / Mann-Whitney tests, Holm correction."),
    ("nashgat/experiment.py", "Single-run driver: train, evaluate (nominal + robustness + transfer), best response."),
    ("nashgat/report.py", "Aggregation into CSV tables and publication figures."),
]

cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def bib_table() -> str:
    txt = (ROOT / "references.bib").read_text(encoding="utf-8")
    rows = []
    for m in re.finditer(r"%\s*Verified:([^\n]*)\n@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        how, kind, key, body = m.groups()
        f = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", body + "\n", re.S)}
        venue = f.get("journal") or f.get("booktitle") or f.get("publisher") or f.get("note", "")
        year = f.get("year", "")
        if year in ("2025", "2026", "2027"):
            doi = f.get("doi", "")
            rows.append(f"| `{key}` | {f.get('title', '').replace('{', '').replace('}', '')} | {venue} | {year} | "
                        f"{('[' + doi + '](https://doi.org/' + doi + ')') if doi else f.get('url', '')} |")
    return "\n".join(["| key | title | venue | year | DOI / URL |", "|---|---|---|---|---|"] + rows)


# --------------------------------------------------------------------------- title
md("""
# NashGAT-MARL: dynamic graph attention and exploitability-aware MARL for traffic-signal control

**Badr El Khalyly**∗ and **Zouhair Ibn Batouta**†<br>
∗ LTIM – Laboratory of Techniques of Information and Modeling, FSBM, Hassan II University, Casablanca, Morocco<br>
† RITM – ESTC/CED, ENSEM, Hassan II University, Casablanca, Morocco<br>
badrelkhalylyphd@gmail.com · Zouhair.ibnbatouta@gmail.com

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/badrelkhalylyphd-boop/NashGAT_MARL_Research/blob/main/notebooks/NashGAT_MARL_Research.ipynb)
Code and released results: https://github.com/badrelkhalylyphd-boop/NashGAT_MARL_Research

**Research notebook.** It is reproducible and self-contained. GPU is optional: the networks are small, so CPU is often as fast.

| Mode | Purpose | Budget |
|---|---|---|
| `FAST_DEMO` | functional verification (≈10–20 min on Colab CPU) | 1 seed, short episodes, few iterations. **Numbers are NOT research results.** |
| `FULL_EXPERIMENT` | the protocol used in the paper | 5 seeds × 250 episodes × 3,600 s, robustness suite, best-response exploitability |

**Integrity rules followed here.** Every number printed is computed in this run; nothing is typed in by hand. Literature metadata was verified against Crossref, OpenAlex, DataCite, arXiv or proceedings pages on 2026-10-19. Scopus could not be accessed, so indexing is marked *requires verification*. Results from **CityFlow** and from the fallback **QueueSim** simulator are separate experimental environments and must never be pooled.
""")
code("""
# ====== experiment mode ======
MODE = "FAST_DEMO"          # "FAST_DEMO" or "FULL_EXPERIMENT"
USE_DRIVE = False           # True: mount Google Drive and copy checkpoints/results there
PREFER_CITYFLOW = True      # try to build CityFlow; fall back to QueueSim if that fails

if MODE == "FAST_DEMO":
    CFG = dict(seeds=[0], episodes=4, horizon=900, eval_episodes=1, robustness=True,
               br_agents=(0, 5), br_episodes=2, br_eval_episodes=2, gg_steps=300, gg_seeds=2,
               ablations=["nashgat_no_nash", "nashgat_static_graph"])
elif MODE == "FULL_EXPERIMENT":
    CFG = dict(seeds=[0, 1, 2, 3, 4], episodes=250, horizon=3600, eval_episodes=3, robustness=True,
               br_agents=(0, 1, 5), br_episodes=20, br_eval_episodes=5, gg_steps=1500, gg_seeds=5,
               ablations=["nashgat_no_gnn", "nashgat_no_attention", "nashgat_no_nash",
                          "nashgat_no_comm_penalty", "nashgat_static_graph"])
else:
    raise ValueError(MODE)
METHODS = ["ippo", "mappo", "gnn_mappo", "nash_mappo", "nashgat"]
print(MODE, CFG)
""")

# --------------------------------------------------------------------------- part 1
md(f"""
## Part 1 — Scientific introduction

### 1.1 Problem and motivation
Adaptive signal control is a multi-agent problem. Each intersection controller acts on local sensor data, but its decisions change the queues of its neighbours through finite road storage and spill-back. Graph-attention MARL for traffic signal control (TSC) is mature: CoLight (2019) and many 2025–2026 variants. Almost all of it optimises a **shared** reward. If controllers optimise **their own** queues, which happens with different operators or local retraining, the system is a *general-sum* Markov game. A jointly trained plan can then be *exploitable*: one controller could improve its own reward by deviating unilaterally. This notebook measures that exploitability next to the usual congestion metrics and communication cost.

### 1.2 Recent peer-reviewed or preprint references (2025–2026), all metadata verified
Scopus record-level indexing and journal quartiles: **requires verification**. Scopus access was not available, so neither is claimed. The full list of 49 verified entries is in `references.bib`, which is written in Part 9.

{bib_table()}

### 1.3 Research gap and what survives scrutiny
* We found no verified paper that combines dynamic GAT, a MAPPO-style actor-critic, an explicit exploitability regulariser and communication-budgeted execution for TSC. Each pair has precedent, though. The closest works are `fei2026gatgame`, `alqudah2026nashhgt`, the preprint `samanta2026gatmappoeig`, and the Nash-based TSC works `zhang2023nashdqn` and `yan2024dcnpg`. Their full texts must be read before any "first" claim.
* **Key negative result (Part 2, Prop. 4):** the natural differentiable exploitability surrogate with a detached critic has *exactly* the gradient of selfish policy improvement. In that form the "Nash-aware" objective is **reward mixing** between team and individual rewards. Part 4 checks this numerically.

### 1.4 Research questions and hypotheses
* **RQ1.** Does graph attention with traffic-coupling edge features improve congestion metrics over MLP actors under an equal budget?
  *H1:* GNN-MAPPO and NashGAT reach a lower average travel time (ATT) than MAPPO.
* **RQ2.** Does the gap regulariser ($\\lambda>0$) reduce measured best-response exploitability, and at what welfare cost?
  *H2:* Nash-MAPPO has a lower sampled NashConv than MAPPO.
* **RQ3.** Can learned gates cut messages while keeping ATT?
  *H3:* NashGAT sends fewer messages than GNN-MAPPO, with no significant ATT loss.
* **RQ4.** Robustness and transfer: noise, link failures, demand shifts, failed controllers, and zero-shot transfer to Hangzhou 4×4.
""")

# --------------------------------------------------------------------------- part 2
md(r"""
## Part 2 — Mathematical foundations

### 2.1 Partially observable general-sum Markov game
$\mathcal{G}=\langle\mathcal{N},\mathcal{S},\{\mathcal{O}_i\},\{\mathcal{A}_i\},P,\{r_i\},Z,\rho,\gamma\rangle$. Its components:
* agents $i\in\mathcal{N}$ are the signalised intersections (12 in Jinan 3×4);
* actions $a_i$ are the 8 admissible light phases (a change triggers a 3 s yellow), plus a communication gate $g_i\in\{0,1\}$ for NashGAT;
* the observation $o_i\in\mathbb{R}^{32}$ holds movement queues, approaching counts and the phase one-hot;
* the reward is $r_i=-\frac1{20}\sum_{m\in\mathcal{M}_i}q_m$ (own queues only, hence **general-sum**);
* the discount is $\gamma=0.99$ per 10 s decision.

Returns are $J_i(\pi)=\mathbb{E}_\pi[\sum_t\gamma^tr_i(t)]$, and welfare is $W=\frac1n\sum_iJ_i$.

### 2.2 Graph neural networks and graph attention
Communication graph: $G_t=(V,E_t)$, with $E_t=\{j\!\to\!i\in E_0: g_{j,t-1}=1,\ \text{link up}\}$. One GATv2 round with edge features $x_{ji}$ (road occupancy, queued share) gives
$$e_{ij}=\mathbf a^\top\mathrm{LeakyReLU}(W_sz_j+W_tz_i+W_ex_{ji}),\quad \alpha_{ij}=\mathrm{softmax}_j(e_{ij}),\quad h_i=[z_i\,\|\,z_i+\mathrm{ReLU}(\textstyle\sum_j\alpha_{ij}W_sz_j)].$$

### 2.3 Multi-agent PPO
Shared-parameter actor $\pi_\theta(a_i\mid h_i)$ (decentralised execution) and a centralised critic (training only). The clipped surrogate is
$$L^{\text{PPO}}=-\mathbb{E}\big[\min(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1\!\pm\!\epsilon)\hat A_t)\big],\qquad \rho_t=\frac{\pi_\theta(a_t|h_t)}{\pi_{\theta_\text{old}}(a_t|h_t)},$$
with GAE($\lambda$) advantages computed on the team reward (MAPPO family) or the individual reward (IPPO).

### 2.4 Nash equilibrium, best response, exploitability
$\pi^\star$ is a Nash equilibrium in the policy class $\Pi$ if, for all $i$ and all $\pi_i'\in\Pi_i$, $J_i(\pi_i',\pi^\star_{-i})\le J_i(\pi^\star)$. Exploitability (NashConv) is
$$\mathcal{E}(\pi)=\sum_i\big[\max_{\pi_i'\in\Pi_i}J_i(\pi_i',\pi_{-i})-J_i(\pi)\big]\ge0.$$
On the traffic game, $\mathcal{E}$ is estimated with **approximate best responses**: PPO fine-tuning of $\pi_k$ on $r_k$ while the other agents are frozen. This gives a noisy **lower** bound.

### 2.5 Local exploitability gap (critic surrogate)
$$\mathrm{gap}_i(s)=\max_a\hat Q_i(s,a)-\textstyle\sum_a\pi_i(a|h_i)\hat Q_i(s,a)\ \ge0 .$$
* **Prop. 1.** $\max_{\pi_i'}J_i(\pi_i',\pi_{-i})-J_i(\pi)\le\frac{C_i}{1-\gamma}\mathbb{E}_{d^\pi}[\mathrm{gap}_i]$, where $C_i$ is a concentrability coefficient. The proof uses the performance-difference lemma.
* **Prop. 2.** A critic error of $\varepsilon_Q$ changes the gap by at most $2\varepsilon_Q$.
* **Prop. 3.** In stateless polymatrix games, $\mathcal{E}=\sum_i\mathrm{gap}_i$ exactly.
* **Prop. 4.** With $\hat Q$ detached, $\nabla_\theta\,\lambda\sum_i\mathrm{gap}_i=-\lambda\sum_i\mathbb{E}\sum_a\nabla\pi_i\hat Q_i$, which is the selfish policy gradient, so the regulariser amounts to reward mixing.

### 2.6 NashGAT-MARL objective (research hypothesis, not a validated result)
$$\mathcal{L}=L^{\text{PPO}}_{\text{team}-\mu c}-c_e\,\mathcal{H}(\pi)+\lambda\,\frac{\mathbb{E}[\mathrm{gap}]}{\mathrm{std}(\hat A)},\qquad c_i=g_i\,\frac{\mathrm{outdeg}(i)}{\max_j\mathrm{outdeg}(j)} .$$
**Limitations.** $\mathcal{E}$ is non-differentiable; the gap is a surrogate and is conservative under partial observability; $C_i$ can be large. There is no claim of convergence to a Nash equilibrium.
""")

# --------------------------------------------------------------------------- part 3
md("""
## Part 3 — Environment set-up
Detect the hardware, install the dependencies, try to build CityFlow, download and validate the **real** CoLight benchmark files, and set the seeds.
""")
code("""
import os, sys, subprocess, json, time, shutil, platform
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
WORK = Path("/content/nashgat_work") if IN_COLAB else Path(os.environ.get("NASHGAT_WORK", "nb_work")).resolve()
RESULTS = Path("/content/results") if IN_COLAB else WORK / "results"
DATA_ROOT = Path(os.environ.get("NASHGAT_DATA_ROOT", str(WORK)))
for p in (WORK, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("python", platform.python_version(), "| colab:", IN_COLAB, "| work dir:", WORK, "| results:", RESULTS)
""")
code("""
# dependencies (torch is preinstalled on Colab; PyG wheels are pure Python since 2.3)
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric", "networkx", "pandas", "scipy", "matplotlib"], check=True)
import torch, numpy as np, pandas as pd
import torch_geometric
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| PyG", torch_geometric.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# the models are tiny (12 nodes); CPU is usually as fast as GPU and avoids host-device copies per step
DEVICE = "cpu"
""")
code("""
# CityFlow: no PyPI wheel exists; build from source (needs cmake + a C++ compiler, both present on Colab)
BACKEND = "queuesim"
if PREFER_CITYFLOW:
    try:
        import cityflow  # noqa
        BACKEND = "cityflow"
    except ImportError:
        if IN_COLAB:
            r = subprocess.run("git clone --depth 1 https://github.com/cityflow-project/CityFlow.git /content/CityFlow && "
                               "pip install -q /content/CityFlow", shell=True, capture_output=True, text=True)
            print(r.stdout[-1500:], r.stderr[-1500:])
            try:
                import importlib; importlib.invalidate_caches()
                import cityflow  # noqa
                BACKEND = "cityflow"
            except ImportError as e:
                print("CityFlow build failed:", e)
        else:
            print("CityFlow not importable outside Colab in this environment.")
print("SIMULATOR BACKEND =", BACKEND,
      "" if BACKEND == "cityflow" else "  <-- QueueSim: a SEPARATE lightweight environment; do not compare its absolute numbers with CityFlow papers")
""")
code("""
# real benchmark data from the CoLight repository (sparse checkout of data/ only)
DATA_DIR = DATA_ROOT / "data/raw/colight"
if not (DATA_DIR / "data/Jinan/3_4/roadnet_3_4.json").exists():
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    cmds = [f"git clone --depth 1 --filter=blob:none --sparse https://github.com/wingsweihua/colight.git {DATA_DIR}",
            f"git -C {DATA_DIR} sparse-checkout set data/Jinan data/Hangzhou data/NewYork/28_7"]
    for c in cmds:
        subprocess.run(c, shell=True, check=True)
commit = subprocess.run(["git", "-C", str(DATA_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("CoLight data commit:", commit)
""")

# --------------------------------------------------------------------------- part 5 (package) placed before use
md("""
## Part 5a — Implementation: write the `nashgat` package
Each module is written to disk with `%%writefile`, so the code that runs is exactly the code shown. The modules are object-oriented and typed. Their sources are identical to the repository versions.
""")
# %%writefile needs existing folders
code("for d in ['nashgat/games', 'nashgat/envs', 'nashgat/algos']:\n    os.makedirs(d, exist_ok=True)")
for rel, desc in MODULES:
    src = (ROOT / rel).read_text(encoding="utf-8")
    if desc:
        md(f"**`{rel}`** — {desc}")
    code(f"%%writefile {rel}\n{src}")
code("""
import importlib, nashgat
importlib.invalidate_caches()
from nashgat.utils import set_seed
from nashgat.envs.roadnet import load_roadnet, load_flows, validate
from nashgat.envs.traffic_env import TrafficMAEnv, default_env_config
from dataclasses import replace
set_seed(0)
print("nashgat", nashgat.__version__, "imported from", Path(nashgat.__file__).parent)
""")
md("### Dataset validation (real files)")
code("""
val = {}
for name, (rn, flows) in {
    "jinan_3x4": ("data/raw/colight/data/Jinan/3_4/roadnet_3_4.json", ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json", "anon_3_4_jinan_real_2500.json"]),
    "hangzhou_4x4": ("data/raw/colight/data/Hangzhou/4_4/roadnet_4_4.json", ["anon_4_4_hangzhou_real.json"]),
    "newyork_28x7": ("data/raw/colight/data/NewYork/28_7/roadnet_28_7.json", ["anon_28_7_newyork_real_double.json"]),
}.items():
    net = load_roadnet(DATA_ROOT / rn)
    for f in flows:
        v = validate(net, load_flows((DATA_ROOT / rn).parent / f))
        val[f] = {k: v[k] for k in ("n_signalised", "n_virtual", "n_roads", "n_phases", "n_movements", "n_vehicles",
                                     "departure_range_s", "invalid_routes")}
        assert v["invalid_routes"] == 0 and not v["issues"], (f, v["issues"])
display(pd.DataFrame(val).T)
assert val["anon_3_4_jinan_real.json"]["n_signalised"] == 12 and val["anon_3_4_jinan_real.json"]["n_vehicles"] == 6295
(RESULTS / "dataset_validation.json").write_text(json.dumps({"commit": commit, "files": val}, indent=1, default=str))
""")

# --------------------------------------------------------------------------- part 4
md("""
## Part 4 — AI patterns, demonstrated on running code
| Pattern | Kind | Role here |
|---|---|---|
| Perception–Decision–Action | architectural | `TrafficMAEnv._obs` → `Actor` → `env.step` |
| Actor–Critic | learning architecture | PPO actor + value/Q critic |
| CTDE | architectural | the critic sees global state at training; the actor sees only $h_i$ |
| Graph message passing | architectural/protocol | one GATv2 round over $E_t$ |
| Strategic best response | algorithmic | PPO best-response oracle for exploitability |
| Nash-aware optimisation | objective | gap regulariser (≡ reward mixing, Prop. 4) |
| Adaptive communication | protocol | Bernoulli gates and message penalty |

Patterns **not** used: experience replay (PPO is on-policy), BDI, Contract Net and MQTT. None has a functional role here.
""")
code("""
# Perception-Decision-Action + CTDE: one decision step; the actor input contains only local + neighbour messages
from nashgat.algos.mappo import MAPPOTrainer, make_config
env = TrafficMAEnv(replace(default_env_config(DATA_ROOT, "jinan", horizon=300), backend=BACKEND))
tr = MAPPOTrainer(env, make_config("nashgat"), seed=0)
obs, _ = env.reset()                                            # perception
o = torch.as_tensor(obs).unsqueeze(0)
mask = torch.as_tensor(tr.graph.active_mask()).unsqueeze(0)
ea = torch.as_tensor(env.edge_features()).unsqueeze(0)
logits, gate_logit, att = tr._actor_forward(tr.actor, o, mask, ea, return_attention=True)   # decision
a = logits[0].argmax(-1).numpy()
obs2, r, *_ = env.step(a)                                       # action
print("obs", obs.shape, "| logits", tuple(logits.shape), "| gates", tuple(gate_logit.shape), "| reward", r.round(2))
# CTDE check: changing a NON-neighbour's observation must not change agent 0's action distribution
nbrs0 = set(env.edge_index[0][env.edge_index[1] == 0].tolist())
far = [j for j in range(env.n_agents) if j not in nbrs0 | {0}][0]
o2 = o.clone(); o2[0, far] += 5.0
l2, _ = tr._actor_forward(tr.actor, o2, mask, ea)
assert torch.allclose(logits[0, 0], l2[0, 0]), "agent 0 must only depend on its 1-hop neighbourhood"
print(f"decentralisation check passed: agent 0 is unaffected by non-neighbour {far}")
""")
code("""
# Graph message passing: attention weights of agent 5 (an interior intersection) over its neighbours
ei, alpha = att[0].numpy(), att[1].mean(-1).detach().numpy()
sel = ei[1] == 5
print(pd.DataFrame({"sender": ei[0][sel], "receiver": ei[1][sel], "alpha(head mean)": alpha[sel].round(3)}))
assert abs(alpha[sel].sum() - 1.0) < 1e-5   # attention is a distribution over the neighbourhood (incl. self-loop)
""")
code("""
# Strategic best response + Nash equilibrium on 2x2 games (closed form)
from nashgat.games.matrix_games import prisoners_dilemma, matching_pennies, battle_of_sexes, pure_nash_equilibria, mixed_nash_2x2
C, D = np.array([1., 0.]), np.array([0., 1.])
pd_game = prisoners_dilemma()
print("PD pure NE:", pure_nash_equilibria(pd_game), "| NashConv(C,C) =", pd_game.nash_conv(C, C), "| NashConv(D,D) =", pd_game.nash_conv(D, D))
x, y = mixed_nash_2x2(matching_pennies()); print("Matching pennies mixed NE:", x, y, "NashConv:", matching_pennies().nash_conv(x, y))
x, y = mixed_nash_2x2(battle_of_sexes()); print("Battle of sexes mixed NE:", x.round(3), y.round(3))
assert pd_game.nash_conv(C, C) == 4.0 and abs(matching_pennies().nash_conv(x * 0 + .5, y * 0 + .5)) < 1e-12
""")
code("""
# Nash-aware optimisation: Prop. 4 checked numerically (gap semi-gradient == reward mixing)
from nashgat.games.polymatrix import networked_prisoners_dilemma, grid_anti_coordination, train_softmax_profile
g = networked_prisoners_dilemma(2, 2)
a_ = train_softmax_profile(g, "welfare_gap", lam=0.7, steps=200, seed=3)
b_ = train_softmax_profile(g, "welfare_mix", lam=0.7, steps=200, seed=3)
print("max |x_gap - x_mix| =", np.abs(a_["x"] - b_["x"]).max())
assert np.allclose(a_["x"], b_["x"], atol=1e-10)
""")
code("""
# Adaptive communication: gates switch off a sender's out-edges at the next step; failures drop links at random
from nashgat.graph import DynamicGraphBuilder
gb = DynamicGraphBuilder(env.edge_index, env.n_agents, gated=True, fail_prob=0.0)
gb.update(np.array([1, 0] * 6))
m = gb.active_mask()
print("active messages:", int(m.sum()), "of", m.size, "| per-agent sends:", gb.comm_cost(m).astype(int))
assert gb.comm_cost(m)[1::2].sum() == 0
""")

# --------------------------------------------------------------------------- unit tests
md("### Part 5b — Unit tests (asserts) on the written package")
code("""
from nashgat.graph import batch_graphs
from nashgat.stats import holm, mean_ci
# simulator: conservation of vehicles and storage limits under random control
env_t = TrafficMAEnv(replace(default_env_config(DATA_ROOT, "jinan", horizon=600), backend="queuesim"))
env_t.reset(); rng = np.random.default_rng(0)
while True:
    _, r, _, tr_, info = env_t.step(rng.integers(0, env_t.n_actions, env_t.n_agents))
    s = env_t.sim
    assert s.next_dep == int((s.finish[:s.next_dep] >= 0).sum()) + int(s.count.sum()) + s.metrics()["entry_backlog"]
    assert (s.count <= s.cap).all() and (r <= 0).all()
    if tr_: break
# graph batching offsets
out, ea_ = batch_graphs(torch.tensor([[0, 1], [1, 0]]), 2, torch.tensor([[True, False], [True, True]]), torch.ones(2, 2, 3))
assert out.tolist() == [[0, 2, 3], [1, 3, 2]]
# NashConv of networked PD profiles
eye = np.eye(2); gpd = networked_prisoners_dilemma(2, 3)
assert gpd.pure_nash_equilibria() == [(1,) * 6] and gpd.nash_conv(eye[[1] * 6]) == 0
# every method performs one update
for mth in ["ippo", "mappo", "gnn_mappo", "nash_mappo", "nashgat", "nashgat_no_attention"]:
    t_ = MAPPOTrainer(TrafficMAEnv(replace(default_env_config(DATA_ROOT, "jinan", horizon=200), backend="queuesim")), make_config(mth), seed=0)
    h_ = t_.train(1)[-1]; assert np.isfinite(h_["loss_v"]) and h_["loss_gap"] >= 0
assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06] or np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
print("all unit tests passed")
""")

# --------------------------------------------------------------------------- part 6
md("""
## Part 6 — Experiments
Every method uses **the same budget**: the same episodes, horizon, evaluation episodes and best-response budget. The hyper-parameters live in `AlgoConfig` and are saved with each run (`config.json`). Shared settings: hidden 64, 4 heads, actor learning rate 5e-4, critic learning rate 1e-3, γ = 0.99, GAE λ = 0.95, clip 0.2, 4 epochs × 4 minibatches, entropy 0.01. The `nashgat` default is λ = 0.5, μ = 0.05.
""")
code("""
from nashgat.algos.mappo import PRESETS, AlgoConfig
from dataclasses import asdict
display(pd.DataFrame({m: PRESETS[m] for m in METHODS + CFG["ablations"]}).T)
(RESULTS / "configs").mkdir(exist_ok=True)
(RESULTS / "configs/experiment_config.json").write_text(json.dumps({"mode": MODE, "cfg": CFG, "backend": BACKEND,
    "defaults": asdict(AlgoConfig()), "presets": PRESETS}, indent=1, default=str))
""")
code("""
from nashgat.experiment import run_single
from nashgat.rule_based import run_rule
torch.set_num_threads(max(1, os.cpu_count() // 2))
TRAFFIC = RESULTS / "traffic"
DRIVE = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive"); DRIVE = Path("/content/drive/MyDrive/nashgat_results"); DRIVE.mkdir(parents=True, exist_ok=True)

def checkpoint_to_drive():
    if DRIVE is not None:
        shutil.copytree(RESULTS, DRIVE / RESULTS.name, dirs_exist_ok=True)

t_start = time.time()
for seed in CFG["seeds"]:
    for m in METHODS + CFG["ablations"]:
        if (TRAFFIC / m / f"seed{seed}" / "eval.json").exists():
            print("skip (exists)", m, seed); continue
        t0 = time.time()
        r = run_single(m, seed, DATA_ROOT, TRAFFIC, episodes=CFG["episodes"], horizon=CFG["horizon"], backend=BACKEND,
                       eval_episodes=CFG["eval_episodes"], robustness=CFG["robustness"], br_agents=CFG["br_agents"],
                       br_episodes=CFG["br_episodes"], br_eval_episodes=CFG["br_eval_episodes"], device=DEVICE,
                       log_every=max(1, CFG["episodes"] // 4))
        print(f"{m:24s} seed {seed}: ATT {r['eval']['nominal_stochastic']['avg_travel_time']:.1f} s  ({time.time()-t0:.0f}s)")
        checkpoint_to_drive()
print(f"total {time.time()-t_start:.0f}s")
""")
code("""
# non-learning references on the same simulator and scenarios
rows = []
scen = {"nominal": default_env_config(DATA_ROOT, "jinan", horizon=CFG["horizon"]),
        "transfer_hangzhou": default_env_config(DATA_ROOT, "hangzhou", horizon=CFG["horizon"])}
for sname, c in scen.items():
    for pol in ("fixed", "maxpressure"):
        rows.append({"scenario": sname, "controller": pol, **run_rule(TrafficMAEnv(replace(c, backend=BACKEND)), pol)})
rule_df = pd.DataFrame(rows); rule_df.to_csv(RESULTS / "rule_based.csv", index=False)
display(rule_df[["scenario", "controller", "avg_travel_time", "avg_waiting_time", "throughput", "avg_queue_per_intersection"]])
""")
code("""
# exact graphical-game study (closed-form exploitability; independent of the simulator)
gg_rows = []
for gname, gm in {"npd_3x4": networked_prisoners_dilemma(3, 4), "anticoord_3x4": grid_anti_coordination(3, 4)}.items():
    for obj, lam in [("independent", 0.), ("welfare", 0.)] + [(o, l) for o in ("welfare_gap", "welfare_mix", "welfare_fullgap") for l in (0.25, 1.0, 4.0)]:
        for s in range(CFG["gg_seeds"]):
            res = train_softmax_profile(gm, obj, lam=lam, steps=CFG["gg_steps"], seed=s, init_scale=1.0)
            gg_rows.append({"game": gname, "objective": obj, "lambda": lam, "seed": s, "welfare": res["welfare"],
                            "nash_conv": res["nash_conv"], "p_action0_mean": float(res["x"][:, 0].mean())})
(RESULTS / "graphical_games").mkdir(exist_ok=True)
gg = pd.DataFrame(gg_rows); gg.to_csv(RESULTS / "graphical_games/final_profiles.csv", index=False)
display(gg.groupby(["game", "objective", "lambda"])[["welfare", "nash_conv"]].mean().round(3))
""")

# --------------------------------------------------------------------------- part 7
md("""
## Part 7 — Visualisations
Every figure is generated from the CSV/JSON files written above. The figures are training curves (team return, ATT, exploitability gap), congestion bars with per-seed points, welfare against measured exploitability, communication against congestion, robustness, ablations, best-response gains, learned attention on the real Jinan geometry, and the graphical-game frontier.
""")
code("""
from nashgat import report
FIGS = RESULTS / "figures"
report.configure(RESULTS, FIGS, DATA_ROOT)
report.main()
from IPython.display import Image, display as _d
for f in ["training_curves", "congestion_comparison", "tradeoffs", "best_response_gains", "robustness", "ablations",
          "attention_graph", "graphical_games_frontier"]:
    p = FIGS / f"{f}.png"
    if p.exists():
        print(f); _d(Image(filename=str(p), width=900))
""")
code("""
# agent communication graph actually used by NashGAT over one evaluation episode (message frequency per edge)
import matplotlib.pyplot as plt, networkx as nx
from nashgat.algos.mappo import MAPPOTrainer
ck = sorted(TRAFFIC.glob("nashgat/seed*/model.pt"))[0]
env_v = TrafficMAEnv(replace(default_env_config(DATA_ROOT, "jinan", horizon=CFG["horizon"]), backend=BACKEND))
trv = MAPPOTrainer(env_v, make_config("nashgat"), seed=0); trv.load(ck)
b = trv.rollout()
freq = b["mask"].mean(0)
raw = json.loads((DATA_ROOT / "data/raw/colight/data/Jinan/3_4/roadnet_3_4.json").read_text())
xy = {it["id"]: (it["point"]["x"], it["point"]["y"]) for it in raw["intersections"]}
pos = {k: xy[a] for k, a in enumerate(env_v.agents)}
G = nx.DiGraph(); G.add_nodes_from(pos)
for (s_, d_), f_ in zip(env_v.edge_index.T, freq): G.add_edge(int(s_), int(d_), f=float(f_))
fig, ax = plt.subplots(figsize=(5, 4))
nx.draw_networkx(G, pos, ax=ax, node_color="#e87ba4", node_size=300, font_size=7, width=[0.3 + 4 * G[u][v]["f"] for u, v in G.edges],
                 edge_color="#2a78d6", connectionstyle="arc3,rad=0.15")
ax.set_title(f"Message frequency per directed edge (mean {freq.mean():.2f})", fontsize=9); ax.set_axis_off()
fig.savefig(FIGS / "communication_graph.png", dpi=200, bbox_inches="tight"); plt.show()
""")

# --------------------------------------------------------------------------- part 8
md("""
## Part 8 — Scientific analysis (computed automatically)
The interpretation below is **generated from the numbers in this run**. In `FAST_DEMO` mode (1 seed, a handful of episodes) no statistical claim is warranted, and the code says so.
""")
code("""
S = pd.read_csv(RESULTS / "tables/summary_all_methods.csv")
cols = ["name", "n", "avg_travel_time_mean", "avg_travel_time_ci_lo", "avg_travel_time_ci_hi", "avg_waiting_time_mean",
        "throughput_mean", "J_sum_mean", "nash_conv_sampled_mean", "comm_per_agent_step_mean", "train_time_s_mean"]
display(S[[c for c in cols if c in S]].round(2))
st_path = RESULTS / "tables/stat_tests_nashgat_vs_baselines.csv"
ST = pd.read_csv(st_path) if st_path.exists() else pd.DataFrame()
if len(ST): display(ST.round(4))
""")
code("""
def interpret(S, ST, n_seeds):
    out = []
    main = S[S.method.isin(METHODS)].set_index("method")
    best = main.avg_travel_time_mean.idxmin()
    out.append(f"Lowest mean ATT: {main.loc[best, 'name']} ({main.loc[best, 'avg_travel_time_mean']:.1f} s, n={int(main.loc[best, 'n'])}).")
    if "nashgat" in main.index:
        for m in main.index:
            if m == "nashgat": continue
            rel = (main.loc["nashgat", "avg_travel_time_mean"] - main.loc[m, "avg_travel_time_mean"]) / main.loc[m, "avg_travel_time_mean"] * 100
            out.append(f"NashGAT vs {main.loc[m, 'name']}: ATT {rel:+.1f}% (negative = NashGAT better).")
    if "nash_conv_sampled_mean" in main:
        nc = main.nash_conv_sampled_mean.dropna()
        if len(nc): out.append("Sampled BR NashConv (lower = less exploitable): " + ", ".join(f"{main.loc[k, 'name']} {v:.2f}" for k, v in nc.items()) + ".")
    if "comm_per_agent_step_mean" in main:
        out.append("Messages/agent/step: " + ", ".join(f"{main.loc[k, 'name']} {v:.2f}" for k, v in main.comm_per_agent_step_mean.items()) + ".")
    if n_seeds < 3:
        out.append(f"CAUTION: only {n_seeds} seed(s) per method; no significance test is meaningful. Differences above are anecdotal.")
    elif len(ST):
        sig = ST[ST.p_welch_holm < 0.05]
        out.append(f"{len(sig)} of {len(ST)} NashGAT-vs-baseline comparisons are significant at 0.05 after Holm correction (Welch t-test).")
        for _, r in sig.iterrows():
            out.append(f"  - {r.metric} vs {r.vs}: {r['rel_change_%']:+.1f}% ({'better' if r.ref_better else 'worse'}), p_Holm={r.p_welch_holm:.3g}")
    return "\\n".join(out)

analysis = interpret(S, ST, len(CFG["seeds"]))
print(analysis)
(RESULTS / "analysis.txt").write_text(analysis)
""")

# --------------------------------------------------------------------------- part 9
md("""
## Part 9 — Research paper export
This part writes the verified BibTeX, the configuration files and an IEEE-style LaTeX draft whose result tables are filled from this run's CSV files. Every interpretive sentence in the draft is marked for author revision.
""")
BIB = (ROOT / "references.bib").read_text(encoding="utf-8")
code("%%writefile references.bib\n" + BIB)
code(r'''
shutil.copy("references.bib", RESULTS / "references.bib")
main_tab = S[S.method.isin(METHODS)]
def fmt(r, c):
    m, lo, hi = r.get(c + "_mean"), r.get(c + "_ci_lo"), r.get(c + "_ci_hi")
    if pd.isna(m): return "--"
    return f"{m:.1f}" if pd.isna(lo) else f"{m:.1f} [{lo:.1f}, {hi:.1f}]"
rows_tex = "\n".join(f"{r['name']} & {fmt(r, 'avg_travel_time')} & {fmt(r, 'avg_waiting_time')} & {fmt(r, 'throughput')} & "
                     f"{fmt(r, 'nash_conv_sampled')} & {fmt(r, 'comm_per_agent_step')} \\\\" for _, r in main_tab.iterrows())
tex = rf"""\documentclass[journal]{{IEEEtran}}
\usepackage{{amsmath,amssymb,graphicx,booktabs}}
\begin{{document}}
\title{{Exploitability-Aware Graph-Attention MARL for Traffic-Signal Control: Draft generated from run ({MODE}, backend={BACKEND})}}
\author{{Badr El Khalyly and Zouhair Ibn Batouta}}
\maketitle
\begin{{abstract}}
AUTHOR: write from the analysis below. Auto-generated facts: {len(CFG['seeds'])} seed(s), {CFG['episodes']} episodes, horizon {CFG['horizon']} s, simulator {BACKEND}.
\end{{abstract}}
\section{{Results}}
\begin{{table}}[t]\centering\caption{{Nominal Jinan 3$\times$4 results (mean [95\% CI] over seeds; simulator: {BACKEND}).}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lccccc}}\toprule
Method & ATT (s) & Wait (s) & Throughput & BR NashConv & Msgs/agent \\ \midrule
{rows_tex}
\bottomrule\end{{tabular}}}}\end{{table}}
\begin{{figure}}[t]\centering\includegraphics[width=\columnwidth]{{figures/training_curves.pdf}}\caption{{Training curves.}}\end{{figure}}
\noindent Auto-generated analysis (verify before use): \begin{{verbatim}}
{analysis}
\end{{verbatim}}
\bibliographystyle{{IEEEtran}}\nocite{{wei2019colight,yu2022mappo,velickovic2018gat,zhang2019cityflow}}\bibliography{{references}}
\end{{document}}
"""
(RESULTS / "manuscript_draft.tex").write_text(tex, encoding="utf-8")
print("wrote", RESULTS / "manuscript_draft.tex")
''')
code("""
# bundle every artefact (figures, CSV tables, configs, checkpoints, BibTeX, LaTeX draft) and download
archive = shutil.make_archive(str(RESULTS.parent / "nashgat_results"), "zip", RESULTS)
print("archive:", archive, f"({Path(archive).stat().st_size / 1e6:.1f} MB)")
checkpoint_to_drive()
if IN_COLAB:
    from google.colab import files
    files.download(archive)
""")
md("""
### Cells needing Colab-only resources
* The CityFlow build cell needs Linux, cmake and a C++ compiler. Outside Colab it reports that CityFlow is unavailable and falls back to **QueueSim**.
* Google Drive mounting (`USE_DRIVE=True`) and `files.download` exist only in Colab. Elsewhere they are skipped.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "accelerator": "GPU",
               "colab": {"provenance": [], "gpuType": "T4", "name": "NashGAT_MARL_Research.ipynb"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
