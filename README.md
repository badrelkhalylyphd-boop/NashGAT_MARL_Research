# NashGAT-MARL: reproducibility code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/badrelkhalylyphd-boop/NashGAT_MARL_Research/blob/main/notebooks/NashGAT_MARL_Research.ipynb)

Code, Colab notebook and released results for **exploitability-aware graph-attention multi-agent reinforcement learning for traffic-signal control**. The repository covers:

* independent PPO, MAPPO, GNN-MAPPO, Nash-MAPPO and NashGAT-MARL;
* ablations and sweeps;
* best-response estimates of exploitability;
* exact graphical games;
* robustness and zero-shot transfer tests on the CoLight Jinan 3×4, Hangzhou 4×4 and New York 28×7 benchmark files.

**Authors**
Badr El Khalyly∗ and Zouhair Ibn Batouta†<br>
∗ LTIM – Laboratory of Techniques of Information and Modeling, FSBM, Hassan II University, Casablanca, Morocco<br>
† RITM – ESTC/CED, ENSEM, Hassan II University, Casablanca, Morocco<br>
badrelkhalylyphd@gmail.com · Zouhair.ibnbatouta@gmail.com

---

## 1. Run in Google Colab (recommended)

Click the **Open in Colab** badge, then run *Runtime → Run all*. The default mode, `REPRODUCE_PAPER`, is the one for reviewers.

| Mode | What it does | Time on Colab |
|---|---|---|
| `REPRODUCE_PAPER` (default) | downloads the 52 released trained models and logs, regenerates every table and figure, and checks them (see below) | ≈ 15–25 min |
| `FAST_DEMO` | functional check of the training pipeline: 1 seed, short episodes. The numbers are not research results | ≈ 15–30 min |
| `FULL_EXPERIMENT` | retrains everything with the paper's protocol: 5 seeds × 250 episodes × 3,600 s | many hours; enable `USE_DRIVE` to keep checkpoints |

**How `REPRODUCE_PAPER` verifies the paper.**

1. The regenerated tables must be identical to the released tables. The notebook asserts this.
2. FixedTime and MaxPressure are recomputed. They are deterministic, so they must match the paper exactly (asserted).
3. Every released model is re-run. The greedy evaluation is deterministic and must equal the paper's value exactly. The stochastic evaluation uses a new random stream and must agree within sampling noise. On the authors' machine this was about ±2 s on average travel time. The comparison is saved to `reproduction_check.csv`.

Retraining from scratch (`FULL_EXPERIMENT`) gives slightly different numbers, because RL training is not bit-reproducible across hardware and library versions. The conclusions should still hold within the reported confidence intervals.

The notebook is self-contained. It runs these steps in order:

1. Install PyTorch Geometric.
2. **Build CityFlow from source.** If the build fails, it falls back to the *QueueSim* simulator and prints a warning.
3. Download and validate the original CoLight data.
4. Write the `nashgat` package from its cells.
5. Run unit tests.
6. Train and evaluate every method with identical budgets.
7. Compute statistics and draw figures.
8. Export everything to `/content/results/` and download it as a zip.

## 2. Run locally

```bash
pip install -r requirements.txt          # Python >= 3.10
bash scripts/fetch_data.sh               # CoLight data (sparse clone; not redistributed here)
python scripts/validate_dataset.py       # -> results/dataset_validation.json
python -m pytest -q                      # 26 tests: Nash/NashConv oracles, Prop. 3-4, simulator, every method
```

To reproduce the released results:

```bash
python scripts/launch_grid.py --workers 8                # 52 runs: 5 methods x 5 seeds + 9 variants x 3 seeds (resumable)
python scripts/run_rule_based.py                         # FixedTime and MaxPressure references
python scripts/run_graphical_games.py --steps 1500       # exact polymatrix games
python scripts/eval_transfer_newyork.py double           # zero-shot New York 28x7 (196 agents)
python scripts/aggregate.py                              # -> results/tables/*.csv, results/figures/*
python scripts/make_tables.py                            # -> results/latex_tables/*.tex
```

Single run or best-response re-evaluation from a released checkpoint:

```bash
python scripts/run_traffic.py --method nashgat --seed 0 --episodes 250
python scripts/run_br_only.py results/traffic/nashgat/seed0
```

To regenerate the notebook after editing the package (CI checks that it stays in sync):

```bash
python scripts/build_notebook.py
```

## 3. Repository layout

```
nashgat/                 package
  envs/roadnet.py          CityFlow roadnet/flow parser and validator
  envs/queue_sim.py        QueueSim: lightweight mesoscopic simulator (separate environment)
  envs/cityflow_backend.py CityFlow backend (same interface)
  envs/traffic_env.py      multi-agent environment, individual rewards, robustness options
  graph.py                 dynamic communication graph with gates and link failures
  models.py                MLP / mean-aggregation / GATv2 encoders, actors, critics
  algos/mappo.py           unified PPO/MAPPO trainer, all methods and ablation presets
  algos/best_response.py   approximate best response, sampled NashConv
  games/                   bimatrix and polymatrix games with exact NashConv
  rule_based.py            FixedTime, MaxPressure
  stats.py, report.py      statistics (t-intervals, Welch, Mann-Whitney, Holm), tables and figures
  experiment.py            single-run driver used by the notebook
scripts/                 data, experiments, aggregation, notebook build and execution
tests/                   unit and mathematical validation tests
notebooks/               NashGAT_MARL_Research.ipynb (Colab)
results/                 released results (see below)
references.bib           verified bibliography used by the notebook
```

## 4. Released results (`results/`)

| Path | Content |
|---|---|
| `traffic/<method>/seed<k>/` | `config.json` (all hyper-parameters), `history.csv` (per-episode training log), `eval.json` (nominal, robustness and transfer), `nashconv.json` (best responses), `attention.npz` (GAT actors), `model.pt` (actor and critic) |
| `tables/` | per-seed and summary tables with 95 % CIs, statistical tests, robustness, best-response gains, graphical games, New York transfer |
| `figures/` | training curves, congestion, trade-offs, robustness, ablations, best-response gains, attention graph, graphical-game frontier |
| `rule_based.csv`, `transfer_newyork_double.csv`, `graphical_games/`, `dataset_validation.json` | reference controllers, transfer, exact games, data checksums |

**Simulator note.** All released traffic results were produced with **QueueSim**, because CityFlow could not be built on the machine used. QueueSim reads the original CityFlow files but simplifies vehicle dynamics. Its absolute travel times are **not comparable** with CityFlow numbers published elsewhere, and results from the two simulators must not be pooled. CityFlow runs can be produced with the notebook on Colab.

## 5. Reproducibility checklist

* Data: original CoLight files at commit `04440b7`, downloaded at run time, with SHA-256 checksums in `results/dataset_validation.json`.
* Seeds: Python, NumPy and PyTorch are seeded (`nashgat.utils.set_seed`), and demand perturbations are seeded. A determinism test is included.
* Equal budget for every method: 250 episodes, identical evaluation and best-response protocols, and the shared hyper-parameters in `AlgoConfig`.
* Statistics: Student-t 95 % intervals over seeds; Welch and Mann–Whitney tests with Holm correction.
* CI: `.github/workflows/tests.yml` installs CPU PyTorch, fetches the data, validates it, runs the tests and checks that the notebook is in sync.
* Hardware for the released runs: Intel Core Ultra 5 235H, CPU only, one thread per run, 10 runs in parallel.

## 6. Data licence and AI assistance

The benchmark files belong to the CoLight authors (https://github.com/wingsweihua/colight). They are downloaded at run time and not redistributed here. The code was developed with substantial assistance from an AI system (Claude); the authors are responsible for its verification and use.
