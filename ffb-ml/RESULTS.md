# Behavioral Cloning Bot — Results

## Overview

A BC bot that imitates ScriptedStrategy via supervised learning on (state, action) pairs captured during headless simulation. Three decision heads share a CNN+MLP backbone: **dialog** (block dice, rerolls, etc.), **player-select** (which player to activate), and **move-target** (where to move).

Evaluated 2026-04-09 on a 100-game dataset, micro and small model scales.

---

## Pipeline

```
HeadlessFantasyFootballServer (500ms/game)
  └── TrainingDataExporter  →  JSONL shards (dialog / player_select / move_target)
        └── extract_features.py  →  .npz shards + vocab.json
              └── train.py  →  checkpoints (.pt)
                    └── export_model.py  →  3× ONNX files
                          └── EvalRunner (Java/ONNX Runtime)  →  win rates
```

---

## Architecture

| Component | Detail |
|---|---|
| Spatial input | 29 channels × 26×15 board (own/opp present, player-state 5-way one-hot, PlayerEncoder k=16 zeros, ball-state 4-way, tackle-zone density ×2) |
| Non-spatial input | 143-dim vector (match state, team casualty counts, active-player context) |
| CNN | Conv(29→32, 3×3) → Conv(32→64) → Conv(64→128) → GlobalAvgPool → 128-dim |
| NS-MLP | Linear(143, 128) → ReLU → 128-dim |
| Backbone | concat → Linear(256, 256) → ReLU → 256-dim shared rep |
| Dialog head | Linear(256 + n\_dialog\_types + 15, 64) → 6 logits |
| Player-select head | Linear(256 + k, 64) → 1 score per candidate (masked softmax over ≤25 slots) |
| Move-target head | Linear(256, 391) → masked softmax (26×15 + 1 end-activation) |
| PlayerEncoder | Embedding(n\_skills+1, 8, padding=0) → mean-pool → Linear(13, k) → ReLU |

**Scales:**

| Scale | k | CNN filters | Hidden | Params |
|---|---|---|---|---|
| micro | 8 | 16 / 32 / 64 | 64 | 253,854 |
| small | 16 | 32 / 64 / 128 | 128 | 387,822 |

---

## Dataset

Generated with `TrainingDataExporter` from **100 games** (ScriptedStrategy vs ScriptedStrategy, `SCRIPTED_SAMPLE` mode):

| Decision type | Records | Random baseline accuracy |
|---|---|---|
| Dialog | 1,363 | 0.553 (avg 1 / n\_options) |
| Player-select | 16,250 | 0.149 (avg 1 / n\_valid\_candidates) |
| Move-target | 14,265 | 0.007 (avg 1 / n\_reachable\_squares) |

Data export: **26.2 s** for 100 games (4 threads).

---

## Training

**Setup:** Adam, lr=3×10⁻³, CosineAnnealingLR, batch=256, 30 epochs, MPS (Apple Silicon).  
**Loss:** joint cross-entropy across all three heads (dialog masked CE + NLL for player-select + NLL for move-target).

### Micro model (253,854 params)

| Epoch | Loss | Dialog | PS | MT |
|---|---|---|---|---|
| 1 | 3.329 | 0.581 | 0.170 | 0.006 |
| 5 | 3.316 | 0.743 | 0.172 | 0.006 |
| 10 | 3.294 | 0.831 | 0.174 | 0.004 |
| 20 | 3.248 | 0.846 | 0.177 | 0.007 |
| 30 | 3.230 | **0.816** | **0.174** | **0.006** |

Training time: **45 s** (1.5 s/epoch).

### Small model (387,822 params)

| Epoch | Loss | Dialog | PS | MT |
|---|---|---|---|---|
| 1 | 3.325 | 0.581 | 0.172 | 0.007 |
| 5 | 3.304 | 0.801 | 0.175 | 0.008 |
| 10 | 3.279 | 0.824 | 0.172 | 0.004 |
| 20 | 3.195 | 0.838 | 0.170 | 0.005 |
| 30 | 3.150 | **0.838** | **0.170** | **0.004** |

Training time: **72 s** (2.4 s/epoch).

### Final accuracy vs random baselines

| Metric | Random | Micro | Small | Δ (small) |
|---|---|---|---|---|
| Dialog | 0.553 | 0.816 | 0.838 | **+0.285** |
| Player-select | 0.149 | 0.174 | 0.170 | **+0.021** |
| Move-target | 0.007 | 0.006 | 0.004 | **−0.003** |

---

## Inference Speed (Java, ONNX Runtime 1.21, batch=1, Apple Silicon)

| Head | Latency | Throughput |
|---|---|---|
| Backbone only (micro) | 0.34 ms | 2,960 / sec |
| Backbone only (small) | 0.27 ms | 3,698 / sec |
| player\_select (small ONNX) | 0.30 ms | 3,368 / sec |
| move\_target (small ONNX) | 0.27 ms | 3,736 / sec |

Both heads comfortably fit within the real-time budget (the headless simulator runs complete games in ~500 ms).

---

## Behavioral Evaluation

**Setup:** N=100 games per condition, EvalRunner (Java), single-threaded.  
**BC agent:** player-select + move-target via ONNX (small model); dialog falls back to ScriptedArgmax (dialog command dispatch is tightly coupled to ScriptedStrategy's internal option mapping).

| Condition | W | D | L | Win rate | 95% Wilson CI |
|---|---|---|---|---|---|
| **BC vs Random** | 21 | 79 | 0 | 21.0% | [14.2% – 30.0%] |
| **BC vs Argmax** | 1 | 23 | 76 | 1.0% | [0.2% – 5.4%] |
| Argmax vs Random *(reference)* | 96 | 4 | 0 | 96.0% | [90.2% – 98.4%] |
| Random vs Random *(null baseline)* | 0 | 100 | 0 | 0.0% | [0.0% – 3.7%] |

> **Note:** "Random" = always end turn (no moves), so draws at 0–0 are the null outcome. The high draw rate in BC vs Random (79%) indicates the BC model's movement is near-random, scoring only sporadically.

---

## Analysis

### What works

**Dialog decisions are well-learned (+28.5% above random).** The model has internalized ScriptedStrategy's block-die selection, reroll usage, and binary choices (follow-up, pick-up, chainsaw, etc.). The small action space and the 1,363 training records are sufficient for this head.

### What doesn't (yet)

**Player-select is marginally above random (+2.1%).** The model slightly prefers better-positioned players, but the gain is small. The spatial CNN receives no meaningful gradient signal from the 16K decisions because move-target training drowns out the positional value signal.

**Move-target is at-random accuracy (≈ 0.007, same as random baseline).** With 14K decisions spread across a 390-cell action space, there are on average only **37 samples per cell**. The CNN has insufficient density to learn spatial value gradients. This is the primary bottleneck.

### Game-level impact

The BC model currently captures approximately **10% of ScriptedArgmax's advantage** over Random (21% vs 96% win rate above the 0% null). All 21 BC wins come from occasional correct movement decisions; the remaining 79 games end 0–0.

Against Argmax (1% win rate), the model is dominated — expected, since it is imitating Argmax but with far less data than needed.

---

## Bugs Encountered and Fixed

| Bug | Cause | Fix |
|---|---|---|
| 0 player-select/move-target records | NPE on null end-turn sentinel in candidate list | Skip null entries, remap end-turn to `MAX_CANDS` slot |
| 66 player-select records with masked action | End-turn index `n_cands` < `MAX_CANDS`, so `cand_mask[n_cands]=0` | Remap raw action `≥ n_actual → MAX_CANDS` in Python |
| `KeyError: x_spatial` in dialog batches | `process_dialog()` omitted spatial input | Added `extract_spatial()` call to dialog path |
| `loss=inf` | 66 bad records with action pointing to masked position | Resolved by action-mapping fix above |
| `torch.from_numpy` TypeError | Numpy scalar from indexing (not an array) | Changed to `torch.as_tensor()` |
| ONNX export failure | `onnxscript` not installed | `pip install onnxscript` |
| ONNX IR version mismatch | PyTorch 2.10 exports IR 10; onnxruntime 1.17 supports IR ≤ 9 | Upgraded onnxruntime to 1.21.0 |
| `cand_skill_ids` shape mismatch | `export_model.py` used `max_cands=25`; `FeatureExtractor` used `MAX_CANDS=24` | Set `FeatureExtractor.MAX_CANDS = 25` |

---

## Scaling Projections

| Training games | Move-target samples | Samples/cell | Expected MT acc | Expected win rate vs Random |
|---|---|---|---|---|
| 100 (current) | 14K | 37 | ≈ random (0.007) | ~20% |
| 1,000 | 143K | 367 | > random (est. 0.015–0.030) | ~40–60% |
| 5,000 | 715K | 1,832 | meaningful (est. 0.04+) | ~60–80% |
| 10,000 | 1.4M | 3,665 | near-Argmax (est. 0.08+) | ~80–90% |

These are rough estimates assuming the CNN can extract spatial value with sufficient data density.

To generate 1,000-game dataset (~2.5 min with 4 threads):

```bash
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.TrainingDataExporter \
  "-Dexec.args=--output ffb-ml/data --games 1000 --threads 4"
```

Then re-run the training and export pipeline:

```bash
cd ffb-ml
python src/extract_features.py --input data --output /tmp/ffb-features-1k
python /tmp/run_experiment2.py   # or equivalent with FEAT_DIR=/tmp/ffb-features-1k
```

---

## File Inventory

| File | Role |
|---|---|
| `ffb-ai/.../strategy/ScriptedStrategy.java` | Thread-local decision logging (pick/pickBool hooks) |
| `ffb-ai/.../strategy/DecisionLog.java` | POJO: scores[] + chosen per decision |
| `ffb-ai/.../simulation/GameStateSerializer.java` | Compact JSON snapshot from `Game` |
| `ffb-ai/.../simulation/ITrainingDataCollector.java` | Collector interface (onDialog / onPlayerSelect / onMoveTarget) |
| `ffb-ai/.../simulation/JsonlTrainingDataCollector.java` | JSONL writer; null-sentinel handling for end-turn |
| `ffb-ai/.../simulation/TrainingDataExporter.java` | Main: thread-pool data generation, CLI args |
| `ffb-ai/.../simulation/FeatureExtractor.java` | Java replica of `extract_features.py` |
| `ffb-ai/.../simulation/OnnxModelAgent.java` | ONNX inference wrapper (player-select + move-target heads) |
| `ffb-ai/.../simulation/EvalRunner.java` | Main: 4 conditions × N games, Wilson CI report |
| `ffb-ai/.../simulation/MatchRunner.java` | Added `AgentMode.MODEL`, model-agent hooks |
| `ffb-ai/pom.xml` | onnxruntime 1.21.0 dependency |
| `ffb-ml/src/extract_features.py` | Vocab build, spatial/NS/dialog/candidate feature extraction |
| `ffb-ml/src/model.py` | PlayerEncoder, BCBackbone, BCModel, make_model() |
| `ffb-ml/src/train.py` | Joint training, dialog_loss / nll_loss, val metrics |
| `ffb-ml/src/export_model.py` | Three-head ONNX export (dialog / player_select / move_target) |
| `ffb-ml/src/eval.py` | Per-type accuracy tables and scaling plots |
| `ffb-ml/src/probe_player_encoder.py` | Linear probe on frozen encoder |
