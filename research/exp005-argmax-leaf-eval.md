# Exp005: Argmax-Quality Leaf Eval + Time-Based Budget

**Date**: 2026-04-16  
**Topic**: Replace static heuristic leaf eval with argmax action-score signal; switch to wall-clock time budget  
**N**: 1 game per condition (inconclusive); full N=20 sweep interrupted  

---

## Motivation

Exp004 showed MCTS at budget=200 barely builds a tree (avg depth 0.4–0.8). The leaf eval is called after at most one activation from root. The key insight: if the leaf eval is **as informative as Argmax's own 1-ply heuristic**, MCTS should at minimum match Argmax (2-ply lookahead for free).

Argmax uses `MoveDecisionEngine.rawScores` — the same scores it uses to pick actions. These come from `ActionScore(probability, value, confidence).softmaxScore()`. The plan: at each MCTS expansion node, call `buildCandidates(game, isHome, rng)` to obtain `rawScores`, then use `max(rawScores)` as the leaf eval. This gives MCTS the same "best-available action" signal Argmax uses.

---

## Changes Made

### 1. Time-based MCTS budget (`MatchRunner.java`, `BbMctsSearch.java`)

Switched from `--mcts-budget N` (iteration count) to `--mcts-time-ms T` (wall-clock ms per decision). Motivation: iteration-based budgets at higher depths caused exponential time blowup. Time-based keeps games within predictable duration.

New conditions added:
- **G / H**: MCTS-UCB / MCTS-Prior vs Random (time-based)
- **G2 / H2**: MCTS-UCB / MCTS-Prior vs Argmax (time-based, requires `--mcts-vs-argmax`)

### 2. Cross-turn search (`BbMctsSearch.java`)

Added `--mcts-cross-turn` flag: after a turn-end in MCTS, simulate the opponent's full scripted turn via `advanceTurnEndScripted(gameState, !isHome)`, then continue the MCTS tree into the next home activation. Guard: `turnCrossings == 0` prevents recursive crossing.

**Result**: Too slow for practical use. `advanceTurnEndScripted` runs Dijkstra for all 11 opponent players (~110ms/crossing). MCTS-Prior with cross-turn took 100–300s/game. Disabled for main sweep.

### 3. Argmax-quality leaf eval at expansion (`BbMctsSearch.java`, lines 665–689)

At expansion (first visit to a node, `child.visitCount == 0`):
1. Call `buildCandidates(game, isHome, rng)` to get `rawScores` from `MoveDecisionEngine`
2. Cache result in `child.cachedSet` (eliminates Dijkstra on second visit — same total cost)
3. Compute `argmaxSignal = sigmoid(2.0 * (maxScore - 1.0))`

**Critical normalization bug found and fixed:**

Initial version used `sigmoid(maxScore / 8.0)`. This was based on a wrong assumption that rawScores were in [-20, 50]. In fact, `rawScores` are `softmaxScore() = 1.0 + p*v*c` values in [0, 2]. With divisor 8.0:
- `sigmoid(2.0 / 8.0)` = 0.562 (best possible TD opportunity)  
- `sigmoid(0.75 / 8.0)` = 0.524 (end-turn only)

Range [0.524, 0.562] — essentially constant noise. MCTS could not distinguish any positions.

Fixed formula: `sigmoid(2.0 * (maxScore - 1.0))`:
- `maxScore = 0.75` (end-turn only): → 0.38  
- `maxScore = 1.0` (neutral): → 0.50  
- `maxScore = 1.5` (good advance opportunity): → 0.73  
- `maxScore = 2.0` (clear TD path): → 0.88  

Range [0.38, 0.88] — comparable scale to `winProb` turn-end signal (±0.45 for ±1 TD).

### 4. GC tuning (`.mvn/jvm.config`)

Added `-Xms2g -XX:+UseG1GC -XX:MaxGCPauseMillis=50 -XX:G1HeapRegionSize=16m` to prevent progressive game slowdown. Without G1GC, game times escalated from ~25s to 67s → 134s → 141s over successive games (heap fragmentation / GC pause accumulation). With G1GC tuning, single-game times are stable at 23–32s per game.

---

## Results (N=1, highly preliminary)

| Condition | W/D/L | Note |
|-----------|-------|------|
| A: Sample vs Random | 0/1/0 | baseline |
| B: Argmax vs Random | 1/0/0 | baseline |
| C: Sample vs Argmax | 0/1/0 | baseline |
| D: Random vs Random | 0/1/0 | baseline |
| G: MCTS-UCB vs Random (200ms) | 0/1/0 | draw |
| H: MCTS-Prior vs Random (200ms) | 1/0/0 | **win** |
| G2: MCTS-UCB vs Argmax (200ms) | 0/1/0 | draw |
| H2: MCTS-Prior vs Argmax (200ms) | 0/0/1 | loss |

**N=1 is statistically meaningless.** The H win vs Random and H2 loss vs Argmax carry 95% CI [0%–79%] / [20%–100%] — entirely uninformative.

---

## MCTS Search Statistics (N=1 at 200ms)

| Condition | iter/s | ms/iter (CPU) | ms/dec (CPU) | avg depth | max depth | nodes/dec |
|-----------|--------|----------------|---------------|-----------|-----------|-----------|
| G: UCB vs Random | 1076 | 0.93 | 2384 | 0.9 | 4 | 61.5 |
| H: Prior vs Random | 921 | 1.09 | 2383 | 0.6 | 4 | 36.1 |
| G2: UCB vs Argmax | 916 | 1.09 | 2436 | 0.9 | 4 | 68.7 |
| H2: Prior vs Argmax | 852 | 1.17 | 2421 | 0.5 | 2 | 18.4 |

Note: `ms/dec` is CPU time (12 threads × 200ms wall-clock). Total iterations per decision = 1076 × 0.2s ≈ 215 (G). `nodes/dec` = new expansions per decision.

Key observations:
- **216 total iterations / 61 new nodes per 200ms wall-clock** (G condition)
- **avg depth still only 0.9** — barely improved from exp004's 0.4–0.8
- The argmax leaf eval calls `buildCandidates` at expansion, not extra vs before (second-visit call moved to first-visit)
- With ~11 candidates per node, depth-2 tree needs 11² = 121 visits minimum — impossible at 61 nodes/dec

---

## Key Findings

### Why avg depth is still ~1

At 200ms/decision with 12 threads, MCTS runs ~216 total iterations. With branching factor ~11:
- Depth 0 (turn-end): ~60–70% of iterations (Blood Bowl turns end frequently after one bad activation)  
- Depth 1 (one activation): ~28–40%

The distribution is dominated by stochastic turn-ends, not by eval quality. Even a perfect leaf eval can't build a depth-2 tree with 216 iterations and 11 branches per node (needs 121+ just to visit depth-2 leaves once each).

### Why MCTS still loses to Argmax

At avg depth < 1, MCTS is making decisions based on noisy 1-ply rollouts:
- Turn-end evals (60% of signal): `winProb(game) - winProbRoot` — evaluates score delta only, almost always 0 at mid-game
- Expansion evals (40% of signal): `argmaxSignal - winProbRoot` — fixed normalization gives real spread, but only sees one activation ahead

Argmax uses the same rawScore signals but deterministically picks the best, without Monte Carlo noise. MCTS's advantage (multi-step lookahead) only materializes at depth 2+, which it can't reach at 200ms.

---

## What Needs to Happen to Beat Argmax

| Approach | Budget Required | Status |
|----------|----------------|--------|
| Time budget → depth 2+ | ~2000ms/decision | Not tested; game duration ~200s (unacceptable) |
| Scripted rollouts (5-10 activations) | 200ms → ~33 rollouts | Not attempted |
| ONNX value function (Phase 2) | 200ms (trained eval) | Requires training data |

The clearest path: **ONNX value function** trained on Argmax self-play data. A trained network that evaluates game states accurately at depth 0 would let MCTS beat Argmax with the current 200ms budget by choosing the activation that leads to the best-valued next state.

Intermediate option: **scripted rollouts** (5–10 activations each, scripted argmax) give much better signal than static eval at the cost of 5–10× fewer iterations. This may beat MCTS+staticEval but will likely still lose to pure Argmax unless the rollout is long enough to score actual TDs.

---

## Debugging History

- **Normalization bug**: `/ 8.0` assumed rawScores in [-20, 50]; actual range [0, 2] → nearly constant signal. Fixed to `2.0 * (maxScore - 1.0)`.
- **GC escalation**: Without G1GC, game 5+ took 67s, 134s, 141s due to heap fragmentation. Fixed by adding G1GC flags to `.mvn/jvm.config`.
- **Cross-turn too slow**: `advanceTurnEndScripted` runs Dijkstra for 11 players (~110ms). In MCTS loop this is called once per iteration → effectively limits budget to 1 iter / 110ms = 9 iters/second. Useless.
- **Turn-end eval regression (prior session)**: Using `leafEval.evaluate` at turn-end (instead of `winProb`) caused games > 200s due to ball-signal over-exploitation. Reverted; comment added.

---

## Status: Incomplete

The N=20 sweep (confirming whether fixed normalization actually helps) was interrupted. The single-game result is insufficient to conclude whether the argmax leaf eval improves vs-Argmax win rate. 

Next step: generate Argmax self-play training data and train Phase 2 ONNX value function.
