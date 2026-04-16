# Exp004: Phase 1 Enhanced Heuristic Eval — Ceiling Analysis

**Date**: 2026-04-14  
**Topic**: Enhanced static eval with ball possession + urgency signals  
**N**: 50 games per condition, budget=200, 12-thread root parallelism

---

## Motivation

MCTS was achieving only 12% vs Random and 4% vs Argmax with a flat sigmoid eval. The plan was to add three new signals: ball possession + field advancement, turn urgency, and on-field player count. Primary goal: improve vs-Argmax win rate toward 60%+ target.

---

## Changes Made

**`BbMctsSearch.java`**:
- Added `BALL_SCALE = 1.0`, `URGENCY_SCALE = 0.5` constants
- Rewrote `staticEval()` to include:
  - Ball possession signal: `±(0.3 + 0.5 × normalizedX)` based on carrier ownership and field position
  - Urgency: `gameProgress × scoreDelta` (amplifies score delta late-game)
  - On-field count (replaces standing-only): counts STANDING | MOVING | PRONE | STUNNED
- Changed `MAX_TRAVERSE_DEPTH` from 15 → 4 (prevents pathological deep trees)

**`MatchRunner.java`**:
- Added `--mcts-vs-argmax` flag enabling conditions E2 (MCTS-UCB vs Argmax) and F2 (MCTS-Prior vs Argmax)

---

## Results

### Final calibrated run (BALL_SCALE=1.0)

| Condition | Wins | Draws | Loss | Win Rate | 95% CI |
|-----------|------|-------|------|----------|--------|
| A: Sample vs Random | 6 | 44 | 0 | 12.0% | [5.6%–23.8%] |
| B: Argmax vs Random | 50 | 0 | 0 | 100.0% | [92.9%–100.0%] |
| C: Sample vs Argmax | 1 | 10 | 39 | 2.0% | [0.4%–10.5%] |
| D: Random vs Random | 0 | 50 | 0 | 0.0% | [0.0%–7.1%] |
| **E: MCTS-UCB vs Random** | **7** | **43** | **0** | **14.0%** | [7.0%–26.2%] |
| **F: MCTS-Prior vs Random** | **5** | **45** | **0** | **10.0%** | [4.3%–21.4%] |
| **E2: MCTS-UCB vs Argmax** | **1** | **17** | **32** | **2.0%** | [0.4%–10.5%] |
| **F2: MCTS-Prior vs Argmax** | **0** | **11** | **39** | **0.0%** | [0.0%–7.1%] |

### Intermediate run (BALL_SCALE=2.5, same depth cap)

| Condition | Win Rate |
|-----------|----------|
| E: MCTS-UCB vs Random | 12.0% |
| F: MCTS-Prior vs Random | 8.0% |
| E2: MCTS-UCB vs Argmax | 4.0% |
| F2: MCTS-Prior vs Argmax | 8.0% |

---

## MCTS Search Statistics (BALL_SCALE=1.0)

| Condition | iter/s | ms/iter | ms/dec | avg depth | max depth | nodes/dec |
|-----------|--------|---------|--------|-----------|-----------|-----------|
| E: UCB vs Random | 1330 | 0.75 | 144 | 0.6 | 4 | 18.2 |
| F: Prior vs Random | 1220 | 0.82 | 157 | 0.4 | 4 | 11.0 |
| E2: UCB vs Argmax | 1244 | 0.80 | 154 | 0.8 | 4 | 32.0 |
| F2: Prior vs Argmax | 1184 | 0.84 | 162 | 0.6 | 4 | 25.0 |

---

## Key Finding: Tree Barely Exists

**Average depth = 0.4–0.8** with budget=200 is the critical observation. With ~11 candidates per node:
- Budget=200 at depth=1: only ~18 root children visited
- Budget=200 at depth=2: would need 200/11² ≈ 1.7 visits/subtree — essentially unexplored

This means MCTS is running **Monte Carlo rollouts** (depth ≤ 1), not tree search. The static eval is called after at most one activation from the current state. Ball position rarely changes in one activation, so the new ball signal contributes almost nothing to the backup values.

**Why BALL_SCALE=2.5 helped vs Argmax (8%) but BALL_SCALE=1.0 didn't (0%)**: With a higher signal, even depth-1 states (one player moved) occasionally put the ball carrier in a better position — the eval difference was large enough to direct PUCT toward those moves. But the high signal also caused over-exploitation when ball carrier states scored Q≈0.88 consistently.

---

## Root Cause of Phase 1 Ceiling

The heuristic Phase 1 eval cannot beat Argmax because:

1. **Tree is too shallow**: Budget=200 with branching factor ~11 means no 2-ply lookahead
2. **Eval only affects root selection**: Since depth ≤ 1, the rollout policy (scripted) does all the work — the eval just reweights which activation to try
3. **Argmax's one-ply greedy is better than rollout noise**: Scripted rollouts introduce variance; Argmax deterministically picks the best immediate action

---

## Debugging History

- **BALL_SCALE=2.5 hang**: PUCT built trees to depth 10–15 (each level ~5ms → 75ms/iter × 200 budget = 15s/decision × 100 decisions = hours/game). Fixed by `MAX_TRAVERSE_DEPTH = 4`.
- **Compile error**: `Team.getPlayers()` returns `Player<?>[]` (array), not `List` — fixed with explicit `==` reference loop.

---

## Conclusion

Phase 1 heuristic improvements raised the ceiling slightly (12% → 14% vs Random) but are insufficient for the 60%+ vs Argmax target. The fundamental bottleneck is that **budget=200 doesn't build a tree deep enough for the eval function to matter** against a one-ply agent. 

Two paths forward (both needed):
1. **Higher budget** (1000+): would enable 2-ply lookahead, giving MCTS structural advantage over greedy
2. **Phase 2 ONNX value function**: trained on Argmax self-play, provides accurate state values even at depth=1 rollout endpoints

See exp005 for Phase 2 results.
