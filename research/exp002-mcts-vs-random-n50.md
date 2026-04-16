# Experiment 002: Can MCTS Beat Random? (N=50)

**Date:** 2026-04-14  
**Status:** Complete  
**Follows from:** [Exp 001](exp001-mcts-speed.md)

## Question

After the speed work in Exp 001, does MCTS actually play better than a random agent? The N=10 results from Exp 001 were too noisy to tell (CI [1.8%–40.4%]).

## Setup

```bash
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  -Dexec.args="/path/to/repo 50 --mcts-budget 200"
```

N=50 games per condition, budget=200 iterations per MCTS decision, Human vs Human roster.

## Results

```
Condition                Wins  Draws   Loss  WinRate ± 95% CI
------------------------------------------------------------------------
A: Sample  vs Random        4     46      0   8.0% [3.2%–18.8%]
B: Argmax  vs Random       50      0      0  100.0% [92.9%–100.0%]
C: Sample  vs Argmax        1      7     42   2.0% [0.4%–10.5%]
D: Random  vs Random        0     50      0   0.0% [0.0%–7.1%]
E: MCTS-UCB    vs Random    8     42      0  16.0% [8.3%–28.5%]
F: MCTS-Prior  vs Random    8     42      0  16.0% [8.3%–28.5%]

MCTS Search Statistics (budget=200):
Condition              iter/s   ms/iter   ms/dec   avg depth   max depth   nodes/dec
E: MCTS-UCB   vs Rand   1232    0.812     155.8      0.6          3          17.7
F: MCTS-Prior vs Rand   1151    0.869     166.8      0.4          3          11.9
```

## Interpretation

**MCTS does reliably beat Random.** The 95% CI lower bound of 8.3% is well above 0, confirming a real win-rate advantage at N=50.

Ranking the agents:

| Agent | Win rate vs Random | Notes |
|-------|--------------------|-------|
| Random | 0% | baseline |
| Sample | 8% | scripted policy, temperature=0.5 |
| MCTS-UCB | 16% | 2× better than Sample |
| MCTS-Prior | 16% | same as UCB at this budget |
| Argmax | 100% | scripted policy, temperature=0 |

MCTS beats the sampled scripted policy (8% → 16%) but is far below the argmax scripted policy (100%). 

## Why is MCTS so much weaker than Argmax?

Three factors explain the gap:

1. **Weak static eval.** The leaf evaluation is `sigmoid(TD_SCALE × scoreDelta + STANDING_SCALE × standing_advantage/11)`. It doesn't account for ball position, cage quality, distance to end zone, turn number, or half. The search signal is too coarse to guide good decisions.

2. **Shallow search.** Avg depth = 0.4–0.6, max = 3. Most iterations terminate at depth 0 (the immediate opponent turn ends the search). The tree barely looks past the current turn. Argmax, by contrast, evaluates every reachable square with full path-probability weighting — effectively a one-ply exhaustive search.

3. **Static eval vs Argmax heuristic.** Argmax uses the full scripted move heuristic (field position, threat detection, block value, etc.) directly as its decision function. The static eval in MCTS is a pale shadow of this. MCTS would need either a better eval or a much larger budget to recover this signal via search.

## What would help

- **Better static eval**: incorporate ball carrier distance to end zone, turn number, cage detection. Even crude proxies would sharpen the search signal.
- **Larger budget**: at budget=200, avg depth ≈ 0.5. At budget=10,000+ the tree would be deeper and UCB could start to reason about multi-turn plans.
- **Argmax as a prior** (MCTS-Argmax): instead of the current softmax prior, use argmax scores directly. Would bias first expansions toward the best scripted moves.
- **Mixed eval**: for non-terminal leaves, blend staticEval with a fast single-call scripted score rather than a full rollout.
