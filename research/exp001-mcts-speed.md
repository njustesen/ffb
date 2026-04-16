# Experiment 001: MCTS Speed — Static Eval, Candidate Caching, Object Pooling

**Date:** 2026-04-14  
**Status:** Complete

## Motivation

The MCTS implementation (`BbMctsSearch`) was running at ~125 iter/s and using scripted rollouts (~5.5ms each) as its leaf evaluation. At budget=1000 with 12 parallel threads, the `MCTS-Prior` condition (condition F) caused an out-of-memory crash — the JVM heap grew to 8GB before G1GC failed to keep up. The goal was to push throughput into the thousands of iterations per second and fix the OOM so budget=1000 becomes usable.

## Changes Made

### Phase 1: Static evaluation + candidate caching + step-chain depth limit

**Problem:** Each MCTS leaf ran a full scripted rollout (~5.5ms), and `buildCandidates()` was called on every iteration (no caching).

1. **Static evaluation at leaf nodes** (`BbMctsSearch.staticEval`):  
   Replaced the scripted rollout with a sigmoid over two signals:  
   `sigmoid(TD_SCALE × scoreDelta + STANDING_SCALE × standingAdvantage/11)`  
   Cost: O(22 players) ≈ 0.05ms vs 5.5ms for a scripted rollout.

2. **Candidate caching** (`BbMctsNode.cachedCandidates` → later `cachedSet`):  
   `buildCandidates()` runs ~55 Dijkstra calls (5 per player × 11 players). Cached per node after first expansion. Invalidated via `areCandidatesValid()` — checks player states still match (players can be knocked down by stochastic dice, making cached actions invalid and causing infinite server step recursion).

3. **Step-chain depth limit** (`GameState.StepChainDepthLimitException`):  
   Blood Bowl server step execution is recursive. Stale cached actions caused infinite recursion (caught at JVM stack overflow after ~10s). Added a ThreadLocal opt-in depth counter that throws after 200 recursive `startNextStep()` calls (~2ms). Only enabled during MCTS budget loops.

4. **`MAX_ACTIVATION_ITERATIONS=500`** in `MatchRunner.advanceToNextActivationOrTurnEnd`:  
   Separate guard for infinite game-loop cases.

**Result after Phase 1:** 125 → 1257 iter/s (10× speedup), ms/iter = 0.796ms.

---

### Phase 2: Eliminate double `selectPlayer()` + ThreadLocal object pooling

**Problem:** At budget=1000 with 12 threads, `MCTS-Prior` (condition F) still OOMed. Root cause: `ScriptedActionPrior.computePrior()` called `MoveDecisionEngine.selectPlayer()` independently at every node expansion, even though `buildCandidates()` in the same expansion had already called it. This doubled the Dijkstra work (2 × ~55 calls = ~110 per expansion).

#### Fix 1: CandidateSet — reuse raw scores for prior computation

New class `CandidateSet` bundles the candidate list with the raw scores from `selectPlayer()`. `buildCandidates()` now returns a `CandidateSet`. `BbMctsNode.cachedSet` stores the set so both candidates and scores survive across iterations.

`IActionPrior` gained a default method:
```java
default double[] computePriorFromScores(List<BbAction> candidates, double[] rawScores, Game game)
```
`ScriptedActionPrior` overrides it to run `softmax(rawScores, T)` directly — no second `selectPlayer()` call:
```java
@Override
public double[] computePriorFromScores(List<BbAction> candidates, double[] rawScores, Game game) {
    if (rawScores == null || rawScores.length != candidates.size()) return computePrior(candidates, game);
    return PolicySampler.softmax(rawScores, T);
}
```

`setPriors()` now takes a `CandidateSet` and calls `computePriorFromScores` instead of `computePrior`.

**Impact:** Halved the number of `selectPlayer()` / Dijkstra calls at each node expansion.

#### Fix 2: ThreadLocal object pooling in `PathProbabilityFinder`

Each `findAllPaths()` call allocated ~15–30KB, of which ~40% came from three reusable objects: `PriorityQueue<PathNode>`, `HashMap<FieldCoordinate,Double> bestProb`, and `HashMap<FieldCoordinate,PathNode> bestNode`. These are now ThreadLocal pools:

```java
private static final ThreadLocal<PriorityQueue<PathNode>> TL_QUEUE = ThreadLocal.withInitial(PriorityQueue::new);
private static final ThreadLocal<HashMap<FieldCoordinate, Double>> TL_BEST_PROB = ThreadLocal.withInitial(HashMap::new);
private static final ThreadLocal<HashMap<FieldCoordinate, PathNode>> TL_BEST_NODE = ThreadLocal.withInitial(HashMap::new);
```

In `findAllPaths()`, replaced `new ...()` with `.get(); .clear()`. The result `HashMap` and `PathEntry` objects are still allocated fresh (they escape the method). `PathNode` objects are temporary and become unreachable once `reconstructPath()` extracts the `FieldCoordinate[]` chain.

**Impact:** ~40% reduction in per-Dijkstra allocation.

## Files Modified

| File | Change |
|------|--------|
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/mcts/BbMctsSearch.java` | Static eval, candidate caching, CandidateSet refactor, setPriors |
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/mcts/BbMctsNode.java` | `cachedCandidates` → `cachedSet` |
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/mcts/CandidateSet.java` | New: bundles actions + rawScores |
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/mcts/IActionPrior.java` | Added `computePriorFromScores` default method |
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/mcts/ScriptedActionPrior.java` | Override `computePriorFromScores` |
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/PathProbabilityFinder.java` | ThreadLocal pooling for queue + two maps |
| `ffb-server/src/main/java/com/fumbbl/ffb/server/GameState.java` | `StepChainDepthLimitException`, depth limit ThreadLocal |
| `ffb-ai/src/main/java/com/fumbbl/ffb/ai/simulation/MatchRunner.java` | `MAX_ACTIVATION_ITERATIONS`, iteration guard |

## Experiment Setup

```bash
mvn install -DskipTests -pl ffb-ai -am
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  -Dexec.args="/path/to/repo N --mcts-budget B"
```

Conditions run per experiment:
- **A**: Sample vs Random
- **B**: Argmax vs Random
- **C**: Sample vs Argmax
- **D**: Random vs Random
- **E**: MCTS-UCB vs Random
- **F**: MCTS-Prior vs Random

All games: Human vs Human roster, 10 games per condition.

## Results

### Throughput over time

| Phase | MCTS-UCB iter/s | MCTS-Prior iter/s |
|-------|-----------------|-------------------|
| Baseline (budget=100, before Phase 1) | 125 | 94 |
| After Phase 1 (budget=200) | 1257 | ~1200 |
| After Phase 2 (budget=200) | 1165 | 1216 |
| After Phase 2 (budget=1000) | 1098 | 823 |

### Budget=200, 10 games/condition (after Phase 2)

```
Condition                Wins  Draws   Loss  WinRate
A: Sample  vs Random        0     10      0  0.0%
B: Argmax  vs Random       10      0      0  100.0%
C: Sample  vs Argmax        0      3      7  0.0%
D: Random  vs Random        0     10      0  0.0%
E: MCTS-UCB    vs Random    1      9      0  10.0%
F: MCTS-Prior  vs Random    0     10      0  0.0%

Condition              iter/s   ms/iter   ms/dec   avg depth   max depth   nodes/dec
E: MCTS-UCB   vs Rand   1165    0.858     164.8      0.6          4          20.7
F: MCTS-Prior vs Rand   1216    0.822     157.8      0.4          3           9.4

Total elapsed: E=24.8s, F=41.7s
```

### Budget=1000, 10 games/condition (after Phase 2)

```
Condition                Wins  Draws   Loss  WinRate
A: Sample  vs Random        1      9      0  10.0%
B: Argmax  vs Random       10      0      0  100.0%
C: Sample  vs Argmax        0      5      5  0.0%
D: Random  vs Random        0     10      0  0.0%
E: MCTS-UCB    vs Random    1      9      0  10.0%
F: MCTS-Prior  vs Random    0     10      0  0.0%

Condition              iter/s   ms/iter   ms/dec    avg depth   max depth   nodes/dec
E: MCTS-UCB   vs Rand   1098    0.911     907.2       0.5          4          18.4
F: MCTS-Prior vs Rand    823    1.215    1209.8       0.4         11          21.7

Total elapsed: E=162.5s, F=204.7s  (both complete, no OOM)
```

The 204.7s for condition F covers 10 complete games. Each game involves roughly 17 MCTS decisions; each decision runs 1000 iterations at 823 iter/s ≈ 1.2s/decision, giving ~20s/game × 10 = 204s.

### Why MCTS-Prior is slower at budget=1000

At budget=1000, MCTS-Prior explores much deeper trees (max depth 11 vs 4 for UCB). Priors guide the search to exploit top candidates, building long chains. Each additional depth level adds one `advanceToNextActivationOrTurnEnd` call (~5 opponent activations × 56 Dijkstra each). Deeper trees → more Dijkstra per iteration → more GC pressure → slightly lower iter/s (823 vs 1098).

## Observations

- **MCTS vs Scripted**: At budget=200 and 1000, MCTS barely beats Random (10% win rate). This is expected — the static eval signal is weak (score + standing count), the search is shallow (avg depth < 1), and the games are short (11-aside scoring can be 0-0 in many games). Blood Bowl is a high-variance game.
- **Win rate noise**: With N=10, confidence intervals are very wide (e.g. 10% win rate has CI [1.8%–40.4%]). Need N≥50 to draw strong conclusions.
- **Variance in MCTS-Prior timing**: ms/match std dev is nearly as large as the mean at budget=1000 (13880ms std dev vs 20452ms mean). Some games involve many long turns; a single unlucky match can dominate. First run at budget=1000 appeared hung but was just a long-tail game.

## Open Questions

- Does MCTS actually improve play quality, and at what budget does it become reliably better than Argmax? (need N≥50 runs)
- Would a better static eval (e.g. ball position, cage formation, turn number) produce stronger search signals?
- Can `advanceToNextActivationOrTurnEnd` be made cheaper during traversal (e.g. lightweight opponent) to reduce GC at high budgets?
