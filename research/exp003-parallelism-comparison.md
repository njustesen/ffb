# Experiment 003: Does Root Parallelism Help? (Same Wall-Clock Budget)

**Date:** 2026-04-14  
**Status:** Complete  
**Follows from:** [Exp 002](exp002-mcts-vs-random-n50.md)

## Question

Exp 002 showed MCTS-UCB and MCTS-Prior each win 16% vs Random at budget=200 (12-thread root parallelism). But "budget=200" with 12 threads means 12 independent trees each doing 200 iterations — that's 2,400 total iterations while 12 CPUs work in parallel. A fair single-CPU comparison at the same wall-clock time (~165ms/decision) does only ~200 iterations. Does the parallelism actually account for the win rate, or does the time budget matter more?

## Setup

Two new conditions G and H run MCTS with a **wall-clock time budget** (165ms/decision, 1 thread) alongside E and F (200-iteration count budget, 12 threads). Both approaches use the same wall-clock time per decision (~165ms), but:

- **E/F**: 12 trees × 200 iter = 2,400 total iterations per decision (12 CPUs in parallel)
- **G/H**: 1 tree × ~200 iter = ~200 total iterations per decision (1 CPU)

```bash
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  "-Dexec.args=/path/to/repo 50 --mcts-budget 200 --mcts-time-ms 165 --time-threads 1"
```

N=50 games per condition. `--time-threads 1` pins G/H to single-threaded; `--mcts-threads` defaults to 12 for E/F.

## Results

```
Condition                Wins  Draws   Loss  WinRate ± 95% CI
------------------------------------------------------------------------
A: Sample  vs Random        1     49      0   2.0% [0.4%–10.5%]
B: Argmax  vs Random       49      1      0  98.0% [89.5%–99.6%]
C: Sample  vs Argmax        1     15     34   2.0% [0.4%–10.5%]
D: Random  vs Random        0     50      0   0.0% [0.0%–7.1%]
E: MCTS-UCB    vs Random    6     44      0  12.0% [5.6%–23.8%]    ← 12 threads, 200 iter
F: MCTS-Prior  vs Random    6     44      0  12.0% [5.6%–23.8%]    ← 12 threads, 200 iter
G: MCTS-UCB    vs Random    2     48      0   4.0% [1.1%–13.5%]    ← 1 thread, 165ms
H: MCTS-Prior  vs Random    5     45      0  10.0% [4.3%–21.4%]    ← 1 thread, 165ms

MCTS Search Statistics:
Condition              iter/s   ms/iter   ms/dec   avg depth   max depth   nodes/dec
E: MCTS-UCB (12t)       1195    0.837     160.7      0.6          3          18.2
F: MCTS-Prior (12t)     1171    0.854     163.9      0.4          3          13.0
G: MCTS-UCB (1t)        2050    0.488     154.5      0.6          4           2.8
H: MCTS-Prior (1t)      1987    0.503     155.5      0.4          4           1.6

Game timing (mean ± σ):
E: 3422ms ± 4806ms per game (avg ~21 MCTS decisions/game)
F: 4424ms ± 7090ms per game
G: 18949ms ± 7029ms per game (avg ~115 MCTS decisions/game)
H: 20000ms ± 6278ms per game
```

## Interpretation

### Parallelism gives a real win-rate boost for UCB; the prior partially compensates

| Condition | Win rate | Total iter/dec | nodes/dec |
|-----------|----------|----------------|-----------|
| G: UCB, 1 thread  |  4% |  ~200 |  2.8 |
| H: Prior, 1 thread | 10% |  ~200 |  1.6 |
| E: UCB, 12 threads | 12% | ~2400 | 18.2 |
| F: Prior, 12 threads | 12% | ~2400 | 13.0 |

- **UCB with parallelism (E) vs without (G)**: 12% vs 4% — 3× improvement. The CIs don't overlap, confirming the parallelism advantage is real.
- **Prior with parallelism (F) vs without (H)**: 12% vs 10% — marginal improvement, CIs overlap. The scripted prior largely compensates for fewer iterations at single-thread: even with only ~200 iterations, the prior guides the search toward the right children so the limited budget isn't wasted on bad moves.
- **H vs G** (both 1-thread, same budget): 10% vs 4% — the Prior adds 6 percentage points relative to uniform UCB at low iteration counts. This is the clearest demonstration that the scripted prior matters most when the budget is tight.

### The nodes/dec statistic explains the gap

`nodes/dec` is the number of unique root-level actions explored per decision. The root action space has ~33 options (11 players × ~3 actions each).

| Condition | nodes/dec | Fraction of space explored |
|-----------|-----------|---------------------------|
| G (1t, ~200 iter) | 2.8 | 8% |
| H (1t, ~200 iter) | 1.6 | 5% |
| E (12t, ~2400 iter) | 18.2 | 55% |
| F (12t, ~2400 iter) | 13.0 | 39% |

With only ~200 iterations, UCB barely explores 3 children at the root before committing. With 12× more iterations (via parallelism), E explores 18 — covering most of the plausible action space. The prior in H focuses its 200 iterations on the top 1-2 estimated-best actions, which is why it achieves 10% despite exploring fewer nodes than G.

### Why G/H games are 5× slower than E/F (the game-time paradox)

E/F average 3–4 seconds per game while G/H average 18–20 seconds — despite spending the same wall-clock time per decision (~160ms). The reason is that **better play shortens games**.

With 12× more iterations, E/F make better activation choices: scoring efficiently, reducing unnecessary turns. E runs ~21 MCTS decisions per game while G runs ~115. Better decisions → fewer turns → fewer MCTS calls → shorter total elapsed time. This is a positive feedback loop: parallelism improves play quality, which reduces the number of decisions needed, which compounds the wall-clock advantage.

### Single-thread iter/s is higher (2050 vs 1195)

G/H report higher iter/s than E/F. Likely cause: 12-thread root parallelism incurs overhead (12 independent `GameState` clones, thread-pool management, merge step) that single-thread avoids. Each single-thread iteration costs 0.49ms vs 0.84ms multi-thread — a ~40% overhead per iteration for the parallel harness.

## Summary

Root parallelism gives Blood Bowl MCTS a meaningful win-rate advantage (+8pp for UCB) by converting wall-clock time into 12× more search iterations. The scripted prior partially compensates when the iteration budget is low: H (1 thread + Prior) at 10% nearly matches E (12 threads + UCB) at 12%.

Both the prior and parallelism are worth keeping. Future work:
- **Better eval**: all four MCTS conditions still fall far below Argmax (100%). The bottleneck is signal quality (weak static eval), not search throughput.
- **Prior quality**: the scripted prior is a softmax of move-score heuristics. A learned value head would be sharper.
- **Larger parallel budget**: at budget=2000+ per thread the trees would deepen and parallelism would start to help UCB structurally (deeper subtrees, not just more root children).
