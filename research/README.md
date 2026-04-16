# FFB Research Diary

Experiments on the Blood Bowl AI engine — MCTS, training data, and simulation.

## Contents

| # | Title | Date | Topic | Key Result |
|---|-------|------|-------|------------|
| [001](exp001-mcts-speed.md) | MCTS Speed: Static Eval, Candidate Caching, Object Pooling | 2026-04-14 | Speed & memory | 10× iter/s speedup; budget=1000 OOM fixed |
| [002](exp002-mcts-vs-random-n50.md) | Can MCTS Beat Random? (N=50) | 2026-04-14 | Search quality | MCTS 16% win rate [8.3%–28.5%] vs Random; 2× better than Sample but far below Argmax |
| [003](exp003-parallelism-comparison.md) | Root Parallelism vs Single-Thread (Same Wall-Clock) | 2026-04-14 | Parallelism | 12-thread (12% win) vs 1-thread (4% UCB / 10% Prior); Prior compensates for fewer iterations |
| [004](exp004-phase1-heuristic-eval.md) | Phase 1 Enhanced Heuristic Eval — Ceiling Analysis | 2026-04-14 | Eval quality | 14% vs Random; 0–2% vs Argmax; avg depth=0.6 confirms tree barely exists at budget=200 |
| [005](exp005-argmax-leaf-eval.md) | Argmax-Quality Leaf Eval + Time-Based Budget | 2026-04-16 | Eval quality | Time budget works; normalization bug fixed; avg depth still 0.9; N=20 sweep interrupted |
