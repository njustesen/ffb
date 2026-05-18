# CLAUDE.md — ffb-ai

Headless AI agent and simulation harness for the FFB Blood Bowl engine. Most active development happens here.

See the root `CLAUDE.md` for project-wide build commands, architecture, and setup.

## Package Structure

```
com.fumbbl.ffb.ai/
├── AiMain.java                          # WebSocket AI agent entry point
├── MoveDecisionEngine.java              # Scores and selects player actions
├── PathProbabilityFinder.java           # Path probability computation
├── ActionScore.java                     # Action scoring data class
├── PolicySampler.java                   # Softmax / argmax / sampleMixed
├── BoardVisualizer.java                 # Debug board printing
├── client/
│   ├── AiDecisionEngine.java            # Polls server state, triggers decisions
│   ├── AiClient.java                    # Headless Swing window wrapper
│   └── MovePolicyState.java
├── mcts/
│   ├── BbMctsSearch.java                # UCB/PUCT search loop
│   ├── BbMctsNode.java                  # Tree node
│   ├── BbMctsStats.java                 # Per-game stats (call computeDerived() first)
│   ├── BbAction.java                    # (player, PlayerAction) pair
│   ├── CandidateSet.java
│   ├── IActionPrior.java                # Interface: double[] computePrior(List<BbAction>, Game)
│   ├── ScriptedActionPrior.java         # Softmax T=0.5 of MoveDecisionEngine scores
│   ├── UniformActionPrior.java          # Uniform → reduces PUCT to UCB
│   ├── ILeafEval.java
│   ├── StaticLeafEval.java
│   └── RolloutSetup.java                # Clones game for rollout; see note below
├── simulation/
│   ├── MatchRunner.java                 # Entry point for headless sim
│   ├── HeadlessFantasyFootballServer.java
│   ├── HeadlessGameSetup.java
│   ├── GameSimulator.java               # cloneGame() — clones Game model only, NOT step stack
│   ├── SimulationLoop.java
│   ├── EvalRunner.java
│   ├── FeatureExtractor.java
│   ├── GameSnapshot.java
│   ├── GameStateSerializer.java
│   ├── RolloutSetup.java
│   ├── OnnxLeafEval.java                # ONNX model for leaf evaluation
│   ├── OnnxModelAgent.java
│   ├── JsonlTrainingDataCollector.java
│   ├── TrainingDataExporter.java
│   └── [replay/download utilities]
├── parity/
│   ├── ParityRunner.java                # Parity test vs Rust engine
│   └── Xoshiro256StarStar.java          # Same RNG as Rust port
└── strategy/
    ├── ScriptedStrategy.java            # Temperature-based dialog policy
    ├── RandomStrategy.java
    └── DecisionLog.java
```

## Key Classes

**`MatchRunner`** — headless simulation entry point. Runs complete games in-process (no server, no DB, ~500 ms/game). Supports `AgentMode`: `SCRIPTED_SAMPLE`, `SCRIPTED_ARGMAX`, `RANDOM`, and custom temperature.

**`RolloutSetup.createFromMidGame(Game, HeadlessFantasyFootballServer)`** — clones `Game` model via `GameSimulator.cloneGame()`, creates fresh `GameState` (empty step stack), pushes a `Select` sequence. Critical: the clone has no step stack — rollout starts at `INIT_SELECTING`.

**`BbMctsSearch`** — UCB or PUCT search. Candidates come from `MoveDecisionEngine.selectPlayer()`. Configure with `--mcts-budget N` (default 10) and `--rollout-activations N` (default 6). Returns most-visited candidate.

**`ScriptedStrategy`** — scores all dialog types (block dice, re-rolls, fouls, apothecary, etc.) with piecewise-linear temperature: T=0 → argmax, T=0.5 → softmax (default), T=1 → uniform.

## Running the Headless Sim

```bash
# Build only ffb-ai and deps (faster than full build)
mvn install -DskipTests -pl ffb-ai -am

# Run headless sim (200 games per condition)
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  -Dexec.args="/path/to/repo 200"

# With MCTS conditions
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  -Dexec.args="/path/to/repo 200 --mcts-budget 20"
```

## Extending the Action Prior

Implement `IActionPrior`:
```java
double[] computePrior(List<BbAction> candidates, Game game);
```
Pass to `BbMctsSearch` constructor. `ScriptedActionPrior` (softmax T=0.5) and `UniformActionPrior` are the existing implementations.

## Parity Testing (vs Rust)

`ParityRunner` uses `Xoshiro256StarStar` to seed games identically to the Rust port, then compares JSONL step logs. Run via `MatchRunner` with parity flags or directly.
