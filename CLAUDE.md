# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FFB is a multi-module Java 8 Maven project implementing a WebSocket-based fantasy football (Blood Bowl) game used by [FUMBBL](https://fumbbl.com). It provides both a server and a Swing desktop client.

## Build & Test Commands

Run from the project root:

```bash
mvn clean install           # Full build of all modules
mvn install -DskipTests     # Build without tests

mvn test                                          # All tests
mvn test -Dtest=ClassName                         # Single test class
mvn test -Dtest=ClassName#methodName              # Single test method
mvn -pl ffb-server test                           # Tests for one module only
```

## Module Architecture

Build order (declared in root `pom.xml`):

1. **ffb-common** — Shared entities, rules, skills, injuries, modifiers, network command serialization, field coordinates, dice. Organized under `com.fumbbl.ffb.*` with sub-packages `bb2016/`, `bb2020/`, `bb2025/` for rule-version-specific implementations.
2. **ffb-tools** — Build-time utilities (icon folder rebuilding, etc.).
3. **ffb-server** — Jetty WebSocket server. Entry point: `com.fumbbl.ffb.server.FantasyFootballServer`. Manages game state (`GameState`, `GameCache`), a MySQL/MariaDB database layer (`db/`), command handlers (`handler/`, ~40+ classes), and session management. Requires an ini file with DB credentials and mode (`standalone` | `fumbbl`).
4. **ffb-client-logic** — Platform-agnostic client logic (the bulk of client code). Handles server command processing (`handler/`), game-phase state machines (`state/`), and 150+ dialog handlers (`dialog/`). Uses Tyrus WebSocket client.
5. **ffb-client** — AWT/Swing UI layer. Entry point: `com.fumbbl.ffb.client.FantasyFootballClientAwt`. Layer-based rendering (`layer/`), `FieldComponent`, `UserInterface`, `IconCache`, `ActionKeyBindings`.
6. **ffb-resources** — Packaged sound and icon assets JAR.
7. **ffb-ai** — Headless AI agent and in-process simulation harness.
   - **WebSocket agent** (`AiMain`): extends `FantasyFootballClientAwt` with a hidden Swing window; `AiDecisionEngine` polls every 100 ms and responds to server dialogs and game states.
   - **Headless simulation** (`simulation/MatchRunner`): runs complete games entirely in-process via `HeadlessFantasyFootballServer` — no server process, no WebSocket, ~500 ms/game. Supports four `AgentMode` values (`SCRIPTED_SAMPLE`, `SCRIPTED_ARGMAX`, `RANDOM`, and custom temperature) and runs comparative experiments with Wilson-score confidence intervals and per-level timing breakdowns.
   - **Move policy** (`MoveDecisionEngine`, `PathProbabilityFinder`, `ActionScore`): scores and selects player actions and target squares using path probability and field-position heuristics.
   - **Dialog policy** (`strategy/ScriptedStrategy`): scores all server-sent dialog types (block dice, re-rolls, fouls, apothecary, etc.) and samples decisions using a piecewise-linear temperature parameter: T=0 → argmax, T=0.5 → raw softmax policy (default), T=1 → uniform random.
   - **Sampling** (`PolicySampler`): softmax, argmax, and `sampleMixed`/`chooseBoolMixed` for the temperature-interpolated distribution.
   - **MCTS agent** (`mcts/` package): Blood Bowl MCTS search operating at the `INIT_SELECTING` granularity — each "action" is a `(player, PlayerAction)` pair. Key classes:
     - `BbMctsSearch` — UCB or PUCT multi-armed bandit; enumerates candidates via `MoveDecisionEngine.selectPlayer()`, runs scripted rollouts via `RolloutSetup` + `MatchRunner.runForActivations()`, returns most-visited candidate. Configure with `--mcts-budget N` (default 10) and `--rollout-activations N` (default 6).
     - `RolloutSetup.createFromMidGame(Game, HeadlessFantasyFootballServer)` — clones the `Game` model via `GameSimulator.cloneGame()`, creates a fresh `GameState` (empty step stack), pushes a `Select` sequence, and calls `startNextStep()` so the rollout starts at `INIT_SELECTING`. Critical: `GameSimulator.cloneGame()` only clones the `Game` model, not the step stack — the rollout GameState's stack starts empty.
     - `IActionPrior` — pluggable prior distribution for PUCT: `double[] computePrior(List<BbAction>, Game)`. Implementations: `ScriptedActionPrior` (softmax T=0.5 of `MoveDecisionEngine` raw scores), `UniformActionPrior` (reduces PUCT to UCB).
     - `BbMctsStats` — per-game statistics (rollout count, timing, branching factor, prior entropy); call `computeDerived()` before reading derived fields.
   - **Headless MCTS benchmark**: Pass `--mcts-budget N` to `MatchRunner` to run two additional conditions: `MCTS_UNIFORM` (UCB) and `MCTS_SCRIPTED` (PUCT with scripted prior), both vs `RANDOM`. Stats table printed after win-rate table.
   - **GUI MCTS**: Pass `-mcts-budget N` to `AiMain` (or use `./play-mcts.sh`) to enable MCTS in the WebSocket agent. `AiDecisionEngine.handleSelectPlayer()` delegates to `BbMctsSearch` when set.

The shade plugin embeds `ffb-common` into the server and client JARs for distribution.

## Key Architectural Patterns

- **Command-based communication**: Clients and server exchange serialized command objects (defined in `ffb-common/net/commands/`), queued and processed sequentially.
- **Step stack (server)**: Game sequences are `Step` classes pushed onto a stack; the top step processes commands and drives the game forward.
- **State machines (client)**: `ClientState` subclasses represent game phases; `ffb-client-logic` transitions between them based on server commands.
- **Factory + rule versions**: `FactoryManager` selects rule-specific implementations (skills, injuries, modifiers) for `bb2016`, `bb2020`, or `bb2025` rulesets.
- **Layer rendering**: The Swing client draws the field using stacked rendering layers, each responsible for a visual concern.

## Running Locally

To start a local two-player game, just run:

```bash
./play.sh
```

This handles everything: starting MariaDB, building if needed, initializing the DB, starting the server, and launching two client windows. In each window enter game name `LocalGame`, password `test`, click **Create**, then pick a team. The game starts when both coaches have chosen a team.

To play against the AI agent (human vs AI):

```bash
./play.sh --ai
```

The human client opens as Kalimar; the AI agent joins headlessly as BattleLore.

To run a fully automated AI-vs-AI game (WebSocket-based, requires MariaDB + server):

```bash
./play-ai-vs-ai.sh
```

Both sides run headlessly. Logs go to `/tmp/ffb-ai-kalimar.log` and `/tmp/ffb-ai-battlelore.log`.

To play against the MCTS AI agent (human vs MCTS AI):

```bash
./play-mcts.sh                  # default budget 10
./play-mcts.sh --mcts-budget 20 # larger search budget
```

To run a visual MCTS AI vs MCTS AI game:

```bash
./play-ai-vs-ai-mcts.sh
```

To run the headless simulation experiment (no server, no DB — all in-process):

```bash
mvn install -DskipTests -pl ffb-ai -am
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  -Dexec.args="/path/to/repo 200"
```

This runs 200 games per condition across four agent pairings (Sample vs Random, Argmax vs Random, Sample vs Argmax, Random vs Random) and prints win rates with 95% Wilson CI and per-level timing statistics. Average game time is ~500 ms. Pass `--mcts-budget N` to also run MCTS conditions (MCTS-UCB and MCTS-Scripted vs Random). The `run-games.sh` script is an older alternative that drives games through a real server.

### One-time setup prerequisites (already done if `play.sh` has been run before)
- MariaDB installed via Homebrew (`brew install mariadb`)
- Maven installed via Homebrew (`brew install maven`)
- `ffb-server/server.ini` placeholders replaced (DB credentials, log paths, admin password)
- `ffb-server/target/lib/` populated (server assembly; `play.sh` handles this)
- `ffblive` database created with schema initialized (coaches: Kalimar, BattleLore, LordCrunchy, LordMisery — all with password `test`)

### Key notes
- The server **must be run from `ffb-server/`** — `GameCache` loads `rosters/`, `teams/`, `setups/` as relative paths.
- The server JAR (`FantasyFootballServer.jar`) only bundles `ffb-common`. All other deps (Jetty, MySQL connector, etc.) are in `target/lib/` next to the JAR.
- The client JAR has no `Class-Path` manifest entry — the classpath must be built manually from `lib/*.jar`. `play.sh` does this.
- **Ruleset**: The server runs **BB2025** rules (set in `UtilServerStartGame.addDefaultGameOptions`). The team files in `teams/` use LRB6 roster definitions, so player positions/stats are LRB6 but game mechanics follow BB2025.
- **Icons**: Player icons are loaded from `http://localhost:2224/icons/` (set in roster XML `<baseIconPath>`), mapped to bundled icons via `icons.ini` in the client JAR. The resources JAR (`ffb-resources`) uses BB2020/BB2025 icon names. The roster XMLs in `ffb-server/rosters/` have been updated to use these names.
- Building on Java 17+ requires a fix in `ClientCommandHandlerGameState.java`: `ForkJoinPool.commonPool().invokeAll(tasks)` now throws `InterruptedException` and must be caught inside the `PrivilegedAction` lambda.

## Type & File Index

Use this to jump directly to the right file instead of grepping.

### Core model types (`ffb-common`)

| Type | Package path |
|------|-------------|
| `Game` | `ffb-common/.../ffb/model/Game.java` |
| `Player` | `ffb-common/.../ffb/model/Player.java` |
| `Team` | `ffb-common/.../ffb/model/Team.java` |
| `FieldModel` | `ffb-common/.../ffb/model/FieldModel.java` |
| `ActingPlayer` | `ffb-common/.../ffb/model/ActingPlayer.java` |
| `GameOptions` | `ffb-common/.../ffb/model/GameOptions.java` |
| Skill definitions | `ffb-common/.../ffb/skill/` (base) and `ffb-common/.../ffb/model/skill/` (model) |
| Rule-version impls | `ffb-common/.../ffb/bb2025/` (active ruleset) |

Full base path: `ffb-common/src/main/java/com/fumbbl/ffb/`

### Network commands (`ffb-common`)

All command objects live in `ffb-common/.../ffb/net/commands/`. Naming convention:
- Client→Server: `ClientCommand<Name>.java`
- Server→Client: `ServerCommand<Name>.java`

### Server step stack (`ffb-server`)

All step classes in `ffb-server/.../ffb/server/step/`. Organized by game phase:
- `step/bb2025/` — BB2025-specific step implementations
- `step/action/`, `step/block/`, `step/move/`, etc. — phase-specific subdirs

To find the step handling a specific game event, grep for the step name or the net command it processes.

### Server command handlers (`ffb-server`)

All handler classes in `ffb-server/.../ffb/server/handler/`. One handler class per command type, named `ServerCommandHandler<Name>.java`.

### Client state machines (`ffb-client-logic`)

State classes in `ffb-client-logic/.../ffb/client/state/`. Dialog handlers in `ffb-client-logic/.../ffb/client/dialog/`.

### AI & simulation (`ffb-ai`)

See `ffb-ai/CLAUDE.md` for the full breakdown.

## Code Style

- Java 8, UTF-8 encoding, no wildcard imports.
- Import order: `com.fumbbl.*` first, then `javax.*`/`java.*`, with blank lines between groups.
- New external dependencies go in the root `pom.xml` first, then referenced in module POMs.
