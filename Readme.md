# Fantasy Football Server/Client (FFB)

FFB is the Fantasy Football software used by [FUMBBL](https://fumbbl.com).

Client and server are both implemented in Java 8 with Swing/AWT.

---

## Quick Start (local game)

```bash
./play.sh                # Two human clients, each gets a login window
./play.sh --ai           # One human client (Kalimar) + AI opponent (BattleLore)
./play-ai-vs-ai.sh       # Fully headless AI-vs-AI game, no windows
```

**Prerequisites (one-time):** MariaDB and Maven via Homebrew. `play.sh` handles
everything else (build, DB creation, schema init, server start).

In each human client window enter:
- Game name: `LocalGame`
- Password: `test`
- Click **Create**, then pick a team.

The game starts when both coaches have chosen a team.

---

## Module Structure

| Module | Description |
|---|---|
| **ffb-common** | Shared entities, rules, skills, injuries, network commands, field coordinates, dice. Rule-version sub-packages: `bb2016/`, `bb2020/`, `bb2025/`. |
| **ffb-tools** | Build-time utilities (icon folder rebuilding, etc.). |
| **ffb-server** | Jetty WebSocket server. Manages game state (`GameState`, `GameCache`), MySQL/MariaDB persistence, 40+ command handlers. Requires `server.ini`. |
| **ffb-client-logic** | Platform-agnostic client logic: server command processing, game-phase state machines, 150+ dialog handlers. Uses Tyrus WebSocket client. |
| **ffb-client** | AWT/Swing UI layer. Layer-based field rendering, `UserInterface`, `IconCache`, `ActionKeyBindings`. |
| **ffb-resources** | Packaged sound and icon assets JAR. |
| **ffb-ai** | Headless AI agent. Extends the client, suppresses the UI window, and drives every game decision programmatically (random strategy). |

---

## Build Commands

```bash
mvn clean install            # Full build of all modules
mvn install -DskipTests      # Build without tests
mvn test                     # All tests
mvn test -Dtest=ClassName    # Single test class
mvn -pl ffb-server test      # Tests for one module only
```

---

## Architecture Summary

### Server

Entry point: `com.fumbbl.ffb.server.FantasyFootballServer`

Game sequences are `Step` classes pushed onto a stack. The top step receives
commands from a queue and either processes them (advancing the stack) or waits
for further input. Field-model and game-data changes are published back to
clients as serialized command objects.

| Argument | Description |
|---|---|
| `[mode]` | `standalone` (local), `fumbbl` (production), `standaloneInitDb` / `fumbblInitDb` (schema setup) |
| `-inifile [filepath]` | Path to server config (`ffb-server/server.ini`). |
| `-override [filepath]` | Environment-specific overrides; same syntax as `inifile`. |

Requires MySQL ≤ 5.6 or MariaDB ≤ 10.4 (connector 5.1.27).

### Client

Entry point: `com.fumbbl.ffb.client.FantasyFootballClientAwt`

Commands from the server are enqueued and processed one-by-one. Mouse and
keyboard events are handled by `ClientState` subclasses that specialize
behaviour for each game phase (movement, setup, out-of-turn sequences, etc.).

| Argument | Description |
|---|---|
| `[mode]` | `-player`, `-spectator`, or `-replay` |
| `-server [hostname]` | Hostname of the server to connect to |
| `-port [port]` | WebSocket port as defined in server config |
| `-coach [coachname]` | Coach name used to log in |
| `-auth [hexstring]` | Pre-encoded login credential (bypasses dialog) |
| `-teamid [teamid]` | ID of a locally stored team (player mode only) |
| `-gameId [gameId]` | Numeric game ID (replay mode only) |

### Communication

Clients and server exchange serialized `NetCommand` objects defined in
`ffb-common/net/commands/`. Each side maintains a command queue; commands are
processed sequentially to keep state consistent.

### Rule Versions

`FactoryManager` selects rule-specific implementations for skills, injuries, and
modifiers. Sub-packages `bb2016/`, `bb2020/`, `bb2025/` contain the
version-specific classes. The local server is configured to run **BB2025** rules.

---

## AI Agent (`ffb-ai`)

The `ffb-ai` module provides a headless AI client that can play a full game
without any human interaction or visible UI.

### Entry Point

```
com.fumbbl.ffb.ai.AiMain
  -coach     <coachname>    Coach name (must exist in the DB)
  -password  <password>     Plain-text password
  -server    <hostname>     Server hostname (default: localhost)
  -port      <port>         Server port (default: 22227)
  -home                     Pass this flag for the home-side player
```

### Key Classes

| Class | Role |
|---|---|
| `AiClient` | Extends `FantasyFootballClientAwt`, hides the UI window, starts `AiDecisionEngine` as a daemon thread. |
| `AiDecisionEngine` | Polls every 100 ms. Handles login (via reflection into `LoginLogicModule`), responds to server dialogs, and drives active-state actions. |
| `RandomStrategy` | Stateless helper — maps every `IDialogParameter` type to a valid (randomly chosen) response sent via `ClientCommunication`. |
| `GameSimulator` | Clones a `Game` state via JSON round-trip for forward-search planning (foundation; search not yet implemented). |

### Running AI Games

```bash
# Human vs AI
./play.sh --ai

# AI vs AI (fully headless)
./play-ai-vs-ai.sh
# Logs: /tmp/ffb-ai-kalimar.log, /tmp/ffb-ai-battlelore.log, /tmp/ffb-server.log
```

---

## Headless Simulation (`ffb-ai/simulation`)

The simulation package provides a fully in-memory game engine — no network, no database,
no Swing window. A complete game runs in ~9 ms.

### Key Classes

| Class | Role |
|---|---|
| `SimulationLoop` | Drives a `GameState` to completion by injecting commands directly into the server-side step stack. Synchronous; no threads or polling delays. Safety cap: 100 000 iterations. |
| `HeadlessGameSetup` | Constructs a fully-initialised `GameState` from XML rosters — mirrors `ServerCommandHandlerJoinApproved`, no DB required. |
| `HeadlessFantasyFootballServer` | Minimal server stub: all network and persistence calls are no-ops. Uses sentinel sessions to distinguish home vs. away. |
| `GameSimulator` | Clones a `Game` via JSON round-trip — foundation for forward-search planning (search not yet implemented). |
| `CapturingClientCommunication` | Intercepts dialog responses from `RandomStrategy` and converts them to server-side `ReceivedCommand` objects instead of sending them over the network. |
| `SimulationBenchmark` | Runs N complete games (default 20) and reports per-step CPU-time breakdowns. |

### Running the benchmark

```bash
java -cp "ffb-ai/target/ffb-ai-*.jar:ffb-server/target/lib/*" \
     com.fumbbl.ffb.ai.simulation.SimulationBenchmark .
```

Output includes per-game setup/kickoff/drive timing, per-turn microseconds, and the
top-20 hottest steps by accumulated CPU time.

### Extending the AI

Replace `RandomStrategy` with your own implementation to plug in a smarter decision engine.
`GameSimulator` is the scaffolding for forward-search (MCTS, minimax, etc.).
