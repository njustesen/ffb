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
| **ffb-ai** | AI agents and headless simulation harness. See below. |
| **ffb-mcp** | Python MCP server exposing game management, replay pipeline, and ML training as Claude tools. |
| **ffb-ml** | Python ML pipeline: feature extraction, BC model training, ONNX export, and evaluation. |

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

The `ffb-ai` module provides two AI agent modes and a headless simulation harness.

### WebSocket Agent

A headless client that connects to a running server over WebSocket. Entry point: `com.fumbbl.ffb.ai.AiMain`.

```
  -coach     <coachname>    Coach name (must exist in the DB)
  -password  <password>     Plain-text password
  -server    <hostname>     Server hostname (default: localhost)
  -port      <port>         Server port (default: 22227)
  -home                     Pass this flag for the home-side player
```

`AiDecisionEngine` polls every 100 ms. It drives all game decisions via:
- **`ScriptedStrategy`** — scores all dialog types (block dice, re-rolls, fouls, apothecary, etc.) and samples decisions using a piecewise-linear temperature: `T=0` → argmax, `T=0.5` → softmax (default), `T=1` → uniform random.
- **`MoveDecisionEngine`** / **`PathProbabilityFinder`** — scores player actions and target squares using path probability and field-position heuristics.

### Headless Simulation

A fully in-process game engine — no network, no database, no Swing. A complete game runs in ~500 ms.

Key classes:

| Class | Role |
|---|---|
| `SimulationLoop` | Drives a `GameState` to completion by injecting commands directly into the step stack. Synchronous; no threads or polling. |
| `HeadlessGameSetup` | Constructs a fully-initialised `GameState` from XML rosters — no DB required. |
| `HeadlessFantasyFootballServer` | Minimal server stub: all network and persistence calls are no-ops. |
| `CapturingClientCommunication` | Intercepts dialog responses and converts them to server-side `ReceivedCommand` objects instead of sending over the network. |
| `MatchRunner` | Runs N games across agent pairings (Sample vs Random, Argmax vs Random, Sample vs Argmax, Random vs Random) and prints win rates with 95% Wilson CI. |
| `ReplayGenerator` | Parallel multi-threaded generator — writes headless games as gzip-compressed `.ffbr` replay files. Supports race filtering and temperature control. |
| `TrainingDataExporter` | Exports per-decision JSONL training data (dialog, player-select, move-target) from headless games. |
| `GameStateSerializer` | Serializes full `GameState` snapshots (board, players, scores) to JSON for training data and replays. |
| `FeatureExtractor` | Java-side feature extraction: converts serialized game state into board channels, non-spatial vectors, and player-encoder inputs matching the Python model's input spec. |
| `OnnxModelAgent` | Loads three exported ONNX BC models (dialog, player-select, move-target) and uses them as the agent policy in evaluation. Falls back to `ScriptedStrategy` for dialog when needed. |
| `EvalRunner` | Evaluates the trained BC model against Random and ScriptedArgmax baselines across multiple conditions. |

### Running the Headless Simulation

```bash
mvn install -DskipTests -pl ffb-ai -am
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
  -Dexec.args="/path/to/repo 200"
```

Runs 200 games per condition and prints win rates with 95% Wilson CI and per-level timing statistics.

```bash
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.ReplayGenerator \
  -Dexec.args="--output ./replays --games 1000 --threads 4"
```

```bash
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.TrainingDataExporter \
  -Dexec.args="--output ffb-ml/data --games 500"
```

---

## ML Pipeline (`ffb-ml`)

Python pipeline for training a Behavioral Cloning (BC) model from headless-simulation data.

### Model

Three decision heads share a CNN+MLP backbone:

| Head | Task |
|---|---|
| `dialog_head` | Per-dialog-type linear → softmax over discrete options |
| `player_select_head` | Scores each candidate player individually via a `PlayerEncoder` |
| `move_target_head` | Spatial CNN head — scores all 26×15 board squares, masked to reachable positions |

`PlayerEncoder` embeds skills (learned embeddings, mean-pooled) and normalised stats (MA/ST/AG/AV/PA) into a shared player representation trained end-to-end.

Three model scales are supported: `micro`, `small`, `medium`.

### Pipeline Steps

```bash
# 1. Extract numpy feature shards from JSONL training data
python ffb-ml/src/extract_features.py --input ffb-ml/data --output ffb-ml/features

# 2. Train the BC model
python ffb-ml/src/train.py \
  --features ffb-ml/features --output ffb-ml/checkpoints --scale small

# 3. Export to ONNX (three files: dialog, player_select, move_target)
python ffb-ml/src/export_model.py \
  --checkpoint ffb-ml/checkpoints/small_best.pt \
  --output /tmp/bc_model

# 4. Evaluate the ONNX model in the Java simulation
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.EvalRunner \
  -Dexec.args="--model /tmp/bc_model --vocab ffb-ml/features/vocab.json --games 100"
```

A value-head scaling sweep (win/score/casualties/SPP prediction) can be run with:

```bash
bash run_value_sweep.sh
```

---

## MCP Server (`ffb-mcp`)

A Python [MCP](https://modelcontextprotocol.io) server that exposes game management,
simulation, and the full ML training pipeline as tools callable by Claude.

### Starting the MCP server

```bash
cd ffb-mcp
pip install mcp
python server.py
```

### Tool Groups

| Group | Description |
|---|---|
| Team management | `list_teams`, `get_team`, `create_team`, `list_rosters`, `get_roster` |
| Coach DB | `list_coaches`, `create_coach`, `delete_coach`, `set_coach_password` |
| Server control | `start_server`, `stop_server`, `server_status`, `build_project` |
| Game management | `schedule_game`, `list_games`, `get_game`, `get_game_state`, `get_game_result`, `close_game`, `delete_game`, `concede_game` |
| Match launchers | `run_human_vs_human`, `run_human_vs_ai`, `run_ai_vs_ai`, `run_match_runner`, `run_games_batch` |
| Replay pipeline | `generate_replays`, `download_replays`, `parse_replays`, `replay_status` |

The MCP server can drive the entire BC training loop — from game generation through
feature extraction, training, and ONNX export — without leaving Claude.
