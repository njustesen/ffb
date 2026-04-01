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
7. **ffb-ai** — Headless AI agent. Entry point: `com.fumbbl.ffb.ai.AiMain`. Extends `FantasyFootballClientAwt` with a hidden Swing window. `AiDecisionEngine` polls every 100 ms and responds to server-sent dialogs and active game states via `RandomStrategy`. Includes `GameSimulator` for JSON-based game state cloning (forward modelling scaffold).

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

To run a fully automated AI-vs-AI game:

```bash
./play-ai-vs-ai.sh
```

Both sides run headlessly. Logs go to `/tmp/ffb-ai-kalimar.log` and `/tmp/ffb-ai-battlelore.log`.

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

## Code Style

- Java 8, UTF-8 encoding, no wildcard imports.
- Import order: `com.fumbbl.*` first, then `javax.*`/`java.*`, with blank lines between groups.
- New external dependencies go in the root `pom.xml` first, then referenced in module POMs.
