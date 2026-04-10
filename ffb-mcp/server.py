"""
FFB MCP Server — exposes Fantasy Football Blood Bowl game management as MCP tools.

Tools are grouped into tiers:
  - Standalone: team/roster XML management, build, headless simulation (no server needed)
  - DB only: coach CRUD via MariaDB
  - Server-required: game management via admin/gamestate HTTP API
  - Match launchers: start/stop server, run matches
  - Replay pipeline: download FUMBBL replays and parse them into JSONL training data
"""

import collections
import glob
import json
import subprocess
import sys
import os
from pathlib import Path

# Ensure sibling modules are importable when run as a script
sys.path.insert(0, os.path.dirname(__file__))

# ── Replay pipeline paths ─────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPLAY_DIR   = _PROJECT_ROOT / "replays" / "fumbbl"
_DATA_DIR     = _PROJECT_ROOT / "ffb-ml" / "data"
_MANIFEST     = _REPLAY_DIR / "manifest.jsonl"
_FETCH_SCRIPT = _PROJECT_ROOT / "ffb-ml" / "fetch_fumbbl_replays.py"

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ffb")

# ── Team management (standalone) ─────────────────────────────────────────────

@mcp.tool()
def list_teams() -> list[dict]:
    """List all teams from ffb-server/teams/*.xml with summary info (id, coach, name, race, playerCount)."""
    from team_xml import list_teams as _fn
    return _fn()


@mcp.tool()
def get_team(team_id: str) -> dict | None:
    """
    Get full details of a team by its XML id attribute (e.g. 'teamHumanKalimar').
    Returns all players with skills, injuries, status, and SPPs.
    """
    from team_xml import get_team as _fn
    return _fn(team_id)


@mcp.tool()
def create_team(
    team_id: str,
    coach: str,
    name: str,
    race: str,
    roster_id: str,
    re_rolls: int,
    fan_factor: int,
    apothecaries: int,
    team_rating: int,
    players: list[dict],
) -> dict:
    """
    Create a new team XML file in ffb-server/teams/.
    Each player dict: {nr: int, name: str, positionId: str, skills?: list[str], gender?: str}.
    roster_id example: 'human.lrb6'. Use list_rosters() to find valid roster ids.
    Returns the filename created.
    """
    from team_xml import create_team as _fn
    path = _fn(team_id, coach, name, race, roster_id, re_rolls,
                fan_factor, apothecaries, team_rating, players)
    return {"created": str(path), "team_id": team_id}


@mcp.tool()
def list_rosters() -> list[dict]:
    """List all race roster definitions from ffb-server/rosters/*.xml (positions, costs, base stats, skills)."""
    from team_xml import list_rosters as _fn
    return _fn()


@mcp.tool()
def get_roster(roster_id: str) -> dict | None:
    """
    Get full roster definition for one race by roster id (e.g. 'human.lrb6', 'orc.lrb6').
    Use list_rosters() to find valid roster ids.
    """
    from team_xml import get_roster as _fn
    return _fn(roster_id)


# ── Coach management (requires MariaDB) ──────────────────────────────────────

@mcp.tool()
def list_coaches() -> list[dict]:
    """List all coaches from the ffblive MariaDB database. Requires MariaDB running."""
    from coach_db import list_coaches as _fn
    return _fn()


@mcp.tool()
def create_coach(name: str, password: str) -> dict:
    """
    Create a new coach in the database. Password is stored as MD5 hash.
    Requires MariaDB running.
    """
    from coach_db import create_coach as _fn
    _fn(name, password)
    return {"created": name}


@mcp.tool()
def delete_coach(name: str) -> dict:
    """Delete a coach from the database by name. Requires MariaDB running."""
    from coach_db import delete_coach as _fn
    _fn(name)
    return {"deleted": name}


@mcp.tool()
def set_coach_password(name: str, new_password: str) -> dict:
    """Reset a coach's login password. Requires MariaDB running."""
    from coach_db import set_coach_password as _fn
    _fn(name, new_password)
    return {"updated": name}


# ── Server lifecycle ──────────────────────────────────────────────────────────

@mcp.tool()
def server_status() -> dict:
    """Check whether MariaDB and the FFB server are running, and what port the server uses."""
    from server_control import server_status as _fn
    return _fn()


@mcp.tool()
def start_server() -> dict:
    """
    Start MariaDB and the FFB server. The FFB JAR must already be built.
    Returns status for each component. Use build_project() first if needed.
    """
    from server_control import start_mariadb, start_ffb_server
    db_result = start_mariadb()
    srv_result = start_ffb_server()
    return {"mariadb": db_result, "ffb_server": srv_result}


@mcp.tool()
def stop_server() -> dict:
    """Stop the FFB server process (leaves MariaDB running)."""
    from server_control import stop_ffb_server
    return stop_ffb_server()


@mcp.tool()
def build_project(skip_tests: bool = True) -> dict:
    """
    Build all Maven modules via 'mvn clean install'.
    Required before first run or after code changes. Takes ~60s cold, ~15s warm.
    """
    from server_control import build_project as _fn
    return _fn(skip_tests)


# ── Game/match management (requires FFB server running) ───────────────────────

@mcp.tool()
def list_games(status: str = "active") -> list[dict]:
    """
    List games by status via the admin HTTP API.
    status: scheduled | starting | active | paused | finished | uploaded | backuped
    Requires FFB server running. Use start_server() first.
    """
    from admin_api import admin_request, parse_game_list_xml
    xml = admin_request("admin.url.list.status", status)
    return parse_game_list_xml(xml)


@mcp.tool()
def get_game(game_id: str) -> list[dict]:
    """
    Get game details by id via the admin HTTP API.
    Requires FFB server running.
    """
    from admin_api import admin_request, parse_game_list_xml
    xml = admin_request("admin.url.list.id", game_id)
    return parse_game_list_xml(xml)


@mcp.tool()
def schedule_game(team_home_id: str, team_away_id: str) -> str:
    """
    Schedule a game between two teams by their XML team ids (e.g. 'teamHumanKalimar').
    Use list_teams() to find valid team ids. Requires FFB server running.
    """
    from admin_api import admin_request
    return admin_request("admin.url.schedule", team_home_id, team_away_id)


@mcp.tool()
def close_game(game_id: str) -> str:
    """Close an active game. Requires FFB server running."""
    from admin_api import admin_request
    return admin_request("admin.url.close", game_id)


@mcp.tool()
def delete_game(game_id: str) -> str:
    """Delete a game record. Requires FFB server running."""
    from admin_api import admin_request
    return admin_request("admin.url.delete", game_id)


@mcp.tool()
def concede_game(game_id: str, team_id: str) -> str:
    """Concede a game for the specified team. Requires FFB server running."""
    from admin_api import admin_request
    return admin_request("admin.url.concede", game_id, team_id)


@mcp.tool()
def get_game_state(game_id: str, from_db: str = "auto", include_log: bool = False) -> str:
    """
    Get full game state JSON for a game.
    from_db: 'true' | 'false' | 'auto'. Requires FFB server running.
    """
    from admin_api import gamestate_request
    return gamestate_request("gamestate.url.get", game_id, from_db, str(include_log).lower())


@mcp.tool()
def get_game_result(game_id: str) -> str:
    """Get the final result of a finished game as JSON. Requires FFB server running."""
    from admin_api import gamestate_request
    return gamestate_request("gamestate.url.result", game_id)


# ── Match launchers (scripts manage their own server lifecycle) ───────────────

@mcp.tool()
def run_human_vs_human() -> dict:
    """
    Launch a human-vs-human game: starts MariaDB + server + two GUI client windows.
    Returns immediately; clients run in the background.
    In each window: enter game name 'LocalGame', password 'test', click Create, pick a team.
    """
    from match_runner import run_human_vs_human as _fn
    return _fn()


@mcp.tool()
def run_human_vs_ai() -> dict:
    """
    Launch a human-vs-AI game: starts MariaDB + server + one GUI client + one AI agent.
    The human plays as Kalimar; the AI joins headlessly as BattleLore.
    """
    from match_runner import run_human_vs_ai as _fn
    return _fn()


@mcp.tool()
def run_ai_vs_ai() -> dict:
    """
    Launch a headless AI-vs-AI game via WebSocket: starts MariaDB + server + two AI agents.
    No GUI windows. Both agents log to /tmp/ffb-ai-kalimar.log and /tmp/ffb-ai-battlelore.log.
    """
    from match_runner import run_ai_vs_ai as _fn
    return _fn()


@mcp.tool()
def run_games_batch(n: int = 5) -> dict:
    """
    Run N sequential AI-vs-AI games and return win rate statistics. Blocking call.
    Uses run-games.sh which manages the server lifecycle automatically.
    Each game takes ~3-10 minutes. Returns scripted-vs-random win counts.
    """
    from match_runner import run_games_batch as _fn
    return _fn(n)


# ── Headless simulation (no server or DB needed) ──────────────────────────────

@mcp.tool()
def run_match_runner(games_per_condition: int = 50) -> dict:
    """
    Run the MatchRunner headless benchmark (no server or DB needed).
    Runs 4 agent pairings (Sample vs Random, Argmax vs Random, Sample vs Argmax, Random vs Random)
    with games_per_condition games each. ~500ms/game after JIT warmup.
    Returns win rates with 95% Wilson confidence intervals and timing stats.
    """
    from headless import run_match_runner as _fn
    return _fn(games_per_condition)


@mcp.tool()
def generate_replays(
    games: int = 100,
    output_dir: str | None = None,
    temperature: float = 0.5,
    races: list[str] | None = None,
    threads: int | None = None,
) -> dict:
    """
    Generate .ffbr replay files using the ReplayGenerator (no server or DB needed).
    The script builds ffb-ai automatically before running.
    races: optional filter list (e.g. ['human', 'orc', 'skaven']).
    Available races: amazon, chaos, chaos_dwarf, dwarf, elf, goblin, high_elf,
      human, lizardman, necromantic, norse, orc, skaven, undead, underworld,
      vampire, wood_elf.
    temperature: 0.0=argmax, 0.5=default mixed policy, 1.0=random.
    Output defaults to ./replays/ in the repo root.
    """
    from headless import run_replay_generator as _fn
    return _fn(games, output_dir, temperature, races, threads)


# ── Replay pipeline ───────────────────────────────────────────────────────────

@mcp.tool()
def replay_status() -> str:
    """
    Return a summary of the current replay dataset:
    how many replays have been downloaded, the date range, race breakdown,
    and a record-type breakdown of parsed JSONL shards.
    """
    manifest_entries = []
    if _MANIFEST.exists():
        with open(_MANIFEST, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        manifest_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    n_replays  = len(manifest_entries)
    dates      = sorted(e.get("started", "") for e in manifest_entries if e.get("started"))
    date_range = f"{dates[0]} → {dates[-1]}" if dates else "—"
    rules_ctr  = collections.Counter(e.get("rules", "?") for e in manifest_entries)

    races = collections.Counter()
    for e in manifest_entries:
        if e.get("home"): races[e["home"]] += 1
        if e.get("away"): races[e["away"]] += 1
    top_races = races.most_common(10)

    shard_files = sorted(glob.glob(str(_DATA_DIR / "shard_fumbbl_*.jsonl")))
    record_types: dict[str, int] = {}
    total_records = 0
    for path in shard_files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec["type"] if rec["type"] != "dialog" else "dialog/" + rec["dialog_id"]
                    record_types[key] = record_types.get(key, 0) + 1
                    total_records += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    lines = [
        f"Replays downloaded : {n_replays}",
        f"Date range         : {date_range}",
        f"Rules              : {dict(rules_ctr)}",
        f"Replay directory   : {_REPLAY_DIR}",
        "",
        "Top races:",
    ]
    for race, cnt in top_races:
        lines.append(f"  {cnt:4d}  {race}")
    lines += [
        "",
        f"JSONL shards       : {len(shard_files)}  ({_DATA_DIR})",
        f"Total records      : {total_records}",
        "",
        "Record type breakdown:",
    ]
    for k, v in sorted(record_types.items()):
        lines.append(f"  {v:6d}  {k}")
    return "\n".join(lines)


@mcp.tool()
def download_replays(
    count: int = 10,
    start_id: int | None = None,
    delay: float = 60.0,
    rules: str = "BB2025",
    after: str | None = None,
    before: str | None = None,
    parse: bool = False,
) -> str:
    """
    Download Blood Bowl replay files from FUMBBL.

    Delegates to fetch_fumbbl_replays.py, which calls FumbblReplayDownloader
    (Java/WebSocket) for each replay and maintains a manifest.jsonl.

    Args:
        count:    Number of successful replays to download (default 10).
        start_id: Replay ID to start counting down from.
                  If omitted, the script auto-discovers the current maximum.
        delay:    Seconds to wait between download attempts (default 60).
                  Set lower (e.g. 5) for testing; FUMBBL may rate-limit below ~30s.
        rules:    Ruleset filter — only download replays with this ruleset (default "BB2025").
        after:    Skip replays started before this date (YYYY-MM-DD).
        before:   Skip replays started after this date (YYYY-MM-DD).
        parse:    If True, also run FumbblReplayParser on each downloaded replay
                  and append records to a new shard in ffb-ml/data/.

    Returns a human-readable summary of what was downloaded.
    """
    cmd = [sys.executable, str(_FETCH_SCRIPT),
           "--count", str(count),
           "--delay", str(delay),
           "--rules", rules,
           "--output", str(_REPLAY_DIR),
           "--parse-output", str(_DATA_DIR)]
    if start_id is not None:
        cmd += ["--start-id", str(start_id)]
    if after:
        cmd += ["--after", after]
    if before:
        cmd += ["--before", before]
    if parse:
        cmd.append("--parse")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=count * (delay + 120) + 60,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: download_replays timed out."
    except Exception as e:
        return f"ERROR launching fetch_fumbbl_replays.py: {e}"

    out = result.stdout.strip()
    err = result.stderr.strip()
    parts = [out]
    if err:
        parts.append(f"\n--- stderr ---\n{err}")
    if result.returncode != 0:
        parts.append(f"\n(exit code {result.returncode})")
    return "\n".join(parts) or "(no output)"


@mcp.tool()
def parse_replays(
    input_path: str | None = None,
    shard: int | None = None,
) -> str:
    """
    Run FumbblReplayParser (Java) to convert .ffbr replay files into
    JSONL training records and write them to ffb-ml/data/.

    Args:
        input_path: Path to a single .ffbr file or a directory of .ffbr files.
                    Defaults to replays/fumbbl/ (all downloaded replays).
        shard:      Integer shard number for the output filename
                    (shard_fumbbl_<N>.jsonl).  If omitted, the next available
                    number is chosen automatically.

    Returns a summary of the parse run (record counts by type).
    """
    if input_path is None:
        input_path = str(_REPLAY_DIR)

    if shard is None:
        shard = 0
        while (_DATA_DIR / f"shard_fumbbl_{shard}.jsonl").exists():
            shard += 1

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mvn", "--no-transfer-progress", "-pl", "ffb-ai", "exec:java",
        "-Dexec.mainClass=com.fumbbl.ffb.ai.simulation.FumbblReplayParser",
        f"-Dexec.args={input_path} {_DATA_DIR} --shard {shard}",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: parse_replays timed out (>10 min)."
    except Exception as e:
        return f"ERROR launching FumbblReplayParser: {e}"

    shard_path = _DATA_DIR / f"shard_fumbbl_{shard}.jsonl"
    record_types: dict[str, int] = {}
    total = 0
    if shard_path.exists():
        with open(shard_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec["type"] if rec["type"] != "dialog" else "dialog/" + rec["dialog_id"]
                    record_types[key] = record_types.get(key, 0) + 1
                    total += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    lines = []
    for line in result.stdout.splitlines():
        if line.startswith("  [") or line.startswith("Done.") or line.startswith("==="):
            lines.append(line)

    if result.returncode != 0:
        lines.append(f"\nBuild FAILED (exit {result.returncode})")
        stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
        lines.append(stderr_tail)
        return "\n".join(lines)

    lines += [
        f"\nOutput : {shard_path}",
        f"Total  : {total} records",
        "",
        "Breakdown:",
    ]
    for k, v in sorted(record_types.items()):
        lines.append(f"  {v:6d}  {k}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
