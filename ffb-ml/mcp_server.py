#!/usr/bin/env python3
"""
MCP server exposing FUMBBL replay pipeline tools:

  download_replays  — fetch BB2025 replays from FUMBBL via fetch_fumbbl_replays.py
  parse_replays     — run FumbblReplayParser to emit JSONL training records
  replay_status     — report manifest counts and record type breakdown

Run (stdio transport, for Claude Code):
    python ffb-ml/mcp_server.py
"""

import collections
import glob
import json
import os
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)           # …/ffb
REPLAY_DIR   = os.path.join(PROJECT_ROOT, "replays", "fumbbl")
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
MANIFEST     = os.path.join(REPLAY_DIR, "manifest.jsonl")
FETCH_SCRIPT = os.path.join(SCRIPT_DIR, "fetch_fumbbl_replays.py")

# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="ffb-replay-pipeline",
    instructions=(
        "Tools for downloading Blood Bowl replays from FUMBBL and parsing them "
        "into JSONL training records for behavioural cloning."
    ),
)


# ── Tool: replay_status ───────────────────────────────────────────────────────

@mcp.tool()
def replay_status() -> str:
    """
    Return a summary of the current replay dataset:
    how many replays have been downloaded, the date range, race breakdown,
    and a record-type breakdown of parsed JSONL shards.
    """
    # Manifest
    manifest_entries = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as fh:
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

    # Shards
    shard_files = sorted(glob.glob(os.path.join(DATA_DIR, "shard_fumbbl_*.jsonl")))
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
        f"Replay directory   : {REPLAY_DIR}",
        "",
        "Top races:",
    ]
    for race, cnt in top_races:
        lines.append(f"  {cnt:4d}  {race}")

    lines += [
        "",
        f"JSONL shards       : {len(shard_files)}  ({DATA_DIR})",
        f"Total records      : {total_records}",
        "",
        "Record type breakdown:",
    ]
    for k, v in sorted(record_types.items()):
        lines.append(f"  {v:6d}  {k}")

    return "\n".join(lines)


# ── Tool: download_replays ────────────────────────────────────────────────────

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
    cmd = [sys.executable, FETCH_SCRIPT,
           "--count", str(count),
           "--delay", str(delay),
           "--rules", rules,
           "--output", REPLAY_DIR,
           "--parse-output", DATA_DIR]

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
            cwd=PROJECT_ROOT,
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


# ── Tool: parse_replays ───────────────────────────────────────────────────────

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
        input_path = REPLAY_DIR

    # Auto-pick shard number
    if shard is None:
        shard = 0
        while os.path.exists(os.path.join(DATA_DIR, f"shard_fumbbl_{shard}.jsonl")):
            shard += 1

    os.makedirs(DATA_DIR, exist_ok=True)

    cmd = [
        "mvn", "--no-transfer-progress", "-pl", "ffb-ai", "exec:java",
        "-Dexec.mainClass=com.fumbbl.ffb.ai.simulation.FumbblReplayParser",
        f"-Dexec.args={input_path} {DATA_DIR} --shard {shard}",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: parse_replays timed out (>10 min)."
    except Exception as e:
        return f"ERROR launching FumbblReplayParser: {e}"

    shard_path = os.path.join(DATA_DIR, f"shard_fumbbl_{shard}.jsonl")
    record_types: dict[str, int] = {}
    total = 0
    if os.path.exists(shard_path):
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
    # Maven stdout has the progress lines
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
