#!/usr/bin/env python3
"""
Orchestrator for downloading FUMBBL Blood Bowl replays.

Downloads replays one at a time via a Java subprocess (FumbblReplayDownloader),
starting from a given ID and walking downward.  Filter-skips (wrong rules/date)
do not consume rate-limit delay; genuine downloads and connection errors do.

Usage:
    python ffb-ml/fetch_fumbbl_replays.py \\
        --start-id 1832851   \\   # ID to start from (omit to auto-discover max)
        --count 100          \\   # number of successful downloads (default: 100)
        --output replays/fumbbl \\  # output directory (default: replays/fumbbl)
        --delay 60           \\   # seconds between real attempts (default: 60)
        --rules BB2025       \\   # only keep replays with this ruleset (default: BB2025)
        --after 2024-01-01   \\   # skip replays started before this date (YYYY-MM-DD)
        --before 2026-12-31  \\   # skip replays started after this date
        --dry-run                # print IDs only, no download
        --parse              \\   # also run FumbblReplayParser after each download
        --parse-output DIR       # where to write JSONL shards (default: ffb-ml/data)

Exit codes: 0 = reached --count, 1 = error.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Maven exec helper
# ---------------------------------------------------------------------------

def _mvn_cmd(main_class: str, args_list: list[str], project_root: str) -> list[str]:
    """Build a 'mvn -pl ffb-ai exec:java ...' command list."""
    exec_args = " ".join(args_list)
    return [
        "mvn", "--no-transfer-progress", "-pl", "ffb-ai", "exec:java",
        f"-Dexec.mainClass={main_class}",
        f"-Dexec.args={exec_args}",
    ]


def run_downloader(replay_id: int, out_path: str | None, project_root: str,
                   rules: str | None, after: str | None, before: str | None,
                   probe_only: bool = False, timeout: int = 120) -> tuple[int, dict | None]:
    """
    Invoke FumbblReplayDownloader for one replay ID.

    Returns (exit_code, metadata_dict_or_None).
      exit_code 0 = success
      exit_code 1 = not found / error
      exit_code 2 = filtered (rules/date mismatch)
    """
    args_list = []
    if probe_only:
        args_list.append("--probe-only")
    args_list.append(str(replay_id))
    if not probe_only and out_path:
        args_list.append(out_path)
    if rules:
        args_list += ["--rules", rules]
    if after:
        args_list += ["--after", after]
    if before:
        args_list += ["--before", before]

    cmd = _mvn_cmd(
        "com.fumbbl.ffb.ai.simulation.FumbblReplayDownloader",
        args_list,
        project_root,
    )

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 1, None
    except Exception as e:
        print(f"  ERROR launching downloader: {e}", file=sys.stderr)
        return 1, None

    metadata = None
    if result.returncode == 0 and not probe_only:
        # Parse the last non-empty line of stdout as metadata JSON
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    metadata = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break

    return result.returncode, metadata


def run_parser(input_path: str, output_dir: str, project_root: str,
               shard_num: int, timeout: int = 300) -> bool:
    """
    Invoke FumbblReplayParser on a single .ffbr file or directory.

    Returns True on success.
    """
    args_list = [input_path, output_dir, "--shard", str(shard_num)]
    cmd = _mvn_cmd(
        "com.fumbbl.ffb.ai.simulation.FumbblReplayParser",
        args_list,
        project_root,
    )

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Binary-search max ID discovery
# ---------------------------------------------------------------------------

def probe_replay(replay_id: int, project_root: str) -> bool:
    """Return True if the replay ID appears to be valid (probe-only connection)."""
    code, _ = run_downloader(
        replay_id, None, project_root,
        rules=None, after=None, before=None,
        probe_only=True, timeout=15,
    )
    return code == 0


def find_max_id(seed: int, project_root: str) -> int:
    """
    Binary-search for the current maximum valid replay ID.

    Probes exponentially upward from `seed` to find an upper bound, then
    binary-searches the gap.  No delay between probes (read-only, fast).
    """
    print(f"  Probing upward from seed {seed} …")
    low = seed
    # Exponential probe upward to find first invalid ID
    high = seed
    step = max(1, seed // 4)
    while probe_replay(high + step, project_root):
        high += step
        step = int(step * 1.5) + 1
        print(f"    still valid at {high} …")

    high = high + step  # first known-invalid
    print(f"  Binary search between {low} and {high} …")
    while high - low > 1:
        mid = (low + high) // 2
        if probe_replay(mid, project_root):
            low = mid
        else:
            high = mid

    print(f"  Max valid ID: {low}")
    return low


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> set[int]:
    """Return the set of already-downloaded replay IDs from manifest.jsonl."""
    seen = set()
    if not os.path.exists(manifest_path):
        return seen
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


def append_manifest(manifest_path: str, entry: dict) -> None:
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download FUMBBL Blood Bowl replays via FumbblReplayDownloader.",
    )
    parser.add_argument("--start-id",     type=int,   default=None,
                        help="Replay ID to start from (default: auto-discover max)")
    parser.add_argument("--count",        type=int,   default=100,
                        help="Number of successful downloads (default: 100)")
    parser.add_argument("--output",       default="replays/fumbbl",
                        help="Output directory for .ffbr files (default: replays/fumbbl)")
    parser.add_argument("--delay",        type=float, default=60.0,
                        help="Seconds between real download attempts (default: 60)")
    parser.add_argument("--rules",        default="BB2025",
                        help="Only download replays with this ruleset (default: BB2025)")
    parser.add_argument("--after",        default=None,
                        help="Skip replays started before YYYY-MM-DD")
    parser.add_argument("--before",       default=None,
                        help="Skip replays started after YYYY-MM-DD")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print IDs only, do not download")
    parser.add_argument("--parse",        action="store_true",
                        help="Run FumbblReplayParser on each download")
    parser.add_argument("--parse-output", default=None,
                        help="JSONL output dir for parser (default: ffb-ml/data)")

    args = parser.parse_args()

    # Locate project root (parent of the directory this script lives in, or CWD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # ffb-ml/../ = project root
    if not os.path.isdir(os.path.join(project_root, "ffb-ai")):
        # Fallback: CWD is the project root
        project_root = os.getcwd()
    if not os.path.isdir(os.path.join(project_root, "ffb-ai")):
        print("ERROR: cannot locate project root (expected ffb-ai/ sibling dir)", file=sys.stderr)
        return 1

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    parse_output_dir = args.parse_output
    if parse_output_dir is None:
        parse_output_dir = os.path.join(script_dir, "data")
    if args.parse:
        os.makedirs(parse_output_dir, exist_ok=True)

    # Load already-downloaded IDs
    seen = load_manifest(manifest_path)
    print(f"Manifest: {len(seen)} already downloaded")

    # Determine starting ID
    start_id = args.start_id
    if start_id is None:
        seed = max(seen) if seen else 1832851
        print(f"Auto-discovering max replay ID (seed={seed}) …")
        if args.dry_run:
            print("  (dry-run: skipping probe, using seed)")
            start_id = seed
        else:
            start_id = find_max_id(seed, project_root)

    print(f"Starting from ID {start_id}, target {args.count} successful downloads")
    print(f"Rules filter : {args.rules or 'none'}")
    print(f"Date range   : {args.after or '*'} → {args.before or '*'}")
    print(f"Delay        : {args.delay}s between real attempts")
    print(f"Output dir   : {output_dir}")
    if args.parse:
        print(f"Parser output: {parse_output_dir}")
    print()

    success_count = 0
    current_id = start_id
    next_shard = 0
    # Find next available shard number
    while os.path.exists(os.path.join(parse_output_dir, f"shard_fumbbl_{next_shard}.jsonl")):
        next_shard += 1

    while success_count < args.count:
        if current_id <= 0:
            print("Exhausted all replay IDs — stopping.")
            break

        if current_id in seen:
            current_id -= 1
            continue

        print(f"[{success_count}/{args.count}] Attempting replay {current_id} …", end=" ", flush=True)

        if args.dry_run:
            print(f"(dry-run)")
            success_count += 1
            current_id -= 1
            continue

        out_filename = f"replay_{current_id}.ffbr"
        out_path = os.path.join(output_dir, out_filename)

        exit_code, metadata = run_downloader(
            current_id, out_path, project_root,
            rules=args.rules,
            after=args.after,
            before=args.before,
        )

        if exit_code == 0 and metadata is not None:
            # Success
            entry = {
                "id":         current_id,
                "file":       out_filename,
                "downloaded": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "started":    metadata.get("started", "unknown"),
                "rules":      metadata.get("rules",   "unknown"),
                "home":       metadata.get("home",    "unknown"),
                "away":       metadata.get("away",    "unknown"),
                "commands":   metadata.get("commands", 0),
            }
            append_manifest(manifest_path, entry)
            seen.add(current_id)
            success_count += 1
            print(f"OK  ({entry['rules']}, {entry['home']} vs {entry['away']}, "
                  f"{entry['commands']} cmds, started {entry['started']})")

            # Optionally parse immediately
            if args.parse:
                ok = run_parser(out_path, parse_output_dir, project_root, next_shard)
                if ok:
                    print(f"  Parsed → shard_fumbbl_{next_shard}.jsonl")
                    next_shard += 1
                else:
                    print(f"  WARN: parser failed for {out_filename}")

            current_id -= 1
            if success_count < args.count:
                time.sleep(args.delay)

        elif exit_code == 2:
            # Filtered — don't count against rate limit, skip immediately
            print(f"filtered (rules/date mismatch)")
            current_id -= 1
            # No sleep for filtered replays

        else:
            # Not found or error — count against rate limit
            print(f"not found or error (exit {exit_code})")
            current_id -= 1
            time.sleep(args.delay)

    print(f"\nDone. Downloaded {success_count} replay(s) to {output_dir}/")
    if args.parse:
        print(f"JSONL shards written to {parse_output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
