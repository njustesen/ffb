#!/usr/bin/env bash
# generate-replays.sh — Mass-produce Blood Bowl replay files using headless simulation.
#
# Builds ffb-ai (if needed) then runs ReplayGenerator, which spins up multiple
# in-process game workers — no server process, no MariaDB, no GUI required.
# Each completed game is written as a gzip-compressed JSON replay (.ffbr) to the
# output directory.  The format is identical to replays saved by the live server.
#
# AGENT BEHAVIOUR
#   Both home and away sides run ScriptedStrategy (SCRIPTED_SAMPLE mode).
#   Temperature controls how deterministic the decisions are:
#     T=0.0  argmax — always picks the highest-scoring option
#     T=0.5  softmax policy — default, balanced exploration (recommended)
#     T=1.0  uniform random — all options equally likely
#
# RACE POOL
#   18 team entries across 17 races (human has two teams, enabling human-vs-human).
#   Each game randomly pairs two distinct entries from the pool.
#   Available races: amazon, chaos, chaos_dwarf, dwarf, elf, goblin, high_elf,
#   human, lizardman, necromantic, norse, orc, skaven, undead, underworld,
#   vampire, wood_elf
#
# PARALLELISM
#   Each worker thread owns its own HeadlessFantasyFootballServer instance.
#   Default threads = min(4, cpus/2) — leaves headroom for other work.
#   Games average ~500 ms each once JIT-warmed; expect ~3 000–8 000 replays/hr
#   at 3–4 threads.  Exotic races (Vampire, Underworld) may occasionally hit the
#   100 000-iteration timeout and produce a partial replay.
#
# USAGE
#   ./generate-replays.sh [options]
#
# OPTIONS
#   --output DIR          Directory to write .ffbr files  (default: ./replays)
#   --games N             Total number of games to run    (default: 100)
#   --temperature T       Agent temperature 0.0–1.0       (default: 0.5)
#   --races r1,r2,...     Comma-separated race filter     (default: all races)
#   --threads K           Parallel worker count           (default: min(4, cpus/2))
#
# EXAMPLES
#   ./generate-replays.sh --games 1000
#   ./generate-replays.sh --games 500 --temperature 0.3 --races human,orc,skaven
#   ./generate-replays.sh --games 2000 --output ~/datasets/replays --threads 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building ffb-ai..."
mvn install -DskipTests -pl ffb-ai -am -q

echo "Starting ReplayGenerator..."
mvn -pl ffb-ai exec:java \
  -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.ReplayGenerator \
  -Dexec.args="$*"
