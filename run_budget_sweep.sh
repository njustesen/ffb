#!/usr/bin/env bash
# run_budget_sweep.sh — MCTS budget sweep: find budget where MCTS beats Argmax.
#
# Tests standard and cross-turn MCTS at three time budgets.
# Uses --mcts-time-ms (wall-clock per decision) for predictable game times.
#
# Usage: bash run_budget_sweep.sh [--skip-baseline]
#   --skip-baseline  skip 100ms level and go straight to 500ms + cross-turn
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
OUT="/tmp/ffb-budget-sweep"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_experiment() {
    local time_ms="$1"
    local n="$2"
    local extra_args="${3:-}"
    local label="time_${time_ms}ms_n${n}${extra_args:+_xt}"
    local outfile="$OUT/result_${label}.txt"

    log "=== Starting time=${time_ms}ms N=$n ${extra_args} ==="
    cd "$REPO"
    mvn -pl ffb-ai exec:java \
        -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner \
        "-Dexec.args=$REPO $n --mcts-time-ms $time_ms --mcts-vs-argmax $extra_args" \
        2>/dev/null \
        | tee "$outfile"
    log "=== Done → $outfile ==="
    echo ""
}

SKIP_BASELINE=0
for arg in "$@"; do
    [[ "$arg" == "--skip-baseline" ]] && SKIP_BASELINE=1
done

log "Budget sweep starting. Results → $OUT"
echo ""

if [[ $SKIP_BASELINE -eq 0 ]]; then
    run_experiment 200  20
fi

run_experiment 500  20
run_experiment 500  20 "--mcts-cross-turn"

log "=== Budget sweep complete ==="
echo ""
echo "Results:"
for f in "$OUT"/result_*.txt; do
    echo "--- $f ---"
    grep -E "G[0-9]*:|H[0-9]*:|iter/s|avg depth|ms/dec|WinRate|Done:" "$f" 2>/dev/null | head -25
    echo ""
done
