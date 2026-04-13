#!/usr/bin/env bash
# run_value_sweep.sh — Value-head scaling experiment.
#
# For each N ∈ {100, 1k, 10k} and scale ∈ {micro, small, medium}:
#   1. Generate N argmax games (reuses existing raw data if already present)
#   2. Extract features (includes value shards via buffered JsonlTrainingDataCollector)
#   3. Train policy + value model
#   4. Collect val metrics from best checkpoint → JSON
#   5. Generate PDF report
#
# Prerequisites:
#   • ffb-ai JAR rebuilt with value-data changes:
#       mvn install -DskipTests -pl ffb-ai -am
#   • Python deps: torch, numpy, matplotlib, onnx, onnxruntime
#
# Usage:
#   bash run_value_sweep.sh [--skip-datagen] [--only-report]
#
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
ML_SRC="$REPO/ffb-ml/src"
SWEEP_ROOT="/tmp/ffb-value-sweep"
RESULTS_JSON="$SWEEP_ROOT/value_scaling_results.json"
REPORT_PDF="$SWEEP_ROOT/value_report.pdf"

GAME_COUNTS=(100 1000 10000)
SCALES=(micro small medium)

SKIP_DATAGEN=0
ONLY_REPORT=0
for arg in "$@"; do
  [[ "$arg" == "--skip-datagen" ]] && SKIP_DATAGEN=1
  [[ "$arg" == "--only-report"  ]] && ONLY_REPORT=1
done

mkdir -p "$SWEEP_ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helper: generate raw game data ────────────────────────────────────────────
generate_data() {
  local n="$1"
  local raw_dir="$SWEEP_ROOT/raw-$n"
  if [[ -d "$raw_dir" ]] && [[ -n "$(ls -A "$raw_dir" 2>/dev/null)" ]]; then
    log "Raw data for $n games already at $raw_dir — skipping generation."
    return
  fi
  log "Generating $n argmax games → $raw_dir ..."
  cd "$REPO"
  mvn -pl ffb-ai exec:java \
    -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.TrainingDataExporter \
    "-Dexec.args=--output $raw_dir --games $n --temperature 0 --threads 4" \
    -q 2>&1 | tee "$SWEEP_ROOT/gen-$n.log"
  log "Data generation done: $n games"
}

# ── Helper: extract features ──────────────────────────────────────────────────
extract_features() {
  local n="$1"
  local raw_dir="$SWEEP_ROOT/raw-$n"
  local feat_dir="$SWEEP_ROOT/features-$n"
  if [[ -d "$feat_dir" ]] && [[ -f "$feat_dir/vocab.json" ]]; then
    log "Features for $n games already at $feat_dir — skipping extraction."
    return
  fi
  log "Extracting features for $n games → $feat_dir ..."
  cd "$REPO"
  python "$ML_SRC/extract_features.py" \
    --input   "$raw_dir" \
    --output  "$feat_dir" \
    --shard-size 10000 \
    2>&1 | tee "$SWEEP_ROOT/extract-$n.log"
  log "Feature extraction done: $n games"
}

# ── Helper: train one (n_games, scale) combination ───────────────────────────
train_model() {
  local n="$1"
  local scale="$2"
  local feat_dir="$SWEEP_ROOT/features-$n"
  local ckpt_dir="$SWEEP_ROOT/$n/$scale"
  local train_log="$SWEEP_ROOT/train-${n}-${scale}.log"
  mkdir -p "$ckpt_dir"

  if [[ -f "$ckpt_dir/${scale}_best.pt" ]]; then
    log "Checkpoint already exists: $ckpt_dir/${scale}_best.pt — skipping training."
    return
  fi

  # Patience tuned per data size to avoid very long runs
  local patience=30
  [[ "$n" -ge 10000 ]] && patience=20

  # For 10k games, stream shards to keep memory reasonable
  local spe_arg=""
  [[ "$n" -ge 10000 ]] && spe_arg="--shards-per-epoch 30"

  log "Training $scale model on $n games → $ckpt_dir ..."
  cd "$REPO"
  python "$ML_SRC/train.py" \
    --features  "$feat_dir" \
    --output    "$ckpt_dir" \
    --scale     "$scale" \
    --epochs    300 \
    --batch-size 256 \
    --lr        5e-4 \
    --w-dialog  2.0 \
    --w-ps      2.0 \
    --w-mt      1.0 \
    --w-value   1.0 \
    --val-frac  0.10 \
    --patience  "$patience" \
    $spe_arg \
    2>&1 | tee "$train_log"
  log "Training done: $n games / $scale"
}

# ── Helper: collect metrics from best checkpoint ───────────────────────────────
collect_metrics() {
  log "Collecting value metrics from all checkpoints → $RESULTS_JSON ..."
  cd "$REPO"
  python "$ML_SRC/collect_value_results.py" \
    --sweep-root "$SWEEP_ROOT" \
    --output     "$RESULTS_JSON"
}

# ── Helper: generate report ───────────────────────────────────────────────────
generate_report() {
  if [[ ! -f "$RESULTS_JSON" ]]; then
    log "No results JSON found — skipping report."
    return
  fi
  log "Generating value scaling report → $REPORT_PDF ..."
  cd "$REPO"
  python "$ML_SRC/generate_value_report.py" \
    --results "$RESULTS_JSON" \
    --output  "$REPORT_PDF"
  log "Report: $REPORT_PDF"
}

# ═══════════════════════════════════════════════════════════════════════════════

if [[ $ONLY_REPORT -eq 1 ]]; then
  generate_report
  exit 0
fi

# ── Step 1: Data generation & feature extraction ─────────────────────────────
if [[ $SKIP_DATAGEN -eq 0 ]]; then
  log "=== Step 1: Data generation ==="
  for n in "${GAME_COUNTS[@]}"; do
    generate_data "$n"
  done
else
  log "Skipping data generation (--skip-datagen)"
fi

log "=== Step 2: Feature extraction ==="
for n in "${GAME_COUNTS[@]}"; do
  extract_features "$n"
done

# ── Step 3: Training sweep ────────────────────────────────────────────────────
log "=== Step 3: Training sweep (${#GAME_COUNTS[@]} data sizes × ${#SCALES[@]} scales) ==="
for n in "${GAME_COUNTS[@]}"; do
  for scale in "${SCALES[@]}"; do
    train_model "$n" "$scale"
  done
done

# ── Step 4: Collect results ───────────────────────────────────────────────────
log "=== Step 4: Collecting results ==="
collect_metrics

# ── Step 5: Report ────────────────────────────────────────────────────────────
log "=== Step 5: Generating report ==="
generate_report

log ""
log "=== Value sweep complete! ==="
log "  Results : $RESULTS_JSON"
log "  Report  : $REPORT_PDF"
log "  Sweep   : $SWEEP_ROOT"
