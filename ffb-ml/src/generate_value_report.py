"""
generate_value_report.py — Value-head scaling report.

Reads the JSON produced by collect_value_results.py and produces a PDF with:
  - 2×2 grid: win accuracy, score RMSE, cas RMSE, SPP RMSE vs n_games
    (one line per model scale, log x-axis)
  - Summary table: best value per (n_games, scale) for each metric
  - Baseline section: what RMSE would a constant-zero predictor achieve?

Usage:
    python src/generate_value_report.py \
        --results ffb-ml/value_scaling_results.json \
        --output  ffb-ml/value_report.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages


SCALE_ORDER  = ["micro", "small", "medium"]
SCALE_LABELS = {"micro": "Micro", "small": "Small", "medium": "Medium"}
SCALE_COLORS = {"micro": "#1f77b4", "small": "#ff7f0e", "medium": "#2ca02c"}
SCALE_MARKERS = {"micro": "o", "small": "s", "medium": "^"}

METRICS = [
    ("win_acc",    "Win Accuracy (val)",    "higher is better", True,  None),
    ("score_rmse", "Score-delta RMSE (val)","lower is better",  False, 0.194),
    ("cas_rmse",   "Cas-delta RMSE (val)",  "lower is better",  False, 0.832),
    ("spp_rmse",   "SPP-delta RMSE (val)",  "lower is better",  False, 0.994),
]
# Constant-zero baseline RMSE (predict 0 for all regression targets)
BASELINES = {
    "score_rmse": 0.194,
    "cas_rmse":   0.832,
    "spp_rmse":   0.994,
    "win_acc":    0.970,  # 94% draws → 97% "correct" by predicting all non-win
}


def load_results(path: Path):
    with open(path) as f:
        return json.load(f)


def organise(results):
    """Returns {scale: {n_games: metrics_dict}}."""
    out = {}
    for r in results:
        sc = r["scale"]
        ng = r["n_games"]
        out.setdefault(sc, {})[ng] = r
    return out


def plot_metric(ax, data_by_scale, key, title, subtitle, higher_is_better):
    scales_present = [s for s in SCALE_ORDER if s in data_by_scale]
    for sc in scales_present:
        pts = sorted(data_by_scale[sc].items())  # (n_games, metrics)
        xs = [p[0] for p in pts]
        ys = [p[1][key] for p in pts]
        ax.plot(xs, ys,
                color=SCALE_COLORS[sc],
                marker=SCALE_MARKERS[sc],
                label=SCALE_LABELS[sc],
                linewidth=1.5, markersize=6)

    # Baseline (constant-zero predictor)
    if key in BASELINES:
        baseline = BASELINES[key]
        all_xs = []
        for sc in scales_present:
            all_xs.extend(data_by_scale[sc].keys())
        if all_xs:
            xmin, xmax = min(all_xs) * 0.7, max(all_xs) * 1.5
            ax.axhline(baseline, color="red", linestyle="--", linewidth=1, alpha=0.7,
                       label=f"Baseline ({baseline:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Training games (N)")
    ax.set_ylabel(title)
    ax.set_title(f"{title}\n({subtitle})", fontsize=9)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, which="both", alpha=0.3)


def build_summary_table(results, metrics):
    """Return (headers, rows) for a summary table."""
    headers = ["N games", "Scale"] + [m[0] for m in metrics]
    rows = []
    for r in sorted(results, key=lambda x: (x["n_games"], SCALE_ORDER.index(x["scale"]) if x["scale"] in SCALE_ORDER else 99)):
        row = [str(r["n_games"]), r["scale"]]
        for key, *_ in metrics:
            v = r.get(key, float("nan"))
            row.append(f"{v:.4f}" if not np.isnan(v) else "—")
        rows.append(row)
    return headers, rows


def add_table_page(pdf, title, headers, rows, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    t = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.2, 1.5)
    # Bold header
    for j in range(len(headers)):
        t[0, j].set_facecolor("#d0d0d0")
        t[0, j].set_text_props(fontweight="bold")
    # Alternate row shading
    for i in range(1, len(rows) + 1):
        color = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            t[i, j].set_facecolor(color)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output",  default="ffb-ml/value_report.pdf")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(results_path)
    if not results:
        print("ERROR: no results found in JSON", file=sys.stderr)
        sys.exit(1)

    data_by_scale = organise(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:

        # ── Page 1: 2×2 metric scaling plots ─────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("Value-head Scaling: prediction quality vs training data",
                     fontsize=13, fontweight="bold")
        for ax, (key, title, subtitle, higher, _) in zip(axes.flat, METRICS):
            plot_metric(ax, data_by_scale, key, title, subtitle, higher)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Policy accuracy scaling (if available) ───────────────────
        policy_metrics = [
            ("dialog_acc", "Dialog Accuracy",        "higher", True,  None),
            ("ps_acc",     "Player-Select Accuracy", "higher", True,  None),
            ("mt_acc",     "Move-Target Accuracy",   "higher", True,  None),
        ]
        if any(r.get("dialog_acc", 0.0) > 0 for r in results):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle("Policy-head Accuracy vs Training Data",
                         fontsize=12, fontweight="bold")
            for ax, (key, title, subtitle, higher, _) in zip(axes, policy_metrics):
                plot_metric(ax, data_by_scale, key, title, subtitle, higher)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 3: Summary table ─────────────────────────────────────────────
        headers, rows = build_summary_table(results, METRICS)
        add_table_page(pdf, "Value-head Metrics Summary", headers, rows)

        # ── Page 4: Notes ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        notes = (
            "Notes\n"
            "─────\n\n"
            "All metrics evaluated on a held-out validation split (10% of value shards).\n\n"
            "Targets are *deltas* from the current game state to the final outcome:\n"
            "  • win_label:  1.0 = acting team won, 0.0 = lost, 0.5 = draw\n"
            "  • delta_score: Δ(own future TDs) − Δ(opp future TDs)\n"
            "  • delta_cas:   Δ(future opp cas suf) − Δ(future own cas suf)\n"
            "  • delta_spp:   Δ(own future SPP earned) − Δ(opp future SPP earned)\n\n"
            "⚠ WIN ACCURACY LIMITATION:\n"
            "  Argmax (T=0) games end 0-0 in ~94% of cases.\n"
            "  Win accuracy appears high (~97%) because predicting 'no win' is trivially\n"
            "  correct in 97% of records (94% draws + 3% losses). This metric is degenerate\n"
            "  for argmax data. Use score/cas/SPP RMSE instead.\n\n"
            "BASELINE (constant-zero predictor, dashed red line):\n"
            "  score RMSE: 0.194\n"
            "  cas RMSE:   0.832\n"
            "  SPP RMSE:   0.994\n\n"
            "Value data: one record per turn (first decision in each half/turn/team tuple)\n"
            "  to avoid correlated labels from multiple decisions in the same turn.\n\n"
            "Model scales:\n"
            "  micro:  k=8,  CNN=(16,32,64),  hidden=64\n"
            "  small:  k=16, CNN=(32,64,128), hidden=128\n"
            "  medium: k=32, CNN=(64,128,256), hidden=256\n\n"
            "Training: policy + value heads jointly; value val RMSE reported.\n"
            "Expert: argmax (T=0), symmetric matchups.\n"
        )
        ax.text(0.05, 0.95, notes, va="top", ha="left", fontsize=9,
                transform=ax.transAxes, fontfamily="monospace")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report written → {out_path}")


if __name__ == "__main__":
    main()
