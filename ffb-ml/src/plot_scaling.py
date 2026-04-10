"""
plot_scaling.py — Generate scaling experiment plots.

Reads result JSON files produced by a scaling run and produces
one figure with two panels:
  - Panel 1: Val accuracy (dialog, player_select, move_target) vs. training games
  - Panel 2: Win rate (BC vs Random, BC vs Argmax) vs. training games

Usage:
    python ffb-ml/src/plot_scaling.py --results /tmp/scaling_results.json
    python ffb-ml/src/plot_scaling.py --results /tmp/scaling_results.json --show

Results JSON format:
  [
    {
      "n_games": 100,
      "val_acc_dialog": 0.75,
      "val_acc_ps": 0.20,
      "val_acc_mt": 0.015,
      "wr_vs_random": 0.04,
      "wr_vs_argmax": 0.00
    },
    ...
  ]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results, out_path, show=False):
    n_games = [r["n_games"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BC Model: Scaling with Training Data", fontsize=14, fontweight="bold")

    # ── Panel 1: Accuracy ─────────────────────────────────────────────────────
    ax1.set_title("Validation Accuracy vs Training Games")
    ax1.plot(n_games, [r["val_acc_dialog"] for r in results],
             "o-", color="steelblue", label="Dialog", linewidth=2, markersize=8)
    ax1.plot(n_games, [r["val_acc_ps"] for r in results],
             "s-", color="darkorange", label="Player Select", linewidth=2, markersize=8)
    ax1.plot(n_games, [r["val_acc_mt"] for r in results],
             "^-", color="forestgreen", label="Move Target", linewidth=2, markersize=8)

    # Reference lines
    ax1.axhline(y=0.838, color="steelblue", linestyle="--", alpha=0.5, label="Dialog ref (0.838)")
    ax1.axhline(y=0.170, color="darkorange", linestyle="--", alpha=0.5, label="PS ref (0.170)")

    ax1.set_xlabel("Training Games")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_games)
    ax1.set_xticklabels([f"{g:,}" for g in n_games])

    # ── Panel 2: Win Rates ────────────────────────────────────────────────────
    ax2.set_title("Win Rate vs Training Games")
    ax2.plot(n_games, [r["wr_vs_random"] * 100 for r in results],
             "o-", color="crimson", label="BC vs Random", linewidth=2, markersize=8)
    ax2.plot(n_games, [r["wr_vs_argmax"] * 100 for r in results],
             "s-", color="purple", label="BC vs Argmax", linewidth=2, markersize=8)

    # Reference lines
    ax2.axhline(y=50, color="purple", linestyle="--", alpha=0.5, label="Argmax target (50%)")
    ax2.axhline(y=96, color="crimson", linestyle="--", alpha=0.5, label="Argmax vs Random (~96%)")

    # Error bars from Wilson CI if available
    for i, r in enumerate(results):
        n = r.get("n_eval_games", 50)
        for wr_key, color, yval in [
            ("wr_vs_random", "crimson", r["wr_vs_random"]),
            ("wr_vs_argmax", "purple", r["wr_vs_argmax"]),
        ]:
            p = yval
            if n > 0 and 0 < p < 1:
                # Wilson CI
                z = 1.96
                denom = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denom
                half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
                ax2.errorbar(n_games[i], p*100, yerr=half*100, fmt="none",
                             color=color, capsize=4, alpha=0.5)

    ax2.set_xlabel("Training Games")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_xscale("log")
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_games)
    ax2.set_xticklabels([f"{g:,}" for g in n_games])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="/tmp/scaling_results.json")
    parser.add_argument("--output",  default="/tmp/scaling_plot.png")
    parser.add_argument("--show",    action="store_true")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    # Sort by n_games
    results.sort(key=lambda r: r["n_games"])
    print(f"Loaded {len(results)} data points:")
    for r in results:
        print(f"  n={r['n_games']:,}: D={r['val_acc_dialog']:.3f} PS={r['val_acc_ps']:.3f} "
              f"MT={r['val_acc_mt']:.3f} | BC/Rand={r['wr_vs_random']:.1%} BC/Arg={r['wr_vs_argmax']:.1%}")

    plot_results(results, args.output, show=args.show)


if __name__ == "__main__":
    main()
