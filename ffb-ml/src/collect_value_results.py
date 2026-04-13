"""
collect_value_results.py — Aggregate value-head metrics from a sweep of checkpoints.

Usage:
    python src/collect_value_results.py \
        --sweep-root ffb-ml/value-sweep \
        --output     ffb-ml/value_scaling_results.json

Expected directory structure under --sweep-root:
    <n_games>/<scale>/small_best.pt   (or micro_best.pt, medium_best.pt)

For example:
    ffb-ml/value-sweep/100/micro/micro_best.pt
    ffb-ml/value-sweep/100/small/small_best.pt
    ffb-ml/value-sweep/1000/medium/medium_best.pt

Each best checkpoint must have a "val_accs" dict with at minimum:
    val_win_acc, val_score_rmse, val_cas_rmse, val_spp_rmse

Output: a JSON array, one entry per (n_games, scale) combination found.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def load_metrics(ckpt_path: Path) -> dict:
    """Load best checkpoint and return its val_accs dict."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    va = ck.get("val_accs", {})
    return {
        "epoch":       ck.get("epoch", -1),
        "win_acc":     va.get("val_win_acc",    va.get("win_val", 0.0)),
        "score_rmse":  va.get("val_score_rmse", 0.0),
        "cas_rmse":    va.get("val_cas_rmse",   0.0),
        "spp_rmse":    va.get("val_spp_rmse",   0.0),
        # Policy head accs (also stored if joint training)
        "dialog_acc":  va.get("dialog",         0.0),
        "ps_acc":      va.get("player_select",  0.0),
        "mt_acc":      va.get("move_target",    0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True,
                        help="Root dir: <n_games>/<scale>/<scale>_best.pt")
    parser.add_argument("--output", default="ffb-ml/value_scaling_results.json",
                        help="Where to write the JSON results array")
    args = parser.parse_args()

    root = Path(args.sweep_root)
    if not root.exists():
        print(f"ERROR: sweep root not found: {root}", file=sys.stderr)
        sys.exit(1)

    results = []
    # Walk: root/<n_games>/<scale>/
    for games_dir in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not games_dir.is_dir():
            continue
        try:
            n_games = int(games_dir.name)
        except ValueError:
            continue

        for scale_dir in sorted(games_dir.iterdir()):
            if not scale_dir.is_dir():
                continue
            scale = scale_dir.name
            ckpt = scale_dir / f"{scale}_best.pt"
            if not ckpt.exists():
                print(f"  SKIP  {n_games:6d}  {scale:8s}  (no {ckpt.name})")
                continue

            try:
                m = load_metrics(ckpt)
                entry = {"n_games": n_games, "scale": scale, **m}
                results.append(entry)
                print(f"  OK    {n_games:6d}  {scale:8s}  "
                      f"win={m['win_acc']:.3f}  "
                      f"sc={m['score_rmse']:.3f}  "
                      f"ca={m['cas_rmse']:.3f}  "
                      f"sp={m['spp_rmse']:.3f}  "
                      f"(epoch {m['epoch']})")
            except Exception as e:
                print(f"  ERR   {n_games:6d}  {scale:8s}  {e}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} entries → {out_path}")


if __name__ == "__main__":
    main()
