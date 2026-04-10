"""
collect_results.py — Read a BC model checkpoint + an EvalRunner log file and
append one JSON record to a scaling results file.

Usage:
    python ffb-ml/src/collect_results.py \
        --checkpoint /tmp/ffb-ckpt-100-v2/small_best.pt \
        --eval-log   /tmp/eval-100.log \
        --n-games    100 \
        --output     /tmp/scaling_results_argmax.json

The output JSON has the schema expected by generate_report.py:
{
    "n_games": 100,
    "val_acc_dialog": 0.960,
    "val_acc_ps":     0.334,
    "val_acc_mt":     0.544,
    "wr_vs_random":   0.82,
    "wr_vs_argmax":   0.38,
    "n_eval_games":   100,
    "best_epoch":     310,
    "train_epochs":   310,
    "time_per_epoch": 6.5
}
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch


def parse_checkpoint(ck_path: Path):
    """Extract best epoch and train accuracies from a checkpoint."""
    ck = torch.load(ck_path, map_location="cpu")
    best_epoch = ck.get("epoch", None)
    # Stored as "val_accs" key but contains train accuracies (naming legacy)
    accs = ck.get("val_accs", {})
    return {
        "best_epoch":   best_epoch,
        "val_acc_dialog": float(accs.get("dialog", 0.0)),
        "val_acc_ps":     float(accs.get("player_select", 0.0)),
        "val_acc_mt":     float(accs.get("move_target", 0.0)),
    }


def parse_eval_log(log_path: Path):
    """
    Parse EvalRunner stdout for win rates.

    EvalRunner prints lines like:
      BC vs Random              98       0       2       98.0%  [ 92.4%– 99.7%]
      BC vs Argmax              38       0       62      38.0%  [ 28.6%– 48.3%]
      Argmax vs Random          96       0       4       96.0%  [ 89.7%– 98.9%]
      Random vs Random          51       0       49      51.0%  [ 41.2%– 60.8%]
    """
    text = log_path.read_text()

    def find_wr(keyword):
        # Matches lines like:
        #   BC vs Random              98       0       2       98.0%  [...]
        #   A: BC      vs Random      98       0       2       98.0%  [...]
        m = re.search(
            rf"^\s*(?:[A-Z]:\s+)?BC\s+vs\s+{re.escape(keyword)}\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if not m:
            return None, None
        wins   = int(m.group(1))
        draws  = int(m.group(2))
        losses = int(m.group(3))
        total  = wins + draws + losses
        wr = wins / total if total > 0 else 0.0
        return wr, total

    wr_rand, n_rand = find_wr("Random")
    wr_arg,  n_arg  = find_wr("Argmax")

    if wr_rand is None:
        print(f"WARNING: could not parse 'BC vs Random' from {log_path}", file=sys.stderr)
        wr_rand = 0.0
        n_rand = 0
    if wr_arg is None:
        print(f"WARNING: could not parse 'BC vs Argmax' from {log_path}", file=sys.stderr)
        wr_arg = 0.0
        n_arg = 0

    return {
        "wr_vs_random": wr_rand,
        "wr_vs_argmax": wr_arg,
        "n_eval_games": n_rand or n_arg,
    }


def parse_train_log(log_path: Path):
    """
    Parse training log for timing: extract average time-per-epoch (in seconds)
    from lines like:
        310    2.9169    0.951    0.315    0.484    6.9s
    and total epoch count.
    """
    if log_path is None or not log_path.exists():
        return {}

    times = []
    for line in log_path.read_text().splitlines():
        m = re.match(r"^\s*\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)s", line)
        if m:
            times.append(float(m.group(1)))

    if not times:
        return {}

    return {
        "time_per_epoch": round(sum(times) / len(times), 1),
        "train_epochs":   len(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to small_best.pt")
    parser.add_argument("--eval-log",   required=True, help="Path to EvalRunner stdout log")
    parser.add_argument("--train-log",  default=None,  help="Path to training stdout log (optional)")
    parser.add_argument("--n-games",    required=True, type=int, help="Number of training games")
    parser.add_argument("--output",     required=True, help="Path to JSON results file (appended to)")
    args = parser.parse_args()

    ck_path    = Path(args.checkpoint)
    eval_path  = Path(args.eval_log)
    train_path = Path(args.train_log) if args.train_log else None

    print(f"Checkpoint : {ck_path}")
    print(f"Eval log   : {eval_path}")

    record = {"n_games": args.n_games}
    record.update(parse_checkpoint(ck_path))
    record.update(parse_eval_log(eval_path))
    record.update(parse_train_log(train_path))

    # Load existing results if file exists
    out_path = Path(args.output)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = []

    # Remove any existing record for the same n_games
    existing = [r for r in existing if r.get("n_games") != args.n_games]
    existing.append(record)
    existing.sort(key=lambda r: r["n_games"])

    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved record to {out_path}:")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
