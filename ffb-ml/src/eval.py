"""
eval.py — Per-type accuracy evaluation + scaling plots.

Usage:
    python src/eval.py --features ffb-ml/features \
                       --checkpoints ffb-ml/checkpoints \
                       --output ffb-ml/eval

Produces:
    accuracy_table.json     — per-type/per-dialog-type accuracy
    accuracy_vs_data.png    — accuracy vs number of training records
    accuracy_vs_scale.png   — accuracy vs model size (requires multiple checkpoints)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from model import BCModel, make_model
from train import NpzShardDataset, dialog_accuracy, nll_accuracy
from torch.utils.data import DataLoader, ConcatDataset


@torch.no_grad()
def evaluate_model(model, feature_dir: Path, device, batch_size=256):
    """Full evaluation on all shards. Returns dict of per-type metrics."""
    model.eval()

    with open(feature_dir / "vocab.json") as f:
        vocab = json.load(f)
    dialog_id_to_name = {v: k for k, v in vocab["dialog_types"].items()}

    results = {}

    # Dialog
    dialog_shards = sorted(feature_dir.glob("dialog_shard_*.npz"))
    if dialog_shards:
        ds = ConcatDataset([NpzShardDataset(p) for p in dialog_shards])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        per_type = {}
        all_acc = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            logits    = out["dialog_logits"]
            action    = batch["action"]
            n_options = batch["n_options"]
            dtype_ids = batch["dialog_type_id"].cpu().numpy()

            for i in range(len(action)):
                dt = dialog_id_to_name.get(int(dtype_ids[i]), "UNKNOWN")
                lg = logits[i:i+1]
                ac = action[i:i+1]
                no = n_options[i:i+1]
                acc_i = dialog_accuracy(lg, ac, no)
                per_type.setdefault(dt, []).append(acc_i)
                all_acc.append(acc_i)

        results["dialog"] = {
            "overall": float(np.mean(all_acc)),
            "n": len(all_acc),
            "per_type": {k: float(np.mean(v)) for k, v in per_type.items()},
        }

    # Player select
    ps_shards = sorted(feature_dir.glob("player_select_shard_*.npz"))
    if ps_shards:
        ds = ConcatDataset([NpzShardDataset(p) for p in ps_shards])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_acc = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            acc = nll_accuracy(out["player_select_log_probs"], batch["action"])
            all_acc.append(acc)
        results["player_select"] = {"overall": float(np.mean(all_acc)), "n": len(ps_shards) * 10000}

    # Move target
    mt_shards = sorted(feature_dir.glob("move_target_shard_*.npz"))
    if mt_shards:
        ds = ConcatDataset([NpzShardDataset(p) for p in mt_shards])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_acc = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            acc = nll_accuracy(out["move_target_log_probs"], batch["action"])
            all_acc.append(acc)
        results["move_target"] = {"overall": float(np.mean(all_acc)), "n": len(mt_shards) * 10000}

    return results


def plot_accuracy_table(results, out_dir: Path):
    """Print a formatted accuracy table and save JSON."""
    print("\n=== Accuracy Table ===")
    for dtype, info in results.items():
        print(f"\n{dtype}: {info['overall']:.3f} (n≈{info.get('n', '?')})")
        if "per_type" in info:
            for t, a in sorted(info["per_type"].items(), key=lambda x: -x[1]):
                print(f"    {t:<40} {a:.3f}")

    with open(out_dir / "accuracy_table.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved accuracy_table.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",     default="ffb-ml/features")
    parser.add_argument("--checkpoints",  default="ffb-ml/checkpoints")
    parser.add_argument("--output",       default="ffb-ml/eval")
    parser.add_argument("--scale",        default="small", choices=["micro", "small", "medium"])
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    feat_dir = Path(args.features)
    ck_dir   = Path(args.checkpoints)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(feat_dir / "vocab.json") as f:
        vocab = json.load(f)
    n_skills       = len(vocab["skills"])
    n_dialog_types = len(vocab["dialog_types"])

    ck_path = ck_dir / f"{args.scale}_best.pt"
    if not ck_path.exists():
        # Fall back to latest epoch checkpoint
        candidates = sorted(ck_dir.glob(f"{args.scale}_epoch*.pt"))
        if not candidates:
            print(f"ERROR: no checkpoint found for scale={args.scale} in {ck_dir}", file=sys.stderr)
            sys.exit(1)
        ck_path = candidates[-1]

    print(f"Loading checkpoint: {ck_path.name}")
    ck = torch.load(ck_path, map_location=device)
    model = make_model(args.scale, n_skills, n_dialog_types)
    model.load_state_dict(ck["model"])
    model.to(device)

    results = evaluate_model(model, feat_dir, device)
    plot_accuracy_table(results, out_dir)

    # Scaling comparison plot (if multiple scale checkpoints exist)
    try:
        import matplotlib.pyplot as plt
        scales = []
        overall_accs = []
        for scale in ["micro", "small", "medium"]:
            p = ck_dir / f"{scale}_best.pt"
            if not p.exists():
                continue
            ck2 = torch.load(p, map_location=device)
            m2  = make_model(scale, n_skills, n_dialog_types)
            m2.load_state_dict(ck2["model"])
            m2.to(device)
            res2 = evaluate_model(m2, feat_dir, device)
            avg = np.mean([res2[t]["overall"] for t in res2])
            from model import count_parameters
            scales.append(f"{scale}\n{count_parameters(m2)//1000}K")
            overall_accs.append(avg)

        if len(scales) > 1:
            fig, ax = plt.subplots()
            ax.bar(scales, overall_accs)
            ax.set_ylabel("Avg accuracy (val)")
            ax.set_title("BC accuracy vs model scale")
            ax.set_ylim(0, 1)
            fig.savefig(out_dir / "accuracy_vs_scale.png", dpi=120, bbox_inches="tight")
            print(f"Saved accuracy_vs_scale.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
