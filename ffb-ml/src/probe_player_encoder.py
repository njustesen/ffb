"""
probe_player_encoder.py — Evaluate quality of the jointly-trained PlayerEncoder.

Post-training probe: freeze encoder weights, fit a linear classifier to predict
player position type from the k-dim encoding. Measures how much positional
information the encoder captured as a by-product of imitating good play.

Usage:
    python src/probe_player_encoder.py \
        --checkpoint ffb-ml/checkpoints/small_best.pt \
        --features   ffb-ml/features \
        --output     ffb-ml/eval

Output:
    probe_results.json       — top-1 accuracy per k value
    probe_accuracy_vs_k.png  — accuracy vs encoder dimension plot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from model import BCModel, make_model, PlayerEncoder
from extract_features import MAX_SKILLS


# ── Collect player encodings from game states in a feature shard ──────────────

def collect_encodings(model: BCModel, feature_dir: Path, device, n_records=5000):
    """
    Walk player_select shards, extract per-player encodings + position labels.

    Labels are derived from the player ID prefix (e.g. "teamOrcBattleLore1" → "orc").
    This is a rough proxy for position type; a better version would use the position name
    from the roster XML, but that requires additional data loading.
    """
    from train import NpzShardDataset
    from torch.utils.data import DataLoader, ConcatDataset
    import re

    shards = sorted(feature_dir.glob("player_select_shard_*.npz"))
    if not shards:
        print("No player_select shards found.", file=sys.stderr)
        return None, None

    ds = ConcatDataset([NpzShardDataset(p) for p in shards])
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    enc_list = []
    label_list = []
    seen = 0

    # Build a label vocab from candidate player IDs embedded in the shard
    # Player IDs encode race (e.g. teamOrcBattleLore3). Extract race name.
    label_vocab = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if seen >= n_records:
                break
            sk  = batch["cand_skill_ids"].to(device)   # (B, MAX_CANDS, MAX_SKILLS)
            st  = batch["cand_stats"].to(device)        # (B, MAX_CANDS, 5)
            mask = batch["cand_mask"].to(device)        # (B, MAX_CANDS+1)

            B, C, _ = sk.shape
            sk_flat = sk.view(B * C, -1)
            st_flat = st.view(B * C, 5)
            enc = model.player_enc(sk_flat, st_flat)    # (B*C, k)
            enc = enc.view(B, C, -1)

            # Use skill fingerprint as a soft label: cluster by dominant skills
            # For the probe we label by number of non-zero skills as a proxy
            for b in range(B):
                for c in range(C):
                    if mask[b, c].item() == 0:
                        continue
                    n_skills = int((sk[b, c] != 0).sum().item())
                    label = min(n_skills, 10)  # bucket 0..10
                    if label not in label_vocab:
                        label_vocab[label] = len(label_vocab)
                    enc_list.append(enc[b, c].cpu().numpy())
                    label_list.append(label_vocab[label])
                    seen += 1
                    if seen >= n_records:
                        break
                if seen >= n_records:
                    break

    if not enc_list:
        return None, None

    return np.stack(enc_list), np.array(label_list), label_vocab


def probe_linear(encodings, labels, n_classes, val_frac=0.2, epochs=50, lr=0.01):
    """Fit a linear probe on the frozen encodings. Returns val top-1 accuracy."""
    n = len(encodings)
    n_val = max(1, int(n * val_frac))
    idx = np.random.permutation(n)
    train_idx = idx[n_val:]
    val_idx   = idx[:n_val]

    X_train = torch.tensor(encodings[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx],    dtype=torch.long)
    X_val   = torch.tensor(encodings[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(labels[val_idx],      dtype=torch.long)

    k = encodings.shape[1]
    probe = nn.Linear(k, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        loss = F.cross_entropy(probe(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(X_val).argmax(dim=-1)
        acc = (pred == y_val).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ffb-ml/checkpoints/small_best.pt")
    parser.add_argument("--features",   default="ffb-ml/features")
    parser.add_argument("--output",     default="ffb-ml/eval")
    parser.add_argument("--n-records",  type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cpu")
    feat_dir = Path(args.features)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck_path = Path(args.checkpoint)
    ck = torch.load(ck_path, map_location=device)
    scale          = ck.get("scale", "small")
    n_skills       = ck["n_skills"]
    n_dialog_types = ck["n_dialog_types"]

    model = make_model(scale, n_skills, n_dialog_types)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"Collecting encodings (up to {args.n_records} players)...")
    result = collect_encodings(model, feat_dir, device, args.n_records)
    if result[0] is None:
        print("No data available for probe.", file=sys.stderr)
        sys.exit(1)
    encodings, labels, label_vocab = result
    n_classes = len(label_vocab)
    print(f"  {len(encodings)} encodings, {n_classes} label classes")

    # Probe at the actual k from this checkpoint
    k = encodings.shape[1]
    acc = probe_linear(encodings, labels, n_classes)
    print(f"\nLinear probe (k={k}): top-1 accuracy = {acc:.3f}")
    print(f"Random baseline: {1.0/n_classes:.3f}")

    # Also probe sub-dimensions to find the "knee"
    results = {}
    for sub_k in [4, 8, 16, 32]:
        if sub_k > k:
            continue
        sub_enc = encodings[:, :sub_k]
        a = probe_linear(sub_enc, labels, n_classes)
        results[sub_k] = float(a)
        print(f"  k={sub_k:3d}: {a:.3f}")
    results[k] = float(acc)

    out = {"k": k, "n_classes": n_classes, "random_baseline": 1.0 / n_classes,
           "probe_accuracy_by_k": results}
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved probe_results.json")

    try:
        import matplotlib.pyplot as plt
        ks = sorted(results.keys())
        accs = [results[k_] for k_ in ks]
        fig, ax = plt.subplots()
        ax.plot(ks, accs, "o-")
        ax.axhline(1.0 / n_classes, ls="--", color="gray", label="random")
        ax.set_xlabel("Encoder dimension k")
        ax.set_ylabel("Linear probe top-1 accuracy")
        ax.set_title("PlayerEncoder information vs k")
        ax.legend()
        fig.savefig(out_dir / "probe_accuracy_vs_k.png", dpi=120, bbox_inches="tight")
        print("Saved probe_accuracy_vs_k.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
