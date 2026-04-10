"""
train.py — Joint BC training across all three decision types.

Usage:
    python src/train.py --features ffb-ml/features --output ffb-ml/checkpoints --scale small

Options:
    --features DIR        directory of npz shards + vocab.json (default: ffb-ml/features)
    --output DIR          checkpoint directory (default: ffb-ml/checkpoints)
    --scale {micro,small,medium}  model scale (default: small)
    --epochs N            training epochs (default: 50)
    --batch-size N        batch size (default: 256)
    --lr FLOAT            learning rate (default: 5e-4)
    --w-dialog FLOAT      dialog loss weight (default: 2.0)
    --w-ps FLOAT          player-select loss weight (default: 2.0)
    --w-mt FLOAT          move-target loss weight (default: 1.0)
    --seed N              random seed (default: 42)
    --device {cpu,cuda,mps}  device (default: auto-detect)
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# Make sure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from model import BCModel, make_model, count_parameters
from extract_features import MAX_SKILLS, NS_DIM, ENCODER_DIM


# ── Dataset classes ───────────────────────────────────────────────────────────

class NpzShardDataset(Dataset):
    """Eagerly loads one npz shard into RAM once; NpzFile is not cached so must do this."""

    def __init__(self, path: Path):
        raw = np.load(path)
        # Materialize all arrays now — NpzFile re-reads zip on every access otherwise
        self.arrays = {k: raw[k] for k in raw.files}
        self.keys   = list(self.arrays.keys())
        self.length = len(self.arrays[self.keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        out = {}
        for k in self.keys:
            arr = self.arrays[k][idx]
            t   = torch.as_tensor(np.array(arr))
            # Spatial boards are stored as float16 to save RAM; cast to float32 for compute
            if t.dtype == torch.float16:
                t = t.float()
            out[k] = t
        return out


def load_shards(feature_dir: Path, prefix: str, max_shards: int = None) -> Dataset:
    paths = sorted(feature_dir.glob(f"{prefix}_shard_*.npz"))
    if not paths:
        return None
    if max_shards is not None:
        paths = paths[:max_shards]
    datasets = [NpzShardDataset(p) for p in paths]
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


# ── Loss functions ────────────────────────────────────────────────────────────

def dialog_loss(logits: torch.Tensor, action: torch.Tensor,
                n_options: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on valid options only (mask out options >= n_options)."""
    B, max_opts = logits.shape
    mask = torch.arange(max_opts, device=logits.device).unsqueeze(0) < n_options.unsqueeze(1)
    logits_masked = logits.masked_fill(~mask, float("-inf"))
    return F.cross_entropy(logits_masked, action.long())


def nll_loss(log_probs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood loss for pre-masked log-softmax outputs."""
    return F.nll_loss(log_probs, action.long())


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def dialog_accuracy(logits, action, n_options):
    mask = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0) < n_options.unsqueeze(1)
    logits_masked = logits.masked_fill(~mask, float("-inf"))
    pred = logits_masked.argmax(dim=-1)
    return (pred == action.long()).float().mean().item()


def nll_accuracy(log_probs, action):
    pred = log_probs.argmax(dim=-1)
    return (pred == action.long()).float().mean().item()


# ── Training loop ─────────────────────────────────────────────────────────────

def _process_batch(model, batch, dtype, optimizer, device, weights):
    """Forward/backward one batch. Returns (loss_val, acc)."""
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    out = model(batch)
    if dtype == "dialog":
        loss = weights["dialog"] * dialog_loss(out["dialog_logits"], batch["action"], batch["n_options"])
        acc  = dialog_accuracy(out["dialog_logits"], batch["action"], batch["n_options"])
    elif dtype == "player_select":
        loss = weights["player_select"] * nll_loss(out["player_select_log_probs"], batch["action"])
        acc  = nll_accuracy(out["player_select_log_probs"], batch["action"])
    else:
        loss = weights["move_target"] * nll_loss(out["move_target_log_probs"], batch["action"])
        acc  = nll_accuracy(out["move_target_log_probs"], batch["action"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), acc


def train_epoch_streaming(model, shard_paths_by_type, optimizer, device, weights, batch_size, epoch_seed,
                          shards_per_epoch=None):
    """Stream through shards, loading and discarding one at a time.

    When shards_per_epoch is set, randomly sample that many shards per type
    (with cycling so all shards are seen over multiple epochs).
    """
    model.train()
    total_loss = 0.0
    counts = {"dialog": 0, "player_select": 0, "move_target": 0}
    accs   = {k: 0.0 for k in counts}

    rng = random.Random(epoch_seed)

    # Build list of (dtype, path) for this epoch
    all_shards = []
    for dtype, paths in shard_paths_by_type.items():
        if not paths:
            continue
        if shards_per_epoch is not None and shards_per_epoch < len(paths):
            # Cycle: deterministically offset by epoch, then take a window
            n = shards_per_epoch
            offset = (epoch_seed * n) % len(paths)
            ordered = list(paths)
            # wrap-around slice
            selected = (ordered + ordered)[offset: offset + n]
        else:
            selected = list(paths)
        for p in selected:
            all_shards.append((dtype, p))

    rng.shuffle(all_shards)

    for dtype, path in all_shards:
        ds     = NpzShardDataset(path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        for batch in loader:
            l, a = _process_batch(model, batch, dtype, optimizer, device, weights)
            total_loss += l
            counts[dtype] += 1
            accs[dtype]   += a
        del ds, loader
        gc.collect()

    avg_accs = {k: accs[k] / max(1, counts[k]) for k in counts}
    return total_loss / max(1, sum(counts.values())), avg_accs


def build_val_loaders_fixed(shard_paths_by_type, val_frac=0.1, max_val_shards=3, batch_size=512, seed=42):
    """Build a small fixed val set from the first few shards of each type."""
    val_loaders = {}
    for dtype, paths in shard_paths_by_type.items():
        if not paths:
            val_loaders[dtype] = None
            continue
        val_datasets = []
        for path in paths[:max_val_shards]:
            ds = NpzShardDataset(path)
            n = len(ds)
            n_val = max(1, int(n * val_frac))
            _, val_ds = torch.utils.data.random_split(
                ds, [n - n_val, n_val],
                generator=torch.Generator().manual_seed(seed)
            )
            val_datasets.append(val_ds)
        combined = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        val_loaders[dtype] = DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loaders


def train_epoch(model, loaders, optimizer, device, weights):
    model.train()
    total_loss = 0.0
    counts = {"dialog": 0, "player_select": 0, "move_target": 0}
    accs = {k: 0.0 for k in counts}

    # Interleave batches from all three loaders
    iters = {k: iter(v) for k, v in loaders.items() if v is not None}
    active = set(iters.keys())

    while active:
        for dtype in list(active):
            try:
                batch = next(iters[dtype])
            except StopIteration:
                active.discard(dtype)
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)

            if dtype == "dialog":
                loss = weights["dialog"] * dialog_loss(out["dialog_logits"], batch["action"], batch["n_options"])
                acc  = dialog_accuracy(out["dialog_logits"], batch["action"], batch["n_options"])
            elif dtype == "player_select":
                loss = weights["player_select"] * nll_loss(out["player_select_log_probs"], batch["action"])
                acc  = nll_accuracy(out["player_select_log_probs"], batch["action"])
            else:
                loss = weights["move_target"] * nll_loss(out["move_target_log_probs"], batch["action"])
                acc  = nll_accuracy(out["move_target_log_probs"], batch["action"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            counts[dtype] += 1
            accs[dtype]   += acc

    avg_accs = {k: accs[k] / max(1, counts[k]) for k in counts}
    n_batches = sum(counts.values())
    return total_loss / max(1, n_batches), avg_accs


@torch.no_grad()
def eval_epoch(model, loaders, device):
    model.eval()
    accs = {"dialog": [], "player_select": [], "move_target": []}

    for dtype, loader in loaders.items():
        if loader is None:
            continue
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            if dtype == "dialog":
                acc = dialog_accuracy(out["dialog_logits"], batch["action"], batch["n_options"])
            elif dtype == "player_select":
                acc = nll_accuracy(out["player_select_log_probs"], batch["action"])
            else:
                acc = nll_accuracy(out["move_target_log_probs"], batch["action"])
            accs[dtype].append(acc)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in accs.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="ffb-ml/features")
    parser.add_argument("--output",     default="ffb-ml/checkpoints")
    parser.add_argument("--scale",      default="small", choices=["micro", "small", "medium"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--w-dialog",   type=float, default=2.0)
    parser.add_argument("--w-ps",       type=float, default=2.0)
    parser.add_argument("--w-mt",       type=float, default=1.0)
    parser.add_argument("--max-shards", type=int,   default=None,
                        help="Limit number of shards per type (default: all)")
    parser.add_argument("--patience",          type=int,   default=None,
                        help="Early stopping patience in epochs (default: no early stopping)")
    parser.add_argument("--shards-per-epoch", type=int,   default=None,
                        help="Shards per type per epoch in streaming mode; cycles through all shards (default: all)")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    feat_dir = Path(args.features)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vocab
    with open(feat_dir / "vocab.json") as f:
        vocab = json.load(f)
    n_skills       = len(vocab["skills"])
    n_dialog_types = len(vocab["dialog_types"])
    print(f"Vocab: {n_skills} skills, {n_dialog_types} dialog types")

    # Discover shards (no loading yet — we'll rotate through them per epoch)
    ms = args.max_shards
    def get_shard_paths(prefix):
        paths = sorted(feat_dir.glob(f"{prefix}_shard_*.npz"))
        return paths[:ms] if ms else paths

    dialog_paths = get_shard_paths("dialog")
    ps_paths     = get_shard_paths("player_select")
    mt_paths     = get_shard_paths("move_target")

    print(f"  dialog shards:        {len(dialog_paths)}")
    print(f"  player_select shards: {len(ps_paths)}")
    print(f"  move_target shards:   {len(mt_paths)}")
    if not dialog_paths and not ps_paths and not mt_paths:
        print("ERROR: no shards found", file=sys.stderr); sys.exit(1)

    # Model
    model = make_model(args.scale, n_skills, n_dialog_types)
    model.to(device)
    print(f"\nModel [{args.scale}]: {count_parameters(model):,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    weights = {
        "dialog":        args.w_dialog,
        "player_select": args.w_ps,
        "move_target":   args.w_mt,
    }
    print(f"\nLoss weights: dialog={weights['dialog']}, player_select={weights['player_select']}, move_target={weights['move_target']}")

    total_shards = len(dialog_paths) + len(ps_paths) + len(mt_paths)
    STREAM_THRESHOLD = 10  # use streaming when total shards exceed this

    shard_paths_by_type = {
        "dialog":        dialog_paths,
        "player_select": ps_paths,
        "move_target":   mt_paths,
    }

    use_streaming = total_shards > STREAM_THRESHOLD
    if use_streaming:
        print(f"\nStreaming mode: {total_shards} shards (one at a time per epoch)")
        print(f"  dialog: {len(dialog_paths)} shards")
        print(f"  player_select: {len(ps_paths)} shards")
        print(f"  move_target: {len(mt_paths)} shards")
        print("  No val split — using full dataset for training, reporting train accuracy.")
    else:
        print("\nLoading full dataset into memory...")
        train_loaders_fixed = {}
        for key, paths in shard_paths_by_type.items():
            if not paths:
                train_loaders_fixed[key] = None; continue
            dsets = [NpzShardDataset(p) for p in paths]
            ds = ConcatDataset(dsets) if len(dsets) > 1 else dsets[0]
            train_loaders_fixed[key] = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            print(f"  {key}: {len(ds):,} samples across {len(paths)} shards")

    best_avg_acc = 0.0
    patience_counter = 0
    if use_streaming and args.shards_per_epoch:
        max_shards_any = max(len(dialog_paths), len(ps_paths), len(mt_paths))
        cycle_epochs = max(1, max_shards_any // args.shards_per_epoch)
        mode_str = f"streaming {args.shards_per_epoch} shards/type/epoch (full cycle every ~{cycle_epochs} epochs)"
    elif use_streaming:
        mode_str = "streaming all shards"
    else:
        mode_str = "full dataset in memory"
    print(f"\nTraining {args.epochs} epochs ({mode_str})...")
    if args.patience:
        print(f"Early stopping patience: {args.patience} epochs (on train accuracy)")
    print(f"{'Epoch':>5}  {'Loss':>8}  {'D-acc':>7}  {'PS-acc':>7}  {'MT-acc':>7}  {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if use_streaming:
            loss, train_accs = train_epoch_streaming(
                model, shard_paths_by_type, optimizer, device, weights,
                args.batch_size, epoch_seed=args.seed + epoch,
                shards_per_epoch=args.shards_per_epoch)
        else:
            loss, train_accs = train_epoch(model, train_loaders_fixed, optimizer, device, weights)
        scheduler.step()

        # Use training accuracies directly (no val set)
        d_acc  = train_accs.get("dialog", 0.0)
        ps_acc = train_accs.get("player_select", 0.0)
        mt_acc = train_accs.get("move_target", 0.0)
        avg_acc = np.mean([a for a in [d_acc, ps_acc, mt_acc] if a > 0])

        elapsed = time.time() - t0
        print(f"{epoch:5d}  {loss:8.4f}  {d_acc:7.3f}  {ps_acc:7.3f}  {mt_acc:7.3f}  {elapsed:5.1f}s")

        # Save checkpoint every 5 epochs and best
        if epoch % 5 == 0 or epoch == args.epochs:
            ck = out_dir / f"{args.scale}_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_accs": train_accs, "scale": args.scale,
                        "n_skills": n_skills, "n_dialog_types": n_dialog_types}, ck)
            print(f"  → Saved {ck.name}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            patience_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_accs": train_accs, "scale": args.scale,
                        "n_skills": n_skills, "n_dialog_types": n_dialog_types},
                       out_dir / f"{args.scale}_best.pt")
        else:
            patience_counter += 1

        if args.patience and patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest avg train accuracy: {best_avg_acc:.3f}")
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
