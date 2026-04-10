"""
export_model.py — Export a trained BCModel checkpoint to ONNX.

Usage:
    python src/export_model.py \
        --checkpoint ffb-ml/checkpoints/small_best.pt \
        --output     ffb-ml/bc_model.onnx

The exported model has three separate ONNX graphs (one per head) because ONNX
doesn't support dynamic dispatch. We export them as three files:
    bc_model_dialog.onnx
    bc_model_player_select.onnx
    bc_model_move_target.onnx

Java inference uses the appropriate file per decision type.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from model import BCModel, make_model
from extract_features import (
    N_BOARD_CHANNELS, BOARD_W, BOARD_H, NS_DIM, MAX_SKILLS, ENCODER_DIM, CAND_POS_DIM,
)


# ── Dialog export wrapper ─────────────────────────────────────────────────────

class DialogInferenceModel(nn.Module):
    """Wraps BCModel for ONNX export of the dialog head."""

    def __init__(self, model: BCModel):
        super().__init__()
        self.model = model
        self.n_dialog_feat = model.dialog_feat_dim

    def forward(self, spatial, non_spatial, dialog_features):
        rep, _ = self.model.compute_representation(spatial, non_spatial)
        logits = self.model.dialog_logits(rep, dialog_features)
        return logits


class PlayerSelectInferenceModel(nn.Module):
    """Wraps BCModel for ONNX export of the player-select head."""

    def __init__(self, model: BCModel):
        super().__init__()
        self.model = model

    def forward(self, spatial, non_spatial, cand_skill_ids, cand_stats, cand_mask, cand_pos):
        rep, _ = self.model.compute_representation(spatial, non_spatial)
        log_probs = self.model.player_select_scores(rep, cand_skill_ids, cand_stats, cand_mask,
                                                     cand_pos=cand_pos)
        return log_probs


class MoveTargetInferenceModel(nn.Module):
    """Wraps BCModel for ONNX export of the move-target head."""

    def __init__(self, model: BCModel):
        super().__init__()
        self.model = model

    def forward(self, spatial, non_spatial, candidate_mask):
        rep, cnn_feat = self.model.compute_representation(spatial, non_spatial)
        log_probs = self.model.move_target_scores(rep, cnn_feat, candidate_mask)
        return log_probs


# ── Export helpers ────────────────────────────────────────────────────────────

def export_dialog(model: BCModel, out_path: Path):
    wrapper = DialogInferenceModel(model).eval()
    spatial      = torch.zeros(1, N_BOARD_CHANNELS, BOARD_W, BOARD_H)
    non_spatial  = torch.zeros(1, NS_DIM)
    dialog_feat  = torch.zeros(1, model.dialog_feat_dim)

    torch.onnx.export(
        wrapper,
        (spatial, non_spatial, dialog_feat),
        str(out_path),
        input_names=["spatial", "non_spatial", "dialog_features"],
        output_names=["dialog_logits"],
        dynamic_axes={
            "spatial":          {0: "batch"},
            "non_spatial":      {0: "batch"},
            "dialog_features":  {0: "batch"},
            "dialog_logits":    {0: "batch"},
        },
        opset_version=18,
    )
    print(f"  Exported {out_path.name}")


def export_player_select(model: BCModel, out_path: Path, max_cands: int = 24):
    from extract_features import MAX_SKILLS
    wrapper = PlayerSelectInferenceModel(model).eval()
    spatial        = torch.zeros(1, N_BOARD_CHANNELS, BOARD_W, BOARD_H)
    non_spatial    = torch.zeros(1, NS_DIM)
    cand_skill_ids = torch.zeros(1, max_cands, MAX_SKILLS, dtype=torch.long)
    cand_stats     = torch.zeros(1, max_cands, 5)
    cand_mask      = torch.ones(1, max_cands + 1)
    cand_pos       = torch.zeros(1, max_cands + 1, CAND_POS_DIM)

    torch.onnx.export(
        wrapper,
        (spatial, non_spatial, cand_skill_ids, cand_stats, cand_mask, cand_pos),
        str(out_path),
        input_names=["spatial", "non_spatial", "cand_skill_ids", "cand_stats", "cand_mask", "cand_pos"],
        output_names=["player_log_probs"],
        dynamic_axes={
            "spatial":          {0: "batch"},
            "non_spatial":      {0: "batch"},
            "cand_skill_ids":   {0: "batch"},
            "cand_stats":       {0: "batch"},
            "cand_mask":        {0: "batch"},
            "cand_pos":         {0: "batch"},
            "player_log_probs": {0: "batch"},
        },
        opset_version=18,
    )
    print(f"  Exported {out_path.name}")


def export_move_target(model: BCModel, out_path: Path):
    wrapper = MoveTargetInferenceModel(model).eval()
    spatial       = torch.zeros(1, N_BOARD_CHANNELS, BOARD_W, BOARD_H)
    non_spatial   = torch.zeros(1, NS_DIM)
    candidate_mask = torch.ones(1, BOARD_W * BOARD_H + 1)

    torch.onnx.export(
        wrapper,
        (spatial, non_spatial, candidate_mask),
        str(out_path),
        input_names=["spatial", "non_spatial", "candidate_mask"],
        output_names=["move_log_probs"],
        dynamic_axes={
            "spatial":       {0: "batch"},
            "non_spatial":   {0: "batch"},
            "candidate_mask":{0: "batch"},
            "move_log_probs":{0: "batch"},
        },
        opset_version=18,
    )
    print(f"  Exported {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ffb-ml/checkpoints/small_best.pt")
    parser.add_argument("--output",     default="ffb-ml/bc_model.onnx",
                        help="Base path; three files will be created with _dialog/player_select/move_target suffix")
    args = parser.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.exists():
        print(f"ERROR: checkpoint not found: {ck_path}", file=sys.stderr)
        sys.exit(1)

    ck = torch.load(ck_path, map_location="cpu")
    scale          = ck.get("scale", "small")
    n_skills       = ck["n_skills"]
    n_dialog_types = ck["n_dialog_types"]

    model = make_model(scale, n_skills, n_dialog_types)
    model.load_state_dict(ck["model"])
    model.eval()

    base = Path(args.output).with_suffix("")
    print(f"Exporting {scale} model (n_skills={n_skills}, n_dialog={n_dialog_types})...")

    export_dialog(model,        Path(str(base) + "_dialog.onnx"))
    export_player_select(model, Path(str(base) + "_player_select.onnx"))
    export_move_target(model,   Path(str(base) + "_move_target.onnx"))

    print("\nDone. Load in Java with OrtSession.")


if __name__ == "__main__":
    main()
