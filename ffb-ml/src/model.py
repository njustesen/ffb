"""
model.py — Behavioral Cloning model for Blood Bowl.

Three heads share a CNN+MLP backbone:
  dialog_head        — per-dialog-type linear → softmax
  player_select_head — scores each player candidate individually
  move_target_head   — scores all board squares, mask to reachable → softmax

PlayerEncoder is trained jointly (end-to-end), not pre-trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from extract_features import (
    N_BOARD_CHANNELS, BOARD_W, BOARD_H, NS_DIM, ENCODER_DIM,
    MAX_SKILLS, PLAYER_ACTIONS, CAND_POS_DIM,
)

# Mid-channels used in SpatialMoveHead
_SPATIAL_HEAD_MID = 32


# ── PlayerEncoder ─────────────────────────────────────────────────────────────

class PlayerEncoder(nn.Module):
    """
    Maps (skill_ids, stats) → k-dim player embedding.

    skill_ids: LongTensor (batch, max_skills)  — 0 = padding
    stats:     FloatTensor (batch, 5)          — MA/ST/AG/AV/PA normalised /10
    """

    def __init__(self, n_skills: int, skill_emb_dim: int = 8, out_dim: int = ENCODER_DIM):
        super().__init__()
        self.skill_emb = nn.Embedding(n_skills + 1, skill_emb_dim, padding_idx=0)
        self.proj = nn.Linear(skill_emb_dim + 5, out_dim)
        self.out_dim = out_dim

    def forward(self, skill_ids: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        # skill_ids: (B, S), stats: (B, 5)
        emb = self.skill_emb(skill_ids)           # (B, S, emb_dim)
        # Mean-pool over the skill dimension, ignoring padding (id == 0)
        mask = (skill_ids != 0).float().unsqueeze(-1)  # (B, S, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)         # (B, 1)
        pooled = (emb * mask).sum(dim=1) / denom       # (B, emb_dim)
        x = torch.cat([pooled, stats], dim=-1)         # (B, emb_dim+5)
        return F.relu(self.proj(x))                    # (B, out_dim)


# ── Spatial move-target head ──────────────────────────────────────────────────

class SpatialMoveHead(nn.Module):
    """
    Data-efficient move-target scoring using the CNN feature map directly.

    Score(x, y) = score_proj(relu(cell_proj(cnn_feat[:, :, x, y])
                                  + global_proj(rep).unsqueeze(-1,-1)))

    This lets the model score each board cell using LOCAL spatial context (from
    the CNN feature at that cell) plus GLOBAL game-state context (from the
    pooled representation). Unlike the old FactoredMoveHead, it does NOT assume
    x-preference and y-preference are independent, and it uses spatial features
    that the GlobalAvgPool would otherwise discard.

    Parameters: cell_proj (c3→mid, 1×1 conv) + global_proj (256→mid)
                + score_proj (mid→1, 1×1 conv) + end_head (256→1)
    """

    def __init__(self, cnn_out_ch: int, rep_dim: int, mid: int = _SPATIAL_HEAD_MID):
        super().__init__()
        self.cell_proj   = nn.Conv2d(cnn_out_ch, mid, 1)  # local features
        self.global_proj = nn.Linear(rep_dim, mid)          # global context
        self.score_proj  = nn.Conv2d(mid, 1, 1)             # → scalar per cell
        self.end_head    = nn.Linear(rep_dim, 1)             # end-activation

    def forward(self, cnn_feat: torch.Tensor, rep: torch.Tensor,
                candidate_mask: torch.Tensor) -> torch.Tensor:
        B = cnn_feat.size(0)
        # Local branch: (B, mid, W, H)
        local = self.cell_proj(cnn_feat)
        # Global branch broadcast: (B, mid) → (B, mid, 1, 1) → broadcasts over W×H
        global_ctx = self.global_proj(rep).unsqueeze(-1).unsqueeze(-1)
        combined   = F.relu(local + global_ctx)               # (B, mid, W, H)
        cell_logits = self.score_proj(combined)               # (B, 1, W, H)
        cell_logits = cell_logits.reshape(B, BOARD_W * BOARD_H)  # (B, 390)
        end_logit   = self.end_head(rep)                       # (B, 1)
        logits = torch.cat([cell_logits, end_logit], dim=1)   # (B, 391)
        logits = logits.masked_fill(candidate_mask == 0, float("-inf"))
        return F.log_softmax(logits, dim=-1)


# ── Backbone ──────────────────────────────────────────────────────────────────

class BCBackbone(nn.Module):
    """
    Shared CNN + MLP backbone → 256-dim representation.

    spatial:     (B, C, 26, 15)  board tensor (CH_ENCODER_BASE channels replaced at runtime)
    non_spatial: (B, NS_DIM)     feature vector
    """

    def __init__(self, cnn_ch=(32, 64, 128), hidden=128, board_channels=N_BOARD_CHANNELS):
        super().__init__()
        c1, c2, c3 = cnn_ch

        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1),              nn.ReLU(),
            nn.Conv2d(c2, c3, 3, padding=1),              nn.ReLU(),
        )
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)  # → (B, c3, 1, 1) → (B, c3)

        self.ns_mlp = nn.Sequential(
            nn.Linear(NS_DIM, hidden),
            nn.ReLU(),
        )

        self.merge = nn.Sequential(
            nn.Linear(c3 + hidden, 256),
            nn.ReLU(),
        )

    def forward(self, spatial: torch.Tensor, non_spatial: torch.Tensor):
        """Returns (rep: (B,256), cnn_feat: (B,c3,W,H)) where cnn_feat is pre-pool."""
        cnn_feat = self.cnn(spatial)                                        # (B, c3, W, H)
        cnn_out  = self.cnn_pool(cnn_feat).squeeze(-1).squeeze(-1)         # (B, c3)
        ns_out   = self.ns_mlp(non_spatial)                                 # (B, hidden)
        rep      = self.merge(torch.cat([cnn_out, ns_out], dim=-1))        # (B, 256)
        return rep, cnn_feat


# ── Full BC model ─────────────────────────────────────────────────────────────

class BCModel(nn.Module):
    """
    Full behavioral cloning model with three decision heads.

    Config (scale examples):
      micro:  k=8,  cnn_ch=(16,32,64),    hidden=64
      small:  k=16, cnn_ch=(32,64,128),   hidden=128
      medium: k=32, cnn_ch=(64,128,256),  hidden=256
    """

    def __init__(
        self,
        n_skills: int,
        n_dialog_types: int,
        skill_emb_dim: int = 8,
        k: int = ENCODER_DIM,           # PlayerEncoder output dim
        cnn_ch: tuple = (32, 64, 128),
        hidden: int = 128,
        dialog_hidden: int = 64,
        player_select_hidden: int = 64,
    ):
        super().__init__()

        self.player_enc = PlayerEncoder(n_skills, skill_emb_dim, k)
        self.backbone   = BCBackbone(cnn_ch, hidden, board_channels=N_BOARD_CHANNELS)

        rep_dim = 256

        # Dialog head: one linear layer per dialog type (lazy init via ModuleDict)
        # Dialog features = n_dialog_types (one-hot) + 15 type-specific floats
        dialog_feat_dim = n_dialog_types + 15
        self.dialog_head = nn.Sequential(
            nn.Linear(rep_dim + dialog_feat_dim, dialog_hidden),
            nn.ReLU(),
            # Max 6 options per dialog type; we output raw logits and mask per sample
            nn.Linear(dialog_hidden, 6),
        )

        # Player-select head: score each candidate individually
        # Input: shared rep (256) + candidate PlayerEncoder output (k) + position (CAND_POS_DIM)
        self.player_select_head = nn.Sequential(
            nn.Linear(rep_dim + k + CAND_POS_DIM, player_select_hidden),
            nn.ReLU(),
            nn.Linear(player_select_hidden, 1),
        )

        # Move-target head: spatial scoring using CNN feature map + global context
        # score(x,y) = score_proj(relu(cell_proj(cnn[x,y]) + global_proj(rep)))
        c3 = cnn_ch[-1]  # output channels of CNN (last entry)
        self.move_target_head = SpatialMoveHead(c3, rep_dim)

        self.k = k
        self.rep_dim = rep_dim
        self.dialog_feat_dim = dialog_feat_dim

    def inject_player_enc(self, spatial: torch.Tensor,
                           skill_ids: torch.Tensor, stats: torch.Tensor,
                           board_x: torch.Tensor, board_y: torch.Tensor) -> torch.Tensor:
        """
        Fill CH_ENCODER_BASE..CH_ENCODER_BASE+k channels of spatial with PlayerEncoder outputs.

        skill_ids:  (B, N, MAX_SKILLS) — N players per sample
        stats:      (B, N, 5)
        board_x:    (B, N) — player x coordinate (or -1 if off-field)
        board_y:    (B, N) — player y coordinate (or -1 if off-field)

        Returns updated spatial tensor (same shape, new channels filled).
        """
        from extract_features import CH_ENCODER_BASE
        B, N = skill_ids.shape[:2]
        sk_flat  = skill_ids.view(B * N, -1)   # (B*N, MAX_SKILLS)
        st_flat  = stats.view(B * N, 5)        # (B*N, 5)
        enc_flat = self.player_enc(sk_flat, st_flat)  # (B*N, k)
        enc      = enc_flat.view(B, N, -1)    # (B, N, k)

        spatial = spatial.clone()
        k = enc.shape[-1]
        for n in range(N):
            x = board_x[:, n]  # (B,)
            y = board_y[:, n]  # (B,)
            valid = (x >= 0) & (y >= 0)
            for b in range(B):
                if valid[b]:
                    spatial[b, CH_ENCODER_BASE:CH_ENCODER_BASE + k, x[b], y[b]] = enc[b, n]
        return spatial

    def compute_representation(self, spatial: torch.Tensor,
                                non_spatial: torch.Tensor):
        """Returns (rep: (B,256), cnn_feat: (B,c3,W,H))."""
        return self.backbone(spatial, non_spatial)

    def dialog_logits(self, rep: torch.Tensor,
                      dialog_features: torch.Tensor) -> torch.Tensor:
        """Returns (B, 6) logits."""
        x = torch.cat([rep, dialog_features], dim=-1)
        return self.dialog_head(x)

    def player_select_scores(self, rep: torch.Tensor,
                              cand_skill_ids: torch.Tensor,
                              cand_stats: torch.Tensor,
                              cand_mask: torch.Tensor,
                              cand_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Score each candidate player.

        cand_skill_ids: (B, MAX_CANDS, MAX_SKILLS)
        cand_stats:     (B, MAX_CANDS, 5)
        cand_mask:      (B, MAX_CANDS+1)  — +1 for end-turn
        cand_pos:       (B, MAX_CANDS+1, CAND_POS_DIM) — optional position features

        Returns (B, MAX_CANDS+1) log-softmax probabilities.
        """
        B, C, _ = cand_skill_ids.shape
        sk_flat   = cand_skill_ids.view(B * C, -1)
        st_flat   = cand_stats.view(B * C, 5)
        enc_flat  = self.player_enc(sk_flat, st_flat)   # (B*C, k)
        enc       = enc_flat.view(B, C, -1)             # (B, C, k)

        rep_exp = rep.unsqueeze(1).expand(-1, C, -1)    # (B, C, 256)
        if cand_pos is not None:
            pos_cands = cand_pos[:, :C, :]              # (B, C, CAND_POS_DIM)
            inp = torch.cat([rep_exp, enc, pos_cands], dim=-1)  # (B, C, 256+k+CAND_POS_DIM)
        else:
            # Fallback: zero position (for backward compat)
            zeros = torch.zeros(B, C, CAND_POS_DIM, device=rep.device)
            inp = torch.cat([rep_exp, enc, zeros], dim=-1)
        scores  = self.player_select_head(inp).squeeze(-1)  # (B, C)

        # End-turn score: a learned bias; use zero for simplicity
        end_score = torch.zeros(B, 1, device=rep.device)
        all_scores = torch.cat([scores, end_score], dim=-1)  # (B, C+1)

        # Apply mask: set invalid positions to -inf
        all_scores = all_scores.masked_fill(cand_mask == 0, float("-inf"))
        return F.log_softmax(all_scores, dim=-1)

    def move_target_scores(self, rep: torch.Tensor,
                            cnn_feat: torch.Tensor,
                            candidate_mask: torch.Tensor) -> torch.Tensor:
        """
        Score all board positions + end-activation using the spatial head.

        rep:            (B, 256)           — global representation
        cnn_feat:       (B, c3, W, H)      — pre-pool CNN feature map
        candidate_mask: (B, 390+1)         — 1 where valid
        Returns (B, 391) log-softmax probabilities.
        """
        return self.move_target_head(cnn_feat, rep, candidate_mask)

    def forward(self, batch: dict) -> dict:
        """
        Flexible forward — handles whichever keys are present in batch.

        Always needs: x_spatial, x_ns
        For dialog:   dialog_features, action (used for loss)
        For player_select: cand_skill_ids, cand_stats, cand_mask, action
        For move_target:   candidate_mask, action
        """
        spatial     = batch["x_spatial"]
        non_spatial = batch["x_ns"]
        rep, cnn_feat = self.compute_representation(spatial, non_spatial)
        out = {"rep": rep}

        if "dialog_features" in batch:
            out["dialog_logits"] = self.dialog_logits(rep, batch["dialog_features"])

        if "cand_skill_ids" in batch:
            out["player_select_log_probs"] = self.player_select_scores(
                rep,
                batch["cand_skill_ids"],
                batch["cand_stats"],
                batch["cand_mask"],
                cand_pos=batch.get("cand_pos"),
            )

        if "candidate_mask" in batch:
            out["move_target_log_probs"] = self.move_target_scores(
                rep, cnn_feat, batch["candidate_mask"])

        return out


# ── Scale presets ─────────────────────────────────────────────────────────────

def make_model(scale: str, n_skills: int, n_dialog_types: int, **kwargs) -> BCModel:
    """Factory: scale in {'micro', 'small', 'medium'}."""
    configs = {
        "micro":  dict(k=8,  skill_emb_dim=4, cnn_ch=(16, 32, 64),    hidden=64,  dialog_hidden=32,  player_select_hidden=32),
        "small":  dict(k=16, skill_emb_dim=8, cnn_ch=(32, 64, 128),   hidden=128, dialog_hidden=64,  player_select_hidden=64),
        "medium": dict(k=32, skill_emb_dim=8, cnn_ch=(64, 128, 256),  hidden=256, dialog_hidden=128, player_select_hidden=128),
    }
    cfg = configs[scale]
    cfg.update(kwargs)
    return BCModel(n_skills=n_skills, n_dialog_types=n_dialog_types, **cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
