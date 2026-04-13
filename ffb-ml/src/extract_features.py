"""
extract_features.py — Convert JSONL training records to numpy feature arrays.

Usage:
    python src/extract_features.py --input ffb-ml/data --output ffb-ml/features

Reads shard_*.jsonl files from --input, produces per-type .npz shards and vocab.json
in --output.

Output files:
    vocab.json                       — {"skills": {"name": id, ...}, "dialog_types": {...},
                                        "player_actions": {...}, "player_states": {...}}
    dialog_shard_N.npz               — x_ns, dialog_type_id, action, n_options, x_skills, x_stats,
                                        dialog_features
    player_select_shard_N.npz        — x_spatial, x_ns, cand_skill_ids, cand_stats,
                                        cand_actions, action, n_candidates
    move_target_shard_N.npz          — x_spatial, x_ns, candidate_mask, action
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

BOARD_W = 26
BOARD_H = 15
MAX_CANDIDATES_MOVE = 400   # max reachable squares (generous upper bound)
MAX_PLAYERS_PER_TEAM = 16
MAX_SKILLS = 12             # max skill entries per player encoding
CAND_POS_DIM = 3            # [std_x/25, y/14, l1_to_ball/25] per candidate

# Board channel indices
CH_OWN_PRESENT   = 0
CH_OPP_PRESENT   = 1
CH_STATE_BASE    = 2        # 5 one-hot channels: standing/prone/stunned/used/off-field
CH_ENCODER_BASE  = 7        # k=16 PlayerEncoder output channels
CH_BALL_BASE     = 23       # 4 one-hot channels: ON_GROUND/CARRIED/IN_AIR/BOUNCING
CH_TACKLE_OWN    = 27
CH_TACKLE_OPP    = 28
CH_ACTING        = 29  # 1.0 at the acting player's board square, 0 elsewhere
N_BOARD_CHANNELS = 30

BALL_STATES = ["ON_GROUND", "CARRIED", "IN_AIR", "BOUNCING"]
PLAYER_STATES = ["STANDING", "MOVING", "PRONE", "STUNNED", "USED", "OFF_FIELD",
                 "KO", "BH", "SI", "RIP", "RESERVE", "MISSING", "FALLING",
                 "BLOCKED", "BANNED", "EXHAUSTED", "OTHER", "UNKNOWN"]
# 5-way one-hot for board encoding
BOARD_STATES_5 = {"STANDING": 0, "PRONE": 1, "STUNNED": 2}  # 3=used(active), 4=off-field

TURN_MODES = ["SETUP", "KICKOFF", "REGULAR", "BETWEEN_TURNS", "KICKOFF_RETURN", "OTHER"]
WEATHER_NAMES = ["NICE", "SUNNY", "POURING_RAIN", "BLIZZARD", "SWELTERING_HEAT",
                 "VERY_SUNNY", "PERFECT_CONDITIONS", "STRONG_WINDS", "UNKNOWN"]
PLAYER_ACTIONS = ["MOVE", "BLOCK", "BLITZ", "PASS", "FOUL", "HANDOFF", "TTM",
                  "STAB", "FOUL_MOVE", "KICK", "OTHER"]

NS_DIM = 143   # non-spatial feature vector dimension (see plan)
ENCODER_DIM = 16  # k

# ── Vocabulary builder ────────────────────────────────────────────────────────

def build_vocab(jsonl_files):
    """Scan all records to enumerate skills, dialog types, player actions, states."""
    skills = set()
    dialog_types = set()
    for path in jsonl_files:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = rec.get("state", {})
                for pid, p in state.get("players", {}).items():
                    for s in p.get("skills", []):
                        skills.add(s)
                t = rec.get("type")
                if t == "dialog":
                    dialog_types.add(rec.get("dialog_id", "UNKNOWN"))
    # ID 0 is reserved as padding for skill embeddings
    skill_vocab = {name: i + 1 for i, name in enumerate(sorted(skills))}
    dialog_vocab = {name: i for i, name in enumerate(sorted(dialog_types))}
    action_vocab = {name: i for i, name in enumerate(PLAYER_ACTIONS)}
    state_vocab = {name: i for i, name in enumerate(PLAYER_STATES)}
    return {
        "skills": skill_vocab,
        "dialog_types": dialog_vocab,
        "player_actions": action_vocab,
        "player_states": state_vocab,
    }


# ── State feature extraction ──────────────────────────────────────────────────

def encode_player_skills(player, skill_vocab):
    """Return fixed-length int array of skill IDs (padded with 0). Shape: (MAX_SKILLS,)"""
    ids = [skill_vocab.get(s, 0) for s in player.get("skills", [])]
    ids = ids[:MAX_SKILLS]
    ids += [0] * (MAX_SKILLS - len(ids))
    return np.array(ids, dtype=np.int32)


def encode_player_stats(player):
    """Return normalised float stats array [ma, st, ag, av, pa] / 10. Shape: (5,)"""
    return np.array([
        player.get("ma", 0) / 10.0,
        player.get("st", 0) / 10.0,
        player.get("ag", 0) / 10.0,
        player.get("av", 0) / 10.0,
        player.get("pa", 0) / 10.0,
    ], dtype=np.float32)


def std_x(x: int, home_playing: bool) -> int:
    """Standardise x coordinate: attack is always toward x=0 (home=no-op; away=flip)."""
    return x if home_playing else (BOARD_W - 1 - x)


def extract_spatial(state, vocab, home_playing, acting_player_id=None):
    """
    Build the (N_BOARD_CHANNELS, BOARD_W, BOARD_H) spatial board tensor.

    Channels:
      0: own player present
      1: opp player present
      2-6: player state one-hot (standing/prone/stunned/used/off-field)
      7-22: PlayerEncoder placeholder zeros (filled at runtime in model)
      23-26: ball state one-hot
      27: own tackle-zone density / 8
      28: opp tackle-zone density / 8
      29: acting player cursor (1.0 at acting player's square)

    Board is always oriented from the ACTING TEAM's perspective:
      x = 0 → opponent end zone (scoring direction)
      x = 25 → own end zone
    For the home team this is the default orientation; for the away team we
    mirror the x-axis so both teams always "attack toward x = 0".
    """
    board = np.zeros((N_BOARD_CHANNELS, BOARD_W, BOARD_H), dtype=np.float16)
    board_cells = state.get("board", [])  # flat list length 26*15, x-major
    players = state.get("players", {})

    own_key  = "home" if home_playing else "away"
    opp_key  = "away" if home_playing else "home"

    # Build lookup: player_id → player dict
    player_map = {pid: p for pid, p in players.items()}

    for x in range(BOARD_W):
        for y in range(BOARD_H):
            pid = board_cells[x * BOARD_H + y] if board_cells else None
            if pid is None:
                continue
            p = player_map.get(pid)
            if p is None:
                continue
            is_own = (p.get("team") == own_key)
            is_opp = (p.get("team") == opp_key)
            xs = std_x(x, home_playing)   # standardised x (flip for away team)
            board[CH_OWN_PRESENT, xs, y] = float(is_own)
            board[CH_OPP_PRESENT, xs, y] = float(is_opp)

            # Player state one-hot (5 classes)
            ps = p.get("state", "UNKNOWN")
            active = p.get("active", False)
            if ps == "STANDING":
                if not active:
                    board[CH_STATE_BASE + 3, xs, y] = 1.0  # used (stood up, not active)
                else:
                    board[CH_STATE_BASE + 0, xs, y] = 1.0  # standing active
            elif ps == "PRONE":
                board[CH_STATE_BASE + 1, xs, y] = 1.0
            elif ps == "STUNNED":
                board[CH_STATE_BASE + 2, xs, y] = 1.0
            else:
                board[CH_STATE_BASE + 4, xs, y] = 1.0  # off-field / other

            # Tackle zones
            if ps == "STANDING" and active:
                tz_ch = CH_TACKLE_OWN if is_own else CH_TACKLE_OPP
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = xs + dx, y + dy
                        if 0 <= nx < BOARD_W and 0 <= ny < BOARD_H:
                            board[tz_ch, nx, ny] += 1.0 / 8.0

    # Ball — use standardised x
    ball = state.get("ball", {})
    bx_raw = ball.get("x")
    by_raw = ball.get("y")
    bstate = ball.get("state", "ON_GROUND")
    if bx_raw is not None and by_raw is not None:
        bxs = std_x(int(bx_raw), home_playing)
        by_i = int(by_raw)
        if 0 <= bxs < BOARD_W and 0 <= by_i < BOARD_H:
            bi = BALL_STATES.index(bstate) if bstate in BALL_STATES else 0
            board[CH_BALL_BASE + bi, bxs, by_i] = 1.0

    # Channel 29: acting player cursor
    if acting_player_id is not None:
        ap_p = player_map.get(acting_player_id)
        if ap_p is not None:
            apx = ap_p.get("x")
            apy = ap_p.get("y")
            if apx is not None and apy is not None:
                apxs = std_x(int(apx), home_playing)
                apy_i = int(apy)
                if 0 <= apxs < BOARD_W and 0 <= apy_i < BOARD_H:
                    board[CH_ACTING, apxs, apy_i] = 1.0

    return board


def extract_nonspatial(state, vocab, home_playing):
    """
    Build ~143-dim non-spatial feature vector.
    Sections:
      1: match state (~38)
      2: team casualty counts (~16)
      3: acting player context (~24)
    Dialog section 4 is handled separately in extract_dialog_features().
    """
    ns = []

    # ── Section 1: match state ────────────────────────────────────────────────
    ns.append(float(state.get("half", 1) - 1))  # 0 or 1

    own_td  = state.get("home" if home_playing else "away", {})
    opp_td  = state.get("away" if home_playing else "home", {})
    ns.append(own_td.get("turn", 0) / 8.0)

    tm = state.get("turn_mode", "OTHER")
    tm_oh = [0.0] * len(TURN_MODES)
    idx = TURN_MODES.index(tm) if tm in TURN_MODES else TURN_MODES.index("OTHER")
    tm_oh[idx] = 1.0
    ns.extend(tm_oh)   # 6

    ns.append(float(home_playing))  # is_home_acting

    score_home = state.get("score_home", 0)
    score_away = state.get("score_away", 0)
    own_score = score_home if home_playing else score_away
    opp_score = score_away if home_playing else score_home
    ns.append(float(own_score))
    ns.append(float(opp_score))
    ns.append(max(-1.0, min(1.0, (own_score - opp_score) / 10.0)))  # clipped diff

    ns.append(own_td.get("rerolls", 0) / 8.0)
    ns.append(opp_td.get("rerolls", 0) / 8.0)
    ns.append(float(state.get("reroll_used", False)))  # own reroll used this turn
    # opp reroll used not tracked per-turn in current data; use 0
    ns.append(0.0)

    ns.append(float(state.get("blitz_used", False)))
    ns.append(float(state.get("foul_used", False)))
    ns.append(float(state.get("pass_used", False)))
    ns.append(float(state.get("handoff_used", False)))

    ns.append(own_td.get("apothecaries", 0) / 2.0)
    ns.append(opp_td.get("apothecaries", 0) / 2.0)
    ns.append(own_td.get("bribes", 0) / 3.0)
    ns.append(opp_td.get("bribes", 0) / 3.0)

    weather = state.get("weather", "NICE")
    w_oh = [0.0] * len(WEATHER_NAMES)
    wi = WEATHER_NAMES.index(weather) if weather in WEATHER_NAMES else WEATHER_NAMES.index("UNKNOWN")
    w_oh[wi] = 1.0
    ns.extend(w_oh)  # 9

    ball = state.get("ball", {})
    bstate = ball.get("state", "ON_GROUND")
    b_oh = [0.0] * 4
    b_oh[BALL_STATES.index(bstate) if bstate in BALL_STATES else 0] = 1.0
    ns.extend(b_oh)  # 4

    bx = ball.get("x")
    by = ball.get("y")
    mid_x = 13.0
    own_endzone_x = 25.0 if home_playing else 0.0
    opp_endzone_x = 0.0 if home_playing else 25.0
    if bx is not None:
        ball_in_own_half = (bx <= 12) if home_playing else (bx >= 13)
        ball_in_opp_endzone = (bx == 0) if home_playing else (bx == 25)
        ball_in_own_endzone = (bx == 25) if home_playing else (bx == 0)
        ball_dist_opp = abs(bx - opp_endzone_x) / 25.0
    else:
        ball_in_own_half = False
        ball_in_opp_endzone = False
        ball_in_own_endzone = False
        ball_dist_opp = 0.5
    ns.append(float(ball_in_own_half))
    ns.append(float(ball_in_opp_endzone))
    ns.append(float(ball_in_own_endzone))
    ns.append(ball_dist_opp)

    # ── Section 2: team casualty counts ──────────────────────────────────────
    # For each team: standing, prone, stunned, used, ko, bh, si, dead
    for team_key in (("home" if home_playing else "away"),
                     ("away" if home_playing else "home")):
        counts = {s: 0 for s in ["STANDING", "PRONE", "STUNNED", "KO", "BH", "SI", "RIP"]}
        used_count = 0
        for pid, p in state.get("players", {}).items():
            if p.get("team") != team_key:
                continue
            ps = p.get("state", "UNKNOWN")
            active = p.get("active", False)
            if ps == "STANDING" and not active:
                used_count += 1
            elif ps in counts:
                counts[ps] += 1
        ns.append(counts["STANDING"] / 11.0)
        ns.append(counts["PRONE"] / 11.0)
        ns.append(counts["STUNNED"] / 11.0)
        ns.append(used_count / 11.0)
        ns.append(counts["KO"] / 11.0)
        ns.append(counts["BH"] / 11.0)
        ns.append(counts["SI"] / 11.0)
        ns.append(counts["RIP"] / 11.0)

    # ── Section 3: acting player context ─────────────────────────────────────
    ap = state.get("acting_player", {})
    ap_id = ap.get("player_id")
    ap_player = state.get("players", {}).get(ap_id) if ap_id else None

    ns.append(float(ap_id is not None))  # player_active

    # Acting player context: position, stats, relative ball (replaces 16-zero placeholder)
    bx_f = ball.get("x")
    by_f = ball.get("y")
    opp_ez_x = 0.0 if home_playing else 25.0
    if ap_player is not None:
        ap_px = ap_player.get("x")
        ap_py = ap_player.get("y")
        if ap_px is not None and ap_py is not None:
            ap_px, ap_py = int(ap_px), int(ap_py)
            ap_pxs = std_x(ap_px, home_playing)           # standardised x (attack toward x=0)
            ns.append(ap_pxs / (BOARD_W - 1))             # 1: col normalised (standardised)
            ns.append(ap_py / (BOARD_H - 1))              # 2: row normalised
            ns.append(ap_player.get("ma", 0) / 10.0)      # 3: MA
            ns.append(ap_player.get("st", 0) / 10.0)      # 4: ST
            ns.append(ap_player.get("ag", 0) / 10.0)      # 5: AG
            ns.append(ap_player.get("av", 0) / 10.0)      # 6: AV
            ns.append(ap_player.get("pa", 0) / 10.0)      # 7: PA
            if bx_f is not None:
                bx_i, by_i = int(bx_f), int(by_f)
                bx_is = std_x(bx_i, home_playing)         # standardised ball x
                ns.append((bx_is - ap_pxs) / 25.0)        # 8: rel ball x (standardised)
                ns.append((by_i - ap_py) / 14.0)          # 9: rel ball y
                ns.append(min(abs(bx_is - ap_pxs) + abs(by_i - ap_py), 25) / 25.0)  # 10: L1
            else:
                ns.extend([0.0, 0.0, 0.5])
            ns.append(ap_pxs / 25.0)                      # 11: dist to opp endzone (std: opp=x=0)
            ns.extend([0.0] * 5)                           # 12-16: reserved
        else:
            ns.extend([0.0] * ENCODER_DIM)
    else:
        ns.extend([0.0] * ENCODER_DIM)

    ns.append(ap.get("current_move", 0) / 9.0)
    ns.append(float(ap.get("has_moved", False)))
    ns.append(float(ap.get("has_blocked", False)))
    ns.append(float(ap.get("has_fouled", False)))

    pa = ap.get("action")
    pa_oh = [0.0] * len(PLAYER_ACTIONS)
    if pa and pa in PLAYER_ACTIONS:
        pa_oh[PLAYER_ACTIONS.index(pa)] = 1.0
    elif pa:
        pa_oh[PLAYER_ACTIONS.index("OTHER")] = 1.0
    ns.extend(pa_oh)  # len(PLAYER_ACTIONS)

    ball_carrier_id = ball.get("carrier_id")
    ns.append(float(ap_id is not None and ap_id == ball_carrier_id))

    # Pad/trim to NS_DIM
    while len(ns) < NS_DIM:
        ns.append(0.0)
    ns = ns[:NS_DIM]

    return np.array(ns, dtype=np.float32)


def extract_dialog_features(rec, vocab):
    """
    Build dialog-type one-hot + up to 15 type-specific floats.
    Returns (dialog_type_id: int, features: float32 array of len 65)
    """
    dialog_vocab = vocab["dialog_types"]
    did = rec.get("dialog_id", "UNKNOWN")
    dtype_id = dialog_vocab.get(did, 0)

    # One-hot over all dialog types
    n_types = len(dialog_vocab)
    oh = [0.0] * n_types
    oh[dtype_id] = 1.0

    # Type-specific features (up to 15)
    specific = [0.0] * 15
    dp = rec.get("dialog_param", {})

    if did == "BLOCK_ROLL" or did == "BLOCK_ROLL_PARTIAL_RE_ROLL":
        specific[0] = dp.get("num_dice", 1) / 3.0
        specific[1] = 1.0  # placeholder: choosing_team_is_own (we don't have this easily)
        dice = dp.get("dice", [])
        for i, d in enumerate(dice[:4]):
            specific[2 + i] = d / 6.0

    elif did == "RE_ROLL":
        specific[0] = dp.get("min_roll", 4) / 6.0
        specific[1] = float(dp.get("is_team_reroll", False))
        specific[2] = float(dp.get("is_fumble", False))

    elif did == "SKILL_USE":
        specific[0] = dp.get("min_roll", 4) / 6.0

    elif did == "USE_APOTHECARY":
        # injury_state: SI → 1, others → 0
        specific[0] = float(dp.get("injury_state", 0) == 5)  # SI base value

    # Combine into flat feature vector
    features = np.array(oh + specific, dtype=np.float32)
    # Pad to fixed length (n_types + 15)
    return dtype_id, features


# ── Per-record processors ─────────────────────────────────────────────────────

def process_dialog(rec, vocab):
    """Returns dict of arrays for one dialog record."""
    state = rec.get("state", {})
    home_playing = state.get("home_playing", True)
    ap_id = state.get("acting_player", {}).get("player_id")

    x_spatial = extract_spatial(state, vocab, home_playing, acting_player_id=ap_id)
    x_ns = extract_nonspatial(state, vocab, home_playing)

    # Acting player skill/stat encoding
    ap = state.get("acting_player", {})
    ap_id = ap.get("player_id")
    ap_player = state.get("players", {}).get(ap_id) if ap_id else {}
    x_skills = encode_player_skills(ap_player, vocab["skills"])
    x_stats = encode_player_stats(ap_player)

    dtype_id, dialog_features = extract_dialog_features(rec, vocab)

    n_scores = len(rec.get("scores", []))
    action = int(rec.get("action", 0))

    return {
        "x_spatial":        x_spatial,
        "x_ns":             x_ns,
        "x_skills":         x_skills,
        "x_stats":          x_stats,
        "dialog_type_id":   np.int32(dtype_id),
        "action":           np.int32(action),
        "n_options":        np.int32(n_scores),
        "dialog_features":  dialog_features,
    }


def process_player_select(rec, vocab):
    """Returns dict of arrays for one player_select record."""
    state = rec.get("state", {})
    home_playing = state.get("home_playing", True)
    ap_id = state.get("acting_player", {}).get("player_id")

    x_spatial = extract_spatial(state, vocab, home_playing, acting_player_id=ap_id)
    x_ns = extract_nonspatial(state, vocab, home_playing)

    cands = rec.get("candidates", [])
    n_cands = len(cands)
    raw_action = int(rec.get("action", n_cands))

    # Per-candidate skill IDs and stats (up to n_cands entries, padded)
    MAX_CANDS = 24  # generous upper bound for player candidates (11 players × 2 actions + some)
    cand_skill_ids = np.zeros((MAX_CANDS, MAX_SKILLS), dtype=np.int32)
    cand_stats = np.zeros((MAX_CANDS, 5), dtype=np.float32)
    cand_actions = np.zeros(MAX_CANDS, dtype=np.int32)
    cand_mask = np.zeros(MAX_CANDS + 1, dtype=np.float32)  # +1 for end-turn

    players = state.get("players", {})
    act_vocab = vocab["player_actions"]
    ball = state.get("ball", {})
    bx_ball = ball.get("x")
    by_ball = ball.get("y")
    n_actual = min(n_cands, MAX_CANDS)
    # Candidate position features: [std_x/25, y/14, l1_to_ball/25]
    cand_pos = np.zeros((MAX_CANDS + 1, CAND_POS_DIM), dtype=np.float32)
    for i, c in enumerate(cands[:MAX_CANDS]):
        pid = c.get("player_id")
        p = players.get(pid, {})
        cand_skill_ids[i] = encode_player_skills(p, vocab["skills"])
        cand_stats[i] = encode_player_stats(p)
        pa = c.get("action", "MOVE")
        cand_actions[i] = act_vocab.get(pa, act_vocab.get("OTHER", 0))
        cand_mask[i] = 1.0
        # Position features (standardised coords)
        cx, cy = p.get("x"), p.get("y")
        if cx is not None and cy is not None:
            cxs = std_x(int(cx), home_playing)
            cy_i = int(cy)
            cand_pos[i, 0] = cxs / (BOARD_W - 1)
            cand_pos[i, 1] = cy_i / (BOARD_H - 1)
            if bx_ball is not None:
                bxs_ball = std_x(int(bx_ball), home_playing)
                l1 = abs(bxs_ball - cxs) + abs(int(by_ball) - cy_i)
                cand_pos[i, 2] = min(l1, 25) / 25.0
    cand_mask[MAX_CANDS] = 1.0  # end-turn is always valid (slot MAX_CANDS)

    # Map action to model output slot:
    #   raw_action < n_actual  → real candidate (direct index, capped at MAX_CANDS-1)
    #   raw_action >= n_cands  → end-turn → slot MAX_CANDS
    #   raw_action in [n_actual, n_cands)  → truncated candidate, treat as end-turn
    if raw_action < n_actual:
        action = raw_action
    else:
        action = MAX_CANDS  # end-turn

    return {
        "x_spatial":      x_spatial,
        "x_ns":           x_ns,
        "cand_skill_ids": cand_skill_ids,
        "cand_stats":     cand_stats,
        "cand_pos":       cand_pos,
        "cand_actions":   cand_actions,
        "cand_mask":      cand_mask,
        "action":         np.int32(action),
        "n_candidates":   np.int32(n_cands),
    }


def process_move_target(rec, vocab):
    """Returns dict of arrays for one move_target record."""
    state = rec.get("state", {})
    home_playing = state.get("home_playing", True)
    ap_id = state.get("acting_player", {}).get("player_id")

    x_spatial = extract_spatial(state, vocab, home_playing, acting_player_id=ap_id)
    x_ns = extract_nonspatial(state, vocab, home_playing)

    cands = rec.get("candidates", [])  # list of [x, y]
    has_end = rec.get("has_end_option", True)
    action = int(rec.get("action", len(cands)))

    # Candidate mask on board (26×15 flat) — using standardised x
    candidate_mask = np.zeros(BOARD_W * BOARD_H + 1, dtype=np.float32)  # +1 for end-activation
    for xy in cands:
        x, y = int(xy[0]), int(xy[1])
        xs = std_x(x, home_playing)
        if 0 <= xs < BOARD_W and 0 <= y < BOARD_H:
            candidate_mask[xs * BOARD_H + y] = 1.0
    if has_end:
        candidate_mask[BOARD_W * BOARD_H] = 1.0  # end-activation option

    # Convert action index (into candidates list) to board position index (standardised x)
    if action < len(cands):
        xy = cands[action]
        xs = std_x(int(xy[0]), home_playing)
        board_action = xs * BOARD_H + int(xy[1])
    else:
        board_action = BOARD_W * BOARD_H  # end-activation

    return {
        "x_spatial":      x_spatial,
        "x_ns":           x_ns,
        "candidate_mask": candidate_mask,
        "action":         np.int32(board_action),
    }


def process_value(rec, vocab):
    """
    Returns dict of arrays for one value-prediction record.

    All targets are DELTAS from the current state to game end, from the acting
    team's perspective. Using deltas avoids the model trivially copying the
    current score/cas/spp from the state features.

      win_label       float32  1.0 = acting team wins, 0.5 = draw, 0.0 = loses
      delta_score     float32  future TDs scored by acting team − future TDs conceded
      delta_cas_suf   float32  future cas suffered by opponent − future cas suffered by acting team
                               (positive = acting team causes more future damage)
      delta_spp       float32  future SPP earned by acting team − future SPP earned by opponent
    """
    state = rec.get("state", {})
    home_playing = state.get("home_playing", True)
    # Value records are snapped at turn start — no specific acting player
    x_spatial = extract_spatial(state, vocab, home_playing, acting_player_id=None)
    x_ns      = extract_nonspatial(state, vocab, home_playing)

    outcome = rec.get("outcome", {})

    # ── Final outcome values (from outcome block) ──────────────────────────────
    final_score_home = float(outcome.get("score_home",    0))
    final_score_away = float(outcome.get("score_away",    0))
    final_cas_suf_h  = float(outcome.get("cas_suf_home",  0))
    final_cas_suf_a  = float(outcome.get("cas_suf_away",  0))
    final_spp_home   = float(outcome.get("spp_home",      0))
    final_spp_away   = float(outcome.get("spp_away",      0))

    # ── Current state values (from serialized game state) ─────────────────────
    curr_score_home = float(state.get("score_home", 0))
    curr_score_away = float(state.get("score_away", 0))

    # Current casualties suffered = BH + SI + RIP count visible in player states
    curr_cas_suf_home = 0.0
    curr_cas_suf_away = 0.0
    for p in state.get("players", {}).values():
        ps = p.get("state", "")
        if ps in ("BH", "SI", "RIP"):
            if p.get("team") == "home":
                curr_cas_suf_home += 1.0
            else:
                curr_cas_suf_away += 1.0

    # Current SPP earned this game (added to serialized team data)
    home_data = state.get("home", {})
    away_data = state.get("away", {})
    curr_spp_home = float(home_data.get("spp_earned", 0))
    curr_spp_away = float(away_data.get("spp_earned", 0))

    # ── Compute deltas ─────────────────────────────────────────────────────────
    delta_score_home = (final_score_home - curr_score_home) - (final_score_away - curr_score_away)
    delta_score_away = (final_score_away - curr_score_away) - (final_score_home - curr_score_home)

    # cas_suf delta: how much more will the OPPONENT suffer (= how much more damage we deal)
    delta_cas_own_h  = (final_cas_suf_a - curr_cas_suf_away) - (final_cas_suf_h - curr_cas_suf_home)
    delta_cas_own_a  = (final_cas_suf_h - curr_cas_suf_home) - (final_cas_suf_a - curr_cas_suf_away)

    delta_spp_home = (final_spp_home - curr_spp_home) - (final_spp_away - curr_spp_away)
    delta_spp_away = (final_spp_away - curr_spp_away) - (final_spp_home - curr_spp_home)

    if home_playing:
        win_label   = 1.0 if final_score_home > final_score_away else (0.5 if final_score_home == final_score_away else 0.0)
        delta_score = delta_score_home
        delta_cas   = delta_cas_own_h
        delta_spp   = delta_spp_home
    else:
        win_label   = 1.0 if final_score_away > final_score_home else (0.5 if final_score_home == final_score_away else 0.0)
        delta_score = delta_score_away
        delta_cas   = delta_cas_own_a
        delta_spp   = delta_spp_away

    return {
        "x_spatial":    x_spatial,
        "x_ns":         x_ns,
        "win_label":    np.float32(win_label),
        "delta_score":  np.float32(np.clip(delta_score, -8, 8)),
        "delta_cas":    np.float32(np.clip(delta_cas,   -8, 8)),
        "delta_spp":    np.float32(np.clip(delta_spp,  -30, 30)),
    }


# ── Shard writer ──────────────────────────────────────────────────────────────

def flush_shard(shard_data, out_dir, prefix, shard_idx):
    """Save accumulated records as a single npz shard."""
    if not shard_data:
        return
    arrays = {}
    for key in shard_data[0]:
        arrays[key] = np.stack([d[key] for d in shard_data])
    path = out_dir / f"{prefix}_shard_{shard_idx:03d}.npz"
    np.savez(path, **arrays)  # uncompressed so mmap_mode='r' works in training
    print(f"  Saved {path.name}: {len(shard_data)} records")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="ffb-ml/data",     help="Directory of shard_*.jsonl files")
    parser.add_argument("--output", default="ffb-ml/features", help="Output directory for npz shards")
    parser.add_argument("--shard-size", type=int, default=10000, help="Records per output shard")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(in_dir.glob("shard_*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no shard_*.jsonl files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL shard(s) in {in_dir}")

    # ── Pass 1: build vocabulary ───────────────────────────────────────────────
    print("Building vocabulary...")
    vocab = build_vocab(jsonl_files)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"  Skills: {len(vocab['skills'])}  Dialog types: {len(vocab['dialog_types'])}")

    # ── Pass 2: extract features ───────────────────────────────────────────────
    ALL_TYPES = ["dialog", "player_select", "move_target", "value"]
    type_counts = {t: 0 for t in ALL_TYPES}
    shard_bufs  = {t: [] for t in ALL_TYPES}
    shard_idxs  = {t: 0  for t in ALL_TYPES}
    prefixes    = {t: t  for t in ALL_TYPES}

    for path in jsonl_files:
        print(f"Processing {path.name}...")
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("type")

                # Dedicated value record — emitted once per turn by JsonlTrainingDataCollector
                if t == "value":
                    try:
                        vdata = process_value(rec, vocab)
                        shard_bufs["value"].append(vdata)
                        type_counts["value"] += 1
                        if len(shard_bufs["value"]) >= args.shard_size:
                            flush_shard(shard_bufs["value"], out_dir, "value", shard_idxs["value"])
                            shard_idxs["value"] += 1
                            shard_bufs["value"] = []
                    except Exception as e:
                        print(f"  WARNING line {line_no} (value): {e}", file=sys.stderr)
                    continue

                if t not in ("dialog", "player_select", "move_target"):
                    continue

                # Policy record
                try:
                    if t == "dialog":
                        data = process_dialog(rec, vocab)
                    elif t == "player_select":
                        data = process_player_select(rec, vocab)
                    else:
                        data = process_move_target(rec, vocab)
                    shard_bufs[t].append(data)
                    type_counts[t] += 1
                    if len(shard_bufs[t]) >= args.shard_size:
                        flush_shard(shard_bufs[t], out_dir, prefixes[t], shard_idxs[t])
                        shard_idxs[t] += 1
                        shard_bufs[t] = []
                except Exception as e:
                    print(f"  WARNING line {line_no} ({t}): {e}", file=sys.stderr)

    # Flush remaining
    for t in ALL_TYPES:
        flush_shard(shard_bufs[t], out_dir, prefixes[t], shard_idxs[t])

    print("\n=== Feature extraction complete ===")
    for t, n in type_counts.items():
        print(f"  {t}: {n} records")


if __name__ == "__main__":
    main()
