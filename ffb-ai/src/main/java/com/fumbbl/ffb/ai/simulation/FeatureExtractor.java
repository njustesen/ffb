package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.TurnData;
import com.fumbbl.ffb.model.skill.Skill;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Java-side feature extractor that replicates the Python {@code extract_features.py} logic.
 *
 * <p>Produces flat float arrays suitable for feeding into the ONNX model:
 * <ul>
 *   <li>{@code buildSpatialBoard()} → float[30 × 26 × 15] channel-major</li>
 *   <li>{@code buildNonSpatial()} → float[143]</li>
 *   <li>{@code buildDialogFeatures()} → float[n_dialog_types + 15]</li>
 *   <li>{@code buildCandidateMask()} → float[26×15 + 1] for move-target</li>
 * </ul>
 *
 * <p>All normalisation constants match the Python extractor exactly.
 */
public final class FeatureExtractor {

    // ── Board dimensions ──────────────────────────────────────────────────────

    public static final int BOARD_W = 26;
    public static final int BOARD_H = 15;
    public static final int N_BOARD_CHANNELS = 30;
    private static final int CH_ACTING       = 29;  // acting player cursor
    public static final int NS_DIM = 143;
    public static final int ENCODER_DIM = 16;
    public static final int MAX_SKILLS = 12;
    public static final int MAX_CANDS = 24; // must match extract_features.py MAX_CANDS=24

    // Channel layout
    private static final int CH_OWN_PRESENT  = 0;
    private static final int CH_OPP_PRESENT  = 1;
    private static final int CH_STATE_BASE   = 2;   // 5 channels
    private static final int CH_ENCODER_BASE = 7;   // 16 channels (zeros, filled at inference)
    private static final int CH_BALL_BASE    = 23;  // 4 channels
    private static final int CH_TACKLE_OWN   = 27;
    private static final int CH_TACKLE_OPP   = 28;

    private static final String[] BALL_STATES    = {"ON_GROUND", "CARRIED", "IN_AIR", "BOUNCING"};
    private static final String[] TURN_MODES     = {"SETUP", "KICKOFF", "REGULAR", "BETWEEN_TURNS", "KICKOFF_RETURN", "OTHER"};
    private static final String[] WEATHER_NAMES  = {"NICE", "SUNNY", "POURING_RAIN", "BLIZZARD", "SWELTERING_HEAT",
                                                     "VERY_SUNNY", "PERFECT_CONDITIONS", "STRONG_WINDS", "UNKNOWN"};
    private static final String[] PLAYER_ACTIONS = {"MOVE", "BLOCK", "BLITZ", "PASS", "FOUL", "HANDOFF", "TTM",
                                                     "STAB", "FOUL_MOVE", "KICK", "OTHER"};

    private final Map<String, Integer> skillVocab;
    private final int nDialogTypes;

    public FeatureExtractor(Map<String, Integer> skillVocab, int nDialogTypes) {
        this.skillVocab    = skillVocab;
        this.nDialogTypes  = nDialogTypes;
    }

    // ── Spatial board ─────────────────────────────────────────────────────────

    /**
     * Build the 29×26×15 spatial board tensor as a flat row-major float array.
     * Layout: board[channel * BOARD_W * BOARD_H + x * BOARD_H + y]
     */
    public float[] buildSpatialBoard(Game game) {
        float[] board = new float[N_BOARD_CHANNELS * BOARD_W * BOARD_H];
        FieldModel fm   = game.getFieldModel();
        boolean homePlaying = game.isHomePlaying();
        Team myTeam  = homePlaying ? game.getTeamHome() : game.getTeamAway();
        Team oppTeam = homePlaying ? game.getTeamAway() : game.getTeamHome();

        // Ball
        FieldCoordinate ballCoord    = fm.getBallCoordinate();
        boolean ballInAir   = game.getPassCoordinate() != null;
        boolean ballBouncing = !ballInAir && fm.isBallMoving();

        for (int x = 0; x < BOARD_W; x++) {
            for (int y = 0; y < BOARD_H; y++) {
                Player<?> p = fm.getPlayer(new FieldCoordinate(x, y));
                if (p == null) continue;
                boolean isOwn = myTeam.hasPlayer(p);
                boolean isOpp = oppTeam.hasPlayer(p);
                if (!isOwn && !isOpp) continue;

                int xs = stdX(x, homePlaying);  // standardised x: attack always toward x=0
                set(board, CH_OWN_PRESENT, xs, y, isOwn ? 1.0f : 0.0f);
                set(board, CH_OPP_PRESENT, xs, y, isOpp ? 1.0f : 0.0f);

                // Player state one-hot (5 classes)
                PlayerState ps = fm.getPlayerState(p);
                int stateBase = ps != null ? ps.getBase() : -1;
                boolean active = ps != null && ps.isActive();

                if (stateBase == PlayerState.STANDING) {
                    set(board, active ? CH_STATE_BASE : CH_STATE_BASE + 3, xs, y, 1.0f);
                } else if (stateBase == PlayerState.PRONE) {
                    set(board, CH_STATE_BASE + 1, xs, y, 1.0f);
                } else if (stateBase == PlayerState.STUNNED) {
                    set(board, CH_STATE_BASE + 2, xs, y, 1.0f);
                } else {
                    set(board, CH_STATE_BASE + 4, xs, y, 1.0f);
                }

                // Tackle zones (use standardised coords for neighbour cells)
                if (stateBase == PlayerState.STANDING && active) {
                    int tzCh = isOwn ? CH_TACKLE_OWN : CH_TACKLE_OPP;
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            int nx = xs + dx, ny = y + dy;
                            if (nx >= 0 && nx < BOARD_W && ny >= 0 && ny < BOARD_H) {
                                add(board, tzCh, nx, ny, 1.0f / 8.0f);
                            }
                        }
                    }
                }
            }
        }

        // Ball state channel (standardised x)
        if (ballCoord != null) {
            int bx = ballCoord.getX(), by = ballCoord.getY();
            int bxs = stdX(bx, homePlaying);
            if (bxs >= 0 && bxs < BOARD_W && by >= 0 && by < BOARD_H) {
                String bstate;
                if (ballInAir) bstate = "IN_AIR";
                else if (ballBouncing) bstate = "BOUNCING";
                else {
                    Player<?> carrier = fm.getPlayer(ballCoord);
                    bstate = carrier != null ? "CARRIED" : "ON_GROUND";
                }
                int bi = indexOf(BALL_STATES, bstate);
                if (bi < 0) bi = 0;
                set(board, CH_BALL_BASE + bi, bxs, by, 1.0f);
            }
        }

        // Channel 29: acting player cursor (standardised x)
        ActingPlayer actingPlayer = game.getActingPlayer();
        if (actingPlayer != null && actingPlayer.getPlayer() != null) {
            FieldCoordinate apCoord = fm.getPlayerCoordinate(actingPlayer.getPlayer());
            if (apCoord != null && apCoord.getX() >= 0 && apCoord.getX() < BOARD_W
                    && apCoord.getY() >= 0 && apCoord.getY() < BOARD_H) {
                int apxs = stdX(apCoord.getX(), homePlaying);
                set(board, CH_ACTING, apxs, apCoord.getY(), 1.0f);
            }
        }

        return board;
    }

    // ── Non-spatial features ──────────────────────────────────────────────────

    /**
     * Build the 143-dim non-spatial feature vector.
     */
    public float[] buildNonSpatial(Game game) {
        float[] ns = new float[NS_DIM];
        int idx = 0;
        boolean home = game.isHomePlaying();
        TurnData ownTd  = home ? game.getTurnDataHome() : game.getTurnDataAway();
        TurnData oppTd  = home ? game.getTurnDataAway() : game.getTurnDataHome();
        FieldModel fm   = game.getFieldModel();

        // Section 1: match state
        ns[idx++] = game.getHalf() - 1;  // 0 or 1
        ns[idx++] = (ownTd != null ? ownTd.getTurnNr() : 0) / 8.0f;

        // Turn mode one-hot (6)
        String tm = game.getTurnMode() != null ? game.getTurnMode().name() : "OTHER";
        int tmi = indexOf(TURN_MODES, tm);
        if (tmi < 0) tmi = TURN_MODES.length - 1;
        ns[idx + tmi] = 1.0f;
        idx += TURN_MODES.length;

        ns[idx++] = home ? 1.0f : 0.0f;

        int scoreHome = game.getGameResult() != null ? game.getGameResult().getScoreHome() : 0;
        int scoreAway = game.getGameResult() != null ? game.getGameResult().getScoreAway() : 0;
        int ownScore = home ? scoreHome : scoreAway;
        int oppScore = home ? scoreAway : scoreHome;
        ns[idx++] = ownScore;
        ns[idx++] = oppScore;
        ns[idx++] = Math.max(-1.0f, Math.min(1.0f, (ownScore - oppScore) / 10.0f));

        ns[idx++] = (ownTd != null ? ownTd.getReRolls() : 0) / 8.0f;
        ns[idx++] = (oppTd != null ? oppTd.getReRolls() : 0) / 8.0f;
        ns[idx++] = game.getTurnData() != null && game.getTurnData().isReRollUsed() ? 1.0f : 0.0f;
        ns[idx++] = 0.0f; // opp reroll used (not tracked)

        TurnData td = game.getTurnData();
        ns[idx++] = td != null && td.isBlitzUsed()   ? 1.0f : 0.0f;
        ns[idx++] = td != null && td.isFoulUsed()    ? 1.0f : 0.0f;
        ns[idx++] = td != null && td.isPassUsed()    ? 1.0f : 0.0f;
        ns[idx++] = td != null && td.isHandOverUsed()? 1.0f : 0.0f;

        ns[idx++] = ownTd != null ? (ownTd.getApothecaries() + ownTd.getWanderingApothecaries()) / 2.0f : 0;
        ns[idx++] = oppTd != null ? (oppTd.getApothecaries() + oppTd.getWanderingApothecaries()) / 2.0f : 0;
        ns[idx++] = 0.0f; // bribes (inducement tracking skipped for Java side)
        ns[idx++] = 0.0f;

        // Weather (9)
        String weather = fm.getWeather() != null ? fm.getWeather().name() : "NICE";
        int wi = indexOf(WEATHER_NAMES, weather);
        if (wi < 0) wi = WEATHER_NAMES.length - 1;
        ns[idx + wi] = 1.0f;
        idx += WEATHER_NAMES.length;

        // Ball state (4)
        FieldCoordinate ballCoord = fm.getBallCoordinate();
        String bstate;
        if (game.getPassCoordinate() != null) bstate = "IN_AIR";
        else if (fm.isBallMoving()) bstate = "BOUNCING";
        else if (ballCoord != null && fm.getPlayer(ballCoord) != null) bstate = "CARRIED";
        else bstate = "ON_GROUND";
        int bi = indexOf(BALL_STATES, bstate);
        if (bi < 0) bi = 0;
        ns[idx + bi] = 1.0f;
        idx += 4;

        // Ball position features (4)
        if (ballCoord != null) {
            int bx = ballCoord.getX();
            boolean ownHalf = home ? (bx <= 12) : (bx >= 13);
            boolean oppEndzone = home ? (bx == 0) : (bx == 25);
            boolean ownEndzone = home ? (bx == 25) : (bx == 0);
            float oppEz = home ? 0.0f : 25.0f;
            float distToOpp = Math.abs(bx - oppEz) / 25.0f;
            ns[idx++] = ownHalf ? 1.0f : 0.0f;
            ns[idx++] = oppEndzone ? 1.0f : 0.0f;
            ns[idx++] = ownEndzone ? 1.0f : 0.0f;
            ns[idx++] = distToOpp;
        } else {
            ns[idx++] = 0.0f; ns[idx++] = 0.0f; ns[idx++] = 0.0f; ns[idx++] = 0.5f;
        }

        // Section 2: team casualty counts (8 × 2)
        Team[] teams = { home ? game.getTeamHome() : game.getTeamAway(),
                         home ? game.getTeamAway() : game.getTeamHome() };
        for (Team team : teams) {
            if (team == null) { idx += 8; continue; }
            int standing = 0, prone = 0, stunned = 0, used = 0, ko = 0, bh = 0, si = 0, rip = 0;
            for (Player<?> p : team.getPlayers()) {
                PlayerState ps = fm.getPlayerState(p);
                if (ps == null) continue;
                int base = ps.getBase();
                boolean active = ps.isActive();
                if (base == PlayerState.STANDING && active)       standing++;
                else if (base == PlayerState.STANDING && !active) used++;
                else if (base == PlayerState.PRONE)               prone++;
                else if (base == PlayerState.STUNNED)             stunned++;
                else if (base == PlayerState.KNOCKED_OUT)         ko++;
                else if (base == PlayerState.BADLY_HURT)          bh++;
                else if (base == PlayerState.SERIOUS_INJURY)      si++;
                else if (base == PlayerState.RIP)                 rip++;
            }
            ns[idx++] = standing / 11.0f;
            ns[idx++] = prone    / 11.0f;
            ns[idx++] = stunned  / 11.0f;
            ns[idx++] = used     / 11.0f;
            ns[idx++] = ko       / 11.0f;
            ns[idx++] = bh       / 11.0f;
            ns[idx++] = si       / 11.0f;
            ns[idx++] = rip      / 11.0f;
        }

        // Section 3: acting player context
        ActingPlayer ap = game.getActingPlayer();
        boolean hasAp = ap != null && ap.getPlayerId() != null;
        ns[idx++] = hasAp ? 1.0f : 0.0f;

        // 16 acting-player features (replaces ENCODER_DIM zeros)
        if (hasAp) {
            Player<?> apPlayer = ap.getPlayer();
            FieldCoordinate apCoord = apPlayer != null ? fm.getPlayerCoordinate(apPlayer) : null;
            int apxRaw = (apCoord != null && apCoord.getX() >= 0) ? apCoord.getX() : 0;
            float apy = (apCoord != null && apCoord.getY() >= 0) ? apCoord.getY() : 0.0f;
            float apxs = stdX(apxRaw, home);                     // standardised x
            ns[idx++] = apxs / (BOARD_W - 1);                   // 1: col (standardised)
            ns[idx++] = apy / (BOARD_H - 1);                    // 2: row
            ns[idx++] = apPlayer != null ? apPlayer.getMovement() / 10.0f : 0.0f;  // 3: MA
            ns[idx++] = apPlayer != null ? apPlayer.getStrength() / 10.0f : 0.0f;  // 4: ST
            ns[idx++] = apPlayer != null ? apPlayer.getAgility()  / 10.0f : 0.0f;  // 5: AG
            ns[idx++] = apPlayer != null ? apPlayer.getArmour()   / 10.0f : 0.0f;  // 6: AV
            ns[idx++] = apPlayer != null ? apPlayer.getPassing()  / 10.0f : 0.0f;  // 7: PA
            // ball-relative features (standardised coords)
            float bxs2 = ballCoord != null ? stdX(ballCoord.getX(), home) : apxs;
            float by2  = ballCoord != null ? ballCoord.getY() : apy;
            ns[idx++] = (bxs2 - apxs) / 25.0f;                  // 8: rel ball x (standardised)
            ns[idx++] = (by2 - apy) / 14.0f;                    // 9: rel ball y
            float l1 = Math.abs(bxs2 - apxs) + Math.abs(by2 - apy);
            ns[idx++] = Math.min(l1, 25.0f) / 25.0f;            // 10: L1 dist to ball
            // In standardised coords opp endzone is always at x=0, so dist = apxs / 25
            ns[idx++] = apxs / 25.0f;                            // 11: dist to opp endzone
            ns[idx++] = 0.0f; // 12: own TZ at ap (not computed here; reserved)
            ns[idx++] = 0.0f; // 13: opp TZ at ap (not computed here; reserved)
            ns[idx++] = 0.0f; // 14: reserved
            ns[idx++] = 0.0f; // 15: reserved
            ns[idx++] = 0.0f; // 16: reserved
        } else {
            idx += ENCODER_DIM;
        }

        ns[idx++] = ap != null ? ap.getCurrentMove() / 9.0f : 0.0f;
        ns[idx++] = ap != null && ap.hasMoved()   ? 1.0f : 0.0f;
        ns[idx++] = ap != null && ap.hasBlocked() ? 1.0f : 0.0f;
        ns[idx++] = ap != null && ap.hasFouled()  ? 1.0f : 0.0f;

        // Player action one-hot
        PlayerAction pa = ap != null ? ap.getPlayerAction() : null;
        String paName = pa != null ? pa.name() : "";
        int paSub = indexOf(PLAYER_ACTIONS, paName);
        if (paSub < 0 && !paName.isEmpty()) paSub = PLAYER_ACTIONS.length - 1; // OTHER
        if (paSub >= 0) ns[idx + paSub] = 1.0f;
        idx += PLAYER_ACTIONS.length;

        // Is ball carrier
        String carrierId = ballCoord != null && fm.getPlayer(ballCoord) != null
            ? fm.getPlayer(ballCoord).getId() : null;
        ns[idx++] = (hasAp && carrierId != null && carrierId.equals(ap.getPlayerId())) ? 1.0f : 0.0f;

        // Rest stays zero
        return ns;
    }

    // ── Dialog features ───────────────────────────────────────────────────────

    /**
     * Build dialog-type one-hot + 15 type-specific floats.
     * Length = nDialogTypes + 15.
     */
    public float[] buildDialogFeatures(IDialogParameter dialog) {
        int len = nDialogTypes + 15;
        float[] features = new float[len];

        if (dialog == null) return features;

        String dialogId = dialog.getId().name();
        // Lookup dialog type ID from vocab (passed in separately to OnnxModelAgent)
        // Just fill the one-hot using the ID from vocab if available
        // This method relies on the caller to know the dialog type index
        // → return just the 15 type-specific floats; one-hot filled by OnnxModelAgent
        return features;
    }

    /**
     * Build dialog features given the pre-looked-up dialog type ID.
     */
    public float[] buildDialogFeatures(IDialogParameter dialog, int dialogTypeId) {
        int len = nDialogTypes + 15;
        float[] features = new float[len];
        if (dialogTypeId >= 0 && dialogTypeId < nDialogTypes) {
            features[dialogTypeId] = 1.0f;
        }
        if (dialog == null) return features;

        int base = nDialogTypes;
        String did = dialog.getId().name();

        try {
            if (did.equals("BLOCK_ROLL") || did.equals("BLOCK_ROLL_PARTIAL_RE_ROLL")) {
                java.lang.reflect.Method m = dialog.getClass().getMethod("getNrOfDice");
                int numDice = (int) m.invoke(dialog);
                features[base] = numDice / 3.0f;
                // dice values if available
                try {
                    java.lang.reflect.Method mr = dialog.getClass().getMethod("getBlockRoll");
                    int[] roll = (int[]) mr.invoke(dialog);
                    if (roll != null) {
                        for (int i = 0; i < Math.min(roll.length, 4); i++) {
                            features[base + 2 + i] = roll[i] / 6.0f;
                        }
                    }
                } catch (Exception ignored) {}
            } else if (did.equals("RE_ROLL")) {
                java.lang.reflect.Method mMin = dialog.getClass().getMethod("getMinimumRoll");
                int minRoll = (int) mMin.invoke(dialog);
                features[base] = minRoll / 6.0f;
                try {
                    java.lang.reflect.Method mTr = dialog.getClass().getMethod("isTeamReRollOption");
                    features[base + 1] = ((boolean) mTr.invoke(dialog)) ? 1.0f : 0.0f;
                    java.lang.reflect.Method mFu = dialog.getClass().getMethod("isFumble");
                    features[base + 2] = ((boolean) mFu.invoke(dialog)) ? 1.0f : 0.0f;
                } catch (Exception ignored) {}
            } else if (did.equals("SKILL_USE")) {
                java.lang.reflect.Method mMin = dialog.getClass().getMethod("getMinimumRoll");
                int minRoll = (int) mMin.invoke(dialog);
                features[base] = minRoll / 6.0f;
            }
        } catch (Exception ignored) {}

        return features;
    }

    // ── Candidate mask (move-target) ──────────────────────────────────────────

    /**
     * Build the candidate mask for move-target decisions.
     * Returns float[BOARD_W × BOARD_H + 1], +1 for end-activation.
     * Positions are stored in standardised x coordinates (stdX) matching training.
     */
    public float[] buildMoveCandidateMask(List<FieldCoordinate> candidates, boolean hasEndOption, boolean home) {
        float[] mask = new float[BOARD_W * BOARD_H + 1];
        for (FieldCoordinate c : candidates) {
            int x = c.getX(), y = c.getY();
            int xs = stdX(x, home);
            if (xs >= 0 && xs < BOARD_W && y >= 0 && y < BOARD_H) {
                mask[xs * BOARD_H + y] = 1.0f;
            }
        }
        if (hasEndOption) {
            mask[BOARD_W * BOARD_H] = 1.0f;
        }
        return mask;
    }

    /**
     * Build candidate skill IDs and stats for player-select decisions.
     *
     * @return flat int array of shape (MAX_CANDS × MAX_SKILLS) row-major
     */
    public int[] buildCandidateSkillIds(List<Player<?>> candidates) {
        int[] ids = new int[MAX_CANDS * MAX_SKILLS];
        for (int i = 0; i < Math.min(candidates.size(), MAX_CANDS); i++) {
            Player<?> p = candidates.get(i);
            if (p == null) continue;
            Skill[] skills = p.getSkills();
            if (skills == null) continue;
            int col = 0;
            for (Skill s : skills) {
                if (col >= MAX_SKILLS) break;
                Integer sid = skillVocab.get(s.getName());
                ids[i * MAX_SKILLS + col++] = sid != null ? sid : 0;
            }
        }
        return ids;
    }

    public float[] buildCandidateStats(List<Player<?>> candidates) {
        float[] stats = new float[MAX_CANDS * 5];
        for (int i = 0; i < Math.min(candidates.size(), MAX_CANDS); i++) {
            Player<?> p = candidates.get(i);
            if (p == null) continue;
            stats[i * 5]     = p.getMovement()    / 10.0f;
            stats[i * 5 + 1] = p.getStrength()    / 10.0f;
            stats[i * 5 + 2] = p.getAgility()     / 10.0f;
            stats[i * 5 + 3] = p.getArmour()      / 10.0f;
            stats[i * 5 + 4] = p.getPassing()     / 10.0f;
        }
        return stats;
    }

    /**
     * Build candidate position features: [std_x/25, y/14, l1_to_ball/25] per candidate.
     * Returns float[(MAX_CANDS+1) × CAND_POS_DIM] row-major (last row = end-turn = zeros).
     * coords is a parallel list of FieldCoordinate (or null if off-field) for each candidate.
     */
    public static final int CAND_POS_DIM = 3;

    public float[] buildCandidatePos(List<FieldCoordinate> coords, FieldCoordinate ballCoord, boolean home) {
        float[] pos = new float[(MAX_CANDS + 1) * CAND_POS_DIM];
        int bxs = ballCoord != null ? stdX(ballCoord.getX(), home) : -1;
        int by  = ballCoord != null ? ballCoord.getY() : -1;
        for (int i = 0; i < Math.min(coords.size(), MAX_CANDS); i++) {
            FieldCoordinate c = coords.get(i);
            if (c == null || c.getX() < 0) continue;
            int cxs = stdX(c.getX(), home);
            int cy  = c.getY();
            pos[i * CAND_POS_DIM + 0] = cxs / (BOARD_W - 1.0f);
            pos[i * CAND_POS_DIM + 1] = cy  / (BOARD_H - 1.0f);
            if (bxs >= 0) {
                float l1 = Math.abs(bxs - cxs) + Math.abs(by - cy);
                pos[i * CAND_POS_DIM + 2] = Math.min(l1, 25.0f) / 25.0f;
            }
        }
        return pos;
    }

    public float[] buildCandidateMaskPs(int nActual) {
        float[] mask = new float[MAX_CANDS + 1];
        for (int i = 0; i < Math.min(nActual, MAX_CANDS); i++) {
            mask[i] = 1.0f;
        }
        mask[MAX_CANDS] = 1.0f; // end-turn always valid
        return mask;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Standardise x coordinate so that "attack toward x=0" for both teams.
     * Home team: identity.  Away team: mirror (BOARD_W - 1 - x).
     * This is its own inverse: stdX(stdX(x, home), home) == x.
     */
    public static int stdX(int x, boolean home) {
        return home ? x : (BOARD_W - 1 - x);
    }

    private static void set(float[] board, int ch, int x, int y, float val) {
        board[ch * BOARD_W * BOARD_H + x * BOARD_H + y] = val;
    }

    private static void add(float[] board, int ch, int x, int y, float val) {
        board[ch * BOARD_W * BOARD_H + x * BOARD_H + y] += val;
    }

    private static int indexOf(String[] arr, String val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(val)) return i;
        }
        return -1;
    }
}
