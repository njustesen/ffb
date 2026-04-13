package com.fumbbl.ffb.ai.simulation;

import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.ai.PathProbabilityFinder;
import com.fumbbl.ffb.ai.strategy.DecisionLog;
import com.fumbbl.ffb.dialog.DialogBlockRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPartialReRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogReRollParameter;
import com.fumbbl.ffb.dialog.DialogSkillUseParameter;
import com.fumbbl.ffb.dialog.DialogUseApothecaryParameter;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.GameResult;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.PlayerResult;
import com.fumbbl.ffb.model.TeamResult;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link ITrainingDataCollector} that buffers one JSON object per decision, then on
 * {@link #onGameEnd} annotates every record with the game outcome and flushes to the writer.
 *
 * <p>Records are discarded (not written) if the game timed out (result == null).
 *
 * <p>Four record types are emitted, distinguished by the {@code "type"} field:
 * <ul>
 *   <li>{@code "dialog"}         — ScriptedStrategy dialog response</li>
 *   <li>{@code "player_select"}  — MoveDecisionEngine player selection</li>
 *   <li>{@code "move_target"}    — MoveDecisionEngine move target</li>
 * </ul>
 *
 * <p>Every record also carries an {@code "outcome"} object (added at game end) with the
 * following fields, all from the <em>home team's perspective</em>:
 * <ul>
 *   <li>{@code score_home} / {@code score_away}  — final touchdown counts</li>
 *   <li>{@code cas_home}   / {@code cas_away}    — casualties <em>inflicted</em> (caused on opponent)</li>
 *   <li>{@code cas_suf_home} / {@code cas_suf_away} — casualties <em>suffered</em> (BH+SI+RIP)</li>
 *   <li>{@code spp_home}   / {@code spp_away}    — total SPP earned this game by each team</li>
 * </ul>
 */
public final class JsonlTrainingDataCollector implements ITrainingDataCollector {

    private final BufferedWriter writer;
    private final List<JsonObject> buffer = new ArrayList<>();

    public JsonlTrainingDataCollector(BufferedWriter writer) {
        this.writer = writer;
    }

    // ── Dialog ────────────────────────────────────────────────────────────────

    @Override
    public void onDialog(IDialogParameter dialog, Game game, DecisionLog log, String agentMode) {
        if (dialog == null || log == null || log.size() == 0) return;

        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  dialog.getId().name());
        rec.add("agent_mode", agentMode);

        // First pick is the primary decision
        rec.add("action", log.firstChosen());
        rec.add("scores", doubleArray(log.firstScores()));

        // All picks (some dialogs have multiple)
        if (log.size() > 1) {
            JsonArray allPicks = new JsonArray();
            for (int i = 0; i < log.size(); i++) {
                JsonObject pick = new JsonObject();
                pick.add("scores", doubleArray(log.getScores(i)));
                pick.add("action", log.getChosen(i));
                allPicks.add(pick);
            }
            rec.add("all_picks", allPicks);
        }

        rec.add("dialog_param", serializeDialogParam(dialog));
        rec.add("state",        GameStateSerializer.serialize(game));

        buffer.add(rec);
    }

    // ── Player select ─────────────────────────────────────────────────────────

    @Override
    public void onPlayerSelect(Game game, MoveDecisionEngine.PlayerSelection sel, String agentMode) {
        if (sel == null) return;

        JsonObject rec = new JsonObject();
        rec.add("type",       "player_select");
        rec.add("agent_mode", agentMode);

        // Chosen index: len(non-null candidates) = end turn
        // The last entry in candidatePlayers/candidateActions is always (null, null) for end-turn.
        // We omit it from the candidates list and use index == candidates.size() to mean end-turn.
        List<Player<?>> cands = sel.candidatePlayers;
        List<PlayerAction> actions = sel.candidateActions;
        // Count only non-null entries (everything except the trailing end-turn sentinel)
        int numReal = 0;
        for (Player<?> p : cands) { if (p != null) numReal++; }
        int chosenIdx = numReal; // default: end-turn
        if (sel.player != null) {
            for (int i = 0; i < cands.size(); i++) {
                Player<?> cp = cands.get(i);
                if (cp != null && cp.getId().equals(sel.player.getId())
                        && actions.get(i) == sel.action) {
                    chosenIdx = i;
                    break;
                }
            }
        }
        rec.add("action", chosenIdx);
        rec.add("scores", doubleArray(sel.rawScores));

        // Candidate list — omit the trailing null end-turn sentinel
        JsonArray candidates = new JsonArray();
        for (int i = 0; i < cands.size(); i++) {
            Player<?> cp = cands.get(i);
            if (cp == null) continue; // end-turn sentinel
            JsonObject c = new JsonObject();
            c.add("player_id", cp.getId());
            PlayerAction pa = actions.get(i);
            c.add("action", pa != null ? pa.name() : "END_TURN");
            candidates.add(c);
        }
        rec.add("candidates", candidates);
        rec.add("end_turn_option", true); // always an option

        rec.add("state", GameStateSerializer.serialize(game));
        buffer.add(rec);
    }

    // ── Move target ───────────────────────────────────────────────────────────

    @Override
    public void onMoveTarget(Game game, ActingPlayer ap, MoveDecisionEngine.MoveResult mr, String agentMode) {
        if (mr == null || ap == null) return;

        JsonObject rec = new JsonObject();
        rec.add("type",       "move_target");
        rec.add("agent_mode", agentMode);

        if (ap.getPlayerId() != null) {
            rec.add("player_id", ap.getPlayerId());
        } else {
            rec.add("player_id", JsonValue.NULL);
        }
        if (ap.getPlayerAction() != null) {
            rec.add("action_type", ap.getPlayerAction().name());
        } else {
            rec.add("action_type", JsonValue.NULL);
        }
        rec.add("has_end_option", mr.hasEndOption);

        // Chosen index: len(candidates) = end activation
        List<FieldCoordinate> cands = mr.candidates;
        int chosenIdx = cands.size(); // default: end
        if (mr.chosen != null) {
            FieldCoordinate dest = lastCoord(mr.chosen);
            if (dest != null) {
                for (int i = 0; i < cands.size(); i++) {
                    if (cands.get(i).equals(dest)) {
                        chosenIdx = i;
                        break;
                    }
                }
            }
        }
        rec.add("action", chosenIdx);
        rec.add("scores", doubleArray(mr.rawScores));

        // Candidate coordinates
        JsonArray coords = new JsonArray();
        for (FieldCoordinate fc : cands) {
            JsonArray xy = new JsonArray();
            xy.add(fc.getX());
            xy.add(fc.getY());
            coords.add(xy);
        }
        rec.add("candidates", coords);

        rec.add("state", GameStateSerializer.serialize(game));
        buffer.add(rec);
    }

    // ── Game end ──────────────────────────────────────────────────────────────

    @Override
    public void onGameEnd(GameResult result) {
        if (result == null) {
            buffer.clear(); // timed-out game — discard
            return;
        }

        JsonObject outcome = buildOutcome(result);
        try {
            // Emit at most one value record per (half, turn_nr, home_playing) tuple.
            // Keep the first record encountered in each turn — the earliest decision
            // in the turn is the least correlated with others in the same turn.
            // All records are still written as policy training examples (unchanged).
            java.util.Set<String> seenTurns = new java.util.HashSet<>();
            for (JsonObject rec : buffer) {
                // Write the policy record unconditionally
                writer.write(rec.toString());
                writer.newLine();

                // Write a value record only for the first decision in each turn
                JsonValue stateVal = rec.get("state");
                if (stateVal != null && stateVal.isObject()) {
                    JsonObject state = stateVal.asObject();
                    int half        = state.getInt("half",        0);
                    int turnNr      = state.getInt("turn_nr",     0);
                    boolean homePlaying = state.getBoolean("home_playing", true);
                    String turnKey  = half + ":" + turnNr + ":" + homePlaying;
                    if (seenTurns.add(turnKey)) {
                        // First decision in this turn — emit a value record
                        JsonObject valueRec = new JsonObject();
                        valueRec.add("type",    "value");
                        valueRec.add("state",   stateVal);
                        valueRec.add("outcome", outcome);
                        writer.write(valueRec.toString());
                        writer.newLine();
                    }
                }
            }
            writer.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            buffer.clear();
        }
    }

    /** Build the outcome object from the home team's perspective. */
    private static JsonObject buildOutcome(GameResult result) {
        JsonObject o = new JsonObject();
        o.add("score_home", result.getScoreHome());
        o.add("score_away", result.getScoreAway());

        TeamResult home = result.getTeamResultHome();
        TeamResult away = result.getTeamResultAway();

        // Casualties suffered (BH + SI + RIP) — happens to the team itself
        int casSufHome = home.getBadlyHurtSuffered()
                       + home.getSeriousInjurySuffered()
                       + home.getRipSuffered();
        int casSufAway = away.getBadlyHurtSuffered()
                       + away.getSeriousInjurySuffered()
                       + away.getRipSuffered();
        o.add("cas_suf_home", casSufHome);
        o.add("cas_suf_away", casSufAway);

        // Casualties inflicted (caused on opponent) — sum over player results
        int casHome = 0, casAway = 0;
        if (home.getTeam() != null) {
            for (com.fumbbl.ffb.model.Player<?> p : home.getTeam().getPlayers()) {
                casHome += home.getPlayerResult(p).getCasualties();
            }
        }
        if (away.getTeam() != null) {
            for (com.fumbbl.ffb.model.Player<?> p : away.getTeam().getPlayers()) {
                casAway += away.getPlayerResult(p).getCasualties();
            }
        }
        o.add("cas_home", casHome);
        o.add("cas_away", casAway);

        // Total SPP earned this game by each team
        int sppHome = 0, sppAway = 0;
        if (home.getTeam() != null) {
            for (com.fumbbl.ffb.model.Player<?> p : home.getTeam().getPlayers()) {
                sppHome += home.getPlayerResult(p).totalEarnedSpps();
            }
        }
        if (away.getTeam() != null) {
            for (com.fumbbl.ffb.model.Player<?> p : away.getTeam().getPlayers()) {
                sppAway += away.getPlayerResult(p).totalEarnedSpps();
            }
        }
        o.add("spp_home", sppHome);
        o.add("spp_away", sppAway);

        return o;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static FieldCoordinate lastCoord(PathProbabilityFinder.PathEntry entry) {
        if (entry == null || entry.path == null || entry.path.length == 0) return null;
        return entry.path[entry.path.length - 1];
    }

    private static JsonArray doubleArray(double[] arr) {
        JsonArray ja = new JsonArray();
        if (arr != null) {
            for (double v : arr) ja.add(v);
        }
        return ja;
    }

    /**
     * Serialize the most useful fields of each dialog type.
     * Unknown/unimplemented types get an empty object — the dialog_id field
     * in the parent record is sufficient for the feature extractor to handle them.
     */
    private static JsonObject serializeDialogParam(IDialogParameter dialog) {
        JsonObject p = new JsonObject();
        switch (dialog.getId()) {

            case BLOCK_ROLL: {
                DialogBlockRollParameter d = (DialogBlockRollParameter) dialog;
                p.add("num_dice",          d.getNrOfDice());
                p.add("choosing_team_id",  d.getChoosingTeamId() != null ? d.getChoosingTeamId() : "");
                int[] roll = d.getBlockRoll();
                if (roll != null) {
                    JsonArray dice = new JsonArray();
                    for (int v : roll) dice.add(v);
                    p.add("dice", dice);
                }
                break;
            }

            case BLOCK_ROLL_PARTIAL_RE_ROLL: {
                DialogBlockRollPartialReRollParameter d = (DialogBlockRollPartialReRollParameter) dialog;
                p.add("num_dice",         d.getNrOfDice());
                p.add("choosing_team_id", d.getChoosingTeamId() != null ? d.getChoosingTeamId() : "");
                int[] roll = d.getBlockRoll();
                if (roll != null) {
                    JsonArray dice = new JsonArray();
                    for (int v : roll) dice.add(v);
                    p.add("dice", dice);
                }
                break;
            }

            case BLOCK_ROLL_PROPERTIES: {
                DialogBlockRollPropertiesParameter d = (DialogBlockRollPropertiesParameter) dialog;
                p.add("num_dice",         d.getNrOfDice());
                p.add("choosing_team_id", d.getChoosingTeamId() != null ? d.getChoosingTeamId() : "");
                break;
            }

            case RE_ROLL: {
                DialogReRollParameter d = (DialogReRollParameter) dialog;
                if (d.getPlayerId() != null)        p.add("player_id",        d.getPlayerId());
                if (d.getReRolledAction() != null)  p.add("rerolled_action",  d.getReRolledAction().getName());
                p.add("min_roll",           d.getMinimumRoll());
                p.add("is_team_reroll",     d.isTeamReRollOption());
                p.add("is_pro_reroll",      d.isProReRollOption());
                p.add("is_fumble",          d.isFumble());
                if (d.getReRollSkill() != null) {
                    p.add("reroll_skill",   d.getReRollSkill().getName());
                }
                break;
            }

            case SKILL_USE: {
                DialogSkillUseParameter d = (DialogSkillUseParameter) dialog;
                if (d.getPlayerId() != null)  p.add("player_id",  d.getPlayerId());
                if (d.getSkill() != null)     p.add("skill",      d.getSkill().getName());
                p.add("min_roll", d.getMinimumRoll());
                break;
            }

            case USE_APOTHECARY: {
                DialogUseApothecaryParameter d = (DialogUseApothecaryParameter) dialog;
                if (d.getPlayerId() != null)         p.add("player_id",   d.getPlayerId());
                if (d.getPlayerState() != null)      p.add("injury_state", d.getPlayerState().getBase());
                if (d.getSeriousInjury() != null)    p.add("serious_injury", d.getSeriousInjury().getName());
                break;
            }

            default:
                // Player ID field is useful for many other dialog types (FOLLOWUP, PILING_ON, etc.)
                try {
                    java.lang.reflect.Method m = dialog.getClass().getMethod("getPlayerId");
                    String playerId = (String) m.invoke(dialog);
                    if (playerId != null) p.add("player_id", playerId);
                } catch (Exception ignored) {}
                break;
        }
        return p;
    }

}
