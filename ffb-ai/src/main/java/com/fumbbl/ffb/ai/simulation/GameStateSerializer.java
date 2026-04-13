package com.fumbbl.ffb.ai.simulation;

import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.SeriousInjury;
import com.fumbbl.ffb.inducement.Inducement;
import com.fumbbl.ffb.inducement.InducementType;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.InducementSet;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.TurnData;
import com.fumbbl.ffb.model.skill.Skill;

/**
 * Serializes a {@link Game} snapshot to a compact {@link JsonObject} for BC training data.
 *
 * <p>Board is encoded as a 26×15 array (x-major) of player IDs (null = empty).
 * Player table maps player ID → stats/state/skills.
 * Ball state is one of {@code CARRIED}, {@code ON_GROUND}, {@code IN_AIR}, {@code BOUNCING}.
 */
public final class GameStateSerializer {

    private GameStateSerializer() {}

    // ── Ball state constants ──────────────────────────────────────────────────

    public static final String BALL_CARRIED   = "CARRIED";
    public static final String BALL_ON_GROUND = "ON_GROUND";
    public static final String BALL_IN_AIR    = "IN_AIR";
    public static final String BALL_BOUNCING  = "BOUNCING";

    // ── Entry point ───────────────────────────────────────────────────────────

    /**
     * Produce a compact JSON snapshot of the current game state.
     *
     * @param game the live game object
     * @return JSON object suitable for writing to a JSONL training record
     */
    public static JsonObject serialize(Game game) {
        JsonObject out = new JsonObject();
        out.add("half",       game.getHalf());
        out.add("turn_nr",    game.getTurnData() != null ? game.getTurnData().getTurnNr() : 0);
        out.add("turn_mode",  game.getTurnMode() == null ? "UNKNOWN" : game.getTurnMode().name());
        out.add("home_playing", game.isHomePlaying());

        FieldModel fm = game.getFieldModel();

        // ── Weather ──────────────────────────────────────────────────────────
        out.add("weather",
            fm.getWeather() != null ? fm.getWeather().name() : "NICE");

        // ── Ball ─────────────────────────────────────────────────────────────
        out.add("ball", serializeBall(game, fm));

        // ── Board: 26×15 array of player IDs ─────────────────────────────────
        out.add("board", serializeBoard(game, fm));

        // ── Players table ─────────────────────────────────────────────────────
        out.add("players", serializePlayers(game, fm));

        // ── Team data ─────────────────────────────────────────────────────────
        out.add("home", serializeTeamData(game, game.getTeamHome(), game.getTurnDataHome()));
        out.add("away", serializeTeamData(game, game.getTeamAway(), game.getTurnDataAway()));

        // ── Scores ────────────────────────────────────────────────────────────
        if (game.getGameResult() != null) {
            out.add("score_home", game.getGameResult().getScoreHome());
            out.add("score_away", game.getGameResult().getScoreAway());
        } else {
            out.add("score_home", 0);
            out.add("score_away", 0);
        }

        // ── Turn-scoped action flags (current team) ───────────────────────────
        TurnData td = game.getTurnData();
        if (td != null) {
            out.add("blitz_used",   td.isBlitzUsed());
            out.add("foul_used",    td.isFoulUsed());
            out.add("pass_used",    td.isPassUsed());
            out.add("handoff_used", td.isHandOverUsed());
            out.add("ttm_used",     td.isTtmUsed());
            out.add("reroll_used",  td.isReRollUsed());
        } else {
            out.add("blitz_used",   false);
            out.add("foul_used",    false);
            out.add("pass_used",    false);
            out.add("handoff_used", false);
            out.add("ttm_used",     false);
            out.add("reroll_used",  false);
        }

        // ── Defender / thrower context ────────────────────────────────────────
        String defenderId = game.getDefenderId();
        out.add("defender_id", defenderId != null ? JsonValue.valueOf(defenderId) : JsonValue.NULL);
        String throwerId = game.getThrowerId();
        out.add("thrower_id", throwerId != null ? JsonValue.valueOf(throwerId) : JsonValue.NULL);

        // ── First-turn-after-kickoff flags ────────────────────────────────────
        TurnData tdHome = game.getTurnDataHome();
        TurnData tdAway = game.getTurnDataAway();
        out.add("first_turn_home", tdHome != null && tdHome.isFirstTurnAfterKickoff());
        out.add("first_turn_away", tdAway != null && tdAway.isFirstTurnAfterKickoff());

        // ── Acting player ─────────────────────────────────────────────────────
        out.add("acting_player", serializeActingPlayer(game));

        return out;
    }

    // ── Ball ──────────────────────────────────────────────────────────────────

    private static JsonObject serializeBall(Game game, FieldModel fm) {
        JsonObject ball = new JsonObject();
        FieldCoordinate ballCoord = fm.getBallCoordinate();

        // Ball state detection (order matters):
        // 1. IN_AIR  — pass is in flight
        // 2. BOUNCING — ball is moving (scatter, kickoff)
        // 3. CARRIED  — player at ball square
        // 4. ON_GROUND — otherwise
        String state;
        String carrierId = null;

        if (game.getPassCoordinate() != null) {
            state = BALL_IN_AIR;
            // Landing target coordinate
            FieldCoordinate landing = game.getPassCoordinate();
            ball.add("target_x", landing.getX());
            ball.add("target_y", landing.getY());
        } else if (fm.isBallMoving()) {
            state = BALL_BOUNCING;
        } else if (ballCoord != null) {
            Player<?> carrierCandidate = fm.getPlayer(ballCoord);
            if (carrierCandidate != null) {
                state = BALL_CARRIED;
                carrierId = carrierCandidate.getId();
            } else {
                state = BALL_ON_GROUND;
            }
        } else {
            // Ball not placed yet (game start / kickoff phase)
            state = BALL_ON_GROUND;
        }

        ball.add("state", state);
        if (ballCoord != null) {
            ball.add("x", ballCoord.getX());
            ball.add("y", ballCoord.getY());
        } else {
            ball.add("x", JsonValue.NULL);
            ball.add("y", JsonValue.NULL);
        }
        if (carrierId != null) {
            ball.add("carrier_id", carrierId);
        } else {
            ball.add("carrier_id", JsonValue.NULL);
        }

        return ball;
    }

    // ── Board ─────────────────────────────────────────────────────────────────

    /**
     * Encodes the board as a flattened array of length 26×15=390.
     * Index = x * 15 + y.  Each entry is the player ID string, or null.
     */
    private static JsonArray serializeBoard(Game game, FieldModel fm) {
        JsonArray board = new JsonArray();
        for (int x = 0; x < 26; x++) {
            for (int y = 0; y < 15; y++) {
                Player<?> p = fm.getPlayer(new FieldCoordinate(x, y));
                if (p != null) {
                    board.add(p.getId());
                } else {
                    board.add(JsonValue.NULL);
                }
            }
        }
        return board;
    }

    // ── Players ───────────────────────────────────────────────────────────────

    private static JsonObject serializePlayers(Game game, FieldModel fm) {
        JsonObject players = new JsonObject();
        serializeTeamPlayers(players, game.getTeamHome(), fm, "home");
        serializeTeamPlayers(players, game.getTeamAway(), fm, "away");
        return players;
    }

    private static void serializeTeamPlayers(JsonObject players, Team team, FieldModel fm, String teamKey) {
        if (team == null) return;
        for (Player<?> p : team.getPlayers()) {
            players.add(p.getId(), serializePlayer(p, fm, teamKey));
        }
    }

    private static JsonObject serializePlayer(Player<?> p, FieldModel fm, String teamKey) {
        JsonObject obj = new JsonObject();
        obj.add("team", teamKey);
        obj.add("nr",   p.getNr());

        // Stats
        obj.add("ma", p.getMovement());
        obj.add("st", p.getStrength());
        obj.add("ag", p.getAgility());
        obj.add("av", p.getArmour());
        obj.add("pa", p.getPassing());

        // Skills
        Skill[] skills = p.getSkills();
        JsonArray skillArray = new JsonArray();
        if (skills != null) {
            for (Skill s : skills) {
                skillArray.add(s.getName());
            }
        }
        obj.add("skills", skillArray);

        // Position (null if off-field)
        FieldCoordinate coord = fm.getPlayerCoordinate(p);
        if (coord != null && coord.getX() >= 0 && coord.getX() < 26
                && coord.getY() >= 0 && coord.getY() < 15) {
            obj.add("x", coord.getX());
            obj.add("y", coord.getY());
        } else {
            obj.add("x", JsonValue.NULL);
            obj.add("y", JsonValue.NULL);
        }

        // State
        PlayerState ps = fm.getPlayerState(p);
        obj.add("state", ps != null ? playerStateString(ps.getBase()) : "UNKNOWN");

        // Active bit
        obj.add("active", ps != null && ps.isActive());

        // Lasting injuries
        SeriousInjury[] lasting = p.getLastingInjuries();
        JsonArray liArr = new JsonArray();
        if (lasting != null) {
            for (SeriousInjury li : lasting) {
                liArr.add(li.getName());
            }
        }
        obj.add("lasting_injuries", liArr);

        return obj;
    }

    /** Map the PlayerState base constant to a short string label. */
    private static String playerStateString(int base) {
        switch (base) {
            case PlayerState.STANDING:        return "STANDING";
            case PlayerState.MOVING:          return "MOVING";
            case PlayerState.PRONE:           return "PRONE";
            case PlayerState.STUNNED:         return "STUNNED";
            case PlayerState.KNOCKED_OUT:     return "KO";
            case PlayerState.BADLY_HURT:      return "BH";
            case PlayerState.SERIOUS_INJURY:  return "SI";
            case PlayerState.RIP:             return "RIP";
            case PlayerState.RESERVE:         return "RESERVE";
            case PlayerState.MISSING:         return "MISSING";
            case PlayerState.FALLING:         return "FALLING";
            case PlayerState.BLOCKED:         return "BLOCKED";
            case PlayerState.BANNED:          return "BANNED";
            case PlayerState.EXHAUSTED:       return "EXHAUSTED";
            default:                          return "OTHER";
        }
    }

    // ── Team data ─────────────────────────────────────────────────────────────

    private static JsonObject serializeTeamData(Game game, Team team, TurnData td) {
        JsonObject obj = new JsonObject();
        if (team == null) {
            obj.add("team_id", JsonValue.NULL);
            obj.add("rerolls", 0);
            obj.add("turn", 0);
            obj.add("apothecaries", 0);
            obj.add("bribes", 0);
            return obj;
        }
        obj.add("team_id",        team.getId());
        obj.add("race",           team.getRace() != null ? team.getRace() : "");
        obj.add("fan_factor",     team.getFanFactor());
        obj.add("dedicated_fans", team.getDedicatedFans());

        if (td != null) {
            // Rerolls remaining this game
            obj.add("rerolls",     td.getReRolls());
            obj.add("turn",        td.getTurnNr());
            obj.add("apothecaries", td.getApothecaries() + td.getWanderingApothecaries());

            // Bribes: find via inducement set
            obj.add("bribes", countInducementUsesLeft(td.getInducementSet(), "bribes"));
        } else {
            obj.add("rerolls",     team.getReRolls());
            obj.add("turn",        0);
            obj.add("apothecaries", team.getApothecaries());
            obj.add("bribes",      0);
        }

        // SPP earned so far this game (summed from live PlayerResult records)
        int sppSoFar = 0;
        if (game.getGameResult() != null) {
            boolean isHome = team == game.getTeamHome();
            com.fumbbl.ffb.model.TeamResult tr = isHome
                ? game.getGameResult().getTeamResultHome()
                : game.getGameResult().getTeamResultAway();
            if (tr != null) {
                for (com.fumbbl.ffb.model.Player<?> p : team.getPlayers()) {
                    sppSoFar += tr.getPlayerResult(p).totalEarnedSpps();
                }
            }
        }
        obj.add("spp_earned", sppSoFar);

        return obj;
    }

    /** Count remaining uses of an inducement by its type name (e.g. "bribes"). */
    private static int countInducementUsesLeft(InducementSet set, String typeName) {
        if (set == null) return 0;
        for (InducementType type : set.getInducementTypes()) {
            if (typeName.equals(type.getName())) {
                Inducement ind = set.get(type);
                return ind != null ? ind.getUsesLeft() : 0;
            }
        }
        return 0;
    }

    // ── Acting player ─────────────────────────────────────────────────────────

    private static JsonObject serializeActingPlayer(Game game) {
        ActingPlayer ap = game.getActingPlayer();
        JsonObject obj = new JsonObject();
        if (ap == null || ap.getPlayerId() == null) {
            obj.add("player_id",    JsonValue.NULL);
            obj.add("action",       JsonValue.NULL);
            obj.add("current_move", 0);
            obj.add("has_moved",    false);
            obj.add("has_blocked",  false);
            obj.add("has_fouled",   false);
            obj.add("has_passed",   false);
            return obj;
        }
        obj.add("player_id",    ap.getPlayerId());
        if (ap.getPlayerAction() != null) {
            obj.add("action", ap.getPlayerAction().name());
        } else {
            obj.add("action", JsonValue.NULL);
        }
        obj.add("current_move", ap.getCurrentMove());
        obj.add("has_moved",    ap.hasMoved());
        obj.add("has_blocked",  ap.hasBlocked());
        obj.add("has_fouled",   ap.hasFouled());
        obj.add("has_passed",   ap.hasPassed());
        return obj;
    }
}
