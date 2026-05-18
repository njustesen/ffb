package com.fumbbl.ffb.ai.parity;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.FieldCoordinateBounds;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.ai.simulation.CapturingClientCommunication;
import com.fumbbl.ffb.ai.simulation.HeadlessFantasyFootballServer;
import com.fumbbl.ffb.ai.simulation.HeadlessGameSetup;
import com.fumbbl.ffb.ai.simulation.MatchRunner;
import com.fumbbl.ffb.ai.strategy.RandomStrategy;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.dialog.DialogReceiveChoiceParameter;
import com.fumbbl.ffb.net.commands.ClientCommandActingPlayer;
import com.fumbbl.ffb.net.commands.ClientCommandCoinChoice;
import com.fumbbl.ffb.net.commands.ClientCommandEndTurn;
import com.fumbbl.ffb.net.commands.ClientCommandKickoff;
import com.fumbbl.ffb.net.commands.ClientCommandReceiveChoice;
import com.fumbbl.ffb.net.commands.ClientCommandStartGame;
import com.fumbbl.ffb.net.commands.ClientCommandTouchback;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.server.step.IStep;
import com.fumbbl.ffb.server.step.StepId;
import com.fumbbl.ffb.server.util.UtilServerSetup;
import com.fumbbl.ffb.util.UtilBox;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Runs a single seeded game and emits a full JSONL decision log.
 *
 * <p>Usage: ParityRunner [serverDir] homeTeamId awayTeamId seed [output.jsonl]
 *
 * <p>The output matches the format emitted by ffb-rust's parity_runner binary.
 * Log lines: game_start, one step per INIT_SELECTING phase-1 decision, game_end.
 *
 * <p>Uses {@link Xoshiro256StarStar} seeded from the given seed so dice rolls
 * are deterministic and can be compared against the Rust engine.
 */
public class ParityRunner {

    private static final int MAX_ITERATIONS = 100_000;

    private final PrintWriter out;
    private final CapturingClientCommunication comm = new CapturingClientCommunication();
    private final List<PendingStep> pending = new ArrayList<>();
    private int stepIndex = 1;

    private static final class PendingStep {
        int i;
        int turn;
        int half;
        String active;
        String stateHash;
        String chosen;
        String postHash = "";

        PendingStep(int i, int turn, int half, String active, String stateHash, String chosen) {
            this.i = i;
            this.turn = turn;
            this.half = half;
            this.active = active;
            this.stateHash = stateHash;
            this.chosen = chosen;
        }
    }

    public ParityRunner(PrintWriter out) {
        this.out = out;
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        if (args.length < 3) {
            System.err.println("Usage: ParityRunner [serverDir] homeTeamId awayTeamId seed [output.jsonl]");
            System.exit(1);
        }

        File serverDir;
        String homeTeamId, awayTeamId;
        long seed;
        String outputPath = null;

        File possibleDir = new File(args[0]);
        if (args.length >= 4 && possibleDir.isDirectory()) {
            serverDir = possibleDir;
            homeTeamId = resolveTeamId(args[1]);
            awayTeamId = resolveTeamId(args[2]);
            seed = Long.parseUnsignedLong(args[3]);
            if (args.length > 4) outputPath = args[4];
        } else {
            File cwd = new File(System.getProperty("user.dir"));
            File candidate = new File(cwd.getParentFile(), "ffb-server");
            serverDir = candidate.exists() ? candidate : new File(cwd, "ffb-server");
            homeTeamId = resolveTeamId(args[0]);
            awayTeamId = resolveTeamId(args[1]);
            seed = Long.parseUnsignedLong(args[2]);
            if (args.length > 3) outputPath = args[3];
        }

        PrintWriter out;
        if (outputPath != null) {
            out = new PrintWriter(new FileOutputStream(outputPath), true);
        } else {
            out = new PrintWriter(new java.io.BufferedWriter(
                new java.io.OutputStreamWriter(System.out, StandardCharsets.UTF_8)), true);
        }

        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();
        GameState gameState = HeadlessGameSetup.create(server, homeTeamId, awayTeamId, serverDir);

        Xoshiro256StarStar rng = new Xoshiro256StarStar(seed);
        server.getFortuna().setDelegate(rng);

        new ParityRunner(out).run(gameState, homeTeamId, awayTeamId, seed);

        out.flush();
        if (outputPath != null) out.close();
    }

    // ── Game loop ─────────────────────────────────────────────────────────────

    public void run(GameState gameState, String homeTeamId, String awayTeamId, long seed) {
        Game game = gameState.getGame();

        String initialHash = stateHash(game);
        out.println(String.format(
            "{\"i\":0,\"type\":\"game_start\",\"home\":\"%s\",\"away\":\"%s\",\"seed\":%d,\"state_hash\":\"%s\"}",
            escJson(homeTeamId), escJson(awayTeamId), seed, initialHash));

        MatchRunner.injectForTeam(gameState, new ClientCommandStartGame(), true);
        MatchRunner.injectForTeam(gameState, new ClientCommandStartGame(), false);

        int iter = 0;
        while (game.getFinished() == null && ++iter < MAX_ITERATIONS) {
            IStep step = gameState.getCurrentStep();
            if (step == null) break;

            IDialogParameter dialog = game.getDialogParameter();
            StepId stepId = step.getId();

            if (dialog != null && stepId != StepId.INIT_SELECTING) {
                handleDialog(dialog, game, gameState);
            } else {
                handleStep(stepId, game, gameState);
            }
        }

        // Finalize: fill post_hashes and flush all pending steps
        String endHash = stateHash(game);
        for (int i = 0; i < pending.size(); i++) {
            pending.get(i).postHash = (i + 1 < pending.size())
                ? pending.get(i + 1).stateHash
                : endHash;
        }
        for (PendingStep s : pending) {
            out.println(String.format(
                "{\"i\":%d,\"type\":\"step\",\"turn\":%d,\"half\":%d,\"active\":\"%s\","
                + "\"dialog\":\"None\",\"state_hash\":\"%s\","
                + "\"actions\":[\"EndTurn\"],\"chosen\":\"%s\","
                + "\"dice\":[],\"post_hash\":\"%s\"}",
                s.i, s.turn, s.half, s.active, s.stateHash, s.chosen, s.postHash));
        }

        int scoreHome = game.getGameResult().getScoreHome();
        int scoreAway = game.getGameResult().getScoreAway();
        out.println(String.format(
            "{\"i\":%d,\"type\":\"game_end\",\"home_score\":%d,\"away_score\":%d,\"state_hash\":\"%s\"}",
            stepIndex, scoreHome, scoreAway, endHash));
    }

    // ── Step handling ─────────────────────────────────────────────────────────

    private void handleStep(StepId stepId, Game game, GameState gameState) {
        switch (stepId) {

            case SETUP:
                resetCurrentTeam(game);
                placeReserves(game, gameState);
                MatchRunner.inject(gameState, new ClientCommandEndTurn(TurnMode.SETUP, null));
                break;

            case KICKOFF: {
                boolean home = game.isHomePlaying();
                FieldCoordinate kickCoord = home
                    ? new FieldCoordinate(13, 8)
                    : new FieldCoordinate(13, 8).transform();
                MatchRunner.inject(gameState, new ClientCommandKickoff(kickCoord));
                break;
            }

            case APPLY_KICKOFF_RESULT:
                MatchRunner.inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                break;

            case INIT_SELECTING: {
                ActingPlayer ap = game.getActingPlayer();
                if (ap == null || ap.getPlayerId() == null) {
                    // Phase 1: only log regular turns (turn >= 1); skip spurious turn=0
                    // that fires during the second-half kickoff setup before begin_turn().
                    boolean homePlaying = game.isHomePlaying();
                    int turn = homePlaying ? game.getTurnDataHome().getTurnNr()
                                          : game.getTurnDataAway().getTurnNr();
                    if (turn >= 1) {
                        recordStep(game);
                    }
                    MatchRunner.inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                } else {
                    // Phase 2: deselect
                    MatchRunner.inject(gameState, new ClientCommandActingPlayer(null, null, false));
                }
                break;
            }

            case KICKOFF_RETURN:
                MatchRunner.inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                break;

            default:
                MatchRunner.inject(gameState, new ClientCommandActingPlayer(null, null, false));
                break;
        }
    }

    private void handleDialog(IDialogParameter dialog, Game game, GameState gameState) {
        switch (dialog.getId()) {
            case KICKOFF_RETURN:
            case SETUP_ERROR:
            case SWARMING_ERROR:
            case INVALID_SOLID_DEFENCE:
                game.setDialogParameter(null);
                break;

            case COIN_CHOICE:
                // Always choose "heads" — deterministic (Xoshiro throws the actual coin)
                MatchRunner.inject(gameState, new ClientCommandCoinChoice(true));
                break;

            case RECEIVE_CHOICE: {
                // Always ensure away kicks (home receives): home winner → receive; away winner → kick.
                // Both branches produce isHomePlaying=false (away is kicking team).
                DialogReceiveChoiceParameter rc = (DialogReceiveChoiceParameter) dialog;
                boolean homeChosen = rc.getChoosingTeamId().equals(game.getTeamHome().getId());
                MatchRunner.inject(gameState, new ClientCommandReceiveChoice(homeChosen));
                break;
            }

            case TOUCHBACK: {
                // Canonical: give ball to receiving team's player nearest to kick-from (13,8).
                // Send from the receiving team's session so StepTouchback uses coord directly.
                boolean homeReceives = !game.isHomePlaying();
                Team recvTeam = homeReceives ? game.getTeamHome() : game.getTeamAway();
                FieldCoordinate kickFrom = new FieldCoordinate(13, 8);
                FieldCoordinate bestCoord = null;
                int bestDist = Integer.MAX_VALUE;
                for (Player<?> p : recvTeam.getPlayers()) {
                    PlayerState ps = game.getFieldModel().getPlayerState(p);
                    FieldCoordinate coord = game.getFieldModel().getPlayerCoordinate(p);
                    boolean onPitch = coord != null && coord.getX() >= 0 && coord.getX() <= 25
                                                     && coord.getY() >= 0 && coord.getY() <= 14;
                    if (ps != null && ps.isStanding() && onPitch) {
                        int dx = coord.getX() - kickFrom.getX();
                        int dy = coord.getY() - kickFrom.getY();
                        int dist = dx * dx + dy * dy;
                        if (dist < bestDist) { bestDist = dist; bestCoord = coord; }
                    }
                }
                if (bestCoord != null) {
                    // Home player: send actual coord (StepTouchback uses directly).
                    // Away player: send coord.transform() (StepTouchback transforms it back).
                    FieldCoordinate cmdCoord = homeReceives ? bestCoord : bestCoord.transform();
                    MatchRunner.injectForTeam(gameState, new ClientCommandTouchback(cmdCoord), homeReceives);
                } else {
                    game.setDialogParameter(null);
                }
                break;
            }

            default:
                comm.clearCaptured();
                RandomStrategy.respondToDialog(dialog, game, comm);
                com.fumbbl.ffb.net.commands.ClientCommand captured = comm.getCapturedCommand();
                if (captured != null) {
                    String teamId = getDialogTeamId(dialog);
                    try {
                        if (teamId != null) {
                            MatchRunner.injectForTeam(gameState, captured,
                                teamId.equals(game.getTeamHome().getId()));
                        } else {
                            MatchRunner.inject(gameState, captured);
                        }
                    } catch (RuntimeException e) {
                        game.setDialogParameter(null);
                    }
                } else {
                    game.setDialogParameter(null);
                }
                break;
        }
    }

    // ── Step recording ────────────────────────────────────────────────────────

    private void recordStep(Game game) {
        boolean homePlaying = game.isHomePlaying();
        int turn = homePlaying
            ? game.getTurnDataHome().getTurnNr()
            : game.getTurnDataAway().getTurnNr();
        int half = game.getHalf();
        String active = homePlaying ? "home" : "away";
        String hash = stateHash(game);
        pending.add(new PendingStep(stepIndex++, turn, half, active, hash, "EndTurn"));
    }

    // ── State hash (FNV-1a 64-bit — must match ffb-rust/crates/ffb-sim/src/parity_log.rs) ──

    public static String stateHash(Game game) {
        boolean homePlaying = game.isHomePlaying();
        // game.getHalf() returns 0 before the first half starts; normalize to 1 so the
        // initial hash matches Rust which represents the pre-kickoff state as Half::First.
        int half = Math.max(1, game.getHalf());
        int turnHome = game.getTurnDataHome().getTurnNr();
        int turnAway = game.getTurnDataAway().getTurnNr();
        String active = homePlaying ? "home" : "away";
        int scoreHome = game.getGameResult().getScoreHome();
        int scoreAway = game.getGameResult().getScoreAway();

        FieldModel fm = game.getFieldModel();
        FieldCoordinate ball = fm.getBallCoordinate();
        int bx = ball != null ? ball.getX() : -1;
        int by = ball != null ? ball.getY() : -1;
        int inPlay = fm.isBallInPlay() ? 1 : 0;

        List<String> playerParts = new ArrayList<>();
        addPlayersFromTeam(game.getTeamHome(), fm, playerParts, "h");
        addPlayersFromTeam(game.getTeamAway(), fm, playerParts, "a");
        playerParts.sort(null);

        StringBuilder sb = new StringBuilder();
        sb.append('h').append(half);
        sb.append('t').append(turnHome).append(turnAway);
        sb.append('a').append(active);
        sb.append('s').append(scoreHome).append(',').append(scoreAway);
        sb.append(" b").append(bx).append(',').append(by).append(',').append(inPlay == 1 ? "true" : "false");
        sb.append(" p");
        for (int i = 0; i < playerParts.size(); i++) {
            if (i > 0) sb.append('|');
            sb.append(playerParts.get(i));
        }

        String canonical = sb.toString();
        long hash = fnv1a64(canonical.getBytes(StandardCharsets.UTF_8));
        return String.format("%016x", hash);
    }

    private static void addPlayersFromTeam(Team team, FieldModel fm, List<String> out, String prefix) {
        if (team == null) return;
        // Sort by jersey number and cap at 11 to match Rust's jersey-sorted positional index.
        List<Player<?>> players = new ArrayList<>(java.util.Arrays.asList(team.getPlayers()));
        players.sort(java.util.Comparator.comparingInt(Player::getNr));
        if (players.size() > 11) players = players.subList(0, 11);
        for (int i = 0; i < players.size(); i++) {
            Player<?> p = players.get(i);
            PlayerState ps = fm.getPlayerState(p);
            FieldCoordinate coord = fm.getPlayerCoordinate(p);
            // Normalize off-pitch coordinates to (-1,-1) so the hash matches Rust
            // (Java uses box coordinates like (30,y) for reserves; Rust uses None → (-1,-1)).
            boolean onPitch = coord != null && coord.getX() >= 0 && coord.getX() <= 25
                                             && coord.getY() >= 0 && coord.getY() <= 14;
            int x = onPitch ? coord.getX() : -1;
            int y = onPitch ? coord.getY() : -1;
            String state = playerStateStr(ps);
            out.add(String.format("%s%02d:%d,%d,%s", prefix, i, x, y, state));
        }
    }

    private static String playerStateStr(PlayerState ps) {
        // Strings must match Rust's parity_log.rs player_state_str()
        if (ps == null) return "Reserve";
        switch (ps.getBase()) {
            case PlayerState.STANDING:       return "Standing";
            case PlayerState.MOVING:         return "Moving";
            case PlayerState.PRONE:          return "Prone";
            case PlayerState.STUNNED:        return "Stunned";
            case PlayerState.KNOCKED_OUT:    return "Ko";
            case PlayerState.BADLY_HURT:     return "Injured";
            case PlayerState.SERIOUS_INJURY: return "Injured";
            case PlayerState.RIP:            return "Injured";
            case PlayerState.RESERVE:        return "Reserve";
            default:                         return "Reserve";
        }
    }

    static long fnv1a64(byte[] data) {
        long hash = 0xcbf29ce484222325L;
        for (byte b : data) {
            hash ^= Byte.toUnsignedLong(b);
            hash *= 1099511628211L;
        }
        return hash;
    }

    // ── Setup helpers (mirrors MatchRunner) ───────────────────────────────────

    private static void resetCurrentTeam(Game game) {
        boolean homePlaying = game.isHomePlaying();
        Team team = homePlaying ? game.getTeamHome() : game.getTeamAway();
        FieldModel fm = game.getFieldModel();
        for (Player<?> p : team.getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            if (ps.canBeSetUpNextDrive()) {
                fm.setPlayerState(p, ps.changeBase(PlayerState.RESERVE));
                UtilBox.putPlayerIntoBox(game, p);
            }
        }
    }

    private static void placeReserves(Game game, GameState gameState) {
        boolean homePlaying = game.isHomePlaying();
        Team team = homePlaying ? game.getTeamHome() : game.getTeamAway();

        FieldModel fm = game.getFieldModel();

        // Sort by jersey number and take first 11 — matches Rust's jersey-sorted allocation.
        List<Player<?>> players = new ArrayList<>(java.util.Arrays.asList(team.getPlayers()));
        players.sort(java.util.Comparator.comparingInt(Player::getNr));
        if (players.size() > 11) players = players.subList(0, 11);

        // Canonical squares matching Rust's setup.rs place_team_for_kickoff().
        // Away coordinates are mirror: server applies .transform() → pass pre-transform.
        int[][] losSquares = {{12,7},{12,6},{12,8},{12,5},{12,9},{12,4},{12,10}};
        int[][] overflowSq = {{5,5},{5,7},{5,9},{6,6},{6,8},{4,6},{4,8},{3,6},{3,8},{2,5},{2,9},{1,7}};
        int li = 0, oi = 0;
        int placed = 0;
        int n = players.size();
        int losNeeded = n >= 3 ? 3 : n;

        for (Player<?> p : players) {
            if (placed >= n) break;

            if (losNeeded > 0) {
                while (li < losSquares.length) {
                    int ox = losSquares[li][0], oy = losSquares[li++][1];
                    FieldCoordinate gc = homePlaying
                        ? new FieldCoordinate(ox, oy)
                        : new FieldCoordinate(ox, oy).transform();
                    if (fm.getPlayer(gc) == null) {
                        UtilServerSetup.setupPlayer(gameState, p.getId(), new FieldCoordinate(ox, oy));
                        losNeeded--;
                        placed++;
                        break;
                    }
                }
            } else {
                while (oi < overflowSq.length) {
                    int ox = overflowSq[oi][0], oy = overflowSq[oi++][1];
                    FieldCoordinate gc = homePlaying
                        ? new FieldCoordinate(ox, oy)
                        : new FieldCoordinate(ox, oy).transform();
                    if (fm.getPlayer(gc) == null) {
                        UtilServerSetup.setupPlayer(gameState, p.getId(), new FieldCoordinate(ox, oy));
                        placed++;
                        break;
                    }
                }
            }
        }
    }

    // ── Dialog team resolution (mirrors MatchRunner) ──────────────────────────

    private static String getDialogTeamId(IDialogParameter dialog) {
        if (dialog instanceof com.fumbbl.ffb.dialog.DialogArgueTheCallParameter) {
            return ((com.fumbbl.ffb.dialog.DialogArgueTheCallParameter) dialog).getTeamId();
        }
        if (dialog instanceof com.fumbbl.ffb.dialog.DialogBriberyAndCorruptionParameter) {
            return ((com.fumbbl.ffb.dialog.DialogBriberyAndCorruptionParameter) dialog).getTeamId();
        }
        if (dialog instanceof com.fumbbl.ffb.dialog.DialogPlayerChoiceParameter) {
            return ((com.fumbbl.ffb.dialog.DialogPlayerChoiceParameter) dialog).getTeamId();
        }
        if (dialog instanceof com.fumbbl.ffb.dialog.DialogBribesParameter) {
            return ((com.fumbbl.ffb.dialog.DialogBribesParameter) dialog).getTeamId();
        }
        return null;
    }

    private static String resolveTeamId(String name) {
        switch (name.toLowerCase()) {
            case "human":    return "teamHumanKalimar";
            case "orc":      return "teamOrcBattleLore";
            case "darkelves":
            case "darkelf":
            case "dark_elf": return "teamDarkElfKalimar";
            default:         return name;
        }
    }

    private static String escJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
