package com.fumbbl.ffb.ai.simulation;

import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;
import com.fumbbl.ffb.Direction;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.SeriousInjury;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.json.UtilJson;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.TurnData;
import com.fumbbl.ffb.model.skill.Skill;
import com.fumbbl.ffb.net.NetCommandFactory;
import com.fumbbl.ffb.net.commands.ServerCommandGameState;
import com.fumbbl.ffb.net.commands.ServerCommandModelSync;
import com.fumbbl.ffb.report.IReport;
import com.fumbbl.ffb.report.ReportApothecaryChoice;
import com.fumbbl.ffb.report.ReportBlockChoice;
import com.fumbbl.ffb.report.ReportBribesRoll;
import com.fumbbl.ffb.report.ReportChainsawRoll;
import com.fumbbl.ffb.report.ReportCoinThrow;
import com.fumbbl.ffb.report.ReportFoul;
import com.fumbbl.ffb.report.ReportHandOver;
import com.fumbbl.ffb.report.ReportId;
import com.fumbbl.ffb.report.ReportInterceptionRoll;
import com.fumbbl.ffb.report.ReportKickoffScatter;
import com.fumbbl.ffb.report.ReportPassDeviate;
import com.fumbbl.ffb.report.ReportPilingOn;
import com.fumbbl.ffb.report.ReportPlayerAction;
import com.fumbbl.ffb.report.ReportPushback;
import com.fumbbl.ffb.report.ReportReceiveChoice;
import com.fumbbl.ffb.report.ReportReRoll;
import com.fumbbl.ffb.report.ReportSkillRoll;
import com.fumbbl.ffb.report.ReportPassBlock;
import com.fumbbl.ffb.report.ReportSkillUse;
import com.fumbbl.ffb.report.ReportWizardUse;
import com.fumbbl.ffb.report.mixed.ReportSelectGazeTarget;
import com.fumbbl.ffb.TurnMode;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Parses FUMBBL replay files ({@code .ffbr}) and extracts human decision records
 * as JSONL in the same format as {@link JsonlTrainingDataCollector}.
 *
 * <h3>Record types produced</h3>
 * <ul>
 *   <li>{@code player_select}      — from PLAYER_ACTION + MDE (with fallback for non-MDE actions)</li>
 *   <li>{@code move_target}        — tracked across model syncs (all *_MOVE actions)</li>
 *   <li>{@code foul_target}        — from FOUL report (target player)</li>
 *   <li>{@code handoff_target}     — from HAND_OVER report (receiver player)</li>
 *   <li>{@code gaze_target}        — from SELECT_GAZE_TARGET report</li>
 *   <li>{@code pass_target}        — from pass coordinate or scatter-inverse</li>
 *   <li>{@code kickoff_target}     — from kickoff scatter inverse</li>
 *   <li>{@code dialog/BLOCK_ROLL_PROPERTIES} — from BLOCK_CHOICE (includes target_id)</li>
 *   <li>{@code dialog/SKILL_USE}             — from SKILL_USE report</li>
 *   <li>{@code dialog/RECEIVE_CHOICE}        — from RECEIVE_CHOICE report</li>
 *   <li>{@code dialog/COIN_CHOICE}           — from COIN_THROW report</li>
 *   <li>{@code dialog/RE_ROLL}               — from failed skill rolls + ReportReRoll</li>
 *   <li>{@code dialog/FOLLOWUP_CHOICE}       — from position changes after PUSHBACK</li>
 *   <li>{@code dialog/USE_APOTHECARY}        — from APOTHECARY_ROLL presence/absence</li>
 *   <li>{@code dialog/APOTHECARY_CHOICE}     — from APOTHECARY_CHOICE report</li>
 *   <li>{@code dialog/PILING_ON}             — from PILING_ON report</li>
 *   <li>{@code dialog/BLOODLUST_ACTION}      — from BLOOD_LUST_ROLL report (action=1: chose to roll)</li>
 *   <li>{@code dialog/USE_CHAINSAW}          — from CHAINSAW_ROLL + pending tracking</li>
 *   <li>{@code dialog/BRIBES}               — from BRIBES_ROLL report (action=0: used)</li>
 *   <li>{@code dialog/ARGUE_THE_CALL}       — from ARGUE_THE_CALL report (action=0: argued)</li>
 *   <li>{@code dialog/BRIBERY_AND_CORRUPTION_RE_ROLL} — from B&amp;C reroll report</li>
 *   <li>{@code dialog/INTERCEPTION}         — from INTERCEPTION_ROLL (action=0: attempted)</li>
 * </ul>
 */
public class FumbblReplayParser {

    private static final String AGENT_MODE = "fumbbl_replay";
    private static final Random RNG = new Random(0);

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        File inputPath = null;
        File outputDir = null;
        int  shardNum  = -1;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--shard": shardNum = Integer.parseInt(args[++i]); break;
                default:
                    if (!args[i].startsWith("-")) {
                        if (inputPath == null)      inputPath = new File(args[i]);
                        else if (outputDir == null) outputDir = new File(args[i]);
                    }
            }
        }

        if (inputPath == null || outputDir == null) {
            System.err.println("Usage: FumbblReplayParser <inputDir|file.ffbr> <outputDir> [--shard N]");
            System.exit(1);
        }

        List<File> files = new ArrayList<>();
        if (inputPath.isDirectory()) {
            File[] found = inputPath.listFiles(f -> f.getName().endsWith(".ffbr"));
            if (found != null) Collections.addAll(files, found);
            files.sort(java.util.Comparator.comparing(File::getName));
        } else {
            files.add(inputPath);
        }

        if (files.isEmpty()) {
            System.err.println("No .ffbr files found in: " + inputPath);
            System.exit(1);
        }

        outputDir.mkdirs();

        if (shardNum < 0) {
            shardNum = 0;
            while (new File(outputDir, "shard_fumbbl_" + shardNum + ".jsonl").exists()) shardNum++;
        }

        File outFile = new File(outputDir, "shard_fumbbl_" + shardNum + ".jsonl");
        System.out.println("=== FumbblReplayParser ===");
        System.out.printf("  Input  : %s (%d file%s)%n", inputPath, files.size(), files.size() == 1 ? "" : "s");
        System.out.printf("  Output : %s%n", outFile.getAbsolutePath());

        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();
        int totalRecords = 0, parsedFiles = 0, failedFiles = 0;

        try (BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(Files.newOutputStream(outFile.toPath()), StandardCharsets.UTF_8))) {
            for (File f : files) {
                try {
                    int count = parseFile(f, server, writer);
                    totalRecords += count;
                    parsedFiles++;
                    if (parsedFiles % 10 == 0 || parsedFiles == files.size()) {
                        System.out.printf("  [%d/%d] records so far: %d%n", parsedFiles, files.size(), totalRecords);
                    }
                } catch (Exception e) {
                    failedFiles++;
                    System.err.printf("  WARN: failed to parse %s: %s%n", f.getName(), e.getMessage());
                }
            }
        }

        System.out.printf("Done. Parsed %d/%d files, %d decision records written to %s%n",
            parsedFiles, files.size(), totalRecords, outFile.getName());
        if (failedFiles > 0) System.out.printf("  (%d files failed)%n", failedFiles);
    }

    // ── File parsing ──────────────────────────────────────────────────────────

    static int parseFile(File file, HeadlessFantasyFootballServer server, BufferedWriter writer)
            throws IOException {

        byte[] raw = Files.readAllBytes(file.toPath());
        JsonObject root = UtilJson.gunzip(raw).asObject();

        JsonArray commandArray = root.get("commandArray").asArray();
        if (commandArray == null || commandArray.size() == 0) return 0;

        NetCommandFactory factory = new NetCommandFactory(server);

        Game game = null;
        int startIdx = 0;
        for (int i = 0; i < commandArray.size(); i++) {
            JsonObject cmdObj = commandArray.get(i).asObject();
            if ("serverGameState".equals(cmdObj.getString("netCommandId", ""))) {
                ServerCommandGameState gsCmd =
                    (ServerCommandGameState) factory.forJsonValue(server, commandArray.get(i));
                if (gsCmd != null) {
                    game = gsCmd.getGame();
                    startIdx = i + 1;
                }
                break;
            }
        }
        if (game == null) return 0;

        int recordCount = 0;

        // ── move_target tracking ──────────────────────────────────────────────
        String     pendingMovePlayerId      = null;
        String     pendingMoveActionType    = null;
        JsonObject pendingMoveState         = null;
        List<FieldCoordinate> pendingMoveCands = null;
        double[]   pendingMoveScores        = null;
        boolean    pendingMoveHasEnd        = false;
        boolean    pendingMoveBallCarrier   = false;
        boolean    pendingMoveBallRetriever = false;
        boolean    pendingMoveReceiver      = false;
        String     nextMovePlayerId         = null;
        PlayerAction nextMoveAction         = null;

        // ── RE_ROLL pending ───────────────────────────────────────────────────
        // Set when a skill roll fails (not rerolled). Cleared when RE_ROLL fires
        // (action=0) or when the next sync has no RE_ROLL (action=1).
        String     rrPlayerId   = null;
        String     rrActionName = null;
        int        rrMinRoll    = 0;
        boolean    rrIsFumble   = false;
        JsonObject rrState      = null;

        // ── FOLLOWUP_CHOICE pending ───────────────────────────────────────────
        String     fuAttackerId = null;
        int        fuAttackerX  = -1, fuAttackerY = -1;
        int        fuTargetX    = -1, fuTargetY   = -1;
        JsonObject fuState      = null;
        int        fuCheckSyncs = 0;

        // ── USE_APOTHECARY + APOTHECARY_CHOICE pending ────────────────────────
        String     apoPlayerId       = null;
        int        apoOldStateBase   = -1;
        String     apoOldSI          = null;
        JsonObject apoDeclineState   = null;
        int        apoCountdown      = 0;
        // For APOTHECARY_CHOICE cross-comparison:
        int        apoNewStateBase   = -1;
        String     apoNewSI          = null;
        String     apoChoicePlayerId = null;

        // ── Pass target pending ───────────────────────────────────────────────
        FieldCoordinate pendingPassCoord  = null;
        JsonObject      pendingPassState  = null;
        boolean         pendingPassDeviated = false;

        // ── USE_CHAINSAW pending ──────────────────────────────────────────────
        boolean    pendingChainsaw      = false;
        JsonObject pendingChainsawState = null;

        // ── PASS_BLOCK pending ────────────────────────────────────────────────
        // When ReportPassBlock fires with isPassBlockAvailable=true, track eligible players.
        // Post-change: if a PASS_BLOCK player moved → action=0; else → action=1.
        String     pbPlayerId    = null;
        JsonObject pbState       = null;
        int        pbOldX        = -1, pbOldY = -1;

        // ── TOUCHBACK pending ─────────────────────────────────────────────────
        // When TurnMode becomes TOUCHBACK pre-change, capture state.
        // Post-change: whoever now has the ball is the touchback receiver.
        JsonObject touchbackState = null;
        boolean    touchbackPending = false;

        // ─────────────────────────────────────────────────────────────────────

        for (int i = startIdx; i < commandArray.size(); i++) {
            JsonObject cmdObj = commandArray.get(i).asObject();
            if (!"serverModelSync".equals(cmdObj.getString("netCommandId", ""))) continue;

            ServerCommandModelSync syncCmd;
            try {
                syncCmd = (ServerCommandModelSync) factory.forJsonValue(game.getRules(), commandArray.get(i));
            } catch (Exception e) { continue; }
            if (syncCmd == null) continue;

            IReport[] reports = syncCmd.getReportList().getReports();

            // ── PRE-SYNC CHECKS ────────────────────────────────────────────────
            // RE_ROLL decline: if pending from last sync and no RE_ROLL in this sync
            if (rrState != null) {
                boolean hasReRoll = false;
                for (IReport r : reports) {
                    if (r.getId() == ReportId.RE_ROLL) { hasReRoll = true; break; }
                }
                if (!hasReRoll) {
                    try {
                        JsonObject rec = buildReRollRecord(rrPlayerId, rrActionName, rrMinRoll, rrIsFumble, rrState, 1);
                        if (rec != null) { writer.write(rec.toString()); writer.newLine(); recordCount++; }
                    } catch (Exception ignored) {}
                    rrState = null; rrPlayerId = null;
                }
            }

            // USE_APOTHECARY decline: countdown
            if (apoPlayerId != null && apoCountdown > 0) {
                boolean hasApoRoll = false;
                for (IReport r : reports) {
                    if (r.getId() == ReportId.APOTHECARY_ROLL) { hasApoRoll = true; break; }
                }
                if (!hasApoRoll) {
                    apoCountdown--;
                    if (apoCountdown <= 0) {
                        try {
                            JsonObject rec = buildUseApothecaryRecord(apoPlayerId, apoOldStateBase, apoOldSI, apoDeclineState, 1);
                            if (rec != null) { writer.write(rec.toString()); writer.newLine(); recordCount++; }
                        } catch (Exception ignored) {}
                        apoPlayerId = null; apoDeclineState = null;
                    }
                }
            }

            // ── PRE-CHANGE PASS ────────────────────────────────────────────────

            for (IReport report : reports) {
                JsonObject rec = null;
                try {
                    switch (report.getId()) {

                        // Block dice — includes target_id
                        case BLOCK_CHOICE:
                            rec = buildBlockRollPropertiesRecord(game, (ReportBlockChoice) report);
                            break;

                        // Skill use
                        case SKILL_USE:
                            rec = buildSkillUseRecord(game, (ReportSkillUse) report);
                            break;

                        // Receive/kick
                        case RECEIVE_CHOICE:
                            rec = buildReceiveChoiceRecord(game, (ReportReceiveChoice) report);
                            break;

                        // Coin toss
                        case COIN_THROW:
                            rec = buildCoinChoiceRecord(game, (ReportCoinThrow) report);
                            break;

                        // Foul target
                        case FOUL:
                            rec = buildFoulTargetRecord(game, (ReportFoul) report);
                            break;

                        // Handoff target
                        case HAND_OVER:
                            rec = buildHandoffTargetRecord(game, (ReportHandOver) report);
                            break;

                        // Gaze target
                        case SELECT_GAZE_TARGET:
                            rec = buildGazeTargetRecord(game, (ReportSelectGazeTarget) report);
                            break;

                        // Re-roll used (action=0)
                        case RE_ROLL: {
                            if (rrState != null) {
                                rec = buildReRollRecord(rrPlayerId, rrActionName, rrMinRoll, rrIsFumble, rrState, 0);
                                rrState = null; rrPlayerId = null;
                            }
                            break;
                        }

                        // Piling On
                        case PILING_ON:
                            rec = buildPilingOnRecord(game, (ReportPilingOn) report);
                            break;

                        // Injury — set up apothecary tracking
                        case INJURY: {
                            String defenderId = getInjuryDefenderId(report);
                            PlayerState injState = getInjuryPlayerState(report);
                            SeriousInjury injSI = getInjurySeriousInjury(report);
                            if (defenderId != null && injState != null) {
                                int base = injState.getBase();
                                if (base == PlayerState.BADLY_HURT
                                        || base == PlayerState.SERIOUS_INJURY
                                        || base == PlayerState.RIP) {
                                    Player<?> injPlayer = game.getPlayerById(defenderId);
                                    if (injPlayer != null) {
                                        boolean injIsHome = game.getTeamHome().hasPlayer(injPlayer);
                                        TurnData injTD = injIsHome
                                            ? game.getTurnDataHome() : game.getTurnDataAway();
                                        int apos = injTD != null
                                            ? (injTD.getApothecaries() + injTD.getWanderingApothecaries()) : 0;
                                        if (apos > 0) {
                                            apoPlayerId     = defenderId;
                                            apoOldStateBase = base;
                                            apoOldSI        = injSI != null ? injSI.getName() : null;
                                            apoDeclineState = GameStateSerializer.serialize(game);
                                            apoCountdown    = 3;
                                            apoNewStateBase = -1; apoNewSI = null;
                                            apoChoicePlayerId = defenderId;
                                        }
                                    }
                                }
                            }
                            break;
                        }

                        // Apothecary rolled (used → emit USE_APOTHECARY action=0)
                        case APOTHECARY_ROLL: {
                            String pid = getApothecaryRollPlayerId(report);
                            PlayerState newState = getApothecaryRollPlayerState(report);
                            SeriousInjury newSI = getApothecaryRollSeriousInjury(report);
                            if (pid != null && apoPlayerId != null && pid.equals(apoPlayerId)) {
                                rec = buildUseApothecaryRecord(apoPlayerId, apoOldStateBase, apoOldSI, apoDeclineState, 0);
                                apoCountdown = 0;
                                apoNewStateBase = newState != null ? newState.getBase() : -1;
                                apoNewSI = newSI != null ? newSI.getName() : null;
                                apoPlayerId = null; apoDeclineState = null;
                            }
                            break;
                        }

                        // Apothecary choice (keep old vs new)
                        case APOTHECARY_CHOICE: {
                            ReportApothecaryChoice acr = (ReportApothecaryChoice) report;
                            if (acr.getPlayerId() != null && apoOldStateBase != -1 && apoNewStateBase != -1) {
                                rec = buildApothecaryChoiceRecord(game, acr, apoOldStateBase, apoOldSI,
                                    apoNewStateBase, apoNewSI);
                            }
                            apoOldStateBase = -1; apoOldSI = null;
                            apoNewStateBase = -1; apoNewSI = null;
                            apoChoicePlayerId = null;
                            break;
                        }

                        // Pushback — set up followup choice tracking
                        case PUSHBACK: {
                            ReportPushback pr = (ReportPushback) report;
                            ActingPlayer ap = game.getActingPlayer();
                            String attackerId = ap != null ? ap.getPlayerId() : null;
                            if (attackerId != null && pr.getDefenderId() != null) {
                                Player<?> att = game.getPlayerById(attackerId);
                                Player<?> def = game.getPlayerById(pr.getDefenderId());
                                if (att != null && def != null) {
                                    FieldCoordinate ac = game.getFieldModel().getPlayerCoordinate(att);
                                    FieldCoordinate dc = game.getFieldModel().getPlayerCoordinate(def);
                                    if (ac != null && dc != null) {
                                        fuAttackerId = attackerId;
                                        fuAttackerX  = ac.getX(); fuAttackerY = ac.getY();
                                        fuTargetX    = dc.getX(); fuTargetY   = dc.getY();
                                        fuState      = GameStateSerializer.serialize(game);
                                        fuCheckSyncs = 3;
                                    }
                                }
                            }
                            break;
                        }

                        // Pass roll — capture intended target from game.getPassCoordinate()
                        case PASS_ROLL: {
                            pendingPassCoord   = game.getPassCoordinate();
                            pendingPassState   = GameStateSerializer.serialize(game);
                            pendingPassDeviated = false;
                            break;
                        }

                        // Pass deviate — compute target via scatter inverse
                        case PASS_DEVIATE: {
                            ReportPassDeviate pd = (ReportPassDeviate) report;
                            JsonObject stateForPass = pendingPassState != null
                                ? pendingPassState : GameStateSerializer.serialize(game);
                            if (!pd.isTtm()) {
                                rec = buildPassTargetRecord(pd, pendingPassCoord, stateForPass);
                            } else {
                                rec = buildTtmTargetRecord(pd, stateForPass);
                            }
                            pendingPassDeviated = true;
                            pendingPassCoord = null; pendingPassState = null;
                            break;
                        }

                        // Kickoff scatter — compute kickoff target via scatter inverse
                        case KICKOFF_SCATTER:
                            rec = buildKickoffTargetRecord(game, (ReportKickoffScatter) report);
                            break;

                        // Bloodlust roll fired → vampire CHOSE to roll (action=1)
                        case BLOOD_LUST_ROLL: {
                            // action=1: chose to roll bloodlust (not feed)
                            String pid = ((ReportSkillRoll) report).getPlayerId();
                            rec = buildSimpleDialogRecord(game, "BLOODLUST_ACTION", 1,
                                pidParam(pid));
                            break;
                        }

                        // Chainsaw roll fired → chainsaw was used (action=0)
                        case CHAINSAW_ROLL: {
                            ReportChainsawRoll cr = (ReportChainsawRoll) report;
                            rec = buildSimpleDialogRecord(game,
                                pendingChainsawState != null ? null : "USE_CHAINSAW", 0,
                                pidParam(cr.getPlayerId()));
                            if (pendingChainsawState != null) {
                                // Use the state captured at activation
                                rec = buildDialogRecordWithState("USE_CHAINSAW", 0,
                                    pidParam(cr.getPlayerId()), pendingChainsawState);
                                // Also emit chainsaw_target
                                JsonObject targetRec = buildChainsawTargetRecord(game, cr);
                                if (targetRec != null) {
                                    writer.write(targetRec.toString()); writer.newLine(); recordCount++;
                                }
                            }
                            pendingChainsaw = false; pendingChainsawState = null;
                            break;
                        }

                        // Bribe used (action=0)
                        case BRIBES_ROLL: {
                            ReportBribesRoll br = (ReportBribesRoll) report;
                            rec = buildSimpleDialogRecord(game, "BRIBES", 0, pidParam(br.getPlayerId()));
                            break;
                        }

                        // Argue the call (action=0)
                        case ARGUE_THE_CALL: {
                            String pid = getArgueTheCallPlayerId(report);
                            rec = buildSimpleDialogRecord(game, "ARGUE_THE_CALL", 0, pidParam(pid));
                            break;
                        }

                        // Bribery & corruption reroll used (action=0)
                        case BRIBERY_AND_CORRUPTION_RE_ROLL: {
                            rec = buildBriberyCorruptionRecord(game, report);
                            break;
                        }

                        // Interception attempted (action=0)
                        case INTERCEPTION_ROLL: {
                            ReportInterceptionRoll ir = (ReportInterceptionRoll) report;
                            rec = buildSimpleDialogRecord(game, "INTERCEPTION", 0, pidParam(ir.getPlayerId()));
                            break;
                        }

                        // Wizard spell cast (action=0, partial — only the used case is detectable)
                        case WIZARD_USE: {
                            ReportWizardUse wu = (ReportWizardUse) report;
                            JsonObject wp = new JsonObject();
                            wp.add("team_id", wu.getTeamId() != null ? wu.getTeamId() : "");
                            if (wu.getWizardSpell() != null) {
                                wp.add("spell_type", wu.getWizardSpell().getName());
                            }
                            rec = buildDialogRecordWithState("WIZARD_SPELL", 0, wp, GameStateSerializer.serialize(game));
                            break;
                        }

                        // Pass Block available — track eligible player for post-change detection
                        case PASS_BLOCK: {
                            ReportPassBlock pb = (ReportPassBlock) report;
                            if (pb.isPassBlockAvailable()) {
                                // Find a player on the team with Pass Block skill
                                Team pbTeam = pb.getTeamId() != null &&
                                    pb.getTeamId().equals(game.getTeamHome().getId())
                                    ? game.getTeamHome() : game.getTeamAway();
                                for (Player<?> p : pbTeam.getPlayers()) {
                                    for (Skill s : p.getSkills()) {
                                        if ("Pass Block".equals(s.getName())) {
                                            FieldCoordinate pc = game.getFieldModel().getPlayerCoordinate(p);
                                            if (pc != null) {
                                                pbPlayerId = p.getId();
                                                pbOldX = pc.getX();
                                                pbOldY = pc.getY();
                                                pbState = GameStateSerializer.serialize(game);
                                            }
                                            break;
                                        }
                                    }
                                    if (pbPlayerId != null) break;
                                }
                            }
                            break;
                        }

                        // Player activation
                        case PLAYER_ACTION: {
                            ReportPlayerAction pa = (ReportPlayerAction) report;

                            // Flush pending move_target
                            if (pendingMovePlayerId != null) {
                                JsonObject moveRec = flushMoveTarget(
                                    game, pendingMovePlayerId, pendingMoveActionType,
                                    pendingMoveState, pendingMoveCands, pendingMoveScores,
                                    pendingMoveHasEnd, pendingMoveBallCarrier,
                                    pendingMoveBallRetriever, pendingMoveReceiver);
                                if (moveRec != null) {
                                    writer.write(moveRec.toString()); writer.newLine(); recordCount++;
                                }
                                pendingMovePlayerId = null;
                            }

                            // Clear chainsaw pending if new player activates
                            // (chainsaw not used; but we need at least one block to detect decline)
                            if (pendingChainsaw) {
                                // Chainsaw player activated but no chainsaw roll yet
                                // The BLOCK_CHOICE should fire next if declined
                                // We keep pendingChainsaw=true until CHAINSAW_ROLL or BLOCK_CHOICE
                            }

                            // Set chainsaw pending if this activation is CHAINSAW
                            if (pa.getPlayerAction() == PlayerAction.CHAINSAW) {
                                pendingChainsaw      = true;
                                pendingChainsawState = GameStateSerializer.serialize(game);
                            } else {
                                pendingChainsaw      = false;
                                pendingChainsawState = null;
                            }

                            // Emit player_select
                            rec = buildPlayerSelectRecord(game, pa);

                            // Setup move tracking
                            if (isMovementAction(pa.getPlayerAction())) {
                                nextMovePlayerId = pa.getActingPlayerId();
                                nextMoveAction   = pa.getPlayerAction();
                            } else {
                                nextMovePlayerId = null;
                                nextMoveAction   = null;
                            }
                            break;
                        }

                        // Turn end — flush any pending move_target
                        case TURN_END: {
                            if (pendingMovePlayerId != null) {
                                JsonObject moveRec = flushMoveTarget(
                                    game, pendingMovePlayerId, pendingMoveActionType,
                                    pendingMoveState, pendingMoveCands, pendingMoveScores,
                                    pendingMoveHasEnd, pendingMoveBallCarrier,
                                    pendingMoveBallRetriever, pendingMoveReceiver);
                                if (moveRec != null) {
                                    writer.write(moveRec.toString()); writer.newLine(); recordCount++;
                                }
                                pendingMovePlayerId = null;
                            }
                            break;
                        }

                        default:
                            break;
                    }
                } catch (Exception ignored) {}

                if (rec != null) {
                    try { writer.write(rec.toString()); writer.newLine(); recordCount++; }
                    catch (Exception ignored) {}
                }

                // ── Post-switch: detect RE_ROLL opportunity from failed skill rolls ──
                // Check after the switch so that if this report IS a RE_ROLL, it was
                // already consumed above (rrState cleared), and we don't re-set pending.
                if (rrState == null && report instanceof ReportSkillRoll
                        && report.getId() != ReportId.BLOOD_LUST_ROLL
                        && report.getId() != ReportId.INTERCEPTION_ROLL
                        && report.getId() != ReportId.CHAINSAW_ROLL) {
                    try {
                        ReportSkillRoll sr = (ReportSkillRoll) report;
                        if (!sr.isSuccessful() && !sr.isReRolled()) {
                            // Check if acting team has rerolls or player has Pro
                            boolean hasReroll = false;
                            TurnData td = game.getTurnData();
                            if (td != null && td.getReRolls() > 0) hasReroll = true;
                            if (!hasReroll) {
                                ActingPlayer ap = game.getActingPlayer();
                                if (ap != null) {
                                    Player<?> actP = game.getPlayerById(ap.getPlayerId());
                                    if (actP != null) {
                                        for (Skill s : actP.getSkills()) {
                                            if ("Pro".equals(s.getName())) { hasReroll = true; break; }
                                        }
                                    }
                                }
                            }
                            if (hasReroll) {
                                rrPlayerId   = sr.getPlayerId();
                                rrActionName = reportIdToReRollAction(report.getId());
                                rrMinRoll    = sr.getMinimumRoll();
                                rrIsFumble   = report.getId() == ReportId.PASS_ROLL
                                               && !sr.isSuccessful();
                                rrState      = GameStateSerializer.serialize(game);
                            }
                        }
                    } catch (Exception ignored) {}
                }

                // ── Post-switch: detect USE_CHAINSAW decline (block happens without chainsaw) ──
                if (pendingChainsaw && report.getId() == ReportId.BLOCK_CHOICE) {
                    try {
                        // Block occurred but no CHAINSAW_ROLL → declined (action=1)
                        JsonObject rec2 = buildDialogRecordWithState("USE_CHAINSAW", 1,
                            new JsonObject(), pendingChainsawState);
                        if (rec2 != null) { writer.write(rec2.toString()); writer.newLine(); recordCount++; }
                    } catch (Exception ignored) {}
                    pendingChainsaw = false; pendingChainsawState = null;
                }

                // ── Post-switch: emit pass_target for accurate pass (no deviate) ──
                if (report.getId() == ReportId.PASS_ROLL && !pendingPassDeviated
                        && pendingPassState != null) {
                    // Will be emitted in post-change below when we know the target.
                    // Keep pendingPassState alive.
                }
            }

            // ── APPLY MODEL CHANGES ────────────────────────────────────────────
            try {
                syncCmd.getModelChanges().applyTo(game, Collections.emptySet());
            } catch (Exception ignored) {}

            // ── POST-CHANGE PASS ───────────────────────────────────────────────

            // FOLLOWUP_CHOICE: check attacker position after pushback
            if (fuAttackerId != null && fuCheckSyncs > 0) {
                fuCheckSyncs--;
                try {
                    Player<?> att = game.getPlayerById(fuAttackerId);
                    if (att != null) {
                        FieldCoordinate ac = game.getFieldModel().getPlayerCoordinate(att);
                        if (ac != null) {
                            int action = -1;
                            if (ac.getX() == fuTargetX && ac.getY() == fuTargetY) {
                                action = 0; // followed up
                            } else if (ac.getX() == fuAttackerX && ac.getY() == fuAttackerY) {
                                action = 1; // stayed
                            }
                            if (action >= 0) {
                                JsonObject rec = buildFollowupRecord(fuState, action);
                                if (rec != null) {
                                    writer.write(rec.toString()); writer.newLine(); recordCount++;
                                }
                                fuAttackerId = null; fuState = null; fuCheckSyncs = 0;
                            }
                        }
                    }
                } catch (Exception ignored) {}
                if (fuCheckSyncs <= 0 && fuAttackerId != null) {
                    fuAttackerId = null; fuState = null; // timeout
                }
            }

            // PASS target (accurate): after PASS_ROLL with no deviate, use passCoordinate
            // which should be set in the game model
            if (pendingPassState != null && !pendingPassDeviated) {
                try {
                    FieldCoordinate passTarget = game.getPassCoordinate();
                    if (passTarget == null) {
                        // Fallback: use pre-change pendingPassCoord if it was captured
                        passTarget = pendingPassCoord;
                    }
                    if (passTarget != null) {
                        JsonObject rec = buildCoordTargetRecord("pass_target", passTarget, pendingPassState, false);
                        if (rec != null) { writer.write(rec.toString()); writer.newLine(); recordCount++; }
                    }
                } catch (Exception ignored) {}
                pendingPassState = null; pendingPassCoord = null;
            }

            // PASS_BLOCK: check if player moved after PassBlock was offered
            if (pbPlayerId != null) {
                try {
                    Player<?> pbP = game.getPlayerById(pbPlayerId);
                    if (pbP != null) {
                        FieldCoordinate nc = game.getFieldModel().getPlayerCoordinate(pbP);
                        if (nc != null) {
                            int action = (nc.getX() != pbOldX || nc.getY() != pbOldY) ? 0 : 1;
                            JsonObject rec = buildDialogRecordWithState("PASS_BLOCK", action, pidParam(pbPlayerId), pbState);
                            if (rec != null) { writer.write(rec.toString()); writer.newLine(); recordCount++; }
                        }
                    }
                } catch (Exception ignored) {}
                pbPlayerId = null; pbState = null;
            }

            // TOUCHBACK: when TurnMode is TOUCHBACK post-change, capture state
            // On the next sync where ball has a carrier, emit the touchback record
            if (game.getTurnMode() == TurnMode.TOUCHBACK && !touchbackPending) {
                try {
                    touchbackState   = GameStateSerializer.serialize(game);
                    touchbackPending = true;
                } catch (Exception ignored) {}
            } else if (touchbackPending && game.getTurnMode() != TurnMode.TOUCHBACK) {
                // TurnMode left TOUCHBACK — ball carrier is now the chosen receiver
                try {
                    FieldCoordinate bc = game.getFieldModel().getBallCoordinate();
                    if (bc != null) {
                        Player<?> carrier = game.getFieldModel().getPlayer(bc);
                        if (carrier != null) {
                            JsonObject rec = buildDialogRecordWithState("TOUCHBACK", 0,
                                pidParam(carrier.getId()), touchbackState);
                            if (rec != null) { writer.write(rec.toString()); writer.newLine(); recordCount++; }
                        }
                    }
                } catch (Exception ignored) {}
                touchbackPending = false; touchbackState = null;
            }

            // Move target setup (post-change)
            if (nextMovePlayerId != null) {
                try {
                    ActingPlayer ap = game.getActingPlayer();
                    if (ap != null && nextMovePlayerId.equals(ap.getPlayerId())) {
                        Player<?> p = game.getPlayerById(nextMovePlayerId);
                        if (p != null) {
                            boolean isHome = game.getTeamHome().hasPlayer(p);
                            Team myTeam  = isHome ? game.getTeamHome() : game.getTeamAway();
                            Team opTeam  = isHome ? game.getTeamAway() : game.getTeamHome();

                            MoveDecisionEngine.MoveResult mr = MoveDecisionEngine.selectMoveTarget(
                                game, ap, myTeam, opTeam, isHome, RNG, /*argmax=*/true);

                            if (mr != null && !mr.candidates.isEmpty()) {
                                pendingMovePlayerId      = nextMovePlayerId;
                                pendingMoveActionType    = nextMoveAction.name();
                                pendingMoveState         = GameStateSerializer.serialize(game);
                                pendingMoveCands         = mr.candidates;
                                pendingMoveScores        = mr.rawScores;
                                pendingMoveHasEnd        = mr.hasEndOption;
                                pendingMoveBallCarrier   = mr.isBallCarrier;
                                pendingMoveBallRetriever = mr.isBallRetriever;
                                pendingMoveReceiver      = mr.isReceiver;
                            }
                        }
                    }
                } catch (Exception ignored) {}
                nextMovePlayerId = null;
                nextMoveAction   = null;
            }
        }

        // Flush remaining pending move
        if (pendingMovePlayerId != null) {
            try {
                JsonObject moveRec = flushMoveTarget(
                    game, pendingMovePlayerId, pendingMoveActionType,
                    pendingMoveState, pendingMoveCands, pendingMoveScores,
                    pendingMoveHasEnd, pendingMoveBallCarrier,
                    pendingMoveBallRetriever, pendingMoveReceiver);
                if (moveRec != null) {
                    writer.write(moveRec.toString()); writer.newLine(); recordCount++;
                }
            } catch (Exception ignored) {}
        }

        return recordCount;
    }

    // ── move_target flush ─────────────────────────────────────────────────────

    private static JsonObject flushMoveTarget(
            Game game, String playerId, String actionType,
            JsonObject stateSnapshot, List<FieldCoordinate> candidates, double[] rawScores,
            boolean hasEndOption, boolean isBallCarrier, boolean isBallRetriever, boolean isReceiver) {

        if (playerId == null || candidates == null || stateSnapshot == null) return null;
        Player<?> p = game.getPlayerById(playerId);
        if (p == null) return null;

        FieldCoordinate finalCoord = game.getFieldModel().getPlayerCoordinate(p);
        int chosenIdx = -1;
        if (finalCoord != null) {
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.get(i).equals(finalCoord)) { chosenIdx = i; break; }
            }
        }
        if (chosenIdx < 0) return null;

        JsonObject rec = new JsonObject();
        rec.add("type",         "move_target");
        rec.add("agent_mode",   AGENT_MODE);
        rec.add("player_id",    playerId);
        rec.add("action_type",  actionType != null ? actionType : JsonValue.NULL.toString());
        rec.add("has_end_option", hasEndOption);
        rec.add("action",       chosenIdx);
        rec.add("scores",       doubleArray(rawScores));

        JsonArray coords = new JsonArray();
        for (FieldCoordinate fc : candidates) {
            JsonArray xy = new JsonArray(); xy.add(fc.getX()); xy.add(fc.getY());
            coords.add(xy);
        }
        rec.add("candidates", coords);
        rec.add("state", stateSnapshot);
        return rec;
    }

    private static boolean isMovementAction(PlayerAction action) {
        if (action == null) return false;
        String n = action.name();
        // All *_MOVE actions
        if (n.endsWith("_MOVE")) return true;
        switch (action) {
            case MOVE: return true;
            default:   return false;
        }
    }

    // ── player_select ─────────────────────────────────────────────────────────

    private static JsonObject buildPlayerSelectRecord(Game game, ReportPlayerAction report) {
        String playerId  = report.getActingPlayerId();
        PlayerAction action = report.getPlayerAction();
        if (playerId == null || action == null) return null;

        Player<?> actingPlayer = game.getPlayerById(playerId);
        if (actingPlayer == null) return null;

        boolean isHome    = game.getTeamHome().hasPlayer(actingPlayer);
        Team myTeam       = isHome ? game.getTeamHome() : game.getTeamAway();
        Team opponentTeam = isHome ? game.getTeamAway() : game.getTeamHome();

        MoveDecisionEngine.PlayerSelection sel = null;
        try {
            sel = MoveDecisionEngine.selectPlayer(game, myTeam, opponentTeam,
                isHome, /*allowBlock=*/true, RNG, /*argmax=*/true);
        } catch (Exception ignored) {}

        // Try to find the chosen action in MDE candidates
        if (sel != null) {
            List<Player<?>> cands = sel.candidatePlayers;
            List<PlayerAction> acts = sel.candidateActions;
            int chosenIdx = -1;
            for (int i = 0; i < cands.size(); i++) {
                Player<?> cp = cands.get(i);
                if (cp != null && cp.getId().equals(playerId) && acts.get(i) == action) {
                    chosenIdx = i; break;
                }
            }
            if (chosenIdx >= 0) {
                JsonObject rec = new JsonObject();
                rec.add("type",            "player_select");
                rec.add("agent_mode",      AGENT_MODE);
                rec.add("action",          chosenIdx);
                rec.add("scores",          doubleArray(sel.rawScores));
                rec.add("in_mde_candidates", true);
                JsonArray candidates = new JsonArray();
                for (int i = 0; i < cands.size(); i++) {
                    Player<?> cp = cands.get(i);
                    if (cp == null) continue;
                    JsonObject c = new JsonObject();
                    c.add("player_id", cp.getId());
                    PlayerAction pa = acts.get(i);
                    c.add("action", pa != null ? pa.name() : "END_TURN");
                    candidates.add(c);
                }
                rec.add("candidates",      candidates);
                rec.add("end_turn_option", true);
                rec.add("state", GameStateSerializer.serialize(game));
                return rec;
            }
        }

        // Fallback: action not in MDE candidates (PASS, HAND_OVER, TTM, STAB, etc.)
        JsonObject rec = new JsonObject();
        rec.add("type",            "player_select");
        rec.add("agent_mode",      AGENT_MODE);
        rec.add("action",          0);
        rec.add("scores",          new JsonArray());
        rec.add("in_mde_candidates", false);
        JsonArray candidates = new JsonArray();
        JsonObject c = new JsonObject();
        c.add("player_id", playerId);
        c.add("action",    action.name());
        candidates.add(c);
        rec.add("candidates",      candidates);
        rec.add("end_turn_option", false);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    // ── Dialog record builders ────────────────────────────────────────────────

    private static JsonObject buildBlockRollPropertiesRecord(Game game, ReportBlockChoice report) {
        int[] roll  = report.getBlockRoll();
        int chosen  = report.getDiceIndex();
        if (roll == null || roll.length == 0) return null;
        if (chosen < 0 || chosen >= roll.length) return null;

        String choosingTeamId = "";
        try {
            if (report.getNrOfDice() < 0) {
                Player<?> defender = game.getPlayerById(report.getDefenderId());
                if (defender != null) {
                    boolean defHome = game.getTeamHome().hasPlayer(defender);
                    choosingTeamId = (defHome ? game.getTeamHome() : game.getTeamAway()).getId();
                }
            } else {
                ActingPlayer ap = game.getActingPlayer();
                if (ap != null && ap.getPlayerId() != null) {
                    Player<?> attacker = game.getPlayerById(ap.getPlayerId());
                    if (attacker != null) {
                        boolean attHome = game.getTeamHome().hasPlayer(attacker);
                        choosingTeamId = (attHome ? game.getTeamHome() : game.getTeamAway()).getId();
                    }
                }
            }
        } catch (Exception ignored) {}

        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "BLOCK_ROLL_PROPERTIES");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     chosen);
        rec.add("scores",     new JsonArray());
        JsonObject param = new JsonObject();
        param.add("num_dice",         report.getNrOfDice());
        param.add("choosing_team_id", choosingTeamId);
        if (report.getDefenderId() != null) param.add("target_id", report.getDefenderId());
        rec.add("dialog_param", param);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildSkillUseRecord(Game game, ReportSkillUse report) {
        if (report.getSkill() == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "SKILL_USE");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     report.isUsed() ? 0 : 1);
        rec.add("scores",     new JsonArray());
        JsonObject param = new JsonObject();
        if (report.getPlayerId() != null) param.add("player_id", report.getPlayerId());
        param.add("skill",    report.getSkill().getName());
        param.add("min_roll", 0);
        rec.add("dialog_param", param);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildReceiveChoiceRecord(Game game, ReportReceiveChoice report) {
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "RECEIVE_CHOICE");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     report.isReceiveChoice() ? 0 : 1);
        rec.add("scores",     new JsonArray());
        rec.add("dialog_param", new JsonObject());
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildCoinChoiceRecord(Game game, ReportCoinThrow report) {
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "COIN_CHOICE");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     report.isCoinChoiceHeads() ? 0 : 1);
        rec.add("scores",     new JsonArray());
        rec.add("dialog_param", new JsonObject());
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildReRollRecord(String playerId, String actionName,
            int minRoll, boolean isFumble, JsonObject state, int action) {
        if (state == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "RE_ROLL");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     action);
        rec.add("scores",     new JsonArray());
        JsonObject param = new JsonObject();
        if (playerId != null) param.add("player_id", playerId);
        param.add("rerolled_action", actionName != null ? actionName : "UNKNOWN");
        param.add("min_roll",        minRoll);
        param.add("is_team_reroll",  true);
        param.add("is_pro_reroll",   false);
        param.add("is_fumble",       isFumble);
        rec.add("dialog_param", param);
        rec.add("state", state);
        return rec;
    }

    private static JsonObject buildFollowupRecord(JsonObject state, int action) {
        if (state == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "FOLLOWUP_CHOICE");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     action);
        rec.add("scores",     new JsonArray());
        rec.add("dialog_param", new JsonObject());
        rec.add("state", state);
        return rec;
    }

    private static JsonObject buildUseApothecaryRecord(String playerId, int injStateBase,
            String seriousInjury, JsonObject state, int action) {
        if (state == null || playerId == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "USE_APOTHECARY");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     action);
        rec.add("scores",     new JsonArray());
        JsonObject param = new JsonObject();
        param.add("player_id",    playerId);
        param.add("injury_state", playerStateBaseToString(injStateBase));
        param.add("serious_injury", seriousInjury != null ? seriousInjury : JsonValue.NULL.toString());
        rec.add("dialog_param", param);
        rec.add("state", state);
        return rec;
    }

    private static JsonObject buildApothecaryChoiceRecord(Game game, ReportApothecaryChoice report,
            int oldBase, String oldSI, int newBase, String newSI) {
        if (report.getPlayerId() == null || report.getPlayerState() == null) return null;
        int chosenBase = report.getPlayerState().getBase();
        String chosenSI = report.getSeriousInjury() != null ? report.getSeriousInjury().getName() : null;
        // action=0: accepted new roll result; action=1: kept old result
        int action;
        if (chosenBase == newBase) {
            action = 0;
        } else if (chosenBase == oldBase) {
            action = 1;
        } else {
            return null; // can't determine
        }
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "APOTHECARY_CHOICE");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     action);
        rec.add("scores",     new JsonArray());
        JsonObject param = new JsonObject();
        param.add("player_id",          report.getPlayerId());
        param.add("old_player_state",   playerStateBaseToString(oldBase));
        param.add("old_serious_injury", oldSI != null ? oldSI : JsonValue.NULL.toString());
        param.add("new_player_state",   playerStateBaseToString(newBase));
        param.add("new_serious_injury", newSI != null ? newSI : JsonValue.NULL.toString());
        rec.add("dialog_param", param);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildPilingOnRecord(Game game, ReportPilingOn report) {
        if (report.getPlayerId() == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "PILING_ON");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     report.isUsed() ? 0 : 1);
        rec.add("scores",     new JsonArray());
        rec.add("dialog_param", pidParam(report.getPlayerId()));
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    // ── Target record builders ────────────────────────────────────────────────

    private static JsonObject buildFoulTargetRecord(Game game, ReportFoul report) {
        String defenderId = report.getDefenderId();
        if (defenderId == null) return null;

        ActingPlayer ap = game.getActingPlayer();
        if (ap == null) return null;
        Player<?> fouler = game.getPlayerById(ap.getPlayerId());
        if (fouler == null) return null;
        FieldCoordinate fc = game.getFieldModel().getPlayerCoordinate(fouler);
        if (fc == null) return null;

        boolean isHome = game.getTeamHome().hasPlayer(fouler);
        Team opTeam = isHome ? game.getTeamAway() : game.getTeamHome();

        // Collect adjacent prone/stunned targets
        List<String> targets = new ArrayList<>();
        for (Player<?> op : opTeam.getPlayers()) {
            FieldCoordinate oc = game.getFieldModel().getPlayerCoordinate(op);
            if (oc == null) continue;
            PlayerState ps = game.getFieldModel().getPlayerState(op);
            if (ps == null) continue;
            int base = ps.getBase();
            if (base != PlayerState.PRONE && base != PlayerState.STUNNED) continue;
            if (Math.abs(oc.getX() - fc.getX()) <= 1 && Math.abs(oc.getY() - fc.getY()) <= 1)
                targets.add(op.getId());
        }

        int chosenIdx = targets.indexOf(defenderId);
        if (chosenIdx < 0) { targets.clear(); targets.add(defenderId); chosenIdx = 0; }

        JsonObject rec = new JsonObject();
        rec.add("type",       "foul_target");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     chosenIdx);
        rec.add("target_id",  defenderId);
        JsonArray cands = new JsonArray(); for (String t : targets) cands.add(t);
        rec.add("candidates", cands);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildHandoffTargetRecord(Game game, ReportHandOver report) {
        String catcherId = report.getCatcherId();
        if (catcherId == null) return null;

        ActingPlayer ap = game.getActingPlayer();
        if (ap == null) return null;
        Player<?> passer = game.getPlayerById(ap.getPlayerId());
        if (passer == null) return null;
        FieldCoordinate fc = game.getFieldModel().getPlayerCoordinate(passer);
        if (fc == null) return null;

        boolean isHome = game.getTeamHome().hasPlayer(passer);
        Team myTeam = isHome ? game.getTeamHome() : game.getTeamAway();

        // Collect adjacent standing teammates
        List<String> targets = new ArrayList<>();
        for (Player<?> mate : myTeam.getPlayers()) {
            if (mate.getId().equals(ap.getPlayerId())) continue;
            FieldCoordinate mc = game.getFieldModel().getPlayerCoordinate(mate);
            if (mc == null) continue;
            PlayerState ps = game.getFieldModel().getPlayerState(mate);
            if (ps == null || ps.getBase() != PlayerState.STANDING) continue;
            if (Math.abs(mc.getX() - fc.getX()) <= 1 && Math.abs(mc.getY() - fc.getY()) <= 1)
                targets.add(mate.getId());
        }

        int chosenIdx = targets.indexOf(catcherId);
        if (chosenIdx < 0) { targets.clear(); targets.add(catcherId); chosenIdx = 0; }

        JsonObject rec = new JsonObject();
        rec.add("type",       "handoff_target");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     chosenIdx);
        rec.add("target_id",  catcherId);
        JsonArray cands = new JsonArray(); for (String t : targets) cands.add(t);
        rec.add("candidates", cands);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildGazeTargetRecord(Game game, ReportSelectGazeTarget report) {
        String defenderId = report.getDefender();
        if (defenderId == null) return null;

        String attackerId = report.getAttacker();
        Player<?> att = attackerId != null ? game.getPlayerById(attackerId) : null;
        FieldCoordinate fc = att != null ? game.getFieldModel().getPlayerCoordinate(att) : null;

        boolean isHome = att != null && game.getTeamHome().hasPlayer(att);
        Team opTeam = isHome ? game.getTeamAway() : game.getTeamHome();

        // Collect adjacent opponents
        List<String> targets = new ArrayList<>();
        if (fc != null) {
            for (Player<?> op : opTeam.getPlayers()) {
                FieldCoordinate oc = game.getFieldModel().getPlayerCoordinate(op);
                if (oc == null) continue;
                if (Math.abs(oc.getX() - fc.getX()) <= 1 && Math.abs(oc.getY() - fc.getY()) <= 1)
                    targets.add(op.getId());
            }
        }

        int chosenIdx = targets.indexOf(defenderId);
        if (chosenIdx < 0) { targets.clear(); targets.add(defenderId); chosenIdx = 0; }

        JsonObject rec = new JsonObject();
        rec.add("type",       "gaze_target");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     chosenIdx);
        rec.add("target_id",  defenderId);
        JsonArray cands = new JsonArray(); for (String t : targets) cands.add(t);
        rec.add("candidates", cands);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildPassTargetRecord(ReportPassDeviate report,
            FieldCoordinate capturedCoord, JsonObject state) {
        if (state == null) return null;
        FieldCoordinate target = capturedCoord;
        if (target == null && report != null && report.getBallCoordinateEnd() != null) {
            target = scatterInverse(report.getBallCoordinateEnd(),
                report.getScatterDirection(), report.getRollScatterDistance());
        }
        if (target == null) return null;
        return buildCoordTargetRecord("pass_target", target, state, false);
    }

    private static JsonObject buildTtmTargetRecord(ReportPassDeviate report, JsonObject state) {
        if (report == null || state == null) return null;
        FieldCoordinate target = report.getBallCoordinateEnd() != null
            ? scatterInverse(report.getBallCoordinateEnd(),
                report.getScatterDirection(), report.getRollScatterDistance())
            : null;
        if (target == null) return null;
        return buildCoordTargetRecord("ttm_target", target, state, false);
    }

    private static JsonObject buildKickoffTargetRecord(Game game, ReportKickoffScatter report) {
        if (report == null || report.getBallCoordinateEnd() == null) return null;
        FieldCoordinate target = scatterInverse(report.getBallCoordinateEnd(),
            report.getScatterDirection(), report.getRollScatterDistance());
        if (target == null) return null;
        return buildCoordTargetRecord("kickoff_target", target, GameStateSerializer.serialize(game), false);
    }

    private static JsonObject buildCoordTargetRecord(String type, FieldCoordinate target,
            JsonObject state, boolean partial) {
        if (target == null || state == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       type);
        rec.add("agent_mode", AGENT_MODE);
        // action encoded as flat board index: x*15 + y (26×15 field)
        rec.add("action",     target.getX() * 15 + target.getY());
        rec.add("target_x",   target.getX());
        rec.add("target_y",   target.getY());
        if (partial) rec.add("partial", true);
        rec.add("state", state);
        return rec;
    }

    private static JsonObject buildChainsawTargetRecord(Game game, ReportChainsawRoll report) {
        if (report.getDefenderId() == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",       "chainsaw_target");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("target_id",  report.getDefenderId());
        rec.add("action",     0);
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    private static JsonObject buildBriberyCorruptionRecord(Game game, IReport report) {
        JsonObject rec = new JsonObject();
        rec.add("type",       "dialog");
        rec.add("dialog_id",  "BRIBERY_AND_CORRUPTION_RE_ROLL");
        rec.add("agent_mode", AGENT_MODE);
        rec.add("action",     0);
        rec.add("scores",     new JsonArray());
        rec.add("dialog_param", new JsonObject());
        rec.add("state", GameStateSerializer.serialize(game));
        return rec;
    }

    // ── Generic dialog helpers ────────────────────────────────────────────────

    private static JsonObject buildSimpleDialogRecord(Game game, String dialogId,
            int action, JsonObject param) {
        if (dialogId == null) return null;
        return buildDialogRecordWithState(dialogId, action, param,
            GameStateSerializer.serialize(game));
    }

    private static JsonObject buildDialogRecordWithState(String dialogId, int action,
            JsonObject param, JsonObject state) {
        if (dialogId == null || state == null) return null;
        JsonObject rec = new JsonObject();
        rec.add("type",         "dialog");
        rec.add("dialog_id",    dialogId);
        rec.add("agent_mode",   AGENT_MODE);
        rec.add("action",       action);
        rec.add("scores",       new JsonArray());
        rec.add("dialog_param", param != null ? param : new JsonObject());
        rec.add("state",        state);
        return rec;
    }

    // ── Injury / apothecary accessors (handle bb2016 and mixed variants) ──────

    private static String getInjuryDefenderId(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportInjury)
                return ((com.fumbbl.ffb.report.mixed.ReportInjury) report).getDefenderId();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportInjury)
                return ((com.fumbbl.ffb.report.bb2016.ReportInjury) report).getDefenderId();
        } catch (Exception ignored) {}
        return null;
    }

    private static PlayerState getInjuryPlayerState(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportInjury)
                return ((com.fumbbl.ffb.report.mixed.ReportInjury) report).getInjury();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportInjury)
                return ((com.fumbbl.ffb.report.bb2016.ReportInjury) report).getInjury();
        } catch (Exception ignored) {}
        return null;
    }

    private static SeriousInjury getInjurySeriousInjury(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportInjury)
                return ((com.fumbbl.ffb.report.mixed.ReportInjury) report).getSeriousInjury();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportInjury)
                return ((com.fumbbl.ffb.report.bb2016.ReportInjury) report).getSeriousInjury();
        } catch (Exception ignored) {}
        return null;
    }

    private static String getApothecaryRollPlayerId(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportApothecaryRoll)
                return ((com.fumbbl.ffb.report.mixed.ReportApothecaryRoll) report).getPlayerId();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportApothecaryRoll)
                return ((com.fumbbl.ffb.report.bb2016.ReportApothecaryRoll) report).getPlayerId();
        } catch (Exception ignored) {}
        return null;
    }

    private static PlayerState getApothecaryRollPlayerState(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportApothecaryRoll)
                return ((com.fumbbl.ffb.report.mixed.ReportApothecaryRoll) report).getPlayerState();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportApothecaryRoll)
                return ((com.fumbbl.ffb.report.bb2016.ReportApothecaryRoll) report).getPlayerState();
        } catch (Exception ignored) {}
        return null;
    }

    private static SeriousInjury getApothecaryRollSeriousInjury(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportApothecaryRoll)
                return ((com.fumbbl.ffb.report.mixed.ReportApothecaryRoll) report).getSeriousInjury();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportApothecaryRoll)
                return ((com.fumbbl.ffb.report.bb2016.ReportApothecaryRoll) report).getSeriousInjury();
        } catch (Exception ignored) {}
        return null;
    }

    private static String getArgueTheCallPlayerId(IReport report) {
        try {
            if (report instanceof com.fumbbl.ffb.report.mixed.ReportArgueTheCallRoll)
                return ((com.fumbbl.ffb.report.mixed.ReportArgueTheCallRoll) report).getPlayerId();
            if (report instanceof com.fumbbl.ffb.report.bb2016.ReportArgueTheCallRoll)
                return ((com.fumbbl.ffb.report.bb2016.ReportArgueTheCallRoll) report).getPlayerId();
        } catch (Exception ignored) {}
        return null;
    }

    // ── Scatter inverse ───────────────────────────────────────────────────────

    /**
     * Reconstruct the origin of a scatter given the end coordinate, direction, and distance.
     * Direction mapping (from UtilServerPushback): NORTH=y-1, SOUTH=y+1, EAST=x+1, WEST=x-1.
     */
    private static FieldCoordinate scatterInverse(FieldCoordinate end, Direction dir, int distance) {
        if (end == null || dir == null || distance <= 0) return end;
        int dx = 0, dy = 0;
        switch (dir) {
            case NORTH:     dx =  0; dy = -1; break;
            case NORTHEAST: dx =  1; dy = -1; break;
            case EAST:      dx =  1; dy =  0; break;
            case SOUTHEAST: dx =  1; dy =  1; break;
            case SOUTH:     dx =  0; dy =  1; break;
            case SOUTHWEST: dx = -1; dy =  1; break;
            case WEST:      dx = -1; dy =  0; break;
            case NORTHWEST: dx = -1; dy = -1; break;
        }
        return new FieldCoordinate(end.getX() - dx * distance, end.getY() - dy * distance);
    }

    // ── Mapping helpers ───────────────────────────────────────────────────────

    private static String reportIdToReRollAction(ReportId id) {
        if (id == null) return "UNKNOWN";
        switch (id) {
            case DODGE_ROLL:   return "DODGE";
            case GO_FOR_IT_ROLL: return "GO_FOR_IT";
            case PASS_ROLL:    return "PASS";
            case CATCH_ROLL:   return "CATCH";
            default:
                String name = id.name();
                if (name.endsWith("_ROLL")) return name.substring(0, name.length() - 5);
                return name;
        }
    }

    private static String playerStateBaseToString(int base) {
        switch (base) {
            case PlayerState.BADLY_HURT:     return "BH";
            case PlayerState.SERIOUS_INJURY: return "SI";
            case PlayerState.RIP:            return "RIP";
            case PlayerState.KNOCKED_OUT:    return "KO";
            default:                         return "UNKNOWN";
        }
    }

    private static JsonObject pidParam(String playerId) {
        JsonObject p = new JsonObject();
        if (playerId != null) p.add("player_id", playerId);
        return p;
    }

    // ── Misc helpers ──────────────────────────────────────────────────────────

    private static JsonArray doubleArray(double[] arr) {
        JsonArray ja = new JsonArray();
        if (arr != null) for (double v : arr) ja.add(v);
        return ja;
    }
}
