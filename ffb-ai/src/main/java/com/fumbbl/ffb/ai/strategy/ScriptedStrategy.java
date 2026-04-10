package com.fumbbl.ffb.ai.strategy;

import com.fumbbl.ffb.ApothecaryType;
import com.fumbbl.ffb.BlockResult;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerChoiceMode;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.ReRolledActions;
import com.fumbbl.ffb.ReRollSources;
import com.fumbbl.ffb.SeriousInjury;
import com.fumbbl.ffb.ai.ActionScore;
import com.fumbbl.ffb.ai.PolicySampler;
import com.fumbbl.ffb.client.net.ClientCommunication;
import com.fumbbl.ffb.dialog.DialogApothecaryChoiceParameter;
import com.fumbbl.ffb.dialog.DialogArgueTheCallParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPartialReRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogBribesParameter;
import com.fumbbl.ffb.dialog.DialogBuyCardsAndInducementsParameter;
import com.fumbbl.ffb.dialog.DialogBuyInducementsParameter;
import com.fumbbl.ffb.dialog.DialogBuyPrayersAndInducementsParameter;
import com.fumbbl.ffb.dialog.DialogFollowupChoiceParameter;
import com.fumbbl.ffb.dialog.DialogInformationOkayParameter;
import com.fumbbl.ffb.dialog.DialogJourneymenParameter;
import com.fumbbl.ffb.dialog.DialogKickSkillParameter;
import com.fumbbl.ffb.dialog.DialogOpponentBlockSelectionParameter;
import com.fumbbl.ffb.dialog.DialogOpponentBlockSelectionPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogPileDriverParameter;
import com.fumbbl.ffb.dialog.DialogPilingOnParameter;
import com.fumbbl.ffb.dialog.DialogPlayerChoiceParameter;
import com.fumbbl.ffb.dialog.DialogReRollBlockForTargetsParameter;
import com.fumbbl.ffb.dialog.DialogReRollForTargetsParameter;
import com.fumbbl.ffb.dialog.DialogReRollParameter;
import com.fumbbl.ffb.dialog.DialogReRollPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogSkillUseParameter;
import com.fumbbl.ffb.dialog.DialogTeamSetupParameter;
import com.fumbbl.ffb.dialog.DialogTouchbackParameter;
import com.fumbbl.ffb.dialog.DialogUseApothecariesParameter;
import com.fumbbl.ffb.dialog.DialogUseApothecaryParameter;
import com.fumbbl.ffb.dialog.DialogUseChainsawParameter;
import com.fumbbl.ffb.dialog.DialogWizardSpellParameter;
import com.fumbbl.ffb.dialog.DialogWinningsReRollParameter;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.InducementSet;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.property.NamedProperties;
import com.fumbbl.ffb.util.UtilCards;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Scripted (probabilistic) decision-making for all server-sent dialog types.
 *
 * Decisions are scored and passed through softmax so that:
 * - argmax(softmax) equals the best deterministic choice
 * - sample(softmax) introduces natural stochasticity while preserving strategic preference
 *
 * Replace {@link RandomStrategy#respondToDialog} calls with this class.
 */
public final class ScriptedStrategy {

    /** Softmax temperature for binary yes/no decisions. */
    private static final double T_BINARY = 0.20;
    /** Softmax temperature for block die selection. */
    private static final double T_BLOCK = 0.10;

    private static final Random RNG = new Random();

    /**
     * Sampling temperature ∈ [0, 1]:
     * 0 = argmax (fully deterministic), 0.5 = raw policy (current default), 1 = uniform (random).
     */
    private static double temperature = 0.5;

    public static void setTemperature(double t) {
        temperature = t;
    }

    // ── Optional per-thread decision logging (for training data collection) ──────

    private static final ThreadLocal<DecisionLog> LOG = new ThreadLocal<>();

    /** Enable decision logging for the current thread. Call before {@link #respondToDialog}. */
    public static void startLogging() { LOG.set(new DecisionLog()); }

    /**
     * Retrieve and clear the decision log for the current thread.
     * Returns {@code null} if logging was not started.
     */
    public static DecisionLog getAndClearLog() {
        DecisionLog log = LOG.get();
        LOG.remove();
        return log;
    }

    /** Sample from the mixed distribution over a score array. */
    private static int pick(double[] scores, double baseTemp) {
        int chosen = PolicySampler.sampleMixed(scores, baseTemp, temperature, RNG);
        DecisionLog log = LOG.get();
        if (log != null) log.add(scores, chosen);
        return chosen;
    }

    /** Binary mixed-distribution choice between scoreTrue and scoreFalse. */
    private static boolean pickBool(double scoreTrue, double scoreFalse, double baseTemp) {
        boolean choice = PolicySampler.chooseBoolMixed(scoreTrue, scoreFalse, baseTemp, temperature, RNG);
        DecisionLog log = LOG.get();
        if (log != null) log.addBool(scoreTrue, scoreFalse, choice);
        return choice;
    }

    private ScriptedStrategy() {}

    // ── Public entry point ──────────────────────────────────────────────────────

    public static void respondToDialog(IDialogParameter param, Game game, ClientCommunication comm) {
        if (param == null) {
            return;
        }
        switch (param.getId()) {

            // ── Binary yes/no choices ────────────────────────────────────────────

            case COIN_CHOICE:
                comm.sendCoinChoice(pickBool(50, 50, T_BINARY));
                break;

            case RECEIVE_CHOICE:
                comm.sendReceiveChoice(pickBool(80, 20, T_BINARY));
                break;

            case FOLLOWUP_CHOICE:
                comm.sendFollowupChoice(pickBool(65, 35, T_BINARY));
                break;

            case PICK_UP_CHOICE:
                comm.sendPickUpChoice(pickBool(90, 10, T_BINARY));
                break;

            case PUNT_TO_CROWD:
                comm.sendPuntToCrowd(pickBool(10, 90, T_BINARY));
                break;

            case USE_CHAINSAW:
                comm.sendUseChainsaw(pickBool(75, 25, T_BINARY));
                break;

            case BLOODLUST_ACTION:
                comm.sendChangeBloodlustAction(pickBool(80, 20, T_BINARY));
                break;

            // ── Piling On (usually decline) ──────────────────────────────────────

            case PILING_ON: {
                DialogPilingOnParameter pilingOn = (DialogPilingOnParameter) param;
                Player<?> pilingOnPlayer = game.getPlayerById(pilingOn.getPlayerId());
                if (pilingOnPlayer != null) {
                    boolean use = pickBool(20, 80, T_BINARY);
                    UtilCards.getSkillWithProperty(pilingOnPlayer,
                        NamedProperties.canPileOnOpponent)
                        .ifPresent(skill -> comm.sendUseSkill(skill, use, pilingOn.getPlayerId()));
                }
                break;
            }

            // ── Bribes (decline) ─────────────────────────────────────────────────

            case BRIBES:
                // Decline the bribe: send UseInducement with the AVOID_BAN inducement type
                // but no player ID, which StepBribes interprets as "don't use a bribe".
                // Cannot use sendUseInducement(null) — NPEs on null type check.
                // Cannot use sendArgueTheCall(null) — that resets fArgueTheCallChoice and loops.
                ((com.fumbbl.ffb.factory.InducementTypeFactory) game.getFactory(
                        com.fumbbl.ffb.FactoryType.Factory.INDUCEMENT_TYPE)).allTypes().stream()
                    .filter(t -> t.hasUsage(com.fumbbl.ffb.inducement.Usage.AVOID_BAN))
                    .findFirst()
                    .ifPresent(t -> comm.sendUseInducement(t, (String) null));
                break;

            // ── Kick Skill (decline) ─────────────────────────────────────────────

            case KICK_SKILL: {
                DialogKickSkillParameter kick = (DialogKickSkillParameter) param;
                Player<?> kickPlayer = game.getPlayerById(kick.getPlayerId());
                if (kickPlayer != null) {
                    boolean use = pickBool(20, 80, T_BINARY);
                    UtilCards.getSkillWithProperty(kickPlayer,
                        NamedProperties.canReduceKickDistance)
                        .ifPresent(skill -> comm.sendUseSkill(skill, use, kick.getPlayerId()));
                }
                break;
            }

            // ── Use Apothecary ───────────────────────────────────────────────────

            case USE_APOTHECARY: {
                DialogUseApothecaryParameter apo = (DialogUseApothecaryParameter) param;
                // Use apothecary if the injury is serious (SeriousInjury != null means MNG or worse)
                boolean useApo = apo.getSeriousInjury() != null
                    ? pickBool(90, 10, T_BINARY)
                    : pickBool(10, 90, T_BINARY);
                List<ApothecaryType> types = apo.getApothecaryTypes();
                ApothecaryType apoType = (types != null && !types.isEmpty()) ? types.get(0) : ApothecaryType.TEAM;
                comm.sendUseApothecary(apo.getPlayerId(), useApo, apoType, apo.getSeriousInjury());
                break;
            }

            // ── Use Igor / Mortuary (decline) ────────────────────────────────────

            case USE_IGOR:
            case USE_MORTUARY_ASSISTANT:
                comm.sendUseInducement((com.fumbbl.ffb.inducement.InducementType) null);
                break;

            // ── Block dice — attacker (pick best die) ────────────────────────────

            case BLOCK_ROLL: {
                DialogBlockRollParameter block = (DialogBlockRollParameter) param;
                int[] roll = block.getBlockRoll();
                int n = Math.abs(block.getNrOfDice());
                if (roll != null && n > 0) {
                    double[] scores = scoreDice(roll, n, game, true);
                    int chosen = pick(scores, T_BLOCK);
                    comm.sendBlockChoice(chosen);
                } else {
                    comm.sendBlockChoice(0);
                }
                break;
            }

            case BLOCK_ROLL_PARTIAL_RE_ROLL: {
                DialogBlockRollPartialReRollParameter partialBlock = (DialogBlockRollPartialReRollParameter) param;
                int[] roll = partialBlock.getBlockRoll();
                int n = Math.abs(partialBlock.getNrOfDice());
                if (roll != null && n > 0) {
                    double[] scores = scoreDice(roll, n, game, true);
                    int chosen = pick(scores, T_BLOCK);
                    comm.sendBlockChoice(chosen);
                } else {
                    comm.sendBlockChoice(0);
                }
                break;
            }

            case BLOCK_ROLL_PROPERTIES: {
                DialogBlockRollPropertiesParameter propsBlock = (DialogBlockRollPropertiesParameter) param;
                int[] roll = propsBlock.getBlockRoll();
                int n = Math.abs(propsBlock.getNrOfDice());
                if (roll != null && n > 0) {
                    double[] scores = scoreDice(roll, n, game, true);
                    int chosen = pick(scores, T_BLOCK);
                    comm.sendBlockChoice(chosen);
                } else {
                    comm.sendBlockChoice(0);
                }
                break;
            }

            // ── Block dice — defender (pick worst die for attacker) ──────────────

            case OPPONENT_BLOCK_SELECTION: {
                DialogOpponentBlockSelectionParameter obs = (DialogOpponentBlockSelectionParameter) param;
                List<com.fumbbl.ffb.model.BlockRoll> blockRolls = obs.getBlockRolls();
                if (blockRolls != null && !blockRolls.isEmpty()) {
                    int[] roll = blockRolls.get(0).getBlockRoll();
                    int n = (roll != null) ? roll.length : 0;
                    if (n > 0) {
                        double[] scores = scoreDice(roll, n, game, false); // false = defender picks argmin
                        int chosen = pick(scores, T_BLOCK);
                        comm.sendBlockChoice(chosen);
                        break;
                    }
                }
                comm.sendBlockChoice(0);
                break;
            }

            case OPPONENT_BLOCK_SELECTION_PROPERTIES: {
                DialogOpponentBlockSelectionPropertiesParameter obs =
                    (DialogOpponentBlockSelectionPropertiesParameter) param;
                List<com.fumbbl.ffb.model.BlockRollProperties> bProps = obs.getBlockRolls();
                if (bProps != null && !bProps.isEmpty()) {
                    int[] roll = bProps.get(0).getBlockRoll();
                    int n = (roll != null) ? roll.length : 0;
                    if (n > 0) {
                        double[] scores = scoreDice(roll, n, game, false);
                        int chosen = pick(scores, T_BLOCK);
                        comm.sendBlockChoice(chosen);
                        break;
                    }
                }
                comm.sendBlockChoice(0);
                break;
            }

            // ── Re-roll ──────────────────────────────────────────────────────────

            case RE_ROLL: {
                DialogReRollParameter rr = (DialogReRollParameter) param;
                // Reroll if it's a fumble/turnover; otherwise conservative
                boolean isHardTurnover = rr.isFumble();
                double scoreYes = isHardTurnover ? 85 : 15;
                double scoreNo  = isHardTurnover ? 15 : 85;
                boolean useRr = pickBool(scoreYes, scoreNo, T_BINARY);
                if (useRr) {
                    if (rr.isTeamReRollOption()) {
                        comm.sendUseReRoll(rr.getReRolledAction(), ReRollSources.TEAM_RE_ROLL);
                    } else if (rr.isProReRollOption()) {
                        comm.sendUseReRoll(rr.getReRolledAction(), ReRollSources.PRO);
                    } else if (rr.getSingleUseReRollSource() != null) {
                        comm.sendUseReRoll(rr.getReRolledAction(), rr.getSingleUseReRollSource());
                    } else if (rr.getReRollSkill() != null) {
                        comm.sendUseSkill(rr.getReRollSkill(), true, rr.getPlayerId(), rr.getReRolledAction());
                    } else {
                        comm.sendUseReRoll(rr.getReRolledAction(), null);
                    }
                } else {
                    comm.sendUseReRoll(rr.getReRolledAction(), null);
                }
                break;
            }

            case RE_ROLL_PROPERTIES: {
                DialogReRollPropertiesParameter rrp = (DialogReRollPropertiesParameter) param;
                // Conservative: usually decline unless all dice are turnovers (heuristic: decline)
                comm.sendUseReRoll(rrp.getReRolledAction(), null);
                break;
            }

            case RE_ROLL_FOR_TARGETS: {
                DialogReRollForTargetsParameter rrft = (DialogReRollForTargetsParameter) param;
                comm.sendUseReRoll(rrft.getReRolledAction(), null);
                break;
            }

            case RE_ROLL_BLOCK_FOR_TARGETS: {
                DialogReRollBlockForTargetsParameter rrbft = (DialogReRollBlockForTargetsParameter) param;
                // Pick the best die index from the block rolls
                com.fumbbl.ffb.model.BlockRoll br = rrbft.getBlockRolls() != null && !rrbft.getBlockRolls().isEmpty()
                    ? rrbft.getBlockRolls().get(0) : null;
                if (br != null && br.getBlockRoll() != null && br.getBlockRoll().length > 0) {
                    double[] scores = scoreDice(br.getBlockRoll(), br.getBlockRoll().length, game, true);
                    comm.sendBlockChoice(PolicySampler.argmax(scores));
                } else {
                    comm.sendBlockChoice(0);
                }
                break;
            }

            // ── Skill use ────────────────────────────────────────────────────────

            case SKILL_USE: {
                DialogSkillUseParameter su = (DialogSkillUseParameter) param;
                boolean use = pickBool(80, 20, T_BINARY);
                comm.sendUseSkill(su.getSkill(), use, su.getPlayerId());
                break;
            }

            // ── Player choice ────────────────────────────────────────────────────

            case PLAYER_CHOICE: {
                DialogPlayerChoiceParameter pc = (DialogPlayerChoiceParameter) param;
                String[] playerIds = pc.getPlayerIds();
                if (pc.getPlayerChoiceMode() == PlayerChoiceMode.SOLID_DEFENCE) {
                    comm.sendPlayerChoice(pc.getPlayerChoiceMode(), new Player[0]);
                    break;
                }
                int minSelects = pc.getMinSelects();
                int maxSelects = (playerIds != null) ? Math.min(pc.getMaxSelects(), playerIds.length) : 0;
                int numToSelect = minSelects;
                if (playerIds == null || playerIds.length == 0) {
                    comm.sendPlayerChoice(pc.getPlayerChoiceMode(), new Player[0]);
                    break;
                }
                numToSelect = Math.min(numToSelect, playerIds.length);
                Player<?>[] chosen = new Player[numToSelect];
                for (int i = 0; i < numToSelect; i++) {
                    chosen[i] = game.getPlayerById(playerIds[i]);
                }
                comm.sendPlayerChoice(pc.getPlayerChoiceMode(), chosen);
                break;
            }

            // ── Touchback ────────────────────────────────────────────────────────

            case TOUCHBACK: {
                FieldCoordinate coord = findBestBallCarrierCoord(game);
                comm.sendTouchback(coord);
                break;
            }

            // ── Interception (decline) ───────────────────────────────────────────

            case INTERCEPTION:
                comm.sendInterceptorChoice(null, null);
                break;

            // ── Apothecary choice ─────────────────────────────────────────────────

            case APOTHECARY_CHOICE: {
                DialogApothecaryChoiceParameter ac = (DialogApothecaryChoiceParameter) param;
                SeriousInjury oldSi = ac.getSeriousInjuryOld();
                SeriousInjury newSi = ac.getSeriousInjuryNew();
                boolean takeNew = isNewInjuryBetter(newSi, oldSi);
                if (pickBool(takeNew ? 85 : 15, takeNew ? 15 : 85, T_BINARY)) {
                    // Accept new result
                    comm.sendApothecaryChoice(ac.getPlayerId(), ac.getPlayerStateNew(), newSi, ac.getPlayerStateOld());
                } else {
                    // Keep old result
                    comm.sendApothecaryChoice(ac.getPlayerId(), ac.getPlayerStateOld(), oldSi, ac.getPlayerStateOld());
                }
                break;
            }

            // ── Team setup ────────────────────────────────────────────────────────

            case TEAM_SETUP: {
                DialogTeamSetupParameter ts = (DialogTeamSetupParameter) param;
                String[] names = ts.getSetupNames();
                if (names != null && names.length > 0) {
                    String chosen = chooseSetup(names, game);
                    comm.sendTeamSetupLoad(chosen);
                }
                break;
            }

            // ── Argue the call (decline) ──────────────────────────────────────────

            case ARGUE_THE_CALL:
                comm.sendArgueTheCall((String) null);
                break;

            // ── Pile driver (use it) ──────────────────────────────────────────────

            case PILE_DRIVER: {
                DialogPileDriverParameter pd = (DialogPileDriverParameter) param;
                List<String> kd = pd.getKnockedDownPlayers();
                if (kd != null && !kd.isEmpty()) {
                    boolean use = pickBool(75, 25, T_BINARY);
                    if (use) {
                        comm.sendPileDriver(kd.get(0));
                    }
                }
                break;
            }

            // ── Journeymen (fill all slots with first position) ───────────────────

            case JOURNEYMEN: {
                DialogJourneymenParameter jm = (DialogJourneymenParameter) param;
                int nrSlots = jm.getNrOfSlots();
                String[] posIds = jm.getPositionIds();
                if (posIds != null && posIds.length > 0 && nrSlots > 0) {
                    int count = Math.min(nrSlots, posIds.length);
                    String[] selectedPositions = new String[count];
                    int[] slots = new int[count];
                    for (int i = 0; i < count; i++) {
                        selectedPositions[i] = posIds[0];
                        slots[i] = i;
                    }
                    comm.sendJourneymen(selectedPositions, slots);
                }
                break;
            }

            // ── Wizard spell (decline) ────────────────────────────────────────────

            case WIZARD_SPELL:
                comm.sendConfirm();
                break;

            // ── Use inducement (decline) ──────────────────────────────────────────

            case USE_INDUCEMENT:
                comm.sendUseInducement((com.fumbbl.ffb.inducement.InducementType) null);
                break;

            // ── Start game / confirm ──────────────────────────────────────────────

            case START_GAME:
                comm.sendStartGame();
                break;

            case CONFIRM_END_ACTION:
                comm.sendConfirm();
                break;

            case INFORMATION_OKAY: {
                DialogInformationOkayParameter info = (DialogInformationOkayParameter) param;
                if (info.isConfirm()) {
                    comm.sendConfirm();
                }
                break;
            }

            case SETUP_ERROR:
            case SWARMING_ERROR:
            case INVALID_SOLID_DEFENCE:
            case PENALTY_SHOOTOUT:
                comm.sendConfirm();
                break;

            // ── Inducements / buy (decline all) ──────────────────────────────────

            case BUY_INDUCEMENTS:
                comm.sendBuyInducements(
                    ((DialogBuyInducementsParameter) param).getTeamId(),
                    ((DialogBuyInducementsParameter) param).getAvailableGold(),
                    new InducementSet(),
                    new String[0], new String[0],
                    new com.fumbbl.ffb.model.skill.Skill[0],
                    new String[0]);
                break;

            case BUY_CARDS:
                comm.sendCardSelection(null);
                break;

            case BUY_CARDS_AND_INDUCEMENTS:
                comm.sendBuyInducements(
                    ((DialogBuyCardsAndInducementsParameter) param).getTeamId(),
                    ((DialogBuyCardsAndInducementsParameter) param).getAvailableGold(),
                    new InducementSet(),
                    new String[0], new String[0],
                    new com.fumbbl.ffb.model.skill.Skill[0],
                    new String[0]);
                break;

            case BUY_PRAYERS_AND_INDUCEMENTS:
                comm.sendBuyInducements(
                    ((DialogBuyPrayersAndInducementsParameter) param).getTeamId(),
                    ((DialogBuyPrayersAndInducementsParameter) param).getAvailableGold(),
                    new InducementSet(),
                    new String[0], new String[0],
                    new com.fumbbl.ffb.model.skill.Skill[0],
                    new String[0]);
                break;

            case PETTY_CASH:
                comm.sendPettyCash(0);
                break;

            // ── Pass block / kickoff return (confirm) ─────────────────────────────

            case PASS_BLOCK:
            case KICKOFF_RETURN:
                comm.sendConfirm();
                break;

            case KICK_OFF_RESULT:
                comm.sendConfirm();
                break;

            // ── Use apothecaries / igors / mortuary (decline) ─────────────────────

            case USE_APOTHECARIES:
                comm.sendUseApothecaries(Collections.emptyList());
                break;

            case USE_IGORS:
                comm.sendUseIgors(Collections.emptyList());
                break;

            case USE_MORTUARY_ASSISTANTS:
                comm.sendConfirm();
                break;

            // ── Bribery and corruption (decline) ──────────────────────────────────

            case BRIBERY_AND_CORRUPTION_RE_ROLL:
                comm.sendUseReRoll(ReRolledActions.ARGUE_THE_CALL, null);
                break;

            // ── Select position (first min available) ─────────────────────────────

            case SELECT_POSITION: {
                com.fumbbl.ffb.dialog.DialogSelectPositionParameter sp =
                    (com.fumbbl.ffb.dialog.DialogSelectPositionParameter) param;
                List<String> positions = sp.getPositions();
                if (positions != null && !positions.isEmpty()) {
                    int count = Math.max(sp.getMinSelect(), 1);
                    count = Math.min(count, positions.size());
                    comm.sendPositionSelection(
                        positions.subList(0, count).toArray(new String[0]),
                        sp.getTeamId());
                }
                break;
            }

            // ── Select keyword ─────────────────────────────────────────────────────

            case SELECT_KEYWORD: {
                com.fumbbl.ffb.dialog.DialogSelectKeywordParameter sk =
                    (com.fumbbl.ffb.dialog.DialogSelectKeywordParameter) param;
                List<com.fumbbl.ffb.model.Keyword> keywords = sk.getKeywords();
                if (keywords != null && !keywords.isEmpty()) {
                    comm.sendKeywordSelection(sk.getPlayerId(),
                        Collections.singletonList(keywords.get(0)));
                }
                break;
            }

            // ── Select skill ───────────────────────────────────────────────────────

            case SELECT_SKILL: {
                com.fumbbl.ffb.dialog.DialogSelectSkillParameter ss =
                    (com.fumbbl.ffb.dialog.DialogSelectSkillParameter) param;
                List<com.fumbbl.ffb.model.skill.Skill> skills = ss.getSkills();
                if (skills != null && !skills.isEmpty()) {
                    comm.sendSkillSelection(ss.getPlayerId(), skills.get(0));
                }
                break;
            }

            // ── Swarming ───────────────────────────────────────────────────────────

            case SWARMING:
                comm.sendConfirm();
                break;

            // ── Winnings re-roll (decline) ─────────────────────────────────────────

            case WINNINGS_RE_ROLL:
                comm.sendUseReRoll(null, null);
                break;

            // ── Defender action (informational) ───────────────────────────────────

            case DEFENDER_ACTION:
                break;

            // ── Game statistics (confirm) ──────────────────────────────────────────

            case GAME_STATISTICS:
                comm.sendConfirm();
                break;

            default:
                System.err.println("[ScriptedStrategy] Unhandled dialog: " + param.getId());
                break;
        }
    }

    // ── Block die scoring ────────────────────────────────────────────────────────

    /**
     * Score each die in the block roll from the attacker's perspective.
     *
     * @param roll         array of die values (1-6 per die)
     * @param n            number of dice to consider
     * @param game         current game state
     * @param attackerPick true = attacker picks (maximize); false = defender picks (minimize scores → invert)
     * @return score array of length n
     */
    public static double[] scoreDice(int[] roll, int n, Game game, boolean attackerPick) {
        boolean attackerHasBlock    = false;
        boolean attackerHasWrestle  = false;
        boolean attackerHasJugg     = false;
        boolean attackerHasTackle   = false;
        boolean attackerHasStripBall = false;
        boolean defenderHasBlock    = false;
        boolean defenderHasDodge    = false;
        boolean defenderIsBallCarrier = false;

        Player<?> attacker = (game.getActingPlayer() != null) ? game.getActingPlayer().getPlayer() : null;
        Player<?> defender = game.getDefender();

        if (attacker != null) {
            attackerHasBlock    = attacker.hasSkillProperty(NamedProperties.preventFallOnBothDown);
            attackerHasWrestle  = attacker.hasSkillProperty(NamedProperties.canTakeDownPlayersWithHimOnBothDown);
            attackerHasJugg     = attacker.hasSkillProperty(NamedProperties.canConvertBothDownToPush);
            attackerHasTackle   = attacker.hasSkillProperty(NamedProperties.ignoreDefenderStumblesResult);
            attackerHasStripBall = attacker.hasSkillProperty(NamedProperties.forceOpponentToDropBallOnPushback);
        }
        if (defender != null) {
            defenderHasBlock    = defender.hasSkillProperty(NamedProperties.preventFallOnBothDown);
            defenderHasDodge    = defender.hasSkillProperty(NamedProperties.canRerollDodge);
            FieldCoordinate ball = game.getFieldModel().getBallCoordinate();
            FieldCoordinate defPos = game.getFieldModel().getPlayerCoordinate(defender);
            defenderIsBallCarrier = (ball != null && defPos != null && ball.equals(defPos));
        }

        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            int dieVal = roll[i];
            ActionScore as = attackerDieActionScore(dieVal,
                attackerHasBlock, attackerHasWrestle, attackerHasJugg, attackerHasTackle, attackerHasStripBall,
                defenderHasBlock, defenderHasDodge, defenderIsBallCarrier);
            // For defender picking: invert value so softmax peaks at worst outcome for attacker
            ActionScore effective = attackerPick ? as : new ActionScore(as.probability, -as.value, as.confidence);
            scores[i] = effective.softmaxScore();
        }
        return scores;
    }

    /**
     * Score a single block die result from the attacker's perspective using p×v×c.
     * Since the die is already rolled, probability = 1.0 for all results.
     * value ∈ [-1,1]: -1 = catastrophic turnover, +1 = best possible outcome.
     * confidence ∈ [0,1]: how certain we are of the value.
     */
    private static ActionScore attackerDieActionScore(int roll,
            boolean attackerHasBlock, boolean attackerHasWrestle, boolean attackerHasJugg,
            boolean attackerHasTackle, boolean attackerHasStripBall,
            boolean defenderHasBlock, boolean defenderHasDodge, boolean defenderIsBallCarrier) {

        BlockResult result = blockResultForRoll(roll);

        switch (result) {
            case POW:
                return new ActionScore(1.0, +0.80, 0.95);  // clean knockdown, clear outcome

            case POW_PUSHBACK:
                if (attackerHasTackle)  return new ActionScore(1.0, +0.70, 0.85); // Tackle ignores Dodge
                if (!defenderHasDodge)  return new ActionScore(1.0, +0.65, 0.80); // defender can't escape
                return new ActionScore(1.0, +0.35, 0.60);                          // defender may dodge away

            case BOTH_DOWN:
                if (attackerHasBlock && !defenderHasBlock) {
                    // Attacker stays up, defender falls — very good
                    return new ActionScore(1.0, +0.55, 0.85);
                }
                if (attackerHasBlock && defenderHasBlock) {
                    // Both stay standing — neutral stalemate
                    return new ActionScore(1.0, 0.0, 0.70);
                }
                if (attackerHasWrestle) {
                    // Both prone (Wrestle bypasses Block) — not our turnover
                    return new ActionScore(1.0, +0.30, 0.70);
                }
                if (attackerHasJugg) {
                    // Jugg converts to push on a blitz — mildly positive
                    return new ActionScore(1.0, +0.10, 0.70);
                }
                if (defenderIsBallCarrier) {
                    // We fall too (turnover), but ball scatters — bad, but some upside
                    return new ActionScore(1.0, -0.30, 0.80);
                }
                // No skills, no ball benefit — turnover, very bad
                return new ActionScore(1.0, -0.80, 0.95);

            case PUSHBACK:
                if (attackerHasStripBall && defenderIsBallCarrier) {
                    return new ActionScore(1.0, +0.55, 0.80); // strip ball on push
                }
                return new ActionScore(1.0, +0.20, 0.50);     // safe push, situational value

            case SKULL:
                return new ActionScore(1.0, -1.0, 1.0);       // attacker down, our turnover — worst possible

            default:
                return new ActionScore(1.0, +0.20, 0.50);
        }
    }

    private static BlockResult blockResultForRoll(int roll) {
        switch (roll) {
            case 1: return BlockResult.SKULL;
            case 2: return BlockResult.BOTH_DOWN;
            case 5: return BlockResult.POW_PUSHBACK;
            case 6: return BlockResult.POW;
            default: return BlockResult.PUSHBACK; // 3, 4
        }
    }

    // ── Touchback / High Kick player selection ────────────────────────────────────

    /**
     * Find the coordinate of the best home player to receive the ball.
     * Uses ballCarrierScore to rate each candidate.
     */
    public static FieldCoordinate findBestBallCarrierCoord(Game game) {
        Player<?> best = findBestBallCarrier(game);
        if (best == null) {
            return null;
        }
        return game.getFieldModel().getPlayerCoordinate(best);
    }

    /**
     * Find the home player most suited to carry the ball.
     * Scores on MA, Dodge, Sure Hands, Block, Side Step, ST, Catch.
     */
    public static Player<?> findBestBallCarrier(Game game) {
        Team homeTeam = game.getTeamHome();
        if (homeTeam == null) {
            return null;
        }
        FieldModel fieldModel = game.getFieldModel();
        Player<?> best = null;
        int bestScore = -1;

        for (Player<?> player : homeTeam.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(player);
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            if (ps == null || coord == null || !ps.isActive()) {
                continue;
            }
            int score = ballCarrierScore(player);
            if (score > bestScore) {
                bestScore = score;
                best = player;
            }
        }
        return best;
    }

    /**
     * Score a player's suitability as a ball carrier.
     */
    public static int ballCarrierScore(Player<?> p) {
        int score = 0;
        score += p.getMovementWithModifiers() * 5;
        if (p.hasSkillProperty(NamedProperties.canRerollDodge))                    score += 25; // Dodge
        if (p.hasSkillProperty(NamedProperties.ignoreTacklezonesWhenPickingUp))    score += 20; // Sure Hands
        if (p.hasSkillProperty(NamedProperties.preventFallOnBothDown))             score += 15; // Block
        if (p.hasSkillProperty(NamedProperties.canChooseOwnPushedBackSquare))      score += 10; // Side Step
        score += p.getStrengthWithModifiers() * 2;
        if (p.hasSkillProperty(NamedProperties.canAttemptCatchInAdjacentSquares))  score += 5;  // Catch
        return score;
    }

    // ── Setup / formation helpers ─────────────────────────────────────────────────

    /**
     * Choose the best setup name for the current game situation.
     * Prefers "receive"/"offense"/"attack" when receiving, "kick"/"defense" when kicking.
     * Falls back to first available setup.
     */
    private static String chooseSetup(String[] names, Game game) {
        boolean receiving = isReceivingTeam(game);
        // Score each setup name by how well it matches the situation
        double[] scores = new double[names.length];
        for (int i = 0; i < names.length; i++) {
            String lower = names[i].toLowerCase();
            if (receiving) {
                if (lower.contains("receiv") || lower.contains("offense") || lower.contains("attack")) {
                    scores[i] = 90;
                } else if (lower.contains("kick") || lower.contains("defense") || lower.contains("defence")) {
                    scores[i] = 20;
                } else {
                    scores[i] = 50;
                }
            } else {
                if (lower.contains("kick") || lower.contains("defense") || lower.contains("defence")) {
                    scores[i] = 90;
                } else if (lower.contains("receiv") || lower.contains("offense") || lower.contains("attack")) {
                    scores[i] = 20;
                } else {
                    scores[i] = 50;
                }
            }
        }
        int chosen = pick(scores, T_BINARY);
        return names[chosen];
    }

    /**
     * Determine whether the home team is currently receiving (not kicking).
     * Heuristic: if any away-team players are already placed on the field,
     * the away team set up first (they kick), meaning we receive.
     */
    public static boolean isReceivingTeam(Game game) {
        Team awayTeam = game.getTeamAway();
        FieldModel fieldModel = game.getFieldModel();
        if (awayTeam == null || fieldModel == null) {
            return false;
        }
        for (Player<?> p : awayTeam.getPlayers()) {
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(p);
            if (coord != null && coord.getX() >= 1 && coord.getX() <= FieldCoordinate.FIELD_WIDTH) {
                return true;
            }
        }
        return false;
    }

    // ── Apothecary helpers ────────────────────────────────────────────────────────

    /**
     * Returns true if newSI represents a less severe injury than oldSI.
     * Severity: null (BH) < MNG < NI < stat reduction < DEAD
     */
    private static boolean isNewInjuryBetter(SeriousInjury newSi, SeriousInjury oldSi) {
        return injurySeverity(newSi) < injurySeverity(oldSi);
    }

    private static int injurySeverity(SeriousInjury si) {
        if (si == null) return 0;                  // BH / healed
        if (si.isDead()) return 10;                // DEAD
        // Use name-based detection for common cases
        String name = si.getName();
        if (name != null && name.contains("MNG")) return 2;   // Miss Next Game
        if (name != null && name.contains("NI"))  return 3;   // Niggling Injury
        return 5;                                             // stat reduction (NI, AV, MA, etc.)
    }
}
