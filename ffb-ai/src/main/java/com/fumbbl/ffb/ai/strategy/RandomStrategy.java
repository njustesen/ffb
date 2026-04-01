package com.fumbbl.ffb.ai.strategy;

import com.fumbbl.ffb.ApothecaryType;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerChoiceMode;
import com.fumbbl.ffb.ReRollSource;
import com.fumbbl.ffb.ReRollSources;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.bb2020.InjuryDescription;
import com.fumbbl.ffb.client.net.ClientCommunication;
import com.fumbbl.ffb.dialog.DialogApothecaryChoiceParameter;
import com.fumbbl.ffb.dialog.DialogArgueTheCallParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPartialReRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogBriberyAndCorruptionParameter;
import com.fumbbl.ffb.dialog.DialogBribesParameter;
import com.fumbbl.ffb.dialog.DialogBuyInducementsParameter;
import com.fumbbl.ffb.dialog.DialogBuyCardsAndInducementsParameter;
import com.fumbbl.ffb.dialog.DialogBuyCardsParameter;
import com.fumbbl.ffb.dialog.DialogBuyPrayersAndInducementsParameter;
import com.fumbbl.ffb.dialog.DialogConfirmEndActionParameter;
import com.fumbbl.ffb.dialog.DialogFollowupChoiceParameter;
import com.fumbbl.ffb.dialog.DialogInformationOkayParameter;
import com.fumbbl.ffb.dialog.DialogInterceptionParameter;
import com.fumbbl.ffb.dialog.DialogInvalidSolidDefenceParameter;
import com.fumbbl.ffb.dialog.DialogJourneymenParameter;
import com.fumbbl.ffb.dialog.DialogKickOffResultParameter;
import com.fumbbl.ffb.dialog.DialogKickSkillParameter;
import com.fumbbl.ffb.dialog.DialogKickoffReturnParameter;
import com.fumbbl.ffb.dialog.DialogOpponentBlockSelectionParameter;
import com.fumbbl.ffb.dialog.DialogOpponentBlockSelectionPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogPassBlockParameter;
import com.fumbbl.ffb.dialog.DialogPenaltyShootoutParameter;
import com.fumbbl.ffb.dialog.DialogPettyCashParameter;
import com.fumbbl.ffb.dialog.DialogPickUpChoiceParameter;
import com.fumbbl.ffb.dialog.DialogPileDriverParameter;
import com.fumbbl.ffb.dialog.DialogPilingOnParameter;
import com.fumbbl.ffb.dialog.DialogPlayerChoiceParameter;
import com.fumbbl.ffb.dialog.DialogPuntToCrowdParameter;
import com.fumbbl.ffb.dialog.DialogReRollBlockForTargetsParameter;
import com.fumbbl.ffb.dialog.DialogReRollForTargetsParameter;
import com.fumbbl.ffb.dialog.DialogReRollParameter;
import com.fumbbl.ffb.dialog.DialogReRollPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogReceiveChoiceParameter;
import com.fumbbl.ffb.dialog.DialogSelectKeywordParameter;
import com.fumbbl.ffb.dialog.DialogSelectPositionParameter;
import com.fumbbl.ffb.dialog.DialogSelectSkillParameter;
import com.fumbbl.ffb.dialog.DialogSetupErrorParameter;
import com.fumbbl.ffb.dialog.DialogSkillUseParameter;
import com.fumbbl.ffb.dialog.DialogStartGameParameter;
import com.fumbbl.ffb.dialog.DialogSwarmingErrorParameter;
import com.fumbbl.ffb.dialog.DialogSwarmingPlayersParameter;
import com.fumbbl.ffb.dialog.DialogTeamSetupParameter;
import com.fumbbl.ffb.dialog.DialogTouchbackParameter;
import com.fumbbl.ffb.dialog.DialogUseApothecariesParameter;
import com.fumbbl.ffb.dialog.DialogUseApothecaryParameter;
import com.fumbbl.ffb.dialog.DialogUseChainsawParameter;
import com.fumbbl.ffb.dialog.DialogUseIgorParameter;
import com.fumbbl.ffb.dialog.DialogUseIgorsParameter;
import com.fumbbl.ffb.dialog.DialogUseInducementParameter;
import com.fumbbl.ffb.dialog.DialogUseMortuaryAssistantParameter;
import com.fumbbl.ffb.dialog.DialogUseMortuaryAssistantsParameter;
import com.fumbbl.ffb.dialog.DialogWinningsReRollParameter;
import com.fumbbl.ffb.dialog.DialogWizardSpellParameter;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.InducementSet;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Stateless random decision-making for all server-sent dialog types.
 *
 * Each dialog is handled by casting the {@link IDialogParameter} and calling
 * the appropriate method on {@link ClientCommunication}.  Where an action has
 * no meaningful "random" interpretation (inducements, Igor, Mortuary) we
 * conservatively decline.
 */
public final class RandomStrategy {

    private static final Random RANDOM = new Random();

    private RandomStrategy() {
        // static utility class
    }

    public static void respondToDialog(IDialogParameter param, Game game, ClientCommunication comm) {
        if (param == null) {
            return;
        }
        switch (param.getId()) {

            // ── Binary yes/no choices ──────────────────────────────────────────
            case COIN_CHOICE:
                comm.sendCoinChoice(RANDOM.nextBoolean());
                break;

            case RECEIVE_CHOICE:
                comm.sendReceiveChoice(RANDOM.nextBoolean());
                break;

            case FOLLOWUP_CHOICE:
                comm.sendFollowupChoice(RANDOM.nextBoolean());
                break;

            case PICK_UP_CHOICE:
                comm.sendPickUpChoice(RANDOM.nextBoolean());
                break;

            case PUNT_TO_CROWD:
                comm.sendPuntToCrowd(RANDOM.nextBoolean());
                break;

            case USE_CHAINSAW:
                comm.sendUseChainsaw(RANDOM.nextBoolean());
                break;

            case BLOODLUST_ACTION:
                comm.sendChangeBloodlustAction(RANDOM.nextBoolean());
                break;

            // Conservatively decline risky / costly options
            case PILING_ON: {
                // Decline piling on — find the piling on skill on the player and send false
                DialogPilingOnParameter pilingOn = (DialogPilingOnParameter) param;
                Player<?> pilingOnPlayer = game.getPlayerById(pilingOn.getPlayerId());
                if (pilingOnPlayer != null) {
                    com.fumbbl.ffb.model.property.NamedProperties.canPileOnOpponent.getName();
                    com.fumbbl.ffb.util.UtilCards.getSkillWithProperty(pilingOnPlayer,
                        com.fumbbl.ffb.model.property.NamedProperties.canPileOnOpponent)
                        .ifPresent(skill -> comm.sendUseSkill(skill, false, pilingOn.getPlayerId()));
                }
                break;
            }

            case BRIBES: {
                // Decline bribes — send null player ID
                DialogBribesParameter bribes = (DialogBribesParameter) param;
                com.fumbbl.ffb.inducement.InducementType bribesType = null;
                // We have no factory here; send null inducement to decline
                comm.sendUseInducement(bribesType);
                break;
            }

            case KICK_SKILL: {
                // Decline kick skill — find the Kick skill on the player and send false
                DialogKickSkillParameter kick = (DialogKickSkillParameter) param;
                Player<?> kickPlayer = game.getPlayerById(kick.getPlayerId());
                if (kickPlayer != null) {
                    com.fumbbl.ffb.util.UtilCards.getSkillWithProperty(kickPlayer,
                        com.fumbbl.ffb.model.property.NamedProperties.canReduceKickDistance)
                        .ifPresent(skill -> comm.sendUseSkill(skill, false, kick.getPlayerId()));
                }
                break;
            }

            case USE_APOTHECARY: {
                DialogUseApothecaryParameter apo = (DialogUseApothecaryParameter) param;
                List<ApothecaryType> types = apo.getApothecaryTypes();
                ApothecaryType apoType = (types != null && !types.isEmpty()) ? types.get(0) : ApothecaryType.TEAM;
                comm.sendUseApothecary(apo.getPlayerId(), RANDOM.nextBoolean(), apoType, apo.getSeriousInjury());
                break;
            }

            case USE_IGOR:
                // Decline — send null player ID for Igor inducement
                comm.sendUseInducement((com.fumbbl.ffb.inducement.InducementType) null);
                break;

            case USE_MORTUARY_ASSISTANT:
                // Decline — send null player ID for Mortuary Assistant inducement
                comm.sendUseInducement((com.fumbbl.ffb.inducement.InducementType) null);
                break;

            // ── Block dice ─────────────────────────────────────────────────────
            case BLOCK_ROLL: {
                DialogBlockRollParameter block = (DialogBlockRollParameter) param;
                int nrOfDice = block.getNrOfDice();
                // Pick a random die index (0 to nrOfDice-1)
                comm.sendBlockChoice(RANDOM.nextInt(nrOfDice));
                break;
            }

            case BLOCK_ROLL_PARTIAL_RE_ROLL: {
                DialogBlockRollPartialReRollParameter partialBlock =
                    (DialogBlockRollPartialReRollParameter) param;
                // Just pick a die index
                comm.sendBlockChoice(RANDOM.nextInt(partialBlock.getNrOfDice()));
                break;
            }

            case BLOCK_ROLL_PROPERTIES: {
                DialogBlockRollPropertiesParameter propsBlock = (DialogBlockRollPropertiesParameter) param;
                int nDice = Math.abs(propsBlock.getNrOfDice());
                if (nDice > 0) {
                    comm.sendBlockChoice(RANDOM.nextInt(nDice));
                }
                break;
            }

            // ── Re-roll ─────────────────────────────────────────────────────────
            case RE_ROLL: {
                DialogReRollParameter rr = (DialogReRollParameter) param;
                // Randomly either decline or use the first available re-roll source
                boolean useRr = RANDOM.nextBoolean();
                if (useRr && rr.isTeamReRollOption()) {
                    comm.sendUseReRoll(rr.getReRolledAction(), ReRollSources.TEAM_RE_ROLL);
                } else if (useRr && rr.isProReRollOption()) {
                    comm.sendUseReRoll(rr.getReRolledAction(), ReRollSources.PRO);
                } else if (useRr && rr.getSingleUseReRollSource() != null) {
                    comm.sendUseReRoll(rr.getReRolledAction(), rr.getSingleUseReRollSource());
                } else if (useRr && rr.getReRollSkill() != null) {
                    comm.sendUseSkill(rr.getReRollSkill(), true, rr.getPlayerId(), rr.getReRolledAction());
                } else {
                    // Decline
                    comm.sendUseReRoll(rr.getReRolledAction(), null);
                }
                break;
            }

            case RE_ROLL_PROPERTIES: {
                DialogReRollPropertiesParameter rrp = (DialogReRollPropertiesParameter) param;
                // Randomly decline
                comm.sendUseReRoll(rrp.getReRolledAction(), null);
                break;
            }

            // ── Skill use ───────────────────────────────────────────────────────
            case SKILL_USE: {
                DialogSkillUseParameter su = (DialogSkillUseParameter) param;
                boolean use = RANDOM.nextBoolean();
                comm.sendUseSkill(su.getSkill(), use, su.getPlayerId());
                break;
            }

            // ── Player choice ───────────────────────────────────────────────────
            case PLAYER_CHOICE: {
                DialogPlayerChoiceParameter pc = (DialogPlayerChoiceParameter) param;
                String[] playerIds = pc.getPlayerIds();
                int maxSelects = Math.min(pc.getMaxSelects(), playerIds.length);
                int minSelects = pc.getMinSelects();
                // For solidDefence: always select 0 players (server immediately ends solid defence
                // when no players are selected, avoiding re-setup complexity).
                if (pc.getPlayerChoiceMode() == PlayerChoiceMode.SOLID_DEFENCE) {
                    comm.sendPlayerChoice(pc.getPlayerChoiceMode(), new Player[0]);
                    break;
                }
                // Pick between minSelects and maxSelects players randomly
                int numToSelect = minSelects + (maxSelects > minSelects
                    ? RANDOM.nextInt(maxSelects - minSelects + 1) : 0);
                numToSelect = Math.min(numToSelect, playerIds.length);
                List<String> shuffled = new ArrayList<>();
                for (String id : playerIds) {
                    shuffled.add(id);
                }
                Collections.shuffle(shuffled, RANDOM);
                Player<?>[] chosen = new Player[numToSelect];
                for (int i = 0; i < numToSelect; i++) {
                    chosen[i] = game.getPlayerById(shuffled.get(i));
                }
                comm.sendPlayerChoice(pc.getPlayerChoiceMode(), chosen);
                break;
            }

            // ── Touchback ───────────────────────────────────────────────────────
            case TOUCHBACK: {
                // Find a standing home player and send their coordinate
                FieldCoordinate coord = findStandingHomePlayerCoord(game);
                comm.sendTouchback(coord);
                break;
            }

            // ── Interception ────────────────────────────────────────────────────
            case INTERCEPTION:
                // Decline interception
                comm.sendInterceptorChoice(null, null);
                break;

            // ── Apothecary choice ───────────────────────────────────────────────
            case APOTHECARY_CHOICE: {
                DialogApothecaryChoiceParameter ac = (DialogApothecaryChoiceParameter) param;
                // Accept the old state (i.e. keep original injury result, decline apothecary)
                comm.sendApothecaryChoice(ac.getPlayerId(), ac.getPlayerStateOld(), ac.getSeriousInjuryOld(), ac.getPlayerStateOld());
                break;
            }

            // ── Team setup ──────────────────────────────────────────────────────
            case TEAM_SETUP: {
                DialogTeamSetupParameter ts = (DialogTeamSetupParameter) param;
                String[] names = ts.getSetupNames();
                if (names != null && names.length > 0) {
                    // Load the saved setup — positions are applied server-side.
                    // sendEndTurn is NOT sent here; the AiDecisionEngine SETUP handler
                    // will detect that players are on the field and confirm in a later tick.
                    comm.sendTeamSetupLoad(names[0]);
                }
                // If no setups exist, we can only end with whatever is on the field
                break;
            }

            // ── Start game / confirm ─────────────────────────────────────────────
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

            // ── Argue the call ───────────────────────────────────────────────────
            case ARGUE_THE_CALL: {
                DialogArgueTheCallParameter atc = (DialogArgueTheCallParameter) param;
                String[] ids = atc.getPlayerIds();
                if (ids != null && ids.length > 0) {
                    comm.sendArgueTheCall(ids[RANDOM.nextInt(ids.length)]);
                } else {
                    comm.sendArgueTheCall((String) null);
                }
                break;
            }

            // ── Pile driver ──────────────────────────────────────────────────────
            case PILE_DRIVER: {
                DialogPileDriverParameter pd = (DialogPileDriverParameter) param;
                List<String> kd = pd.getKnockedDownPlayers();
                if (kd != null && !kd.isEmpty()) {
                    comm.sendPileDriver(kd.get(0));
                }
                break;
            }

            // ── Select position ──────────────────────────────────────────────────
            case SELECT_POSITION: {
                DialogSelectPositionParameter sp = (DialogSelectPositionParameter) param;
                List<String> positions = sp.getPositions();
                if (positions != null && !positions.isEmpty()) {
                    int count = Math.max(sp.getMinSelect(), 1);
                    count = Math.min(count, positions.size());
                    List<String> selected = new ArrayList<>(positions.subList(0, count));
                    comm.sendPositionSelection(selected.toArray(new String[0]), sp.getTeamId());
                }
                break;
            }

            // ── Select keyword ───────────────────────────────────────────────────
            case SELECT_KEYWORD: {
                DialogSelectKeywordParameter sk = (DialogSelectKeywordParameter) param;
                List<com.fumbbl.ffb.model.Keyword> keywords = sk.getKeywords();
                if (keywords != null && !keywords.isEmpty()) {
                    List<com.fumbbl.ffb.model.Keyword> selected = Collections.singletonList(keywords.get(0));
                    comm.sendKeywordSelection(sk.getPlayerId(), selected);
                }
                break;
            }

            // ── Select skill ─────────────────────────────────────────────────────
            case SELECT_SKILL: {
                DialogSelectSkillParameter ss = (DialogSelectSkillParameter) param;
                List<com.fumbbl.ffb.model.skill.Skill> skills = ss.getSkills();
                if (skills != null && !skills.isEmpty()) {
                    comm.sendSkillSelection(ss.getPlayerId(), skills.get(0));
                }
                break;
            }

            // ── Swarming ─────────────────────────────────────────────────────────
            case SWARMING:
                comm.sendConfirm();
                break;

            // ── Winnings re-roll ─────────────────────────────────────────────────
            case WINNINGS_RE_ROLL:
                // Decline
                comm.sendUseReRoll(null, null);
                break;

            // ── Inducements / buy ────────────────────────────────────────────────
            case BUY_INDUCEMENTS:
                // Send empty inducement purchase
                comm.sendBuyInducements(
                    ((DialogBuyInducementsParameter) param).getTeamId(),
                    ((DialogBuyInducementsParameter) param).getAvailableGold(),
                    new InducementSet(),
                    new String[0], new String[0],
                    new com.fumbbl.ffb.model.skill.Skill[0],
                    new String[0]);
                break;

            case BUY_CARDS:
                // Decline: send null card selection
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

            // ── Petty cash ───────────────────────────────────────────────────────
            case PETTY_CASH:
                comm.sendPettyCash(0);
                break;

            // ── Journeymen ───────────────────────────────────────────────────────
            case JOURNEYMEN: {
                DialogJourneymenParameter jm = (DialogJourneymenParameter) param;
                int nrSlots = jm.getNrOfSlots();
                String[] posIds = jm.getPositionIds();
                if (posIds != null && posIds.length > 0 && nrSlots > 0) {
                    int count = Math.min(nrSlots, posIds.length);
                    String[] selectedPositions = new String[count];
                    int[] slots = new int[count];
                    for (int i = 0; i < count; i++) {
                        selectedPositions[i] = posIds[0]; // pick same position for all
                        slots[i] = i;
                    }
                    comm.sendJourneymen(selectedPositions, slots);
                }
                break;
            }

            // ── Wizard spell ─────────────────────────────────────────────────────
            case WIZARD_SPELL:
                // Decline
                comm.sendConfirm();
                break;

            // ── Use inducement ───────────────────────────────────────────────────
            case USE_INDUCEMENT:
                // Decline — send null inducement type
                comm.sendUseInducement((com.fumbbl.ffb.inducement.InducementType) null);
                break;

            // ── Pass block / kickoff return ──────────────────────────────────────
            case PASS_BLOCK:
                comm.sendConfirm();
                break;

            case KICKOFF_RETURN:
                comm.sendConfirm();
                break;

            // ── Kickoff result ───────────────────────────────────────────────────
            case KICK_OFF_RESULT:
                // The server expects sendKickOffResultChoice but we don't know the available
                // results from the dialog alone.  sendConfirm() is the safe fallback.
                comm.sendConfirm();
                break;

            // ── Re-roll for targets ──────────────────────────────────────────────
            case RE_ROLL_FOR_TARGETS: {
                DialogReRollForTargetsParameter rrft = (DialogReRollForTargetsParameter) param;
                // Decline re-roll for all targets
                comm.sendUseReRoll(rrft.getReRolledAction(), null);
                break;
            }

            case RE_ROLL_BLOCK_FOR_TARGETS: {
                DialogReRollBlockForTargetsParameter rrbft = (DialogReRollBlockForTargetsParameter) param;
                // Pick index 0 for the first block roll
                comm.sendBlockChoice(0);
                break;
            }

            // ── Opponent block selection ─────────────────────────────────────────
            case OPPONENT_BLOCK_SELECTION:
                comm.sendBlockChoice(0);
                break;

            case OPPONENT_BLOCK_SELECTION_PROPERTIES:
                comm.sendBlockChoice(0);
                break;

            // ── Use apothecaries / igors / mortuary ──────────────────────────────
            case USE_APOTHECARIES: {
                DialogUseApothecariesParameter uapo = (DialogUseApothecariesParameter) param;
                // Decline all — send empty list
                comm.sendUseApothecaries(Collections.emptyList());
                break;
            }

            case USE_IGORS: {
                // Decline all
                comm.sendUseIgors(Collections.emptyList());
                break;
            }

            case USE_MORTUARY_ASSISTANTS:
                // No sendUseMortuaryAssistants method in ClientCommunication — send confirm
                comm.sendConfirm();
                break;

            // ── Bribery and corruption ───────────────────────────────────────────
            case BRIBERY_AND_CORRUPTION_RE_ROLL:
                // Decline
                comm.sendUseReRoll(null, null);
                break;

            // ── Defender action (informational only) ─────────────────────────────
            case DEFENDER_ACTION:
                // No action required — informational dialog
                break;

            // ── Game statistics (informational) ──────────────────────────────────
            case GAME_STATISTICS:
                comm.sendConfirm();
                break;

            default:
                System.err.println("[RandomStrategy] Unhandled dialog: " + param.getId());
                break;
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────────

    private static FieldCoordinate findStandingHomePlayerCoord(Game game) {
        Team homeTeam = game.getTeamHome();
        if (homeTeam == null) {
            return null;
        }
        FieldModel fieldModel = game.getFieldModel();
        for (Player<?> player : homeTeam.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(player);
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            if (ps != null && coord != null && ps.isActive()) {
                return coord;
            }
        }
        return null;
    }
}
