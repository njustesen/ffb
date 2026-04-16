package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.FactoryType;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.server.factory.SequenceGeneratorFactory;
import com.fumbbl.ffb.server.step.generator.Select;
import com.fumbbl.ffb.server.step.generator.SequenceGenerator;
import com.fumbbl.ffb.server.util.UtilSkillBehaviours;

/**
 * Bootstraps a mid-game {@link GameState} from a cloned {@link Game} model,
 * ready to accept a player-activation command at {@code INIT_SELECTING}.
 *
 * <p>This is the foundation for MCTS rollouts: clone the live game, push the
 * {@code Select} sequence, inject the candidate action, then drive simulation
 * with the scripted strategy for a fixed number of turns.
 */
public final class RolloutSetup {

    private RolloutSetup() {}

    /**
     * Create a {@link GameState} from a JSON-cloned mid-game snapshot.
     *
     * <p>The returned state has the {@code Select} sequence on the step stack
     * so that the first {@code handleCommand()} call will advance to
     * {@code INIT_SELECTING}, ready to accept a {@code ClientCommandActingPlayer}.
     *
     * @param sourceGame game model to clone (the live game, not yet mutated)
     * @param server     shared headless server instance
     * @return a fresh {@link GameState} at the {@code INIT_SELECTING} entry point
     */
    public static GameState createFromMidGame(Game sourceGame,
                                              HeadlessFantasyFootballServer server) {
        // 1. Deep-clone the Game model via JSON round-trip.
        GameSimulator simulator = new GameSimulator(
            server.getFactorySource(), server.getFactoryManager());
        Game clone = simulator.cloneGame(sourceGame);

        // 2. Re-initialize transient rule state that is not in the JSON snapshot
        //    (ModifierAggregator, EnhancementRegistry, skill rule tables).
        clone.initializeRules();
        UtilSkillBehaviours.registerBehaviours(clone, server.getDebugLog());

        // 3. Wrap in a fresh GameState and replace its stub game with the clone.
        GameState gameState = new GameState(server);
        gameState.setGame(clone);

        // 4. Create the StepFactory now that the real game is attached.
        gameState.initRulesDependentMembers();

        // 5. Push the Select sequence and prime INIT_SELECTING as the current step.
        //    The GameState's step stack is fresh (empty — we only cloned the Game model),
        //    so there is no stale stack state from the source game.
        SequenceGeneratorFactory seqFactory =
            clone.getFactory(FactoryType.Factory.SEQUENCE_GENERATOR);
        ((Select) seqFactory.forName(SequenceGenerator.Type.Select.name()))
            .pushSequence(new Select.SequenceParams(gameState, false));

        // Pop INIT_SELECTING from the stack and call start() so that
        // getCurrentStep() returns it and the first handleCommand() call works.
        gameState.startNextStep();

        return gameState;
    }

    /**
     * Reset a previously created rollout {@link GameState} for the next MCTS
     * iteration by restoring the {@link GameSnapshot} and re-pushing the
     * {@code Select} sequence.
     *
     * <p>This is ~50× faster than calling {@link #createFromMidGame} again: the
     * expensive JSON clone + {@code initializeRules()} + {@code initRulesDependentMembers()}
     * are done only once per {@code selectActivation}; each iteration just restores
     * the mutable fields (O(n_players)) and resets the step stack.
     *
     * @param gameState the rollout state to reset (originally from {@link #createFromMidGame})
     * @param snapshot  the snapshot taken immediately after {@code createFromMidGame}
     */
    public static void resetForIteration(GameState gameState, GameSnapshot snapshot) {
        Game rolloutGame = gameState.getGame();

        // 1. Restore mutable game model fields from the snapshot.
        snapshot.restore(rolloutGame);

        // 2. Reset the step stack to empty.
        gameState.getStepStack().clear();

        // 3. Re-push the Select sequence so the next step is INIT_SELECTING.
        SequenceGeneratorFactory seqFactory =
            rolloutGame.getFactory(FactoryType.Factory.SEQUENCE_GENERATOR);
        ((Select) seqFactory.forName(SequenceGenerator.Type.Select.name()))
            .pushSequence(new Select.SequenceParams(gameState, false));

        // 4. Pop INIT_SELECTING from the stack so getCurrentStep() returns it.
        gameState.startNextStep();
    }

    /**
     * Sync a persistent rollout {@link GameState} to the current live game state
     * without re-running the expensive JSON clone + rule initialization.
     *
     * <p>This is the key speed optimization for MCTS: instead of calling
     * {@link #createFromMidGame} (29 ms) at the start of every decision, the caller
     * creates the rollout state <em>once</em> (per game) and calls this method (~0.1 ms)
     * at each new decision point to bring the rollout game in sync with the live game.
     *
     * <p>Safety: The rollout game's player objects are <em>separate</em> from the live
     * game's players (they were cloned at construction time).  {@link GameSnapshot} uses
     * player IDs to match, so restoring a snapshot taken from {@code liveGame} into
     * {@code rolloutGameState} is safe.
     *
     * @param rolloutGameState  the persistent rollout state (created once with
     *                          {@link #createFromMidGame})
     * @param liveGame          the current live game whose mutable state should be
     *                          reflected in the rollout game
     * @return a fresh {@link GameSnapshot} of the synced rollout game, ready for use
     *         with {@link #resetForIteration} in the MCTS iteration loop
     */
    public static GameSnapshot syncFromLiveGame(GameState rolloutGameState, Game liveGame) {
        // Take a snapshot of the live game's current mutable state.
        GameSnapshot liveSnapshot = GameSnapshot.take(liveGame);

        // Apply it to the rollout game (player IDs match since the rollout game is a
        // clone of the same game; positions/states/scores/turn-data are updated).
        liveSnapshot.restore(rolloutGameState.getGame());

        // Reset the step stack to INIT_SELECTING (same as resetForIteration steps 2–4).
        rolloutGameState.getStepStack().clear();
        SequenceGeneratorFactory seqFactory =
            rolloutGameState.getGame().getFactory(FactoryType.Factory.SEQUENCE_GENERATOR);
        ((Select) seqFactory.forName(SequenceGenerator.Type.Select.name()))
            .pushSequence(new Select.SequenceParams(rolloutGameState, false));
        rolloutGameState.startNextStep();

        // Return a snapshot of the now-synced rollout game for use as the per-iteration base.
        return GameSnapshot.take(rolloutGameState.getGame());
    }
}
