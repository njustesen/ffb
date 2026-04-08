package com.fumbbl.ffb.ai.client;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MovePolicyStateTest {

    // ── Core mechanics ─────────────────────────────────────────────────────────

    @Test void initiallyDoesNotForceEnd() {
        assertFalse(new MovePolicyState().shouldEndNow());
    }

    @Test void forcesEndAfterMoveWithoutDice() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();
        assertTrue(s.shouldEndNow());
    }

    @Test void diceRollResetsConstraint() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();
        s.reset(); // any dice-roll dialog
        assertFalse(s.shouldEndNow());
    }

    @Test void multipleDiceRollsDuringPathStillGrantOnlyOneMove() {
        // e.g. dodge + re-roll during a single sendPlayerMove path
        MovePolicyState s = new MovePolicyState();
        s.recordMove();
        s.reset(); // dodge dialog
        s.reset(); // re-roll dialog
        assertFalse(s.shouldEndNow()); // one move window still open
        s.recordMove();
        assertTrue(s.shouldEndNow());
    }

    @Test void newActivationResetsConstraint() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();
        s.reset(); // handleSelectPlayer called for next player
        assertFalse(s.shouldEndNow());
    }

    // ── Game-sequence scenarios ────────────────────────────────────────────────

    /**
     * Blitz: player moves adjacent to target, block dice resolve (reset),
     * HitAndRun skill grants one more move.
     * Sequence: move → BLOCK_ROLL dialog (reset) → move → end
     */
    @Test void blitzMoveBlockDiceThenHitAndRunMove() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();                        // pre-block positioning move
        assertTrue(s.shouldEndNow());          // no more moves until dice
        s.reset();                             // BLOCK_ROLL dialog responded to
        assertFalse(s.shouldEndNow());         // HitAndRun window open
        s.recordMove();                        // post-block HitAndRun move
        assertTrue(s.shouldEndNow());          // done
    }

    /**
     * Foul with Sneaky Git / Dirty Player: player moves next to prone opponent,
     * armour/injury dice resolve (reset), player may reposition.
     * Sequence: move → ARMOUR_ROLL dialog (reset) → move → end
     */
    @Test void foulMoveThenInjuryDiceThenRepositionMove() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();                        // move to adjacent square
        s.reset();                             // ARMOUR_ROLL / INJURY_ROLL dialog
        assertFalse(s.shouldEndNow());         // foul resolved → bonus move
        s.recordMove();
        assertTrue(s.shouldEndNow());
    }

    /**
     * Pass with Give and Go: player moves, throws ball (pass roll + catch roll),
     * then may move again.
     * Sequence: move → PASS_ROLL dialog (reset) → CATCH_ROLL dialog (reset) → move → end
     */
    @Test void passRollThenGiveAndGoMove() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();                        // move before passing
        s.reset();                             // PASS_ROLL dialog
        s.reset();                             // CATCH_ROLL dialog (receiver)
        assertFalse(s.shouldEndNow());         // Give and Go window open
        s.recordMove();
        assertTrue(s.shouldEndNow());
    }

    /**
     * Hand-off: player moves, hands ball to adjacent teammate (catch roll),
     * may move again with Give and Go.
     * Sequence: move → CATCH_ROLL dialog (reset) → move → end
     */
    @Test void handOffCatchRollThenGiveAndGoMove() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();
        s.reset();                             // CATCH_ROLL dialog for hand-off
        assertFalse(s.shouldEndNow());
        s.recordMove();
        assertTrue(s.shouldEndNow());
    }

    /**
     * Ball pickup: player moves to ball square, PICK_UP_ROLL dialog fires (reset),
     * player may continue running with the ball.
     * Sequence: move → PICK_UP_ROLL dialog (reset) → move → end
     */
    @Test void pickUpRollThenContinueRunning() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();                        // move to ball square
        s.reset();                             // PICK_UP_ROLL dialog
        assertFalse(s.shouldEndNow());         // pickup bonus — keep running
        s.recordMove();                        // run with ball
        assertTrue(s.shouldEndNow());          // done
    }

    /**
     * Dodge during move: player enters a tackle zone, DODGE_ROLL fires (reset),
     * player may reposition one more square.
     * Sequence: move → DODGE_ROLL dialog (reset) → move → end
     */
    @Test void dodgeDuringMoveThenOneMoreStep() {
        MovePolicyState s = new MovePolicyState();
        s.recordMove();
        s.reset();                             // DODGE_ROLL dialog mid-path
        assertFalse(s.shouldEndNow());
        s.recordMove();
        assertTrue(s.shouldEndNow());
    }
}
