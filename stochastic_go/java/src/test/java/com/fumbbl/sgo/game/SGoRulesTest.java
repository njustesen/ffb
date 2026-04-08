package com.fumbbl.sgo.game;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SGoRulesTest {

    @Test
    void initialHashConsistency() {
        SGoState s = SGoState.initial();
        assertEquals(s.stateHash, s.computeHash());
    }

    @Test
    void initialStateProperties() {
        SGoState s = SGoState.initial();
        assertEquals(SGoState.P1, s.currentPlayer);
        assertEquals(SGoState.TOTAL_TURNS, s.p1TurnsRemaining);
        assertEquals(SGoState.TOTAL_TURNS, s.p2TurnsRemaining);
        assertFalse(s.isTurnEnd);
        assertEquals((1L << SGoState.TOTAL_CELLS) - 1L, s.emptyCells); // all TOTAL_CELLS bits set
        assertFalse(s.isTerminal());
    }

    @Test
    void placementSuccessRules() {
        // roll 1 always fails
        assertFalse(SGoRules.placementSuccess(1, 0));
        assertFalse(SGoRules.placementSuccess(1, 5));
        // roll 6 always succeeds
        assertTrue(SGoRules.placementSuccess(6, 5));
        assertTrue(SGoRules.placementSuccess(6, 0));
        // roll > k succeeds
        assertTrue(SGoRules.placementSuccess(4, 3));
        assertTrue(SGoRules.placementSuccess(3, 2));
        // roll <= k fails (and roll != 1 or 6)
        assertFalse(SGoRules.placementSuccess(3, 3));
        assertFalse(SGoRules.placementSuccess(2, 3));
        assertFalse(SGoRules.placementSuccess(4, 4));
    }

    @Test
    void adjacentFriendlyCount() {
        SGoState s = SGoState.initial();
        assertEquals(0, SGoRules.adjacentFriendlyCount(s.board, 0, SGoState.P1));
        s.board[1] = SGoState.P1;
        assertEquals(1, SGoRules.adjacentFriendlyCount(s.board, 0, SGoState.P1));
    }

    @Test
    void adjacentOpponentCountEmptyBoard() {
        SGoState s = SGoState.initial();
        assertEquals(0, SGoRules.adjacentOpponentCount(s.board, 0, SGoState.P1));
        // Cell 12 = center of 5x5 board (row=2,col=2)
        assertEquals(0, SGoRules.adjacentOpponentCount(s.board, 12, SGoState.P1));
    }

    @Test
    void adjacentOpponentCountWithOpponents() {
        SGoState s = SGoState.initial();
        // Place P2 at cell 1 (row=0,col=1 in 5x5)
        s.board[1] = SGoState.P2;
        // Cell 0 (row=0,col=0) has neighbors: 1,5,6 — P2 at cell 1 → count=1
        assertEquals(1, SGoRules.adjacentOpponentCount(s.board, 0, SGoState.P1));
        // Cell 6 (row=1,col=1) has neighbors: 0,1,2,5,7,10,11,12 — P2 at cell 1 → count=1
        assertEquals(1, SGoRules.adjacentOpponentCount(s.board, 6, SGoState.P1));
    }

    @Test
    void applyPlacementSuccess() {
        SGoState s = SGoState.initial();
        // Roll 6 always succeeds; k_dice=0 on empty board
        SGoState next = SGoRules.applyPlacement(s, 0, 6);
        assertEquals(SGoState.P1, next.board[0]);
        assertFalse(next.isTurnEnd);
        assertTrue((next.emptyCells & 1L) == 0); // cell 0 cleared
        // Hash updated correctly
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyPlacementFailureEndsTurn() {
        SGoState s = SGoState.initial();
        // Roll 1 always fails
        SGoState next = SGoRules.applyPlacement(s, 0, 1);
        assertEquals(SGoState.EMPTY, next.board[0]);
        assertTrue(next.isTurnEnd);
        assertTrue((next.emptyCells & 1L) != 0); // cell 0 still empty
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyPlacementNonFumbleFailureCapturesAndEndsTurn() {
        // 5x5 board: cell 10 = row=2,col=0; neighbors are 5,6,11,15,16.
        // Set up: P1 tries cell 10 with 2 P2 neighbors (k_opp=2, k_fri=0 → k_dice=max(0,3+2-0)=5).
        // roll=2 <= k_dice=5 AND roll!=1 → non-fumble failure → capture + turn ends.
        // Lowest-index P2 neighbor of cell 10 = cell 5.
        SGoState s = SGoState.initial();
        s.board[5] = SGoState.P2;
        s.board[6] = SGoState.P2;
        s.emptyCells &= ~(1L << 5);
        s.emptyCells &= ~(1L << 6);
        s.stateHash = s.computeHash();

        SGoState next = SGoRules.applyPlacement(s, 10, 2);
        assertEquals(SGoState.EMPTY, next.board[10]); // piece NOT placed
        assertTrue(next.isTurnEnd);                   // turn ends
        assertEquals(SGoState.EMPTY, next.board[5]);  // P2 stone at cell 5 captured
        assertEquals(SGoState.P2, next.board[6]);     // P2 stone at cell 6 untouched
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyPlacementFumbleNoCapture() {
        // roll=1 is a fumble: turn ends but no capture occurs
        SGoState s = SGoState.initial();
        s.board[1] = SGoState.P2;
        s.emptyCells &= ~(1L << 1);
        s.stateHash = s.computeHash();

        SGoState next = SGoRules.applyPlacement(s, 0, 1);
        assertTrue(next.isTurnEnd);
        assertEquals(SGoState.P2, next.board[1]); // not captured (fumble, not non-fumble failure)
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyPlacementFriendlySupportReducesKDice() {
        // In 5x5, cell 0's neighbors include cells 1 and 5.
        // P1 has 2 friendlies adjacent to cell 0 (cells 1 and 5). No opponents.
        // k_opp=0, k_fri=2 → k_dice = max(0, 3+0-2) = 1.
        // Roll 2 should succeed (roll=2 > k_dice=1 and roll!=1).
        SGoState s = SGoState.initial();
        s.board[1] = SGoState.P1;
        s.board[5] = SGoState.P1;
        s.emptyCells &= ~(1L << 1);
        s.emptyCells &= ~(1L << 5);
        s.stateHash = s.computeHash();

        SGoState next = SGoRules.applyPlacement(s, 0, 2);
        assertEquals(SGoState.P1, next.board[0]);
        assertFalse(next.isTurnEnd);
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyEndTurn() {
        SGoState s = SGoState.initial();
        SGoState next = SGoRules.applyEndTurn(s);
        assertTrue(next.isTurnEnd);
        assertFalse(s.isTurnEnd); // original unchanged
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void advanceTurn() {
        SGoState s = SGoState.initial();
        SGoState ended = SGoRules.applyEndTurn(s);
        SGoState advanced = SGoRules.advanceTurn(ended);
        assertEquals(SGoState.P2, advanced.currentPlayer);
        assertEquals(SGoState.TOTAL_TURNS - 1, advanced.p1TurnsRemaining);
        assertEquals(SGoState.TOTAL_TURNS, advanced.p2TurnsRemaining);
        assertFalse(advanced.isTurnEnd);
        assertEquals(advanced.stateHash, advanced.computeHash());
    }

    @Test
    void terminalConditionTurnsExhausted() {
        SGoState s = SGoState.initial();
        s.p1TurnsRemaining = 0;
        s.p2TurnsRemaining = 0;
        assertTrue(s.isTerminal());
    }

    @Test
    void terminalConditionBoardFullNoLongerTerminal() {
        // Captures prevent permanent board fill, so board-full is no longer terminal
        SGoState s = SGoState.initial();
        s.emptyCells = 0L;
        assertFalse(s.isTerminal());
    }

    @Test
    void scoring() {
        SGoState s = SGoState.initial();
        assertEquals(0, s.score());
        s.board[0] = SGoState.P1;
        s.board[1] = SGoState.P1;
        s.board[2] = SGoState.P2;
        assertEquals(1, s.score());
    }

    @Test
    void hashDeterminism() {
        SGoState a = SGoState.initial();
        SGoState b = SGoState.initial();
        assertEquals(a.stateHash, b.stateHash);
        assertEquals(a.computeHash(), b.computeHash());
    }

    @Test
    void incrementalHashConsistency() {
        SGoState s = SGoState.initial();
        // Place P1 at cell 5 with roll 6 (success)
        SGoState next = SGoRules.applyPlacement(s, 5, 6);
        assertEquals(next.stateHash, next.computeHash(),
                "Incremental hash must match full recompute after placement");

        // End turn
        SGoState ended = SGoRules.applyEndTurn(next);
        assertEquals(ended.stateHash, ended.computeHash(),
                "Incremental hash must match full recompute after end_turn");

        // Advance turn
        SGoState advanced = SGoRules.advanceTurn(ended);
        assertEquals(advanced.stateHash, advanced.computeHash(),
                "Incremental hash must match full recompute after advance_turn");
    }

    @Test
    void winProb() {
        SGoState s = SGoState.initial();
        double wp = SGoRules.winProb(s);
        assertEquals(0.5, wp, 1e-9, "Empty board: win prob should be 0.5");

        s.board[0] = SGoState.P1;
        double wpP1Ahead = SGoRules.winProb(s);
        assertTrue(wpP1Ahead > 0.5, "P1 ahead: win prob > 0.5");

        s.board[0] = SGoState.P2;
        double wpP2Ahead = SGoRules.winProb(s);
        assertTrue(wpP2Ahead < 0.5, "P2 ahead: win prob < 0.5");
    }
}
