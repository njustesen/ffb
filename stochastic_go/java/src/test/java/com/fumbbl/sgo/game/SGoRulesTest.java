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
        long expectedEmpty = SGoState.TOTAL_CELLS < 64 ? (1L << SGoState.TOTAL_CELLS) - 1L : -1L;
        assertEquals(expectedEmpty, s.emptyCells); // all TOTAL_CELLS bits set
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
        // Cell 24 = (row=3,col=3) in 7x7
        assertEquals(0, SGoRules.adjacentOpponentCount(s.board, 24, SGoState.P1));
    }

    @Test
    void adjacentOpponentCountWithOpponents() {
        SGoState s = SGoState.initial();
        // Place P2 at cell 1 (row=0,col=1 in 8x8)
        s.board[1] = SGoState.P2;
        // Cell 0 (row=0,col=0) has neighbors: 1,8,9 — P2 at cell 1 → count=1
        assertEquals(1, SGoRules.adjacentOpponentCount(s.board, 0, SGoState.P1));
        // Cell 9 (row=1,col=1) has neighbors: 0,1,2,8,10,16,17,18 — P2 at cell 1 → count=1
        assertEquals(1, SGoRules.adjacentOpponentCount(s.board, 9, SGoState.P1));
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
    void applyPlacementNonFumbleFailureNoAdjacentAttackerStone() {
        // 8x8 board: cell 10 = row=1,col=2; neighbors include 9=(1,1) and 11=(1,3).
        // P1 tries cell 10 with 2 P2 neighbors (k_opp=2, k_fri=0 → k_dice=5).
        // roll=2 → non-fumble failure. P1 has NO adjacent own stones → just end turn.
        SGoState s = SGoState.initial();
        s.board[9] = SGoState.P2;
        s.board[11] = SGoState.P2;
        s.emptyCells &= ~(1L << 9);
        s.emptyCells &= ~(1L << 11);
        s.stateHash = s.computeHash();

        SGoState next = SGoRules.applyPlacement(s, 10, 2);
        assertEquals(SGoState.EMPTY, next.board[10]); // piece NOT placed
        assertTrue(next.isTurnEnd);                   // turn ends
        assertEquals(SGoState.P2, next.board[9]);     // P2 stones untouched (only attacker's own removed)
        assertEquals(SGoState.P2, next.board[11]);
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyPlacementNonFumbleFailureRemovesAttackerOwnStone() {
        // 8x8 board: P1 tries cell 10=(1,2) with 1 P2 neighbor (cell 9=(1,1)) and
        // 1 P1 own stone (cell 11=(1,3)). k_opp=1, k_fri=1 → k_dice=max(0,3+1-1)=3.
        // roll=2 ≤ k_dice=3 AND roll≠1 → non-fumble failure → P1 loses own stone at cell 11.
        SGoState s = SGoState.initial();
        s.board[9] = SGoState.P2;
        s.board[11] = SGoState.P1;
        s.emptyCells &= ~(1L << 9);
        s.emptyCells &= ~(1L << 11);
        s.stateHash = s.computeHash();

        SGoState next = SGoRules.applyPlacement(s, 10, 2);
        assertEquals(SGoState.EMPTY, next.board[10]); // piece NOT placed
        assertTrue(next.isTurnEnd);                   // turn ends
        assertEquals(SGoState.P2, next.board[9]);     // P2 stone untouched
        assertEquals(SGoState.EMPTY, next.board[11]); // P1's own stone removed
        assertTrue((next.emptyCells & (1L << 11)) != 0); // cell 11 now empty
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
        assertEquals(SGoState.P2, next.board[1]); // not captured
        assertEquals(next.stateHash, next.computeHash());
    }

    @Test
    void applyPlacementFriendlySupportReducesKDice() {
        // In 8x8, cell 0's neighbors include cells 1=(0,1) and 8=(1,0).
        // P1 has 2 friendlies adjacent to cell 0 (cells 1 and 8). No opponents.
        // k_opp=0, k_fri=2 → k_dice = max(0, 3+0-2) = 1.
        // Roll 2 should succeed (roll=2 > k_dice=1 and roll!=1).
        SGoState s = SGoState.initial();
        s.board[1] = SGoState.P1;
        s.board[8] = SGoState.P1;
        s.emptyCells &= ~(1L << 1);
        s.emptyCells &= ~(1L << 8);
        s.stateHash = s.computeHash();

        SGoState next = SGoRules.applyPlacement(s, 0, 2);
        assertEquals(SGoState.P1, next.board[0]); // placed
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
