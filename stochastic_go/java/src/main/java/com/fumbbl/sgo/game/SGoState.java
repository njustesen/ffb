package com.fumbbl.sgo.game;

/**
 * Stochastic Go game state.
 *
 * board: flat int[64], board[row*8+col] in {EMPTY=0, P1=1, P2=2}.
 * emptyCells: 64-bit bitmask, bit i set means cell i is empty.
 * stateHash: incrementally maintained Zobrist hash.
 */
public final class SGoState {

    public static final int EMPTY = 0;
    public static final int P1 = 1;
    public static final int P2 = 2;
    public static final int BOARD_SIZE = 5;
    public static final int TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE;
    public static final int TOTAL_TURNS = 20;

    public final int[] board;          // [row*BOARD_SIZE+col]
    public int currentPlayer;
    public int p1TurnsRemaining;
    public int p2TurnsRemaining;
    public boolean isTurnEnd;
    public long emptyCells;            // bitmask: bit i = cell i is empty
    public long stateHash;

    private SGoState(int[] board, int currentPlayer, int p1TurnsRemaining,
                     int p2TurnsRemaining, boolean isTurnEnd, long emptyCells, long stateHash) {
        this.board = board;
        this.currentPlayer = currentPlayer;
        this.p1TurnsRemaining = p1TurnsRemaining;
        this.p2TurnsRemaining = p2TurnsRemaining;
        this.isTurnEnd = isTurnEnd;
        this.emptyCells = emptyCells;
        this.stateHash = stateHash;
    }

    public static SGoState initial() {
        int[] board = new int[TOTAL_CELLS];
        long empty = (1L << TOTAL_CELLS) - 1L; // only bits 0..TOTAL_CELLS-1
        SGoState s = new SGoState(board, P1, TOTAL_TURNS, TOTAL_TURNS, false, empty, 0L);
        s.stateHash = s.computeHash();
        return s;
    }

    public SGoState clone() {
        int[] newBoard = new int[TOTAL_CELLS];
        System.arraycopy(board, 0, newBoard, 0, TOTAL_CELLS);
        return new SGoState(newBoard, currentPlayer, p1TurnsRemaining, p2TurnsRemaining,
                isTurnEnd, emptyCells, stateHash);
    }

    public boolean isTerminal() {
        return p1TurnsRemaining == 0 && p2TurnsRemaining == 0;
    }

    public int score() {
        int p1 = 0, p2 = 0;
        for (int i = 0; i < TOTAL_CELLS; i++) {
            if (board[i] == P1) p1++;
            else if (board[i] == P2) p2++;
        }
        return p1 - p2;
    }

    /** Recompute full Zobrist hash (used for testing). */
    public long computeHash() {
        long h = 0L;
        for (int i = 0; i < TOTAL_CELLS; i++) {
            int piece = board[i];
            if (piece != EMPTY) {
                h ^= Zobrist.BOARD[i * 3 + piece];
            }
        }
        h ^= Zobrist.CURRENT_PLAYER[currentPlayer];
        h ^= Zobrist.TURN_END[isTurnEnd ? 1 : 0];
        h ^= Zobrist.P1_TURNS[p1TurnsRemaining];
        h ^= Zobrist.P2_TURNS[p2TurnsRemaining];
        return h;
    }
}
