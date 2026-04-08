package com.fumbbl.sgo.game;

/**
 * Immutable action for Stochastic Go.
 * Singletons are pre-created: PLACE_ACTIONS[cell] and END_TURN.
 * id 0..(TOTAL_CELLS-1) = place at cell (row*BOARD_SIZE+col), id TOTAL_CELLS = end_turn.
 */
public final class SGoAction {

    public static final int END_TURN_ID = SGoState.TOTAL_CELLS;

    public static final SGoAction END_TURN = new SGoAction(END_TURN_ID, -1, -1);

    /** Indexed by cell = row*BOARD_SIZE+col. */
    public static final SGoAction[] PLACE_ACTIONS;

    static {
        PLACE_ACTIONS = new SGoAction[SGoState.TOTAL_CELLS];
        for (int row = 0; row < SGoState.BOARD_SIZE; row++) {
            for (int col = 0; col < SGoState.BOARD_SIZE; col++) {
                int cell = row * SGoState.BOARD_SIZE + col;
                PLACE_ACTIONS[cell] = new SGoAction(cell, row, col);
            }
        }
    }

    public final int id;
    public final int row;  // -1 for END_TURN
    public final int col;  // -1 for END_TURN

    private SGoAction(int id, int row, int col) {
        this.id = id;
        this.row = row;
        this.col = col;
    }

    public boolean isPlace() {
        return id != END_TURN_ID;
    }

    @Override
    public String toString() {
        if (id == END_TURN_ID) return "end_turn";
        return "place(" + row + "," + col + ")";
    }
}
