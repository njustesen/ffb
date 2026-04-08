package com.fumbbl.ffb.ai.client;

/**
 * Tracks whether the active player has already moved this activation without
 * an intervening dice-roll dialog. Enforces the one-move-between-dice-rolls rule:
 * a player may only make consecutive move steps when a dice roll (pickup, dodge,
 * block, pass, catch, etc.) resets the constraint between them.
 */
class MovePolicyState {

    private boolean hasMoved = false;
    /** True after a block or foul command has been sent this activation. */
    private boolean hasActed = false;

    /**
     * Returns {@code true} when the player has moved and no dice-roll dialog has
     * been responded to since — the player must end their activation immediately.
     */
    boolean shouldEndNow() {
        return hasMoved;
    }

    /** Call when the AI sends a move command. */
    void recordMove() {
        hasMoved = true;
    }

    /** Call when the AI sends a block or foul command. */
    void recordAction() {
        hasActed = true;
    }

    /** True after a block/foul command was sent — prevents repeating visualization and command. */
    boolean hasActed() {
        return hasActed;
    }

    /**
     * Call after any dice-roll dialog response (pickup roll, dodge, block dice,
     * pass roll, injury roll, etc.) or when a new player is activated.
     * Opens a fresh move window: one more move is allowed.
     */
    void reset() {
        hasMoved = false;
        hasActed = false;
    }
}
