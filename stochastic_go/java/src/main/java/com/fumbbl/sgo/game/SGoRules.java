package com.fumbbl.sgo.game;

/**
 * Stateless game logic for Stochastic Go. All methods are static.
 */
public final class SGoRules {

    /** Precomputed orthogonal+diagonal neighbors for each cell. */
    public static final int[][] NEIGHBORS;

    static {
        int N = SGoState.BOARD_SIZE;
        NEIGHBORS = new int[SGoState.TOTAL_CELLS][];
        for (int cell = 0; cell < SGoState.TOTAL_CELLS; cell++) {
            int row = cell / N;
            int col = cell % N;
            int count = 0;
            int[] tmp = new int[8];
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    int r2 = row + dr;
                    int c2 = col + dc;
                    if (r2 >= 0 && r2 < N && c2 >= 0 && c2 < N) {
                        tmp[count++] = r2 * N + c2;
                    }
                }
            }
            NEIGHBORS[cell] = new int[count];
            System.arraycopy(tmp, 0, NEIGHBORS[cell], 0, count);
        }
    }

    /** Count adjacent cells occupied by the opponent. */
    public static int adjacentOpponentCount(int[] board, int cell, int player) {
        int opponent = (player == SGoState.P1) ? SGoState.P2 : SGoState.P1;
        int count = 0;
        int[] neighbors = NEIGHBORS[cell];
        for (int i = 0; i < neighbors.length; i++) {
            if (board[neighbors[i]] == opponent) count++;
        }
        return count;
    }

    /** Count adjacent cells occupied by the same player (friendly support). */
    public static int adjacentFriendlyCount(int[] board, int cell, int player) {
        int count = 0;
        int[] neighbors = NEIGHBORS[cell];
        for (int i = 0; i < neighbors.length; i++) {
            if (board[neighbors[i]] == player) count++;
        }
        return count;
    }

    /** True if a 1D6 roll succeeds for a placement with k adjacent opponents. */
    public static boolean placementSuccess(int roll, int k) {
        if (roll == 1) return false;
        if (roll == 6) return true;
        return roll > k;
    }

    /**
     * Estimated win probability for P1.
     *
     * Combines stone count with a clustering bonus: each adjacent same-player pair
     * (counted once) represents sustained future placement-rate advantage because
     * k_fri reduces k_dice for all future placements near the cluster. Without this
     * bonus, MCTS gets zero gradient in the early game when all cells have k_dice=0
     * and stone counts are symmetric, preventing it from learning cluster-building.
     *
     * Weight 0.5: two adjacent pairs ≈ one extra stone equivalent.
     */
    public static double winProb(SGoState state) {
        int p1 = 0, p2 = 0;
        int p1Pairs = 0, p2Pairs = 0;
        int[] board = state.board;
        for (int i = 0; i < SGoState.TOTAL_CELLS; i++) {
            if (board[i] == SGoState.P1) {
                p1++;
                int[] neighbors = NEIGHBORS[i];
                for (int j = 0; j < neighbors.length; j++) {
                    int nb = neighbors[j];
                    if (nb > i && board[nb] == SGoState.P1) p1Pairs++;
                }
            } else if (board[i] == SGoState.P2) {
                p2++;
                int[] neighbors = NEIGHBORS[i];
                for (int j = 0; j < neighbors.length; j++) {
                    int nb = neighbors[j];
                    if (nb > i && board[nb] == SGoState.P2) p2Pairs++;
                }
            }
        }
        double diff = (p1 - p2) + 0.5 * (p1Pairs - p2Pairs);
        return 1.0 / (1.0 + Math.exp(-0.15 * diff));
    }

    /**
     * Apply placement at cell with given roll. Returns new state with incremental hash.
     *
     * Uses base-3 net-k mechanic: k_dice = max(0, 3 + k_opponents - k_friendlies).
     * Isolated placement (k_opp=0, k_fri=0): k_dice=3 → 3/6 success, ~1 placement/turn.
     * Two friendlies nearby (k_fri=2): k_dice=1 → 5/6 success, ~5 placements/turn.
     * Three friendlies (k_fri=3): k_dice=0 → 5/6 success (capped).
     *
     * This base difficulty ensures clustering (building 2+ adjacent friendlies) is the
     * dominant strategy — discoverable by MCTS within hundreds of iterations but requiring
     * thousands to execute optimally (precise cluster direction and shape).
     *
     * - success (roll > k_dice or roll = 6): piece placed, turn continues.
     * - any failure (roll <= k_dice or roll = 1): turn ends.
     */
    public static SGoState applyPlacement(SGoState state, int cell, int roll) {
        int kOpp = adjacentOpponentCount(state.board, cell, state.currentPlayer);
        int kFri = adjacentFriendlyCount(state.board, cell, state.currentPlayer);
        int kDice = Math.max(0, Math.min(5, 3 + kOpp - kFri));
        boolean success = placementSuccess(roll, kDice);

        SGoState next = state.clone();
        if (success) {
            next.board[cell] = state.currentPlayer;
            next.emptyCells &= ~(1L << cell);
            next.stateHash ^= Zobrist.BOARD[cell * 3 + state.currentPlayer];
        } else {
            // Any failure ends the turn.
            next.isTurnEnd = true;
            next.stateHash ^= Zobrist.TURN_END[0] ^ Zobrist.TURN_END[1];

            // Non-fumble failure (roll != 1, so roll <= k_dice): capture the lowest-index
            // adjacent opponent stone. This prevents board-fill while punishing risky play
            // (attacker loses their turn but gains one opponent stone captured).
            if (roll != 1) {
                int opponent = (state.currentPlayer == SGoState.P1) ? SGoState.P2 : SGoState.P1;
                int[] neighbors = NEIGHBORS[cell];
                for (int i = 0; i < neighbors.length; i++) {
                    int nb = neighbors[i];
                    if (next.board[nb] == opponent) {
                        next.board[nb] = SGoState.EMPTY;
                        next.emptyCells |= (1L << nb);
                        next.stateHash ^= Zobrist.BOARD[nb * 3 + opponent];
                        break;
                    }
                }
            }
        }
        return next;
    }

    /** Voluntarily end the current player's turn. */
    public static SGoState applyEndTurn(SGoState state) {
        SGoState next = state.clone();
        next.isTurnEnd = true;
        next.stateHash ^= Zobrist.TURN_END[0] ^ Zobrist.TURN_END[1];
        return next;
    }

    /**
     * Advance to the next player's turn. Decrements current player's turn counter,
     * switches player, resets isTurnEnd.
     */
    public static SGoState advanceTurn(SGoState state) {
        SGoState next = state.clone();
        long h = next.stateHash;

        // Flip isTurnEnd: true → false
        h ^= Zobrist.TURN_END[1] ^ Zobrist.TURN_END[0];
        next.isTurnEnd = false;

        if (state.currentPlayer == SGoState.P1) {
            h ^= Zobrist.CURRENT_PLAYER[SGoState.P1] ^ Zobrist.CURRENT_PLAYER[SGoState.P2];
            h ^= Zobrist.P1_TURNS[state.p1TurnsRemaining] ^ Zobrist.P1_TURNS[state.p1TurnsRemaining - 1];
            next.p1TurnsRemaining--;
            next.currentPlayer = SGoState.P2;
        } else {
            h ^= Zobrist.CURRENT_PLAYER[SGoState.P2] ^ Zobrist.CURRENT_PLAYER[SGoState.P1];
            h ^= Zobrist.P2_TURNS[state.p2TurnsRemaining] ^ Zobrist.P2_TURNS[state.p2TurnsRemaining - 1];
            next.p2TurnsRemaining--;
            next.currentPlayer = SGoState.P1;
        }
        next.stateHash = h;
        return next;
    }

    private SGoRules() {}
}
