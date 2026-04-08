package com.fumbbl.sgo.mcts;

import com.fumbbl.sgo.game.SGoState;

/**
 * Open-addressing hash map from long state hash to StateNode.
 * Eliminates Long boxing vs HashMap<Long, StateNode>.
 */
public final class TranspositionTable {

    private static final long EMPTY_KEY = Long.MIN_VALUE; // sentinel

    private long[] keys;
    private StateNode[] values;
    private int capacity;
    private int mask;
    private int size;

    public int totalAttempts;
    public int totalHits;

    public TranspositionTable() {
        this(1 << 16); // 65536 initial slots
    }

    public TranspositionTable(int initialCapacity) {
        capacity = nextPowerOfTwo(initialCapacity);
        mask = capacity - 1;
        keys = new long[capacity];
        values = new StateNode[capacity];
        java.util.Arrays.fill(keys, EMPTY_KEY);
    }

    public StateNode getOrCreate(long hash, SGoState state, boolean isTurnEnd) {
        totalAttempts++;
        int idx = probe(hash);
        if (keys[idx] == hash) {
            totalHits++;
            return values[idx];
        }
        // New entry
        StateNode node = new StateNode(hash, state, isTurnEnd);
        keys[idx] = hash;
        values[idx] = node;
        size++;
        if (size * 2 > capacity) grow();
        return node;
    }

    public StateNode lookup(long hash) {
        int idx = probe(hash);
        return keys[idx] == hash ? values[idx] : null;
    }

    public int size() { return size; }

    /** Returns all StateNodes (for stats collection). */
    public StateNode[] allNodes() {
        StateNode[] result = new StateNode[size];
        int j = 0;
        for (int i = 0; i < capacity; i++) {
            if (keys[i] != EMPTY_KEY) result[j++] = values[i];
        }
        return result;
    }

    private int probe(long hash) {
        int idx = (int)(hash ^ (hash >>> 32)) & mask;
        while (keys[idx] != EMPTY_KEY && keys[idx] != hash) {
            idx = (idx + 1) & mask;
        }
        return idx;
    }

    private void grow() {
        long[] oldKeys = keys;
        StateNode[] oldValues = values;
        int oldCap = capacity;

        capacity <<= 1;
        mask = capacity - 1;
        keys = new long[capacity];
        values = new StateNode[capacity];
        java.util.Arrays.fill(keys, EMPTY_KEY);

        for (int i = 0; i < oldCap; i++) {
            if (oldKeys[i] != EMPTY_KEY) {
                int idx = probe(oldKeys[i]);
                keys[idx] = oldKeys[i];
                values[idx] = oldValues[i];
            }
        }
    }

    private static int nextPowerOfTwo(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }
}
