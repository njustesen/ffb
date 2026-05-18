package com.fumbbl.ffb.ai.parity;

import com.fumbbl.ffb.server.util.rng.Fortuna;

/**
 * Xoshiro256** PRNG — exactly matches the Rust implementation in ffb-rust/crates/ffb-core/src/rng.rs.
 *
 * <p>Used to replace Fortuna with a seeded, deterministic RNG for cross-engine parity testing.
 * Both Java and Rust must produce identical roll sequences for the same seed.
 *
 * <p>Initialization uses SplitMix64 (same as Rust's {@code SmallRng::seed_from_u64}).
 * Die rolls use rejection sampling to produce uniformly distributed results.
 */
public class Xoshiro256StarStar implements Fortuna.IDiceRoller {

    private final long[] state = new long[4];

    public Xoshiro256StarStar(long seed) {
        // SplitMix64 — must match Rust's SplitMix64 seeding exactly
        long s = seed;
        for (int i = 0; i < 4; i++) {
            s += 0x9e3779b97f4a7c15L;
            long z = s;
            z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
            z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
            state[i] = z ^ (z >>> 31);
        }
    }

    /** Generate next 64-bit value using xoshiro256** algorithm. Matches Rust's next_u64(). */
    public long nextLong() {
        long result = Long.rotateLeft(state[1] * 5L, 7) * 9L;
        long t = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= t;
        state[3] = Long.rotateLeft(state[3], 45);
        return result;
    }

    /**
     * Roll a fair die with {@code sides} faces, returning a value in [1, sides].
     *
     * <p>Uses rejection sampling — identical to Rust's {@code Xoshiro256StarStar::roll(sides)}.
     * All arithmetic treats the 64-bit values as unsigned.
     */
    @Override
    public int getDieRoll(int sides) {
        return getDieRollInternal(sides);
    }

    private int getDieRollInternal(int sides) {
        long s = Integer.toUnsignedLong(sides);
        // threshold = u64::MAX - (u64::MAX % sides) — using unsigned arithmetic
        long threshold = -1L - Long.remainderUnsigned(-1L, s);
        long v;
        do {
            v = nextLong();
        } while (Long.compareUnsigned(v, threshold) >= 0);
        return (int) (Long.remainderUnsigned(v, s) + 1L);
    }
}
