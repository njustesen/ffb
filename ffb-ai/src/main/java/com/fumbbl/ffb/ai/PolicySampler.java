package com.fumbbl.ffb.ai;

import java.util.Random;

/**
 * Softmax-based probability utilities for the scripted AI policy.
 *
 * All decision scoring uses softmax so that:
 *  - argmax() always picks the highest-scored action (same as the deterministic tables)
 *  - sample() produces natural stochasticity proportional to the scores
 */
public final class PolicySampler {

    private PolicySampler() {}

    /**
     * Convert raw scores into a probability distribution via softmax.
     *
     * @param scores      array of raw scores (all must be finite; ≥ 1 recommended)
     * @param temperature controls spread: low T → near-deterministic, high T → uniform
     * @return probability array (sums to 1.0)
     */
    public static double[] softmax(double[] scores, double temperature) {
        if (scores == null || scores.length == 0) {
            return new double[0];
        }
        double[] probs = new double[scores.length];
        double maxScore = scores[0];
        for (double s : scores) {
            if (s > maxScore) maxScore = s;
        }
        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            probs[i] = Math.exp((scores[i] - maxScore) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    /**
     * Sample an index from the softmax distribution of the given scores.
     *
     * @param scores      raw scores
     * @param temperature softmax temperature
     * @param rng         random number generator
     * @return sampled index
     */
    public static int sample(double[] scores, double temperature, Random rng) {
        double[] probs = softmax(scores, temperature);
        double r = rng.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                return i;
            }
        }
        return probs.length - 1;
    }

    /**
     * Return the index with the highest score (argmax), breaking ties by returning
     * the first occurrence.
     *
     * @param scores raw scores
     * @return index of the highest score
     */
    public static int argmax(double[] scores) {
        if (scores == null || scores.length == 0) {
            return 0;
        }
        int best = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Return the index with the lowest score (argmin).
     *
     * @param scores raw scores
     * @return index of the lowest score
     */
    public static int argmin(double[] scores) {
        if (scores == null || scores.length == 0) {
            return 0;
        }
        int best = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] < scores[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Choose true or false from a binary softmax.
     *
     * @param scoreTrue  score for the "true" option
     * @param scoreFalse score for the "false" option
     * @param temperature softmax temperature
     * @param rng        random number generator
     * @return true if the true option is sampled
     */
    public static boolean chooseBool(double scoreTrue, double scoreFalse, double temperature, Random rng) {
        int idx = sample(new double[]{scoreTrue, scoreFalse}, temperature, rng);
        return idx == 0;
    }

    /**
     * Sample from the piecewise-linear mixture distribution parameterised by {@code temperature}:
     * <ul>
     *   <li>T = 0   → argmax (fully deterministic)</li>
     *   <li>T = 0.5 → softmax(scores, baseTemp) — the unmodified policy distribution</li>
     *   <li>T = 1   → uniform (fully random)</li>
     * </ul>
     *
     * @param scores      raw scores
     * @param baseTemp    softmax temperature used to compute the raw policy distribution at T=0.5
     * @param temperature mixture parameter ∈ [0, 1]
     * @param rng         random number generator
     * @return sampled index
     */
    public static int sampleMixed(double[] scores, double baseTemp, double temperature, Random rng) {
        if (scores.length == 1) return 0;
        double[] raw = softmax(scores, baseTemp);
        double[] p = mixedDist(raw, temperature);
        double r = rng.nextDouble();
        double cum = 0.0;
        for (int i = 0; i < p.length - 1; i++) {
            cum += p[i];
            if (r < cum) return i;
        }
        return p.length - 1;
    }

    /**
     * Binary version of {@link #sampleMixed}: T=0 → pick higher-scored option,
     * T=0.5 → chooseBool, T=1 → coin flip.
     */
    public static boolean chooseBoolMixed(double scoreTrue, double scoreFalse, double baseTemp,
                                          double temperature, Random rng) {
        double[] raw = softmax(new double[]{scoreTrue, scoreFalse}, baseTemp);
        double[] p = mixedDist(raw, temperature);
        return rng.nextDouble() < p[0];
    }

    /**
     * Compute the piecewise-linear mixed distribution between the argmax one-hot (T=0),
     * the raw policy (T=0.5), and the uniform distribution (T=1).
     */
    private static double[] mixedDist(double[] raw, double temperature) {
        int n = raw.length;
        int best = argmax(raw);
        double[] p = new double[n];
        double uniform = 1.0 / n;
        if (temperature <= 0.0) {
            p[best] = 1.0;
        } else if (temperature < 0.5) {
            double alpha = 2.0 * temperature;  // 0..1 as T goes 0..0.5
            for (int i = 0; i < n; i++) {
                p[i] = (1 - alpha) * (i == best ? 1.0 : 0.0) + alpha * raw[i];
            }
        } else if (temperature < 1.0) {
            double alpha = 2.0 * (temperature - 0.5);  // 0..1 as T goes 0.5..1
            for (int i = 0; i < n; i++) {
                p[i] = (1 - alpha) * raw[i] + alpha * uniform;
            }
        } else {
            for (int i = 0; i < n; i++) p[i] = uniform;
        }
        return p;
    }
}
