package com.fumbbl.ffb.ai;

/**
 * Represents a scored AI decision as the product: probability × value × confidence.
 *
 * <ul>
 *   <li>probability ∈ [0,1]: how likely the action is to succeed (1.0 = always succeeds;
 *       dice-based for risky actions)</li>
 *   <li>value ∈ [-1,1]: how good/bad the outcome is when it happens
 *       (-1.0 = catastrophic like skull turnover; +1.0 = touchdown)</li>
 *   <li>confidence ∈ [0,1]: how confident we are in the value estimate
 *       (1.0 = clear outcome; 0.1 = highly situational)</li>
 * </ul>
 *
 * The product p×v×c preserves sign: a high-confidence bad action stays negative;
 * an uncertain good action stays positive but reduced. This prevents confident
 * bad choices from beating uncertain good ones.
 *
 * For use with {@link PolicySampler}, scores are shifted via {@link #softmaxScore()}
 * from [-1,1] into [0,2] so that all softmax inputs are non-negative.
 */
public final class ActionScore {

    public final double probability;  // [0, 1]
    public final double value;        // [-1, 1]
    public final double confidence;   // [0, 1]

    public ActionScore(double probability, double value, double confidence) {
        this.probability = clamp(probability, 0.0, 1.0);
        this.value       = clamp(value,       -1.0, 1.0);
        this.confidence  = clamp(confidence,  0.0, 1.0);
    }

    /** Raw product: p × v × c ∈ [-1, 1]. */
    public double score() {
        return probability * value * confidence;
    }

    /**
     * Softmax-compatible score: 1.0 + score(), mapping [-1,1] → [0,2].
     * Preserves relative ordering while ensuring all values are non-negative
     * for use with {@link PolicySampler}.
     */
    public double softmaxScore() {
        return 1.0 + score();
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    @Override
    public String toString() {
        return String.format("ActionScore{p=%.3f v=%+.3f c=%.3f => %+.4f}",
            probability, value, confidence, score());
    }
}
