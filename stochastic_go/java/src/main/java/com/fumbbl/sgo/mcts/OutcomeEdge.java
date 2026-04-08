package com.fumbbl.sgo.mcts;

public final class OutcomeEdge {
    public double probability;
    public StateNode childState;

    public OutcomeEdge(double probability, StateNode childState) {
        this.probability = probability;
        this.childState = childState;
    }
}
