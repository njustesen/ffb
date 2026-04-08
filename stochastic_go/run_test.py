"""
Tournament runner for Stochastic Go.

Usage:
  python run_test.py              # MCTS-FT vs. random, 100 games
  python run_test.py --mirror     # MCTS-FT vs. MCTS-FT, 100 games
  python run_test.py --games 20 --budget 50
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List

from game import (
    SGoState, SGoAction, END_TURN, P1, P2,
    legal_actions, apply_placement, apply_end_turn, advance_turn,
)
from mcts_ft import mcts_ft, SearchContext
from sgo_interface import SGoGameInterface
from stats import SearchStats

GAME = SGoGameInterface()


def random_agent(state: SGoState, player: int) -> SGoAction:
    """Pick a random legal action. Never places more than 3 stones to avoid endless turns."""
    actions = legal_actions(state)
    # Bias: end turn with 20% probability to avoid infinite greedy placement.
    if random.random() < 0.20 and END_TURN in actions:
        return END_TURN
    return random.choice(actions)


def mcts_agent(state: SGoState, player: int, budget: int,
               ctx: SearchContext) -> SGoAction:
    action, stats = mcts_ft(state, budget, GAME, ctx=ctx)
    return action


def play_game(p1_agent, p2_agent) -> int:
    """Play a full game. Returns final score (P1_stones - P2_stones)."""
    state = SGoState.initial()

    while not state.is_terminal():
        player = state.current_player
        if player == P1:
            action = p1_agent(state, P1)
        else:
            action = p2_agent(state, P2)

        if action.type == "end_turn":
            state = apply_end_turn(state)
        else:
            # In real play we'd roll dice; here we roll randomly.
            roll = random.randint(1, 6)
            state = apply_placement(state, action.coord, roll)

        if state.is_turn_end:
            state = advance_turn(state)

    return state.score()


def run_tournament(games: int, budget: int, mirror: bool, verbose: bool) -> None:
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_stats: List[SearchStats] = []

    for g in range(games):
        ctx1 = SearchContext()
        ctx2 = SearchContext() if mirror else None

        if mirror:
            p1 = lambda s, pl: mcts_agent(s, pl, budget, ctx1)
            p2 = lambda s, pl: mcts_agent(s, pl, budget, ctx2)
        else:
            p1 = lambda s, pl: mcts_agent(s, pl, budget, ctx1)
            p2 = random_agent

        score = play_game(p1, p2)

        if score > 0:
            p1_wins += 1
        elif score < 0:
            p2_wins += 1
        else:
            draws += 1

        if verbose:
            result = "P1" if score > 0 else ("P2" if score < 0 else "Draw")
            print(f"  Game {g+1:3d}: score={score:+d}  winner={result}")

    print()
    print("=" * 50)
    mode = "MCTS vs MCTS" if mirror else "MCTS vs Random"
    print(f"Tournament: {mode}, {games} games, budget={budget}")
    print(f"  P1 wins:  {p1_wins} ({100*p1_wins/games:.1f}%)")
    print(f"  P2 wins:  {p2_wins} ({100*p2_wins/games:.1f}%)")
    print(f"  Draws:    {draws} ({100*draws/games:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Stochastic Go tournament")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Running {args.games} games (budget={args.budget}, mirror={args.mirror}) ...")
    run_tournament(args.games, args.budget, args.mirror, args.verbose)

    # Print a single-search stats sample.
    print()
    print("Sample search stats (1 search from initial state, budget=200):")
    ctx = SearchContext()
    _, stats = mcts_ft(SGoState.initial(), 200, GAME, ctx=ctx)
    print(stats.summary())


if __name__ == "__main__":
    main()
