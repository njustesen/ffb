"""
MCTS-FT: Full-Turn Monte Carlo Tree Search.

A generic MCTS variant for multi-action adversarial turn-based games with chance nodes.
Game-specific logic is injected via the GameInterface protocol.

Key properties:
  - Each pass traverses a full turn for both players.
  - Combined expansion and simulation (no rollout phase).
  - Chance nodes group dice values by resulting state hash, with summed probabilities.
  - Entropy-minimizing outcome selection for chance nodes.
  - Transposition table: identical states share a single StateNode.
  - Persistent SearchContext for subtree reuse across re-searches.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from stats import SearchStats, entropy, kl_divergence

# Type aliases
StateHash = int
ActionKey = Any   # hashable action


# ─────────────────────────────────────────────
# Tree data structures
# ─────────────────────────────────────────────

@dataclass
class OutcomeEdge:
    probability: float          # sum of P(dice_val) for all dice values → this state
    child_state: "StateNode"    # shared via transposition table


@dataclass
class ChanceNode:
    # Keyed by StateHash of the resulting child state, NOT by dice face value.
    outcome_edges: Dict[StateHash, OutcomeEdge] = field(default_factory=dict)
    visit_counts: Dict[StateHash, int] = field(default_factory=dict)
    total_visits: int = 0

    def theoretical_probs(self) -> Dict[StateHash, float]:
        return {h: e.probability for h, e in self.outcome_edges.items()}


@dataclass
class ActionEdge:
    action: ActionKey
    chance_node: Optional[ChanceNode] = None    # None for deterministic transitions
    deterministic_child: Optional["StateNode"] = None
    visit_count: int = 0
    value_sum: float = 0.0


@dataclass
class StateNode:
    state: Any                              # game state object
    state_hash: StateHash
    is_turn_end: bool
    action_edges: Dict[ActionKey, ActionEdge] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def value_estimate(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class TranspositionTable:
    def __init__(self):
        self.table: Dict[StateHash, StateNode] = {}
        self.total_attempts: int = 0
        self.total_hits: int = 0

    def get_or_create(self, state_hash: StateHash, state: Any,
                      is_turn_end: bool = False) -> StateNode:
        self.total_attempts += 1
        if state_hash in self.table:
            self.total_hits += 1
            return self.table[state_hash]
        node = StateNode(state=state, state_hash=state_hash, is_turn_end=is_turn_end)
        self.table[state_hash] = node
        return node

    def lookup(self, state_hash: StateHash) -> Optional[StateNode]:
        return self.table.get(state_hash)


@dataclass
class SearchContext:
    """Persists across multiple MCTS_FT() calls for subtree reuse."""
    tt: TranspositionTable = field(default_factory=TranspositionTable)
    chance_node_count: int = 0


# ─────────────────────────────────────────────
# Game interface (to be implemented per game)
# ─────────────────────────────────────────────

class GameInterface:
    """
    Abstract interface a game must implement for MCTS-FT.
    Subclass and implement all methods for your specific game.
    """

    def legal_actions(self, state: Any, player: int) -> List[ActionKey]:
        """Return all legal actions for player from state."""
        raise NotImplementedError

    def is_stochastic(self, state: Any, action: ActionKey) -> bool:
        """True if action from state leads to a chance node (random event)."""
        raise NotImplementedError

    def apply_deterministic(self, state: Any, action: ActionKey) -> Any:
        """Apply a deterministic action. Only called when is_stochastic() is False."""
        raise NotImplementedError

    def dice_distribution(self, state: Any, action: ActionKey) -> List[Tuple[Any, float]]:
        """
        Return list of (dice_value, probability) pairs summing to 1.0.
        Only called when is_stochastic() is True.
        """
        raise NotImplementedError

    def apply_dice_outcome(self, state: Any, action: ActionKey, dice_value: Any) -> Any:
        """Apply action + dice_value to state. Returns resulting state."""
        raise NotImplementedError

    def is_turn_end(self, state: Any, player: int) -> bool:
        """True if the given state represents the end of player's turn."""
        raise NotImplementedError

    def is_terminal(self, state: Any) -> bool:
        """True if the game is over."""
        raise NotImplementedError

    def current_player(self, state: Any) -> int:
        """Return the player whose turn it is."""
        raise NotImplementedError

    def hash_state(self, state: Any) -> StateHash:
        """Return a hash for state. Identical states must return identical hashes."""
        raise NotImplementedError

    def win_prob(self, state: Any) -> float:
        """Estimate P1's win probability from state. In (0, 1)."""
        raise NotImplementedError

    def advance_turn(self, state: Any) -> Any:
        """Called after is_turn_end() is True: switch to next player, reset turn state."""
        raise NotImplementedError

    def opponent(self, player: int) -> int:
        raise NotImplementedError


# ─────────────────────────────────────────────
# UCB constant
# ─────────────────────────────────────────────

C_ACTION = 1.41  # sqrt(2); tune per game


# ─────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────

def expand_outcomes(cn: ChanceNode, state: Any, action: ActionKey,
                    game: GameInterface, ctx: SearchContext) -> None:
    """
    Enumerate all dice values, apply each, group by resulting state hash.
    Dice values producing the same game state are merged into one OutcomeEdge
    with summed probability. The tree never has per-dice-face edges.
    """
    groups: Dict[StateHash, Tuple[Any, float]] = {}

    for dice_val, prob in game.dice_distribution(state, action):
        result = game.apply_dice_outcome(state, action, dice_val)
        h = game.hash_state(result)
        if h not in groups:
            groups[h] = (result, 0.0)
        prev_state, prev_prob = groups[h]
        groups[h] = (prev_state, prev_prob + prob)

    for h, (result_state, total_prob) in groups.items():
        turn_end = game.is_turn_end(result_state, game.current_player(state))
        child = ctx.tt.get_or_create(h, result_state, is_turn_end=turn_end)
        cn.outcome_edges[h] = OutcomeEdge(probability=total_prob, child_state=child)


def select_outcome(cn: ChanceNode) -> Tuple[StateHash, OutcomeEdge]:
    """
    Entropy-minimizing outcome selection.
    Drives visit frequency toward theoretical probability distribution.
    """
    unexplored = [h for h in cn.outcome_edges if h not in cn.visit_counts]
    if unexplored:
        # First visits: explore highest-probability outcomes first.
        best = max(unexplored, key=lambda h: cn.outcome_edges[h].probability)
        return best, cn.outcome_edges[best]

    # All explored: pick most underrepresented relative to probability.
    N = cn.total_visits
    best = max(
        cn.outcome_edges.keys(),
        key=lambda h: cn.outcome_edges[h].probability / (cn.visit_counts[h] / N),
    )
    return best, cn.outcome_edges[best]


def select_action(node: StateNode, player: int, p1: int) -> Tuple[ActionKey, ActionEdge]:
    """
    UCB action selection.
    P1 maximizes value, P2 minimizes (so we flip sign for P2).
    """
    N = node.visit_count + 1
    log_N = math.log(N)          # computed once; reused across all edges
    c = C_ACTION
    best_score = -math.inf
    best: List[Tuple[ActionKey, ActionEdge]] = []

    for action, edge in node.action_edges.items():
        n = edge.visit_count
        v = edge.value_sum / n if n > 0 else 0.0
        if player != p1:
            v = -v
        score = v + c * (log_N / (n + 1)) ** 0.5
        if score > best_score:
            best_score = score
            best = [(action, edge)]
        elif score == best_score:
            best.append((action, edge))

    return random.choice(best)


def traverse_turn(
    start: StateNode,
    player: int,
    game: GameInterface,
    ctx: SearchContext,
    depth_counter: List[int],
) -> Tuple[List[Tuple], StateNode]:
    """
    Walk one player's complete turn, expanding nodes lazily.
    Returns the path and the turn-end StateNode.
    """
    node = start
    path: List[Tuple] = []

    while not node.is_turn_end:
        if game.is_terminal(node.state):
            # Game ended mid-turn; treat as turn end.
            node.is_turn_end = True
            break

        # Lazy action expansion.
        if not node.action_edges:
            for action in game.legal_actions(node.state, player):
                edge = ActionEdge(action=action)
                if game.is_stochastic(node.state, action):
                    edge.chance_node = ChanceNode()
                    ctx.chance_node_count += 1
                node.action_edges[action] = edge

        if not node.action_edges:
            # No legal actions (shouldn't happen in normal play; treat as turn end).
            node.is_turn_end = True
            break

        action, edge = select_action(node, player, p1=1)
        depth_counter[0] += 1

        if edge.chance_node is not None:
            cn = edge.chance_node
            if not cn.outcome_edges:
                expand_outcomes(cn, node.state, action, game, ctx)

            out_hash, out_edge = select_outcome(cn)
            cn.visit_counts[out_hash] = cn.visit_counts.get(out_hash, 0) + 1
            cn.total_visits += 1
            path.append((node, action, cn, out_hash))
            node = out_edge.child_state

        else:
            if edge.deterministic_child is None:
                next_state = game.apply_deterministic(node.state, action)
                h = game.hash_state(next_state)
                turn_end = game.is_turn_end(next_state, player)
                edge.deterministic_child = ctx.tt.get_or_create(h, next_state, is_turn_end=turn_end)
            path.append((node, action, None, None))
            node = edge.deterministic_child

    return path, node


def backpropagate(path: List[Tuple], value: float, iteration_id: int) -> None:
    """
    Walk path in reverse, updating visit counts and value sums.
    Skip re-updating a StateNode already visited this iteration (transposition safety).
    """
    visited: Set[StateHash] = set()

    for node, action, cn, out_hash in reversed(path):
        edge = node.action_edges[action]
        edge.visit_count += 1
        edge.value_sum += value

        if node.state_hash not in visited:
            node.visit_count += 1
            node.value_sum += value
            visited.add(node.state_hash)


def best_action(root: StateNode) -> ActionKey:
    """Return most-visited action at root (robust to outlier value estimates)."""
    return max(root.action_edges.keys(), key=lambda a: root.action_edges[a].visit_count)


def collect_stats(root: StateNode, ctx: SearchContext, elapsed_ms: float,
                  iterations: int, max_depth: int, game: GameInterface) -> SearchStats:
    stats = SearchStats()
    stats.total_time_ms = elapsed_ms
    stats.iterations = iterations
    stats.total_state_nodes = len(ctx.tt.table)
    stats.total_chance_nodes = ctx.chance_node_count
    stats.total_transposition_hits = ctx.tt.total_hits
    stats.total_transposition_attempts = ctx.tt.total_attempts
    stats.max_depth_reached = max_depth

    # Branching factor: avg action edges among expanded nodes.
    expanded = [n for n in ctx.tt.table.values() if n.action_edges]
    if expanded:
        stats.avg_branching_factor = sum(len(n.action_edges) for n in expanded) / len(expanded)

    # Root visit distribution.
    stats.root_visit_distribution = {
        str(a): e.visit_count for a, e in root.action_edges.items()
    }
    stats.root_visit_entropy = entropy(stats.root_visit_distribution)

    # Average KL divergence across all chance nodes.
    kl_sum = 0.0
    kl_count = 0
    for node in ctx.tt.table.values():
        for edge in node.action_edges.values():
            if edge.chance_node and edge.chance_node.total_visits > 0:
                cn = edge.chance_node
                theoretical = cn.theoretical_probs()
                kl = kl_divergence(cn.visit_counts, theoretical)
                kl_sum += kl
                kl_count += 1
    stats.avg_chance_kl = kl_sum / kl_count if kl_count > 0 else 0.0
    stats.chance_node_count_for_kl = kl_count

    stats.finalize()
    return stats


def mcts_ft(
    root_state: Any,
    budget: int,
    game: GameInterface,
    ctx: Optional[SearchContext] = None,
    p1: int = 1,
    p2: int = 2,
) -> Tuple[ActionKey, SearchStats]:
    """
    Run MCTS-FT from root_state for `budget` iterations.

    Args:
        root_state: Current game state.
        budget: Number of full-turn passes to run.
        game: Game interface implementation.
        ctx: Persistent search context (reused for subtree reuse). Created fresh if None.
        p1, p2: Player identifiers (default 1 and 2).

    Returns:
        (best_action, SearchStats)
    """
    if ctx is None:
        ctx = SearchContext()

    root_hash = game.hash_state(root_state)
    root_turn_end = game.is_turn_end(root_state, game.current_player(root_state))
    root = ctx.tt.get_or_create(root_hash, root_state, is_turn_end=root_turn_end)
    root_wp = game.win_prob(root_state)

    max_depth = 0
    start = time.perf_counter()

    for iteration in range(budget):
        depth_counter = [0]

        # P1's turn
        path_p1, p1_terminal = traverse_turn(root, p1, game, ctx, depth_counter)

        # After P1's turn ends, advance to P2's turn.
        if not game.is_terminal(p1_terminal.state):
            p2_start_state = game.advance_turn(p1_terminal.state)
            p2_start_hash = game.hash_state(p2_start_state)
            p2_turn_end = game.is_turn_end(p2_start_state, p2)
            p2_start_node = ctx.tt.get_or_create(p2_start_hash, p2_start_state,
                                                  is_turn_end=p2_turn_end)

            path_p2, p2_terminal = traverse_turn(p2_start_node, p2, game, ctx, depth_counter)
            terminal_state = p2_terminal.state
        else:
            path_p2 = []
            terminal_state = p1_terminal.state

        value = game.win_prob(terminal_state) - root_wp
        full_path = path_p1 + path_p2
        backpropagate(full_path, value, iteration)
        max_depth = max(max_depth, depth_counter[0])

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    action = best_action(root)
    stats = collect_stats(root, ctx, elapsed_ms, budget, max_depth, game)
    return action, stats
