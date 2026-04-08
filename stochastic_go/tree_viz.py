"""
Interactive search tree visualization for MCTS-FT.

Generates an HTML file using pyvis (vis.js). Falls back to a text summary if pyvis
is not installed.

Usage:
  python tree_viz.py              # 200-iteration search, opens tree_viz.html
  python tree_viz.py --budget 50
  python tree_viz.py --max-nodes 150 --output my_tree.html
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import webbrowser
from typing import Any, Dict, Optional, Set

from game import SGoState, BOARD_SIZE, P1, P2, EMPTY
from mcts_ft import (
    StateNode, ChanceNode, ActionEdge, OutcomeEdge,
    SearchContext, mcts_ft, StateHash
)
from sgo_interface import SGoGameInterface

GAME = SGoGameInterface()


# ─────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────

def lerp_color(c0: tuple, c1: tuple, t: float) -> str:
    """Linearly interpolate between two RGB tuples; return hex string."""
    t = max(0.0, min(1.0, t))
    r = int(c0[0] + (c1[0] - c0[0]) * t)
    g = int(c0[1] + (c1[1] - c0[1]) * t)
    b = int(c0[2] + (c1[2] - c0[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def value_color(v: float) -> str:
    """Map value_estimate to red-white-green. v in [-1, 1]."""
    RED = (220, 50, 50)
    WHITE = (240, 240, 240)
    GREEN = (50, 180, 80)
    if v < 0:
        return lerp_color(RED, WHITE, v + 1.0)   # -1→RED, 0→WHITE
    else:
        return lerp_color(WHITE, GREEN, v)        # 0→WHITE, 1→GREEN


def prob_color(p: float) -> str:
    """Map probability to pale-yellow → deep-orange."""
    PALE = (255, 245, 180)
    DEEP = (220, 100, 20)
    return lerp_color(PALE, DEEP, p)


def visit_blue(frac: float) -> str:
    """Map visit fraction (0–1) to light→dark blue."""
    LIGHT = (180, 210, 255)
    DARK = (20, 60, 180)
    return lerp_color(LIGHT, DARK, frac)


# ─────────────────────────────────────────────
# ASCII board helper
# ─────────────────────────────────────────────

def board_ascii(state: SGoState) -> str:
    symbols = {EMPTY: ".", P1: "●", P2: "○"}
    rows = []
    for r in range(BOARD_SIZE):
        rows.append(" ".join(symbols[state.board[r][c]] for c in range(BOARD_SIZE)))
    header = "  " + " ".join(str(c) for c in range(BOARD_SIZE))
    lines = [header] + [f"{r} {row}" for r, row in enumerate(rows)]
    p1 = sum(state.board[r][c] == P1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    p2 = sum(state.board[r][c] == P2 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    lines.append(f"P1(●)={p1}  P2(○)={p2}  diff={p1-p2:+d}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Pyvis visualization
# ─────────────────────────────────────────────

def build_pyvis_graph(root: StateNode, ctx: SearchContext,
                      max_nodes: int = 200) -> "Network":
    from pyvis.network import Network

    net = Network(
        height="900px", width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 120,
          "levelSeparation": 100
        }
      },
      "physics": { "enabled": false },
      "edges": { "smooth": { "enabled": false } }
    }
    """)

    # Determine visit threshold so we show at most max_nodes.
    all_nodes = list(ctx.tt.table.values())
    if len(all_nodes) > max_nodes:
        sorted_visits = sorted((n.visit_count for n in all_nodes), reverse=True)
        threshold = sorted_visits[max_nodes - 1]
    else:
        threshold = 0

    max_visits = max((n.visit_count for n in all_nodes), default=1) or 1

    visited_state_ids: Set[StateHash] = set()
    edge_counter = [0]

    def add_state_node(node: StateNode, label_extra: str = ""):
        if node.state_hash in visited_state_ids:
            return
        if node.visit_count < threshold:
            return
        visited_state_ids.add(node.state_hash)

        size = 10 + 3 * math.sqrt(node.visit_count)
        color = value_color(node.value_estimate)
        opacity = 0.4 + 0.6 * (node.visit_count / max_visits)

        state: SGoState = node.state
        p1 = sum(state.board[r][c] == P1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        p2 = sum(state.board[r][c] == P2 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        label = f"P1:{p1} P2:{p2}\nv={node.visit_count}\nval={node.value_estimate:+.2f}"
        if label_extra:
            label += f"\n{label_extra}"

        title = board_ascii(state)

        net.add_node(
            node.state_hash,
            label=label,
            title=title,
            shape="dot",
            size=size,
            color={"background": color, "border": "#ffffff"},
            font={"size": 9, "color": "white"},
            opacity=opacity,
        )

    def add_chance_node(cn_id: str, cn: ChanceNode):
        net.add_node(
            cn_id,
            label="🎲",
            shape="diamond",
            size=12,
            color={"background": "#888888", "border": "#cccccc"},
            font={"size": 10, "color": "white"},
        )

    def traverse(node: StateNode, depth: int = 0):
        if node.visit_count < threshold:
            return
        add_state_node(node)

        max_child_visits = max(
            (e.visit_count for e in node.action_edges.values()), default=1
        ) or 1

        for action, edge in node.action_edges.items():
            if edge.visit_count == 0:
                continue

            visit_frac = edge.visit_count / max_child_visits
            edge_width = 1 + 4 * visit_frac
            edge_color = visit_blue(visit_frac)
            action_label = str(action)[:12]

            if edge.chance_node is not None:
                cn = edge.chance_node
                cn_id = f"cn_{id(cn)}"

                if cn_id not in visited_state_ids:
                    visited_state_ids.add(cn_id)
                    add_chance_node(cn_id, cn)
                    net.add_edge(
                        node.state_hash, cn_id,
                        label=action_label,
                        width=edge_width,
                        color=edge_color,
                        arrows="to",
                        font={"size": 8, "color": "#aaaaff"},
                    )

                for out_hash, out_edge in cn.outcome_edges.items():
                    child = out_edge.child_state
                    if child.visit_count < threshold:
                        continue
                    add_state_node(child)
                    traverse(child, depth + 1)

                    prob_w = 1 + 5 * out_edge.probability
                    prob_c = prob_color(out_edge.probability)
                    eid = f"oe_{edge_counter[0]}"
                    edge_counter[0] += 1
                    net.add_edge(
                        cn_id, child.state_hash,
                        label=f"p={out_edge.probability:.2f}",
                        width=prob_w,
                        color=prob_c,
                        arrows="to",
                        font={"size": 8, "color": "#ffcc88"},
                        id=eid,
                    )
            else:
                child = edge.deterministic_child
                if child is None or child.visit_count < threshold:
                    continue
                add_state_node(child)
                traverse(child, depth + 1)
                net.add_edge(
                    node.state_hash, child.state_hash,
                    label=action_label,
                    width=edge_width,
                    color=edge_color,
                    arrows="to",
                    font={"size": 8, "color": "#aaaaff"},
                )

    traverse(root)
    return net


def visualize(budget: int, max_nodes: int, output: str) -> None:
    print(f"Running MCTS-FT for {budget} iterations ...")
    ctx = SearchContext()
    action, stats = mcts_ft(SGoState.initial(), budget, GAME, ctx=ctx)
    print(f"Best action: {action}")
    print(stats.summary())

    root = ctx.tt.lookup(GAME.hash_state(SGoState.initial()))

    try:
        net = build_pyvis_graph(root, ctx, max_nodes=max_nodes)
        net.save_graph(output)
        print(f"\nTree saved to: {output}")
        webbrowser.open(f"file://{os.path.abspath(output)}")
    except ImportError:
        print("\npyvis not installed. Install with: pip install pyvis")
        print("Falling back to text summary...")
        _text_summary(root, ctx)


def _text_summary(root: StateNode, ctx: SearchContext) -> None:
    print(f"\nTree has {len(ctx.tt.table)} state nodes.")
    print(f"Root visit count: {root.visit_count}")
    print("Top actions at root:")
    sorted_edges = sorted(root.action_edges.items(),
                          key=lambda kv: kv[1].visit_count, reverse=True)
    for action, edge in sorted_edges[:10]:
        print(f"  {str(action):<20} visits={edge.visit_count:4d}  "
              f"value={edge.value_sum/edge.visit_count:+.3f}" if edge.visit_count > 0
              else f"  {str(action):<20} visits=0")


def main():
    parser = argparse.ArgumentParser(description="MCTS-FT tree visualization")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--output", default="tree_viz.html")
    args = parser.parse_args()
    visualize(args.budget, args.max_nodes, args.output)


if __name__ == "__main__":
    main()
