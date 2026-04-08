# MCTS-FT: Full-Turn Monte Carlo Tree Search

A variant of MCTS designed for **multi-action adversarial turn-based games** — games where
each player takes multiple sequential actions before handing the turn to the opponent, and
random events (dice rolls) can cut a turn short at any point.

## Algorithm Overview

Standard MCTS expands one ply per pass. MCTS-FT expands **both players' complete turns**
in a single pass, making each iteration a full round of play. This is important for games
like Blood Bowl or Stochastic Go where:

- The *value* of a position only becomes clear after the opponent has responded.
- A player's turn can end involuntarily (turnover/failed roll), making single-ply evaluation
  misleading.

### Key Properties

**1. Full-turn traversal**  
Each pass covers exactly one turn for the acting player followed by one turn for the
opponent. The tree depth grows by two "turn-levels" per search, but each level can span
arbitrarily many action steps.

**2. Combined expansion and simulation**  
There is no separate expansion/rollout phase. Every node encountered for the first time is
added to the tree on the spot. Later, its action edges are populated lazily when first
visited. The tree is always fully grown up to the depth explored.

**3. Chance nodes with outcome aggregation**  
When an action involves a random event (dice roll), the game engine is called once per
possible dice value to determine the resulting state. Results with identical state hashes
are **merged into a single outcome edge** with summed probability. The tree never contains
per-dice-face edges — only per-distinct-outcome edges. This keeps the branching factor
manageable and makes probability tracking exact.

**4. Entropy-minimizing chance node selection**  
When revisiting a chance node, the algorithm selects the outcome most underexplored
relative to its probability:

```
score(outcome) = probability / (visit_count / total_visits)
```

This drives the explored distribution toward the theoretical one, minimizing
KL(explored || theoretical).

**5. Transposition table**  
Identical game states reachable via different action sequences share a single `StateNode`
in the tree. A global hash map (state hash → StateNode) is consulted on every state
creation. This prevents combinatorial blowup in games with many equivalent paths and
enables value estimates to be shared across branches.

**6. Win-rate delta evaluation**  
Each pass is scored as the *change* in estimated win probability from the root state to the
terminal state at the end of P2's turn: `win_prob(terminal) - win_prob(root)`. Using the
root as a baseline reduces variance in backpropagated values, making UCB statistics more
reliable early in search.

**7. Re-search after every random event (agent execution)**  
When executing a chosen action, the agent sends one action and stops as soon as a random
event is triggered. After the real dice outcome arrives from the game engine, the agent
re-searches from the updated state — never committing to a plan that assumed a specific
dice outcome.

**8. Subtree reuse**  
The transposition table persists across consecutive searches within a turn. When the agent
re-searches after observing a dice outcome, the new root state was almost certainly already
explored in the previous search. That subtree retains its visit counts and value estimates,
giving the new search a head start.

---

## Stochastic Go

A test bed designed to exercise all of the above properties.

### Rules

- **Board**: 8×8 grid
- **Players**: P1 (●, Black) and P2 (○, White), alternating turns
- **Actions per turn**: Place as many stones as desired (one at a time), or voluntarily end
  the turn at any point. **No per-turn stone limit** — the turn continues as long as
  placements succeed.
- **Stone placement**:
  - Choose any empty intersection
  - Roll 1D6
  - **Success**: roll > number of adjacent opponent stones (orthogonal + diagonal, up to 8)
  - **Roll of 1**: always fails regardless of adjacency
  - **Roll of 6**: always succeeds regardless of adjacency
  - **Failure**: stone not placed, turn ends immediately (turnover)
- **Game end**: After 10 turns each, or when the board is full
- **Score**: P1 stones − P2 stones

### Success Probabilities

| Adjacent opponent stones (k) | P(success) | Dice that succeed |
|---|---|---|
| 0 | 5/6 | 2,3,4,5,6 |
| 1 | 5/6 | 2,3,4,5,6 (roll=1 always fails; 2>1) |
| 2 | 4/6 | 3,4,5,6 |
| 3 | 3/6 | 4,5,6 |
| 4 | 2/6 | 5,6 |
| 5 | 1/6 | 6 only |
| 6+ | 1/6 | 6 only |

Each placement generates exactly **2 outcome edges** in its chance node (success and
failure), regardless of how many dice faces exist.

### Why This Mirrors Blood Bowl

| Stochastic Go | Blood Bowl |
|---|---|
| Unlimited placements until failure | Team turn continues until turnover |
| Roll > k adjacent opponents | Block/dodge rolls modified by skills and assists |
| Failure ends turn immediately | Turnover ends team turn immediately |
| Risk/reward: keep placing vs. end safely | Risk/reward: risky blocks vs. conservative play |

---

## File Structure

```
stochastic_go/
  README.md         # This file
  game.py           # Game state, rules, actions, hashing
  evaluate.py       # win_prob() heuristic: sigmoid(k * stone_count_diff)
  mcts_ft.py        # MCTS-FT algorithm (generic, game-agnostic)
  sgo_interface.py  # Bridges game.py and mcts_ft.py
  stats.py          # SearchStats dataclass and KL/entropy helpers
  tree_viz.py       # Interactive HTML search tree visualization (pyvis)
  run_test.py       # Tournament runner
  tests/
    test_game.py          # Game logic unit tests
    test_chance_node.py   # Chance node aggregation and selection tests
    test_mcts_ft.py       # Tree structure, backprop, and search property tests
    test_stats.py         # Stats collection tests
```

---

## Running

### Install dependencies

```bash
pip install pyvis   # optional, for tree visualization
```

No other dependencies beyond Python 3.8+ standard library.

### Run the tournament

```bash
cd stochastic_go
python run_test.py                          # MCTS-FT vs. random, 100 games
python run_test.py --mirror                 # MCTS-FT vs. MCTS-FT
python run_test.py --games 20 --budget 50   # Fewer games, smaller budget
python run_test.py --verbose                # Print each game result
```

### Visualize the search tree

```bash
python tree_viz.py              # 200-iteration search, opens tree_viz.html in browser
python tree_viz.py --budget 50  # Smaller tree, easier to inspect
```

The tree visualization shows:
- **Node size**: proportional to visit count
- **Node color**: red = negative value (bad for P1), white = neutral, green = positive (good for P1)
- **Action edges**: thicker and darker blue = more visited
- **Chance edges**: thicker and more orange = higher probability
- **Hover**: displays board state as ASCII art

### Run unit tests

```bash
python -m pytest tests/ -v
```

---

## Future Extension: Blood Bowl

MCTS-FT is designed to extend to Blood Bowl with the following component changes:

| Component | Stochastic Go | Blood Bowl |
|---|---|---|
| `legal_actions()` | place stone, end turn | move, block, blitz, pass, hand-off, foul, end turn |
| `is_stochastic()` | all placements | block/dodge/pickup/pass/injury rolls |
| `apply_dice_outcome()` | place/fail based on 1D6 | block results (push/down/both-down etc.), dodge success/fail, etc. |
| `hash_state()` | board array + player + turn count | Zobrist over player positions × player states × game phase |
| `win_prob()` | sigmoid(stone_count_diff) | weighted combo of TD differential, player count, field position (or learned model) |
| Turnover detection | failure outcome → `is_turn_end=True` | turnover step in server Step stack → `is_turn_end=True` |
| Forward simulation | pure Python | `GameSimulator.cloneGame()` + `SimulationLoop` in Java (ffb-ai module) |

### Integration points in the ffb codebase

- **`ffb-ai/.../AiDecisionEngine.java`** — replace `RandomStrategy` calls with MCTS-FT search result
- **`ffb-ai/.../GameSimulator.java`** — extend with a `ServerStub` to drive cloned game states through the Step stack
- **`ffb-common/.../net/commands/`** — command objects define the action space for the search tree
- **`ffb-server/.../model/step/`** — turnover/failure steps in the step stack signal involuntary turn end

The Blood Bowl implementation requires a Java port of the MCTS-FT algorithm (or a JNI/subprocess bridge to Python), plus a forward-simulation layer that executes game Steps on a cloned `GameState` without network I/O.
