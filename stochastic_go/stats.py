"""
Search statistics for MCTS-FT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SearchStats:
    # Speed
    total_time_ms: float = 0.0
    iterations: int = 0
    iterations_per_second: float = 0.0

    # Tree size
    total_state_nodes: int = 0
    total_chance_nodes: int = 0
    total_transposition_hits: int = 0
    total_transposition_attempts: int = 0
    transposition_hit_rate: float = 0.0

    # Exploration shape
    max_depth_reached: int = 0
    avg_branching_factor: float = 0.0

    # Exploration quality
    root_visit_distribution: Dict[str, int] = field(default_factory=dict)
    root_visit_entropy: float = 0.0
    avg_chance_kl: float = 0.0
    chance_node_count_for_kl: int = 0

    def finalize(self):
        """Compute derived fields after search completes."""
        if self.total_time_ms > 0:
            self.iterations_per_second = self.iterations / (self.total_time_ms / 1000.0)
        if self.total_transposition_attempts > 0:
            self.transposition_hit_rate = self.total_transposition_hits / self.total_transposition_attempts

    def summary(self) -> str:
        lines = [
            f"  Iterations:       {self.iterations}",
            f"  Time:             {self.total_time_ms:.1f} ms",
            f"  Iter/sec:         {self.iterations_per_second:.0f}",
            f"  State nodes:      {self.total_state_nodes}",
            f"  Chance nodes:     {self.total_chance_nodes}",
            f"  Transpos. hits:   {self.total_transposition_hits} / {self.total_transposition_attempts}"
            f"  ({self.transposition_hit_rate * 100:.1f}%)",
            f"  Max depth:        {self.max_depth_reached}",
            f"  Avg branching:    {self.avg_branching_factor:.2f}",
            f"  Root entropy:     {self.root_visit_entropy:.3f}",
            f"  Avg KL div:       {self.avg_chance_kl:.4f}",
        ]
        return "\n".join(lines)


def entropy(distribution: Dict) -> float:
    """Shannon entropy of a distribution (dict of counts or probs)."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in distribution.values():
        p = v / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def kl_divergence(visit_counts: Dict, theoretical_probs: Dict) -> float:
    """KL(visit_distribution || theoretical_probs). Lower = better-calibrated."""
    total = sum(visit_counts.values())
    if total == 0:
        return 0.0
    kl = 0.0
    for key, count in visit_counts.items():
        q = count / total
        p = theoretical_probs.get(key, 0.0)
        if q > 0 and p > 0:
            kl += q * math.log(q / p)
    return kl
