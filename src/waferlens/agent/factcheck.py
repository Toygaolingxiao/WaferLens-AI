"""Numeric fact-check for the explanation agent.

The agent may only state numbers that were actually computed by L3/L4. Before
any explanation is shown, the numbers it claims are extracted and compared
against the computed values; a mismatch blocks or rewrites the output. The
agent explains — it never invents a figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FactCheckResult:
    ok: bool
    mismatches: list[str]


def check_numbers(
    claimed: dict[str, float],
    computed: dict[str, float],
    rel_tol: float = 0.01,
) -> FactCheckResult:
    """Compare claimed figures against computed ones.

    Every claimed key must exist in ``computed`` and match within ``rel_tol``
    (relative). Unknown keys count as mismatches: citing a number nobody
    computed is exactly the failure this guard exists to catch.
    """
    mismatches: list[str] = []
    for key, value in claimed.items():
        if key not in computed:
            mismatches.append(f"{key}: claimed {value}, not a computed value")
            continue
        reference = computed[key]
        tolerance = max(abs(reference) * rel_tol, 1e-9)
        if abs(value - reference) > tolerance:
            mismatches.append(f"{key}: claimed {value}, computed {reference}")
    return FactCheckResult(ok=not mismatches, mismatches=mismatches)
