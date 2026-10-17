"""Small-sample statistics used in the analysis."""

import math


def wilson(p, n, z=1.96):
    """Wilson score interval for a proportion p observed over n trials."""
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def mcnemar(pairs):
    """Exact two-sided McNemar test on paired (baseline, treatment) 0/1 outcomes.

    Returns (lost, gained, p): problems the treatment broke, problems it fixed.
    """
    lost = sum(1 for a, b in pairs if a == 1 and b == 0)
    gained = sum(1 for a, b in pairs if a == 0 and b == 1)
    n = lost + gained
    if n == 0:
        return lost, gained, 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1))
    return lost, gained, min(1.0, 2 * tail / 2 ** n)
