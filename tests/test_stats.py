import pytest

from jspace.stats import mcnemar, wilson


def test_wilson_contains_estimate_and_stays_in_unit_interval():
    for p, n in [(0.0, 30), (1.0, 30), (0.2, 150), (0.927, 150)]:
        lo, hi = wilson(p, n)
        assert -1e-12 <= lo <= p <= hi <= 1 + 1e-12


def test_wilson_reference_value():
    lo, hi = wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_mcnemar_counts_and_exact_p():
    # AIME CoT, heavy ablation: 6 problems lost, none gained
    pairs = [(1, 0)] * 6 + [(1, 1)] * 2 + [(0, 0)] * 22
    assert mcnemar(pairs) == (6, 0, pytest.approx(0.03125))


def test_mcnemar_symmetric_and_degenerate():
    assert mcnemar([(1, 0)] * 5 + [(0, 1)] * 5)[2] == 1.0
    assert mcnemar([(1, 1), (0, 0)]) == (0, 0, 1.0)
