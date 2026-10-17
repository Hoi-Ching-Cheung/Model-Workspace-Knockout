"""Report figures and the paired significance table, built from results/*.csv.

Safe on partial CSVs: any cell with fewer than MIN_N paired rows shows as n/a.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

from .stats import mcnemar, wilson

MIN_N = 10
ABLATIONS = ["random", "light", "medium", "heavy"]
DATASETS = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("aime24", "AIME 2024")]
ARMS = [("cot", "CoT"), ("no_cot", "no CoT")]

COLORS = {"clean": "#404040", "random": "#898781", "light": "#eda36a",
          "medium": "#e06720", "heavy": "#9c3403"}
INK_ON = {"clean": "white", "random": "white", "light": "#52514e",
          "medium": "white", "heavy": "white"}
LABELS = {"clean": "clean", "random": "random control",
          "light": "light (27–28)", "medium": "medium (26–29)",
          "heavy": "heavy (24–32)"}
MUTED, GRID = "#52514e", "#e1e0d9"


def load_results(results_dir):
    out = {}
    for name, _ in DATASETS:
        p = os.path.join(results_dir, f"{name}.csv")
        if os.path.exists(p):
            out[name] = pd.read_csv(p)
    return out


def _style(ax):
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _legend(ax, conds):
    hs = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in conds]
    ax.legend(hs, [LABELS[c] for c in conds], ncol=len(conds), fontsize=7,
              frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))


def _bars(ax, groups, conds, width):
    """groups: [(label, {cond: (value, err, n) | None})]"""
    gap = width * 0.1
    for gi, (_, vals) in enumerate(groups):
        for ci, cond in enumerate(conds):
            x = gi + (ci - (len(conds) - 1) / 2) * (width + gap)
            v = vals.get(cond)
            if v is None:
                ax.text(x, 0.01, "n/a", ha="center", va="bottom",
                        rotation=90, fontsize=6, color="#898781")
                continue
            val, err, n = v
            ax.bar(x, val, width, color=COLORS[cond], yerr=err, capsize=2,
                   error_kw=dict(lw=0.8, ecolor=MUTED))
            ax.text(x, 0.01, f"n={n}", ha="center", va="bottom", rotation=90,
                    fontsize=6, color=INK_ON[cond] if val > 0.12 else MUTED)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=8)
    _style(ax)


def _paired(sub, cond):
    a = sub[sub.condition == "clean"].set_index("id").correct
    b = sub[sub.condition == cond].set_index("id").correct
    ids = a.index.intersection(b.index)
    return a.loc[ids], b.loc[ids]


def _relative(sub, cond):
    """Accuracy under cond / paired clean accuracy, with a 95% CI half-width."""
    a, b = _paired(sub, cond)
    n = len(a)
    if n < MIN_N or a.mean() == 0:
        return None
    bm = b.mean()
    se = 3.0 / n if bm == 0 else 1.96 * math.sqrt(bm * (1 - bm) / n)
    return bm / a.mean(), se / a.mean(), n


def _absolute(sub, cond):
    b = sub[sub.condition == cond].correct
    n = len(b)
    if n < MIN_N:
        return None
    m = b.mean()
    se = 3.0 / n if m in (0.0, 1.0) else 1.96 * math.sqrt(m * (1 - m) / n)
    return m, se, n


def fig_cot_gap(res, path=None):
    """Clean accuracy along the difficulty axis, CoT vs no-CoT (H1)."""
    tiers = []
    def add(df, label):
        tiers.append((label, {arm: (df[df.arm == arm].correct.mean(),
                                    (df.arm == arm).sum())
                              for arm, _ in ARMS if (df.arm == arm).any()}))
    clean = {k: v[v.condition == "clean"] for k, v in res.items()}
    add(clean["gsm8k"], "GSM8K")
    for lvl in sorted(clean["math500"].level.dropna().unique()):
        add(clean["math500"][clean["math500"].level == lvl], f"MATH\nL{int(lvl)}")
    add(clean["aime24"], "AIME\n2024")

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for arm, color, dx, name in [("cot", "#2a78d6", -0.07, "CoT"),
                                 ("no_cot", "#898781", 0.07, "no CoT")]:
        pts = [(x + dx, *d[arm]) for x, (_, d) in enumerate(tiers) if arm in d]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cis = [wilson(acc, n) for _, acc, n in pts]
        err = [[y - lo for y, (lo, _) in zip(ys, cis, strict=True)],
               [hi - y for y, (_, hi) in zip(ys, cis, strict=True)]]
        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.5, zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt="o", ms=5, color=color,
                    capsize=2, lw=0.9, zorder=3, label=name)
    for x, (_, d) in enumerate(tiers):
        if "cot" in d and "no_cot" in d:
            c, nc = d["cot"][0], d["no_cot"][0]
            txt = "∞" if nc == 0 else (f"{c / nc:.0f}×" if c / nc >= 10 else f"{c / nc:.1f}×")
            ax.text(x, -0.005, txt, ha="center", va="top", fontsize=7,
                    color=MUTED, fontweight="bold")
        ns = "/".join(str(d[k][1]) for k, _ in ARMS if k in d)
        ax.text(x, -0.045, f"n={ns}", ha="center", va="top", fontsize=6, color="#898781")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([t for t, _ in tiers], fontsize=8)
    ax.set_ylim(-0.09, 1.05)
    _style(ax)
    ax.set_ylabel("accuracy (Wilson 95% CI)", fontsize=9)
    ax.set_title("Accuracy by difficulty — CoT vs no CoT (unablated)", fontsize=11, pad=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_ablation_vs_difficulty(res, path=None):
    """Each ablation's accuracy relative to its own arm's clean baseline (H2)."""
    groups = []
    for name, label in DATASETS:
        if name not in res:
            continue
        for arm, alabel in ARMS:
            sub = res[name][res[name].arm == arm]
            if not len(sub):
                continue
            base = sub[sub.condition == "clean"].correct.mean()
            groups.append((f"{label}\n({alabel})\nclean {base:.0%}",
                           {c: _relative(sub, c) for c in ABLATIONS}))
    fig, ax = plt.subplots(figsize=(9.8, 4.1))
    _bars(ax, groups, ABLATIONS, width=0.19)
    ax.axhline(1.0, color="#c3c2b7", lw=0.8, ls="--", zorder=0)
    ax.set_ylim(0, None)
    _legend(ax, ABLATIONS)
    ax.set_ylabel("accuracy / clean accuracy", fontsize=9)
    ax.set_title("J-space ablation vs task difficulty", fontsize=11, pad=30)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fig_math_levels(res, path=None):
    """MATH-500 by level, absolute accuracy. Relative accuracy is meaningless
    near the no-CoT floor (clean L3 is 1/30)."""
    conds = ["clean"] + ABLATIONS
    m = res["math500"]
    groups = []
    for lvl in sorted(m.level.dropna().unique()):
        for arm, alabel in ARMS:
            s = m[(m.level == lvl) & (m.arm == arm)]
            groups.append((f"L{int(lvl)}\n({alabel})", {c: _absolute(s, c) for c in conds}))
    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    _bars(ax, groups, conds, width=0.155)
    ax.set_ylim(0, 1.12)
    _legend(ax, conds)
    ax.set_ylabel("accuracy (95% CI)", fontsize=9)
    ax.set_title("MATH-500 by level", fontsize=11, pad=30)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def mcnemar_table(res):
    """Every ablation vs its arm's paired clean run."""
    rows = []
    for name, _ in DATASETS:
        if name not in res:
            continue
        for arm, _ in ARMS:
            sub = res[name][res[name].arm == arm]
            for cond in ABLATIONS:
                a, b = _paired(sub, cond)
                if not len(a):
                    continue
                lost, gained, p = mcnemar(list(zip(a, b, strict=True)))
                rows.append({"dataset": name, "arm": arm, "condition": cond,
                             "n": len(a), "clean": a.mean(), "ablated": b.mean(),
                             "lost": lost, "gained": gained, "p": p})
    return pd.DataFrame(rows).round(3)
