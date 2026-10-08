#!/usr/bin/env python3
"""Total serving cost of the three regeneration policies vs storing everything.

For each of the three serving policies (fully regenerate, hot cache + regen on
miss, two-tier cache + IR cold store), we plot the cumulative 2025--2045 dollar
cost of serving each of the three modalities (image, music, molecule) under
both the steady 1 PB/yr corpus and the 20%/yr growing corpus.

x-axis: the three policies.  Within each policy group, every modality gets two
side-by-side bars (steady, growing).  Bars show the total cost only, no
component breakdown.  Two dashed lines mark the store-everything baseline
(steady and growing), so a bar below its line is the break-even win region.

Figure: figures/breakeven_policy_cost.pdf
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import cost_model as cm  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"

# modality: label, color, S [PB], c [TFLOP], k1, k2  (from working_examples.tex)
MODALITIES = [
    ("Image",    "#1f77b4", 3.0e-9,  2100.0, 1.0 / 6,   1.0 / 200),
    ("Music",    "#2ca02c", 5.1e-9,    21.0, 1.0 / 212, 1.0 / 470),
]
POLICIES = ["Regenerate\nfrom Prompt", "Cache +\nRegen from Prompt",
            "LazIR"]
CORPORA = [("steady", 0.0), ("growing", 0.20)]

YEARS = np.arange(cm.YEAR0, cm.HORIZON + 1)
AGE = YEARS - cm.YEAR0
PHDD = cm.p_hdd(YEARS)
PTF = cm.p_tflop(YEARS)
ALPH = cm.alpha(YEARS)          # $/PB/yr HDD electricity, declining 10%/yr
BET = cm.beta(YEARS)            # $/TFLOP GPU electricity, declining 30%/yr
H = cm.HIT_RATIO

# Costs are reported normalized to the first-year (2025) store-everything HDD
# total cost = 1000 * p_hdd(2025) [$/PB] + alpha(2025) [$/PB/yr] for 1 PB. This
# is the same anchor used in plot_cumulative_cost_breakdown.py, so the two
# figures share a common unit.
ANCHOR = 1000.0 * cm.p_hdd(cm.YEAR0) + cm.alpha(cm.YEAR0)


def storage_total(g):
    """Store-everything baseline cost [$]: replacement HDD capital + electricity."""
    new_data = (1.0 + g) ** AGE
    A = cm._replacement_series(new_data)
    cap = np.sum(1000.0 * A * PHDD)
    v_full = (AGE + 1.0) if g == 0 else ((1.0 + g) ** (AGE + 1) - 1.0) / g
    elec = np.sum(ALPH * v_full)
    return cap, elec


def policy_costs(g, S, c, k1, k2):
    """Return (full_regen, cache, two_tier) cumulative serving cost [$]."""
    new_data = (1.0 + g) ** AGE
    A = cm._replacement_series(new_data)
    cap = np.sum(1000.0 * A * PHDD)
    v_full = (AGE + 1.0) if g == 0 else ((1.0 + g) ** (AGE + 1) - 1.0) / g
    elec_full = np.sum(ALPH * v_full)

    n_cum = (AGE + 1.0) / S if g == 0 else ((1.0 + g) ** (AGE + 1) - 1.0) / (g * S)
    R = cm.REQ_PER_OBJ * n_cum
    ptf_beta = PTF + BET

    # fully regenerate every request
    full_regen = c * np.sum(R * ptf_beta)

    # hot cache + regenerate full object on miss
    frac = np.array([cm.cache_fraction(n) for n in n_cum])
    v_cache = frac * n_cum * S
    dv = np.diff(np.concatenate([[0.0], v_cache]))
    A_cache = cm._replacement_series(np.maximum(dv, 0.0))
    cache_hdd = np.sum(1000.0 * A_cache * PHDD)
    cache_elec = np.sum(ALPH * v_cache)
    miss_cost = np.sum((1.0 - H) * R * ptf_beta)        # per unit c
    cache = cache_hdd + cache_elec + c * miss_cost

    # two-tier: IR cold store (k1) + hot cache, decode miss at k2*c
    two_tier = (k1 * cap + cache_hdd) + (k1 * elec_full + cache_elec) + k2 * c * miss_cost
    return full_regen, cache, two_tier


def main():
    apply_style()
    FIGDIR.mkdir(exist_ok=True)

    base = {g: sum(storage_total(g)) / ANCHOR for _lbl, g in CORPORA}
    base_tape = {g: cm.tape_storage_cost_full(g)["total"] / ANCHOR
                 for _lbl, g in CORPORA}

    # cost[policy_idx][modality_idx][corpus_idx], normalized by ANCHOR
    cost = np.zeros((3, len(MODALITIES), len(CORPORA)))
    for mi, (_lbl, _col, S, c, k1, k2) in enumerate(MODALITIES):
        for ci, (_clbl, g) in enumerate(CORPORA):
            cost[:, mi, ci] = np.array(policy_costs(g, S, c, k1, k2)) / ANCHOR

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(3)                      # three policies
    bw = 0.13
    # (modality, corpus) bars per group, centred on each policy tick
    n_bars = len(MODALITIES) * len(CORPORA)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bw

    for mi, (_lbl, col, *_rest) in enumerate(MODALITIES):
        for ci, (clbl, _g) in enumerate(CORPORA):
            slot = mi * len(CORPORA) + ci
            hatch = "" if clbl == "steady" else "////"
            ax.bar(x + offsets[slot], cost[:, mi, ci], bw,
                   color=col, edgecolor="white", linewidth=0.5,
                   hatch=hatch, alpha=1.0 if clbl == "steady" else 0.85)

    # store-everything baselines: HDD (gray) and tape (brown)
    ax.axhline(base[0.0], ls="--", lw=1.0, color="0.35")
    ax.axhline(base[0.20], ls="-.", lw=1.0, color="0.35")
    ax.text(2.62, base[0.0] * 1.2, "HDD (linear)", fontsize=9,
            color="0.35", ha="right")
    ax.text(2.62, base[0.20] * 1.1, "HDD (exponential)", fontsize=9,
            color="0.35", ha="right")

    ax.axhline(base_tape[0.0], ls="--", lw=1.0, color="#8c564b")
    ax.axhline(base_tape[0.20], ls="-.", lw=1.0, color="#8c564b")
    ax.text(2.62, base_tape[0.0] * 0.7, "tape (linear)", fontsize=9,
            color="#8c564b", ha="right")
    ax.text(2.62, base_tape[0.20] * 0.7, "tape (exponential)", fontsize=9,
            color="#8c564b", ha="right")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES)
    ax.set_ylabel("Cumulative Cost\n(normalized to first year cost)",
                  fontsize=11)
    ax.grid(True, axis="y", ls=":", alpha=0.45)

    mod_handles = [Patch(facecolor=col, label=lbl)
                   for lbl, col, *_ in MODALITIES]
    corpus_handles = [
        Patch(facecolor="0.7", label="linear (1 PB/yr)"),
        Patch(facecolor="0.7", hatch="////", label="exponential (20%/yr)"),
    ]
    leg1 = ax.legend(handles=mod_handles, loc="upper right",
                     bbox_to_anchor=(1.0, 1.0), fontsize=8,
                     title="modality", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=corpus_handles, loc="upper right",
              bbox_to_anchor=(0.82, 1.0), fontsize=8,
              title="corpus", framealpha=0.9)

    fig.tight_layout()
    out = FIGDIR / "breakeven_policy_cost.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # echo numbers for the paper (normalized to first-year store-everything = 1)
    print(f"anchor (first-year store-everything HDD total) = ${ANCHOR:,.0f}")
    for mi, (lbl, *_r) in enumerate(MODALITIES):
        for ci, (clbl, _g) in enumerate(CORPORA):
            fr, ca, tt = cost[:, mi, ci]
            print(f"{lbl:9s} {clbl:8s}  full={fr:,.1f}x  cache={ca:,.1f}x  two-tier={tt:,.1f}x")
    print(f"store-all HDD  steady={base[0.0]:,.1f}x  growing={base[0.20]:,.1f}x")
    print(f"store-all tape steady={base_tape[0.0]:,.1f}x  growing={base_tape[0.20]:,.1f}x")


if __name__ == "__main__":
    main()
