#!/usr/bin/env python3
"""Cumulative cost breakdown by modality (paper Figure 3).

Standalone script for Figure 3 (fig:modality-stack). It writes two panels that
sit side by side in a single column:

* ``cumcost_image_stack.pdf`` -- the stacked-bar cumulative cost breakdown for
  the image modality (FLUX.1-dev) at g = 20%/yr, one triplet of bars per year
  (full HDD, full tape, two-tier IR), each stacked into its cost components,
  with a shared legend laid out in two rows below the axes.

* ``cumcost_lines.pdf`` -- a line chart of the cumulative cost for every year
  from 2025 to 2045 for storing everything on HDD, storing everything on tape,
  and the two-tier IR (cache + IR cold store, recompute on miss) policy for the
  image, music, and molecule modalities. HDD and tape are modality-independent
  storage baselines; only the two-tier IR curve differs by modality.

Carbon cost is excluded (we report carbon as emissions, not dollar cost).

The generation cost c and the two-tier store factors k1 (store size) and k2
(decode compute) are the per-modality values of Table tab:modality-ex; the
object size S differs by modality.
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import cost_model as cm  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"

# one representative model per modality (Table tab:modality-ex):
# file id, label, c [TFLOP], S [PB], k1, k2
MODALITIES = [
    ("image",    "Image (FLUX.1-dev)",     2100.0, 3.0e-9,  1.0 / 6,   1.0 / 200),
    ("music",    "Music (MusicGen Large)",   21.0, 5.1e-9,  1.0 / 212, 1.0 / 470),
    ("molecule", "Molecule (Uni-Mol)",        5.0, 5.0e-12, 1.0 / 80,  1.0 / 7000),
]
G = 0.20
YEARS = [2030, 2035, 2040, 2045, 2050]    # bar-chart years
LINE_YEARS = list(range(2025, 2046))      # line-chart years, 2025..2045

# Costs are reported normalized to the first-year (2025) store-everything HDD
# total cost, so the y-axis is dimensionless and the HDD line starts at 1.0.
# The anchor is 1000 * p_hdd(2025) [$/PB] + alpha(2025) [$/PB/yr] for 1 PB,
# which is modality-independent and identical to the anchor used in
# breakeven_policy_cost.py.
ANCHOR = 1000.0 * cm.p_hdd(cm.YEAR0) + cm.alpha(cm.YEAR0)

# cost components, stacked bottom-to-top, with colours (carbon excluded).
# A bar fills only the components it has: the HDD/two-tier bars use HDD capital,
# GPU compute, and electricity; the tape bar splits into media, operational
# infrastructure (drives + library), and electricity.
COMPONENTS = [
    ("hdd_cap",    "HDD capital",       "#9ecae1"),
    ("tape_media", "Tape media",        "#9467bd"),
    ("tape_infra", "Drives + library",  "#8c564b"),
    ("gpu",        "GPU compute",       "#e34a33"),
    ("elec",       "Electricity",       "#bcbd22"),
]

# line-chart colours: gray/brown for the two storage baselines, one colour per
# modality for the two-tier IR curves.
LINE_COLORS = {
    "hdd":  "0.45",
    "tape": "#8c564b",
    "image":    "#e34a33",
    "music":    "#1f77b4",
    "molecule": "#2ca02c",
}


def stacked_bar(ax, x, breakdown, width, hatch):
    """Draw one stacked bar at position x from a component dict (values in $,
    normalized by ANCHOR to first-year store-everything cost = 1).
    Components absent from ``breakdown`` contribute nothing."""
    bottom = 0.0
    for key, _label, color in COMPONENTS:
        h = breakdown.get(key, 0.0) / ANCHOR
        if h == 0.0:
            continue
        ax.bar(x, h, width, bottom=bottom, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.4)
        bottom += h
    return bottom


def tape_components(g, year):
    """Tape store-everything breakdown through ``year``, split into media,
    operational infrastructure (drives + library), and electricity (the small
    mounted-drive and controller power)."""
    tb = cm.tape_storage_cost_full(g, year)
    return {"tape_media": tb["media"],
            "tape_infra": tb["drives"] + tb["library"],
            "elec": tb["power"]}


def legend_handles():
    """Legend entries: the five cost components only.  The three bar types
    (full HDD, full tape, IR regen) are identified by an in-plot key drawn by
    ``bar_type_key`` instead of the legend."""
    return [Patch(facecolor=col, label=lab) for _k, lab, col in COMPONENTS]


# The three bars in every year group, in left-to-right order, with the hatch
# that identifies each and a decreasing height for the upper-left key so the
# key mirrors the usual HDD > tape > IR-regen ordering of the real bars.
BAR_TYPES = [
    ("Full HDD",  "",    1.00),
    ("Full tape", "..",  0.70),
    ("LazIR",     "///", 0.38),
]


def bar_type_key(ax):
    """Draw a small key in the upper-left corner: three mini bars of decreasing
    height, one per storage/regen choice, hatched to match the three bars in
    each year group so a reader maps each bar to its choice directly."""
    key = ax.inset_axes([0.02, 0.66, 0.36, 0.20])
    xs = np.arange(len(BAR_TYPES))
    for x, (_lbl, hatch, h) in zip(xs, BAR_TYPES):
        key.bar(x, h, 1.0, color="0.82", edgecolor="0.25",
                linewidth=0.6, hatch=hatch)
    key.set_xlim(-0.55, len(BAR_TYPES) - 0.45)
    key.set_ylim(0, 1.05)
    key.set_xticks(xs)
    key.set_xticklabels([lbl for lbl, _h, _v in BAR_TYPES], fontsize=10)
    key.tick_params(axis="x", length=0, pad=3)
    key.set_yticks([])
    for spine in key.spines.values():
        spine.set_visible(False)
    key.patch.set_alpha(0.0)


def plot_image_stack():
    """Panel 1: image stacked-bar breakdown with a two-row legend below."""
    fid, label, c, S, k1, k2 = MODALITIES[0]
    # wide but tall enough that the two-row y-label fits along the axis
    fig, ax = plt.subplots(figsize=(7.7, 3.3))
    width = 0.27
    xs = np.arange(len(YEARS))
    for i, year in enumerate(YEARS):
        full, two = cm.cost_breakdown(G, S, k1, k2, c, year)
        full_cap = {"hdd_cap": full["hdd"], "gpu": full["gpu"], "elec": full["elec"]}
        two_cap = {"hdd_cap": two["hdd"], "gpu": two["gpu"], "elec": two["elec"]}
        stacked_bar(ax, xs[i] - width, full_cap, width, hatch="")
        stacked_bar(ax, xs[i], tape_components(G, year), width, hatch="..")
        stacked_bar(ax, xs[i] + width, two_cap, width, hatch="///")

    ax.set_xticks(xs)
    # x axis is elapsed duration in years from the 2025 start, so 2030..2050
    # read as 5, 10, 15, 20, 25.
    ax.set_xticklabels([y - 2025 for y in YEARS])
    ax.set_xlabel("Duration (in years)")
    # two-row y-label so the full wording fits along the axis without clipping.
    ax.set_ylabel("Cumulative Cost\n(normalized to first year cost)",
                  fontsize=11)
    ax.grid(True, axis="y", ls=":", alpha=0.45)

    # legend: the five cost components only, one row just above the axes
    ax.legend(handles=legend_handles(), loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=10,
              frameon=False, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.0)

    # bar-type key (upper-left): maps each of the three bars in a group to its
    # storage/regen choice, replacing the bar-type entries in the legend
    bar_type_key(ax)

    fig.tight_layout()
    out = FIGDIR / "cumcost_image_stack.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_cost_lines():
    """Panel 2: cumulative cost per year (2025-2045) for full HDD, full tape,
    and two-tier IR for image, music, and molecule."""
    fig, ax = plt.subplots(figsize=(3.6, 3.7))
    years = np.array(LINE_YEARS)

    # storage baselines are modality-independent; compute once (use image S).
    _, _, c0, S0, k10, k20 = MODALITIES[0]
    hdd = np.array([sum(cm.cost_breakdown(G, S0, k10, k20, c0, y)[0][k]
                        for k in ("hdd", "gpu", "elec")) for y in years]) / ANCHOR
    tape = np.array([cm.tape_storage_cost_full(G, y)["total"]
                     for y in years]) / ANCHOR

    ax.plot(years, hdd, color=LINE_COLORS["hdd"], lw=1.8, marker="o", ms=3,
            label="full HDD")
    ax.plot(years, tape, color=LINE_COLORS["tape"], lw=1.8, marker="s", ms=3,
            label="full tape")

    markers = {"image": "^", "music": "D", "molecule": "v"}
    for fid, label, c, S, k1, k2 in MODALITIES:
        two = np.array([sum(cm.cost_breakdown(G, S, k1, k2, c, y)[1][k]
                            for k in ("hdd", "gpu", "elec")) for y in years]) / ANCHOR
        name = label.split(" (")[0]
        ax.plot(years, two, color=LINE_COLORS[fid], lw=1.6,
                marker=markers[fid], ms=3, label=f"two-tier {name}")

    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cost (First Year = 1)")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1.6)

    fig.tight_layout()
    out = FIGDIR / "cumcost_lines.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    apply_style()
    FIGDIR.mkdir(exist_ok=True)
    plot_image_stack()
    plot_cost_lines()


if __name__ == "__main__":
    main()
