#!/usr/bin/env python3
"""Stacked-bar policy cost comparison for the Civitai image-retrieval replay
(Section 5.3, sec:ir-replay).

For three LRU cache sizes measured on the trace (1%, 5%, 10% of the working
set, with whole-trace miss ratios 33.8%, 18.6%, 12.8%), we compare the total
cost of five serving policies:

  1) Full storage on HDD   -- persist every distinct image once on HDD.
  2) Full storage on tape  -- persist every distinct image once on LTO tape.
  3) Full regeneration     -- keep nothing, regenerate every request at c.
  4) Cache + regen on miss  -- hot cache, regenerate a full object on a miss.
  5) Cache + IR cold store  -- hot cache plus a compact latent cold store,
                               decode a miss at k2*c instead of the full c.

The two storage baselines and full regeneration do not depend on the cache
size, so they repeat across the three cache-size groups; only the two
cache-backed policies move, and each is costed with the miss ratio of its own
cache size.  Every bar is stacked into its cost components (HDD capital, tape
media, tape drives + library, GPU compute, electricity).  Prices are held at
their 2025 anchors (as in the section text), matched to the 2.87-year trace.

Figure: figures/civitai_policy_cost_stack.pdf
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

# ── trace facts (sec:ir-replay) ─────────────────────────────────────────────
N_DISTINCT = 92.3e6          # distinct images in the trace
N_REQ = 2.07e9               # total retrieval requests
SPAN_YR = 2.87               # trace span [yr]
S_IMG_TB = 3.0e-6            # image size [TB] (3 MB)
FULL_TB = N_DISTINCT * S_IMG_TB          # 276.9 TB to store everything
FULL_PB = FULL_TB / 1000.0

# generator (FLUX.1-dev, Table tab:modality-ex)
C_GEN = 2100.0               # full generation cost [TFLOP]
K1 = 1.0 / 6                 # latent store fraction
K2 = 1.0 / 200               # decode-on-miss compute fraction

# LRU cache sizes measured on the trace: fraction of distinct objects and the
# corresponding whole-trace miss ratio (Figure fig:civitai-trace).
CACHE_SIZES = [
    (0.01, 0.338, "1%"),
    (0.05, 0.186, "5%"),
    (0.10, 0.128, "10%"),
]

# ── prices, held at 2025 anchors (single-point trace replay) ────────────────
P_HDD = cm.p_hdd(cm.YEAR0)                # $/TB
P_TF = cm.p_tflop(cm.YEAR0)              # $/TFLOP amortised GPU capital
BETA = cm.BETA                           # $/TFLOP GPU electricity
ALPHA_TB_YR = cm.ALPHA / 1000.0          # $/TB/yr HDD electricity (700 $/PB/yr)

# stacked-bar policies: the three low-cost strategies, so a linear axis is
# readable (full regeneration and cache+regen dwarf these and go in the line
# chart instead).  Order is left-to-right within each cache-size group; each
# mode gets a hatch so the legend, not the x-axis, identifies the bars.
BAR_POLICIES = ["HDD", "Tape", "LazIR"]
MODE_HATCH = {"HDD": "", "Tape": "///", "LazIR": "xxx"}

# stacked cost components, bottom-to-top, with colours
COMPONENTS = [
    ("hdd_cap",    "HDD capital",      "#9ecae1"),
    ("tape_media", "Tape media",       "#9467bd"),
    ("tape_infra", "Drives + library", "#8c564b"),
    ("gpu",        "GPU compute",      "#e34a33"),
    ("elec",       "Electricity",      "#bcbd22"),
]


def hdd_storage(volume_tb):
    """HDD capital + electricity to keep ``volume_tb`` powered over the trace."""
    cap = volume_tb * P_HDD
    elec = volume_tb * ALPHA_TB_YR * SPAN_YR
    return cap, elec


def full_hdd():
    cap, elec = hdd_storage(FULL_TB)
    return {"hdd_cap": cap, "elec": elec}


def full_tape():
    """Full tape store of a fixed FULL_PB corpus over the trace span: cartridge
    media, a minimum drive complement, library slots, and mounted-drive power."""
    media = FULL_TB * cm.P_TAPE0
    drives = cm.N_DRIVE_MIN * cm.P_DRIVE           # < drive life, no re-buy
    slots = np.ceil(FULL_TB / cm.TAPE_CART_TB)
    library = slots * cm.C_SLOT
    power = cm.alpha_tape(cm.YEAR0) * FULL_PB * SPAN_YR
    return {"tape_media": media, "tape_infra": drives + library, "elec": power}


def full_regen():
    """Regenerate every request from scratch at the full generation cost."""
    tflop = N_REQ * C_GEN
    return {"gpu": tflop * P_TF, "elec": tflop * BETA}


def cache_regen(frac, miss):
    """Hot cache of ``frac`` of distinct objects; misses regenerate in full."""
    cache_tb = frac * FULL_TB
    cap, elec_hdd = hdd_storage(cache_tb)
    miss_tflop = miss * N_REQ * C_GEN
    return {"hdd_cap": cap,
            "gpu": miss_tflop * P_TF,
            "elec": elec_hdd + miss_tflop * BETA}


def cache_ir(frac, miss):
    """Hot cache plus a k1-sized latent cold store; misses decode at k2*c."""
    cache_tb = frac * FULL_TB
    ir_tb = K1 * FULL_TB
    cap, elec_cache = hdd_storage(cache_tb)
    ir_cap, elec_ir = hdd_storage(ir_tb)
    miss_tflop = miss * N_REQ * (K2 * C_GEN)
    return {"hdd_cap": cap + ir_cap,
            "gpu": miss_tflop * P_TF,
            "elec": elec_cache + elec_ir + miss_tflop * BETA}


def policy_breakdown(name, frac, miss):
    """Component breakdown dict for a single named policy at one cache size."""
    return {
        "HDD": full_hdd(),
        "Tape": full_tape(),
        "Full regen": full_regen(),
        "Cache + regen": cache_regen(frac, miss),
        "LazIR": cache_ir(frac, miss),
    }[name]


def _total(bd):
    return sum(bd.get(k, 0.0) for k, _l, _c in COMPONENTS)


def stacked_bar(ax, x, breakdown, width, hatch=""):
    """Draw one stacked bar at ``x`` on a linear axis; return its total."""
    bottom = 0.0
    for key, _label, color in COMPONENTS:
        h = breakdown.get(key, 0.0)
        if h <= 0.0:
            continue
        ax.bar(x, h, width, bottom=bottom, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.4)
        bottom += h
    return bottom


# ── figure 1: stacked-bar breakdown of the three low-cost policies ──────────
def plot_stack():
    """Linear-scale stacked cost breakdown. HDD and Tape are cache-independent,
    so each is drawn as a single bar named on the x-axis; LazIR is drawn once per
    cache size, ticked by cache size with a "LazIR" group label on the row below.
    Only the cost components remain in the legend."""
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    bw = 0.7
    x_hdd, x_tape = 0.0, 1.0
    x_laz = [2.4, 3.4, 4.4]                     # LazIR group, one bar per cache size

    totals = []

    def _bar(x, name, frac, miss):
        total = stacked_bar(ax, x, policy_breakdown(name, frac, miss), bw,
                            hatch=MODE_HATCH[name])
        totals.append(total)
        ax.annotate(_money(total), xy=(x, total), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=12)

    # HDD and Tape do not depend on cache size -> a single bar each
    _bar(x_hdd, "HDD", *CACHE_SIZES[0][:2])
    _bar(x_tape, "Tape", *CACHE_SIZES[0][:2])
    # LazIR moves with cache size -> one bar per cache size
    for xi, (frac, miss, _lbl) in zip(x_laz, CACHE_SIZES):
        _bar(xi, "LazIR", frac, miss)

    ax.set_ylim(0, max(totals) * 1.14)
    ax.set_ylabel("Total Cost [USD]", fontsize=17)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.spines["top"].set_visible(False)     # no borderline box around the plot
    ax.spines["right"].set_visible(False)

    # x ticks: HDD / Tape name their single bars; the three LazIR bars share one
    # "LazIR" tick, their cache sizes being annotated above the bars instead.
    x_laz_c = sum(x_laz) / len(x_laz)
    ax.set_xticks([x_hdd, x_tape, x_laz_c])
    ax.set_xticklabels(["HDD", "Tape", "LazIR"])
    ax.set_xlim(-0.6, 5.0)

    # light grey band behind the LazIR group; "Cache size" heads the band with
    # the three cache sizes directly beneath it, one above each LazIR bar.
    trans = ax.get_xaxis_transform()
    ax.axvspan(x_laz[0] - 0.6, x_laz[-1] + 0.55, facecolor="0.9",
               edgecolor="none", zorder=0)
    ax.text(x_laz_c, 0.90, "Cache size", transform=trans,
            ha="center", va="top", fontsize=15, color="0.35")
    for xi, (_f, _m, lbl) in zip(x_laz, CACHE_SIZES):
        ax.text(xi, 0.80, lbl, transform=trans, ha="center", va="top",
                fontsize=15, color="0.2")

    # legend: cost components only (HDD / Tape / LazIR are now on the x-axis),
    # broken into three rows via ncol=2 so the font can be larger.
    present = ["hdd_cap", "tape_media", "tape_infra", "gpu", "elec"]
    comp_handles = [Patch(facecolor=col, label=lab)
                    for k, lab, col in COMPONENTS if k in present]
    ax.legend(handles=comp_handles, loc="lower center",
              bbox_to_anchor=(0.5, 0.80), bbox_transform=fig.transFigure,
              ncol=3, fontsize=15,
              frameon=False, handlelength=1.1, handletextpad=0.35,
              columnspacing=1.0, borderaxespad=0.0)

    fig.subplots_adjust(left=0.215, right=0.98, top=0.79, bottom=0.15)
    out = FIGDIR / "civitai_policy_cost_stack.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ── figure 2: total cost vs cache size for the regeneration policies ────────
def plot_lines():
    """Total serving cost (no breakdown) vs cache size for the three
    regeneration policies, on a log y-axis."""
    lines = [
        ("Full regen",       "Full regen",       "#e34a33", "o"),
        ("Cache + regen",    "Cache + regen",    "#1f77b4", "s"),
        ("LazIR", "LazIR", "#0d6217", "^"),
    ]
    x = np.arange(len(CACHE_SIZES))

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for name, label, color, marker in lines:
        y = [_total(policy_breakdown(name, frac, miss))
             for frac, miss, _lbl in CACHE_SIZES]
        ax.plot(x, y, color=color, marker=marker, markersize=7,
                linewidth=2.0, label=label)
        for xi, yi in zip(x, y):
            ax.annotate(_money(yi), xy=(xi, yi), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        va="bottom", fontsize=10, color=color)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(miss {miss:.1%})"
                        for _f, miss, lbl in CACHE_SIZES])
    ax.set_xlabel("Cache Size (Fraction of Working Set)")
    ax.set_ylabel("Total Cost [USD]")
    ax.set_xlim(-0.3, len(CACHE_SIZES) - 0.7)
    ax.grid(True, axis="both", ls=":", alpha=0.45)
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = FIGDIR / "civitai_policy_cost_lines.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_bars():
    """Same data as ``plot_lines`` (total serving cost vs cache size for the
    three regeneration policies) drawn as grouped bars on a log y-axis, written
    to a separate PDF.  ``plot_lines`` is intentionally left untouched."""
    # same palette as Figure 4 (breakeven_policy_cost.py), one colour + hatch
    # per policy so the three groups read consistently with that figure.
    series = [
        ("Full regen",    "Regen from Prompt",         "#1f77b4", "///"),
        ("Cache + regen", "Cache + Regen from Prompt", "#2ca02c", "..."),
        ("LazIR",         "LazIR",                     "#ff7f0e", "xxx"),
    ]
    x = np.arange(len(CACHE_SIZES))
    n = len(series)
    width = 0.8 / n
    ymin = 1e3

    # figsize chosen so the tight-cropped PDF matches the stacked-bar panel
    # (plot_stack, Fig. 8a); at equal 0.49-column width the two panels then
    # render at the same size with their x-axes aligned. 8b's single-row legend
    # crops smaller than 8a's two-row legend at equal figsize, so 8b is enlarged
    # to compensate.
    fig, ax = plt.subplots(figsize=(5.97, 4.72))
    for i, (name, label, color, hatch) in enumerate(series):
        y = [_total(policy_breakdown(name, frac, miss))
             for frac, miss, _lbl in CACHE_SIZES]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, y, width, color=color, label=label,
               edgecolor="white", linewidth=0.5, hatch=hatch, zorder=3)
        # nudge the green/orange labels rightward so the taller neighbouring
        # bar does not shade them; the first (blue) series stays centred.
        dx, ha = (0, "center") if i == 0 else (-12, "left")
        for xi, yi in zip(x + offset, y):
            ax.annotate(_money(yi), xy=(xi, yi), xytext=(dx, 3),
                        textcoords="offset points", ha=ha,
                        va="bottom", fontsize=10, color=color)

    ax.set_yscale("log")
    ax.set_ylim(ymin, 5e6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(miss {miss:.1%})"
                        for _f, miss, lbl in CACHE_SIZES])
    ax.set_xlabel("Cache Size (Fraction of Working Set)", fontsize=15)
    ax.set_ylabel("Total Cost [USD]", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(-0.5, len(CACHE_SIZES) - 0.5)
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.spines["top"].set_visible(False)     # no borderline box around the plot
    ax.spines["right"].set_visible(False)

    # single legend row placed at the same height as Fig. 8a's SECOND (component)
    # row, leaving the first-row space above it empty as padding, so both panels
    # reserve the same top space and their x-axes align.
    ROW2_Y = 0.815
    # labels are long, so use two columns spanning the reserved two-row top space
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, ROW2_Y),
              bbox_transform=fig.transFigure, ncol=2,
              fontsize=15, frameon=False, columnspacing=1.2,
              handlelength=1.4, handletextpad=0.4, borderaxespad=0.0)

    # fixed axes rectangle identical to plot_stack (Fig. 8a), saved without a
    # tight bbox so both PDFs are the same size and the x-axes line up.
    fig.subplots_adjust(left=0.18, right=0.975, top=0.80, bottom=0.21)
    out = FIGDIR / "civitai_policy_cost_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _money(v):
    """Compact dollar label for a bar top or marker."""
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def main():
    apply_style()
    FIGDIR.mkdir(exist_ok=True)
    plot_stack()
    plot_lines()
    plot_bars()

    # echo numbers for the paper text
    all_names = ["HDD", "Tape", "Full regen",
                 "Cache + regen", "LazIR"]
    for frac, miss, lbl in CACHE_SIZES:
        print(f"\ncache {lbl} (miss {miss:.1%}):")
        for name in all_names:
            bd = policy_breakdown(name, frac, miss)
            parts = "  ".join(f"{k}=${bd[k]:,.0f}" for k, _l, _c in COMPONENTS
                              if bd.get(k, 0.0) > 0.0)
            print(f"  {name:16s} total=${_total(bd):,.0f}   [{parts}]")


if __name__ == "__main__":
    main()
