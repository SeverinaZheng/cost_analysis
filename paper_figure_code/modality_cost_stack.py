#!/usr/bin/env python3
"""Stacked-bar cost breakdown by modality, one figure per modality.

Mimics ``image_cost_stack.py`` but sweeps the three generative modalities we
study, one representative model each, instead of three image models:
FLUX.1-dev (image), MusicGen Large (music), and Uni-Mol (molecule).

For each modality we plot the cumulative cost through 2030, 2035, 2040, 2045 at
a corpus growth rate g = 20%/yr.  Each year shows two stacked bars: the
full-fidelity baseline that stores every object on HDD, and the two-tier
design that keeps a Zipfian hot cache plus a compact intermediate-representation
cold store and recomputes on a miss.  Each bar is stacked into its cost
components: HDD capital, amortised GPU compute, and electricity.

Carbon cost is excluded (we report carbon as emissions, not dollar cost).

The generation cost c and the two-tier store factors k1 (store size) and k2
(decode compute) are the per-modality values of Table tab:modality-ex; the
object size S differs by modality.

Figures: figures/stack_image.pdf, stack_music.pdf, stack_molecule.pdf
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
YEARS = [2030, 2035, 2040, 2045]

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


def stacked_bar(ax, x, breakdown, width, hatch):
    """Draw one stacked bar at position x from a component dict (values in $M).
    Components absent from ``breakdown`` contribute nothing."""
    bottom = 0.0
    for key, _label, color in COMPONENTS:
        h = breakdown.get(key, 0.0) / 1e6
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


def main():
    apply_style()
    FIGDIR.mkdir(exist_ok=True)
    width = 0.27

    for fid, label, c, S, k1, k2 in MODALITIES:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        xs = np.arange(len(YEARS))
        for i, year in enumerate(YEARS):
            full, two = cm.cost_breakdown(G, S, k1, k2, c, year)
            full_cap = {"hdd_cap": full["hdd"], "gpu": full["gpu"], "elec": full["elec"]}
            two_cap = {"hdd_cap": two["hdd"], "gpu": two["gpu"], "elec": two["elec"]}
            stacked_bar(ax, xs[i] - width, full_cap, width, hatch="")
            stacked_bar(ax, xs[i], tape_components(G, year), width, hatch="..")
            stacked_bar(ax, xs[i] + width, two_cap, width, hatch="///")

        ax.set_xticks(xs)
        ax.set_xticklabels(YEARS)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Cost [million USD]")
        ax.grid(True, axis="y", ls=":", alpha=0.45)

        fig.tight_layout()
        out = FIGDIR / f"stack_{fid}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    write_legend()


def write_legend():
    """Write a single shared legend (cost components + bar types) used across
    all three modality panels in the LaTeX figure."""
    fig = plt.figure(figsize=(13.0, 0.4))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # all entries on a single row: 5 cost components + 3 bar types
    handles = [Patch(facecolor=col, label=lab) for _k, lab, col in COMPONENTS]
    handles += [
        Patch(facecolor="0.8", edgecolor="white", label="full HDD"),
        Patch(facecolor="0.8", edgecolor="white", hatch="..", label="full tape"),
        Patch(facecolor="0.8", edgecolor="white", hatch="///", label="two-tier"),
    ]
    ax.legend(handles=handles, loc="center", ncol=len(handles), fontsize=9,
              frameon=False, handlelength=1.3, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.0)

    out = FIGDIR / "stack_legend.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
