#!/usr/bin/env python3
"""Stacked-bar cost breakdown for image generation, one figure per model.

For each of three representative image-generation models we plot the
cumulative cost through 2025, 2030, ..., 2045 at a corpus growth rate
g = 20%/yr.  Each year shows two stacked bars: the full-fidelity baseline that
stores every object on HDD, and the two-tier design that keeps a Zipfian hot
cache plus a compact intermediate-representation cold store and recomputes on
a miss.  Each bar is stacked into its cost components: HDD capital, amortised
GPU compute, electricity, and carbon.

The generation cost c and the two-tier store factors k1 (store size) and k2
(decode compute) are the per-model values of Table tab:gen-cost in the Image
Generation subsection; the object size S = 3 MB is common.

Figures: figures/stack_sd15.pdf, stack_sd3med.pdf, stack_flux1dev.pdf
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

# image-generation models from Table tab:gen-cost:
# file id, label, c [TFLOP], S [PB], k1, k2
MODELS = [
    ("sd15",     "SD 1.5",      41.0, 3.0e-9, 1.0 / 24, 1.0 / 16),
    # ("sd3med",   "SD 3 Med",   501.0, 3.0e-9, 1.0 / 6,  1.0 / 50),
    # SDXL: 4-channel VAE (k1 = 1/24, like SD 1.5) but 1024px decode keeps
    # k2*c ~= 10 TFLOP, so it differs from FLUX in the storage tier only.
    ("sdxl",     "SDXL",       800.0, 3.0e-9, 1.0 / 24, 1.0 / 80),
    ("flux1dev", "FLUX.1-dev", 2100.0, 3.0e-9, 1.0 / 6,  1.0 / 200),
]
G = 0.20
YEARS = [2030, 2035, 2040, 2045]

# cost components, stacked bottom-to-top, with colours
COMPONENTS = [
    ("hdd",    "HDD capital", "#9ecae1"),
    ("gpu",    "GPU compute", "#e34a33"),
    ("elec",   "Electricity", "#bcbd22"),
    ("carbon", "Carbon",      "#7f7f7f"),
]


def stacked_bar(ax, x, breakdown, width, hatch):
    """Draw one stacked bar at position x from a component dict (values in $M)."""
    bottom = 0.0
    for key, _label, color in COMPONENTS:
        h = breakdown[key] / 1e6
        ax.bar(x, h, width, bottom=bottom, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.4)
        bottom += h
    return bottom


def main():
    apply_style()
    FIGDIR.mkdir(exist_ok=True)
    width = 0.38

    for fid, label, c, S, k1, k2 in MODELS:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        xs = np.arange(len(YEARS))
        for i, year in enumerate(YEARS):
            full, two = cm.cost_breakdown(G, S, k1, k2, c, year)
            stacked_bar(ax, xs[i] - width / 2, full, width, hatch="")
            stacked_bar(ax, xs[i] + width / 2, two, width, hatch="///")

        ax.set_xticks(xs)
        ax.set_xticklabels(YEARS)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Cost [million USD]")
        ax.set_title(f"{label}   c = {c:.0f} TF,  g = {int(G*100)}%",
                     fontsize=12)
        ax.grid(True, axis="y", ls=":", alpha=0.45)

        comp_handles = [Patch(facecolor=col, label=lab)
                        for _k, lab, col in COMPONENTS]
        strat_handles = [
            Patch(facecolor="0.8", edgecolor="white", label="full HDD"),
            Patch(facecolor="0.8", edgecolor="white", hatch="///",
                  label="two-tier"),
        ]
        leg1 = ax.legend(handles=comp_handles, loc="upper left", fontsize=9,
                         title="component")
        ax.add_artist(leg1)
        ax.legend(handles=strat_handles, loc="upper center", fontsize=9,
                  title="left vs right bar")

        fig.tight_layout()
        out = FIGDIR / f"stack_{fid}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
