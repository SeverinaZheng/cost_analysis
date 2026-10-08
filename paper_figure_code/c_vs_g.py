#!/usr/bin/env python3
"""Ablation: break-even generation cost c* vs corpus growth rate g.

For each of the three representative modalities (Image / Audio / Molecule)
we fix that modality's object size S and two-tier factors (k1, k2) and Zipf
skew s = 1, then sweep the annual corpus growth rate g.  The curve is the
break-even per-object generation cost c*: objects cheaper than c* favour the
two-tier (cache + IR cold store, regenerate on miss) design.  Each modality's
actual generation cost is drawn as a horizontal reference line.

Figure: figures/c_vs_g.pdf
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import cost_model as cm  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"
COLORS = {"Image": "#e34a33", "Audio": "#0d6217", "Molecule": "#6a51a3"}
MARKERS = {"Image": "o", "Audio": "s", "Molecule": "^"}


def main():
    apply_style()
    g_grid = np.linspace(0.0, 0.40, 41)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, model, c_gen, S, k1, k2 in [m for m in cm.MODALITIES if m[0] != "Molecule"]:
        cstar = np.array([cm.breakeven_c(g=g, S=S, k1=k1, k2=k2, s=1.0)
                          for g in g_grid])
        ax.plot(g_grid * 100, cstar, color=COLORS[name],
                marker=MARKERS[name], markevery=8, markersize=5,
                label=f"{name} ({model})")
        ax.axhline(c_gen, color=COLORS[name], ls=":", lw=2.4, alpha=0.95)

    ax.set_yscale("log")
    ax.set_xlabel(r"Corpus Growth Rate $g$ [%/yr]", fontsize=18)
    ax.set_ylabel("Break-even #Flops", fontsize=18)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 20, 40])                      # fewer x ticks
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(NullLocator())       # drop minor ticks
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="both", ls=":", alpha=0.45)
    ax.tick_params(axis="both", which="major", labelsize=18)  # larger tick font
    # legend is emitted once by ablation_legend.py, not repeated per panel
    fig.tight_layout()

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "c_vs_g.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
