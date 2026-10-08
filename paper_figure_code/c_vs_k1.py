#!/usr/bin/env python3
"""Ablation: break-even generation cost c* vs IR-store size fraction k1.

k1 is the size of the stored intermediate representation as a fraction of the
full object.  For each modality we fix S, k2, growth g = 0.20 and Zipf skew
s = 1, sweep k1, and plot the break-even per-object generation cost c*.  A
filled marker shows each modality's actual operating point (its real k1),
and a horizontal dotted line shows its actual generation cost.

Figure: figures/c_vs_k1.pdf
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
    k1_grid = np.logspace(-3, 0, 60)        # 1e-3 .. 1

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, model, c_gen, S, k1_act, k2 in [m for m in cm.MODALITIES if m[0] != "Molecule"]:
        cstar = np.array([cm.breakeven_c(g=cm.G_DEFAULT, S=S, k1=k1, k2=k2,
                                         s=1.0) for k1 in k1_grid])
        ax.plot(k1_grid, cstar, color=COLORS[name], label=f"{name} ({model})")
        ax.axhline(c_gen, color=COLORS[name], ls=":", lw=2.4, alpha=0.95)
        # actual operating point
        c_at = cm.breakeven_c(g=cm.G_DEFAULT, S=S, k1=k1_act, k2=k2, s=1.0)
        ax.plot(k1_act, c_at, color=COLORS[name], marker=MARKERS[name],
                markersize=8, ls="none", markeredgecolor="black",
                markeredgewidth=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"IR-store Size Fraction $k_{\text{size}}$", fontsize=18)
    ax.set_ylabel("Break-even #Flops", fontsize=18)
    ax.set_xticks([1e-3, 1e-1, 1e0])                # fewer x ticks
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(NullLocator())       # drop minor ticks
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="both", ls=":", alpha=0.45)
    ax.tick_params(axis="both", which="major", labelsize=18)  # larger tick font
    # legend is emitted once by ablation_legend.py, not repeated per panel
    fig.tight_layout()

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "c_vs_k1.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
