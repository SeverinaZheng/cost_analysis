#!/usr/bin/env python3
r"""Standalone shared legend for the ablation panels (paper Figure 5).

The three ablation panels c_vs_g / c_vs_k1 / c_vs_k2 are now emitted without
their own legend so it is not repeated three times.  This produces a single
horizontal legend to place once above the row of subfigures.

Figure: figures/c_vs_ablation_legend.pdf
"""

import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cost_model as cm  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"
COLORS = {"Image": "#e34a33", "Audio": "#0d6217", "Molecule": "#6a51a3"}
MARKERS = {"Image": "o", "Audio": "s", "Molecule": "^"}


def main():
    apply_style()
    handles = [Line2D([0], [0], color=COLORS[n], marker=MARKERS[n], ms=6, lw=1.4,
                      markeredgecolor="black", markeredgewidth=0.5, label=n)
               for n, *_ in cm.MODALITIES if n != "Molecule"]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=2.4, label=r"actual $c$"))

    fig = plt.figure(figsize=(6.5, 0.38))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=11,
               frameon=False, columnspacing=1.4, handletextpad=0.5)

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "c_vs_ablation_legend.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
