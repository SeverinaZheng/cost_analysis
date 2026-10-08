#!/usr/bin/env python3
"""Ablation: break-even generation cost c* vs Zipf skew s.

The request distribution is Zipf(s); a larger s concentrates requests on a
smaller hot set, shrinking the cache needed for a 90% hit ratio.  For each
modality we fix S, k1, k2 and growth g = 0.20, sweep s, and plot the
break-even per-object generation cost c*.  A filled marker shows the paper's
default operating point (s = 1); the horizontal dotted line is each modality's
actual generation cost.

Figure: figures/c_vs_zipf.pdf
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import cost_model as cm  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"
COLORS = {"Image": "#e34a33", "Audio": "#0d6217", "Molecule": "#6a51a3"}
MARKERS = {"Image": "o", "Audio": "s", "Molecule": "^"}


def main():
    apply_style()
    s_grid = np.linspace(0.6, 1.4, 33)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, model, c_gen, S, k1, k2 in cm.MODALITIES:
        cstar = np.array([cm.breakeven_c(g=cm.G_DEFAULT, S=S, k1=k1, k2=k2,
                                         s=s) for s in s_grid])
        ax.plot(s_grid, cstar, color=COLORS[name], label=f"{name} ({model})")
        ax.axhline(c_gen, color=COLORS[name], ls=":", lw=1.2, alpha=0.7)
        c_at = cm.breakeven_c(g=cm.G_DEFAULT, S=S, k1=k1, k2=k2, s=1.0)
        ax.plot(1.0, c_at, color=COLORS[name], marker=MARKERS[name],
                markersize=8, ls="none", markeredgecolor="black",
                markeredgewidth=0.6)

    ax.set_yscale("log")
    ax.set_xlabel(r"Zipf Skew $s$")
    ax.set_ylabel(r"Break-even Gen. Cost $c^{*}$ [TFLOP]")
    ax.set_xlim(0.6, 1.4)
    ax.grid(True, which="both", axis="both", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "c_vs_zipf.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
