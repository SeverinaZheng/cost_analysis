#!/usr/bin/env python3
"""(Amount) Data-centre carbon emissions and electricity demand rise over time.

Left axis: IEA Base-Case data-centre electricity demand [TWh/yr], which more
than doubles from 270 TWh (2020) to 945 TWh (2030).
Right axis: IEA Base-Case data-centre CO2 emissions [Mt/yr], rising from
~180 Mt today toward a ~320 Mt peak by 2030.  Emissions grow more slowly than
electricity because the grid keeps decarbonising, but the absolute tonnes of
CO2 from compute and storage still climb steeply.

Reported IEA anchors are drawn as filled markers; interpolated/derived points
as open markers.  Sources are documented in carbon_data.py.

Figure: figures/carbon_amount_trend.pdf
"""

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import carbon_data as cd  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"
C_ELEC = "#0d6217"   # green  : electricity
C_CO2 = "#7b3294"    # purple : emissions


def _split(series):
    yr_r = [y for y, _v, r in series if r]
    v_r = [v for _y, v, r in series if r]
    yr_i = [y for y, _v, r in series if not r]
    v_i = [v for _y, v, r in series if not r]
    yr_all = [y for y, _v, _r in series]
    v_all = [v for _y, v, _r in series]
    return yr_all, v_all, yr_r, v_r, yr_i, v_i


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    ya, va, yr, vr, yi, vi = _split(cd.DC_ELECTRICITY_TWH)
    ax.plot(ya, va, color=C_ELEC, lw=2.0, zorder=2)
    ax.plot(yr, vr, "o", color=C_ELEC, ms=7, zorder=3, label="electricity (IEA)")
    ax.plot(yi, vi, "o", mfc="white", mec=C_ELEC, mew=1.5, ms=6, zorder=3)
    ax.set_ylabel("Data-centre Electricity [TWh/yr]", color=C_ELEC)
    ax.tick_params(axis="y", labelcolor=C_ELEC)
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1050)

    ax2 = ax.twinx()
    ya2, va2, yr2, vr2, yi2, vi2 = _split(cd.DC_CO2_MT)
    ax2.plot(ya2, va2, color=C_CO2, lw=2.0, ls="--", zorder=2)
    ax2.plot(yr2, vr2, "s", color=C_CO2, ms=7, zorder=3, label="CO$_2$ (IEA)")
    ax2.plot(yi2, vi2, "s", mfc="white", mec=C_CO2, mew=1.5, ms=6, zorder=3)
    ax2.set_ylabel("CO$_2$ Emissions [Mt/yr]", color=C_CO2)
    ax2.tick_params(axis="y", labelcolor=C_CO2)
    ax2.set_ylim(0, 360)
    ax2.grid(False)

    # colour-coded axis spines: green electricity on the left, purple CO2 on
    # the right drawn as a full vertical line rather than bare ticks.
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_ELEC)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_CO2)
    ax2.spines["right"].set_linewidth(1.6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=11)
    ax.set_title("Data-centre energy and carbon, IEA Base Case", fontsize=12)

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "carbon_amount_trend.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


def electricity_only():
    """Plot only the data-centre electricity demand per year, without the CO2
    series or its right-hand axis. Same IEA Base-Case electricity data as
    main(); reported IEA anchors are filled markers, interpolated points are
    open markers.

    Figure: figures/electricity_amount_trend.pdf
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    ya, va, yr, vr, yi, vi = _split(cd.DC_ELECTRICITY_TWH)
    ax.plot(ya, va, color=C_ELEC, lw=3.2, zorder=2)
    ax.plot(yr, vr, "o", color=C_ELEC, ms=13, zorder=3, label="electricity (IEA)")
    ax.plot(yi, vi, "o", mfc="white", mec=C_ELEC, mew=2.2, ms=12, zorder=3)
    ax.set_ylabel("Electricity [TWh/yr]", color=C_ELEC, fontsize=19)
    ax.set_xlabel("Year", fontsize=19)
    ax.tick_params(axis="y", labelcolor=C_ELEC, labelsize=18)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_ylim(0, 1050)
    ax.set_yticks([0, 500, 1000])          # fewer ticks, less crowded
    ax.set_xticks([2020, 2025, 2030])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_ELEC)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    ax.legend(loc="upper left", fontsize=17)
    # in-plot title removed: duplicated the \subfigure caption in the paper

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "electricity_amount_trend.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    electricity_only()
