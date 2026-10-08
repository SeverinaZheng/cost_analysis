#!/usr/bin/env python3
"""(Price) The dollar charged per tonne of CO2 rises over the years.

Two independent valuations are shown on a single USD-per-tonne axis:
  - EU ETS allowance, annual average (traded market price), up ~13x from
    2017 to its 2022-23 peak before easing to ~EUR 65 in 2024.
  - US social cost of carbon, central regulatory estimate, up ~9x from the
    first 2010 IWG value ($21) to the 2023 EPA update ($190).
The paper's modelling anchor (a conservative $50/tonne voluntary-market
price) is drawn as a horizontal reference, sitting well below the regulatory
social cost.

Sources are documented in carbon_data.py.

Figure: figures/carbon_price_trend.pdf
"""

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import carbon_data as cd  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"
C_ETS = "#0d6217"   # green : market price
C_SCC = "#6a51a3"   # purple: social cost


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    # EU ETS annual average, converted to USD/tonne.
    ets_yr = [y for y, _ in cd.EU_ETS_EUR]
    ets_usd = [v * cd.EUR_TO_USD for _, v in cd.EU_ETS_EUR]
    ax.plot(ets_yr, ets_usd, "-o", color=C_ETS, ms=6, lw=2.0,
            label="EU ETS market price")

    # US social cost of carbon central estimates.
    scc_yr = [y for y, _v, _l in cd.SCC_USD]
    scc_v = [v for _y, v, _l in cd.SCC_USD]
    ax.plot(scc_yr, scc_v, "-^", color=C_SCC, ms=8, lw=2.0,
            label="US social cost of carbon")
    for y, v, lab in cd.SCC_USD:
        ax.annotate(f"${v:.0f}", (y, v), textcoords="offset points",
                    xytext=(6, 4), fontsize=10, color=C_SCC)

    # Paper modelling anchor.
    ax.axhline(cd.VOLUNTARY_USD, color="#7f7f7f", ls=":", lw=1.6)
    ax.annotate(f"paper anchor ${cd.VOLUNTARY_USD:.0f}/t (voluntary mkt)",
                (2010.5, cd.VOLUNTARY_USD), textcoords="offset points",
                xytext=(0, 5), fontsize=10, color="#7f7f7f")

    ax.set_xlabel("Year")
    ax.set_ylabel("Carbon Price [USD / tonne CO$_2$]")
    ax.set_ylim(0, 210)
    ax.set_xlim(2009, 2025)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("Price per tonne of CO$_2$ over time", fontsize=12)

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "carbon_price_trend.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
