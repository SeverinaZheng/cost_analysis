#!/usr/bin/env python3
"""(Amount x Price) Implied annual carbon dollar cost of data centres.

The dollar cost of data-centre carbon is the product of two quantities that
are BOTH rising:  emissions [tonnes] x carbon price [USD/tonne].  We multiply
the IEA Base-Case emission trajectory by each carbon-price valuation to get the
implied annual carbon bill, in billions of USD, from 2020 to 2030.

  cost(year) = CO2_Mt(year) * 1e6 [t/Mt] * price(year) [USD/t] / 1e9 [USD/B]

Three valuations: the EU ETS market price, the US social cost of carbon, and
the paper's conservative $50/tonne anchor.  Because amount roughly doubles and
price has risen several-fold, the implied bill grows much faster than either
factor alone.  The 2025-2030 region (dashed) extends the IEA emission anchors
and holds the carbon price flat at its latest reported value, so it is a
deliberately conservative lower bound on the future bill.

Sources are documented in carbon_data.py.

Figure: figures/carbon_dollar_cost.pdf
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

import carbon_data as cd  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"
SPLIT = 2024.0   # last year with reported carbon price; beyond is projection

VALUATIONS = [
    ("US social cost of carbon", cd.scc_usd, "#6a51a3", "^"),
    ("EU ETS market price", cd.eu_ets_usd, "#0d6217", "o"),
    (f"voluntary mkt (${cd.VOLUNTARY_USD:.0f}/t)",
     lambda y: cd.VOLUNTARY_USD, "#7f7f7f", "s"),
]


def annual_cost_billion(price_fn, year):
    """CO2_Mt * 1e6 t/Mt * $/t  ->  divided by 1e9 to get billion USD."""
    return cd.co2_mt(year) * 1e6 * price_fn(year) / 1e9


def main():
    apply_style()
    years = np.arange(2020, 2031)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for label, price_fn, color, marker in VALUATIONS:
        cost = np.array([annual_cost_billion(price_fn, y) for y in years])
        solid = years <= SPLIT
        ax.plot(years[solid], cost[solid], "-", color=color, lw=2.0,
                marker=marker, ms=6, label=label)
        # projection segment (dashed), bridging from the last solid point
        proj = years >= SPLIT
        ax.plot(years[proj], cost[proj], "--", color=color, lw=2.0,
                marker=marker, ms=6, mfc="white")

    ax.axvspan(SPLIT, 2030, color="0.9", alpha=0.5, zorder=0)
    ax.text(2027.0, 4.0, "projection\n(price held flat)", fontsize=10,
            color="0.4", ha="center", va="bottom")

    ax.set_xlabel("Year")
    ax.set_ylabel("Implied Carbon Cost [billion USD/yr]")
    ax.set_xlim(2020, 2030)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("Annual data-centre carbon bill = amount $\\times$ price",
                 fontsize=12)

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "carbon_dollar_cost.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
