#!/usr/bin/env python3
"""Per-subsection break-even generation cost c* [TFLOP] for Sections 3.1-3.4.

Capital only (HDD buy-in + amortised GPU compute); electricity and carbon are
added later in Sections 3.6 and 4, so they are excluded here to match the
break-even inequalities as written in each subsection.

Running example: image object size S = 3e-9 PB, g = 20%, horizon 2025-2045,
hit ratio h = 0.90, Zipf skew s = 1.  Prices from the Table 1 anchors:
HDD 20 $/TB at 10%/yr decline, GPU 18 $/(TFLOP/s) at 30%/yr decline;
HDD power alpha = 315 $/PB/yr (0.45 W/TB), GPU beta = 9.78e-9 $/TFLOP.
"""

import numpy as np
import cost_model as cm

S = 3.0e-9           # PB per object (3 MB image)
G = 0.20
H = 0.90
HORIZON = 2045

years = np.arange(cm.YEAR0, HORIZON + 1)
age = years - cm.YEAR0
phdd = cm.p_hdd(years)
ptf = cm.p_tflop(years)


def breakeven_store_all(g, with_replacement):
    """3.1-3.3: store-everything vs regenerate-everything, capital only."""
    if abs(g) < 1e-12:
        n_cum = (age + 1) / S
        new_data = np.ones_like(age, dtype=float)
    else:
        n_cum = ((1.0 + g) ** (age + 1) - 1.0) / (g * S)
        new_data = (1.0 + g) ** age
    R = cm.REQ_PER_OBJ * n_cum
    if with_replacement:
        A = cm._replacement_series(new_data)
    else:
        A = new_data
    C_hdd_cum = np.sum(1000.0 * A * phdd)
    denom = np.sum(R * ptf)
    return C_hdd_cum / denom, C_hdd_cum


def breakeven_cache(g):
    """3.4: hot cache vs full HDD, capital only."""
    n_cum = ((1.0 + g) ** (age + 1) - 1.0) / (g * S)
    R = cm.REQ_PER_OBJ * n_cum
    new_data = (1.0 + g) ** age
    A = cm._replacement_series(new_data)
    C_hdd_cum = np.sum(1000.0 * A * phdd)

    frac = np.array([cm.cache_fraction(n, s=1.0, hit=H) for n in n_cum])
    v_cache = frac * n_cum * S
    dv = np.diff(np.concatenate([[0.0], v_cache]))
    A_cache = cm._replacement_series(np.maximum(dv, 0.0))
    C_cache_cum = np.sum(1000.0 * A_cache * phdd)

    denom = np.sum((1.0 - H) * R * ptf)
    return (C_hdd_cum - C_cache_cum) / denom, C_hdd_cum, C_cache_cum


c31, hdd31 = breakeven_store_all(0.0, with_replacement=False)
c32, hdd32 = breakeven_store_all(G, with_replacement=False)
c33, hdd33 = breakeven_store_all(G, with_replacement=True)
c34, hdd34, cache34 = breakeven_cache(G)

print(f"S = {S:g} PB, g = {G:.0%}, h = {H:.0%}, horizon {cm.YEAR0}-{HORIZON}\n")
print(f"3.1 steady 1 PB/yr (g=0):       C_HDD,cum = ${hdd31:,.0f}   c* = {c31:,.1f} TFLOP")
print(f"3.2 growing corpus (g=20%):     C_HDD,cum = ${hdd32:,.0f}   c* = {c32:,.1f} TFLOP")
print(f"3.3 + HDD replacement:          C_HDD,cum = ${hdd33:,.0f}   c* = {c33:,.1f} TFLOP")
print(f"3.4 + hot cache:                C_cache   = ${cache34:,.0f}   c* = {c34:,.1f} TFLOP")
