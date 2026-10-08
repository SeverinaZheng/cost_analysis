#!/usr/bin/env python3
r"""Heatmaps: how much more (or less) does storing everything on HDD cost than
regenerating, as a function of *when* generation begins and *how long* the corpus
is kept, for the image modality under a growing corpus.

We fix the image modality (FLUX.1-dev) and the growing corpus (g=20%/yr) and
sweep two axes:

  * x-axis: the first year we start to generate, any year in 2025..2050.
  * y-axis: the storage duration in years, 0..20, so a corpus started in year
            ``t0`` is kept through ``t0 + duration``; a 2050 start held for the
            full 20 years runs to 2070.

For every (start, duration) cell we form a cost ratio, store-everything on HDD
over a regeneration policy, and color the cell by it:

  * store-all-on-HDD / full regeneration -> every request decoded from scratch
    at the full generation cost c, keeping nothing.
  * store-all-on-HDD / two-tier IR regeneration -> a k1-sized IR cold store plus
    a Zipf hot cache, decoding misses at k2*c.

A ratio above 1 means storing everything costs more than regenerating, so
regeneration wins; a ratio below 1 means storing is the cheaper option. Every
hardware price still declines against the real calendar year, so a later start
buys cheaper HDD, GPU, and tape, and because GPU compute falls faster (30%/yr)
than HDD (10%/yr), a later start shifts the balance toward regeneration.

Figures: figures/image_growing_full_regen.pdf   (store-all / full regeneration)
         figures/image_growing_two_tier.pdf      (store-all / two-tier IR regen)
"""

import math
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext

sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cost_model as cm  # noqa: E402

FIGDIR = pathlib.Path(__file__).parent / "figures"

# image modality only: label, model, c [TFLOP], S [PB], k1, k2
IMAGE = [m for m in cm.MODALITIES if m[0] == "Image"][0]

G = 0.20                       # growing corpus, 20%/yr
YEAR0 = cm.YEAR0               # 2025 price anchor

START_MIN, START_MAX = 2025, 2050          # x-axis: first generation year
DURATION_MAX = 20                          # y-axis: storage duration [yr]

START_YEARS = np.arange(START_MIN, START_MAX + 1)       # 26 columns
DURATIONS = np.arange(0, DURATION_MAX + 1)              # 21 rows (0..20 yr)


# ---------------------------------------------------------------------------
# Per-window cumulative costs, generation starting at ``t0``
# ---------------------------------------------------------------------------
def window_costs(g, S, c, k1, k2, t0, s=1.0, duration=DURATION_MAX):
    """Cumulative cost [$] of each strategy for a corpus whose ingest *starts*
    in year ``t0`` and is kept for ``duration`` years, returned as prefix arrays
    indexed by the duration offset ``k`` (kept through year ``t0 + k``).

    Returns ``(two_tier, hdd, full_regen, cache)`` where every entry is the
    cumulative cost from ``t0`` through year ``t0 + k`` inclusive, and ``cache``
    is a Zipf hot cache backed by full prompt regeneration on a miss
    (Section 4.2). Hardware prices are always
    evaluated at the real calendar year, so starting later benefits from the
    ongoing HDD/GPU price decline.
    """
    years = np.arange(t0, t0 + duration + 1)
    age = years - t0                          # 0, 1, 2, ... within the window
    phdd = cm.p_hdd(years)                     # $/TB, real-year price
    ptf = cm.p_tflop(years)                    # $/TFLOP, real-year price
    alph = cm.alpha(years)                     # $/PB/yr HDD elec, real-year (10%/yr decline)
    bet = cm.beta(years)                       # $/TFLOP GPU elec, real-year (30%/yr decline)

    new_data = (1.0 + g) ** age                # PB ingested each year
    A = cm._replacement_series(new_data, cm.L_HDD)     # HDD manufactured w/ replace
    if abs(g) < 1e-12:
        v_full = (age + 1.0)                   # live volume [PB]
    else:
        v_full = ((1.0 + g) ** (age + 1) - 1.0) / g
    n_cum = v_full / S                         # cumulative object count
    R = cm.REQ_PER_OBJ * n_cum                 # annual request rate
    ptf_beta = ptf + bet                       # GPU capital + electricity

    # hot cache (Zipf) provisioned to its yearly increment, with replacement
    frac = np.array([cm.cache_fraction(nn, s) for nn in n_cum])
    v_cache = frac * n_cum * S                 # cache volume [PB]
    dv_cache = np.diff(np.concatenate([[0.0], v_cache]))
    A_cache = cm._replacement_series(np.maximum(dv_cache, 0.0), cm.L_HDD)

    # per-year contributions
    hdd_cap_y = 1000.0 * A * phdd              # HDD capital
    hdd_elec_y = alph * v_full                 # HDD electricity (powered live vol)
    cache_hdd_y = 1000.0 * A_cache * phdd      # hot-cache capital
    cache_elec_y = alph * v_cache              # hot-cache electricity
    miss_y = (1.0 - cm.HIT_RATIO) * R * ptf_beta   # miss regen per unit (k2*c)

    # --- store-everything on HDD ---
    hdd_cum = np.cumsum(hdd_cap_y) + np.cumsum(hdd_elec_y)

    # --- two-tier: IR cold store (k1 of full) + hot cache, decode miss at k2*c ---
    two_tier_cum = (
        k1 * np.cumsum(hdd_cap_y)              # IR store capital = k1 * HDD capital
        + np.cumsum(cache_hdd_y)              # hot-cache capital
        + k1 * np.cumsum(hdd_elec_y)          # IR store electricity
        + np.cumsum(cache_elec_y)            # hot-cache electricity
        + k2 * c * np.cumsum(miss_y)         # miss regeneration (compute + GPU elec)
    )

    # --- full regeneration: keep nothing, decode every request at full cost c ---
    full_regen_y = c * R * ptf_beta            # every request at compute + GPU elec
    full_regen_cum = np.cumsum(full_regen_y)

    # --- hot cache + full regeneration on miss (Section 4.2 "Caching"):
    #     a Zipf hot cache serves the hits at HDD cost, and the (1-h) misses
    #     regenerate fully from the prompt at the full cost c (no IR store). ---
    cache_cum = (
        np.cumsum(cache_hdd_y)                # hot-cache HDD capital
        + np.cumsum(cache_elec_y)             # hot-cache electricity
        + c * np.cumsum(miss_y)               # miss regeneration at full cost c
    )

    return two_tier_cum, hdd_cum, full_regen_cum, cache_cum


def build_ratios(g, S, c, k1, k2, s=1.0):
    """Return ``(ratio_full, ratio_two, ratio_cache)`` NxM arrays indexed
    ``[duration, start]``.

    ``ratio_full[d, i]``  = store-all-on-HDD / full regeneration,
    ``ratio_two[d, i]``   = store-all-on-HDD / two-tier IR regeneration, and
    ``ratio_cache[d, i]`` = store-all-on-HDD / (Zipf hot cache + full prompt
    regeneration on a miss), for a corpus first generated in ``START_YEARS[i]``
    and kept ``DURATIONS[d]`` years.
    """
    nd, ns = len(DURATIONS), len(START_YEARS)
    ratio_full = np.full((nd, ns), np.nan)
    ratio_two = np.full((nd, ns), np.nan)
    ratio_cache = np.full((nd, ns), np.nan)
    for i, t0 in enumerate(START_YEARS):
        tt, hdd, fr, ca = window_costs(g, S, c, k1, k2, int(t0), s=s)
        for d in range(nd):
            ratio_full[d, i] = hdd[d] / fr[d]
            ratio_two[d, i] = hdd[d] / tt[d]
            ratio_cache[d, i] = hdd[d] / ca[d]
    return ratio_full, ratio_two, ratio_cache


# ---------------------------------------------------------------------------
# Carbon: cumulative lifecycle CO2 over the same (start year, duration) sweep
# ---------------------------------------------------------------------------
GAMMA_OVER_E = cm.GAMMA_E / cm.E_RATE   # 3.75 kg CO2 per electricity-dollar
PHI_HDD_EMB = cm.CAP_PHI_HDD            # 5,000 kg CO2/PB manufactured (w/ replace)
PSI_GPU = cm.PSI_GPU                    # 1.32e-8 kg CO2/TFLOP executed


def window_carbon(g, S, c, k1, k2, t0, s=1.0, duration=DURATION_MAX):
    """Cumulative lifecycle carbon [t CO2] of each strategy for a corpus whose
    ingest starts in year ``t0`` and is kept for ``duration`` years, returned as
    prefix arrays exactly like ``window_costs``:
    ``(two_tier, hdd, full_regen, cache)``.

    Follows the carbon section of the paper. Operational carbon converts each
    policy's electricity dollars at gamma_e/e = 3.75 kg CO2/$, so it inherits
    the 10%/yr HDD and 30%/yr GPU energy-intensity declines evaluated at the
    real calendar year. Embodied carbon charges 5,000 kg CO2 per manufactured
    petabyte, replacements included, and 1.32e-8 kg CO2 per TFLOP executed,
    both held at their 2025 values. Because the embodied coefficients do not
    decline while the operational ones do, a later start shifts the carbon
    balance far less than the dollar balance.
    """
    years = np.arange(t0, t0 + duration + 1)
    age = years - t0
    alph = cm.alpha(years)                     # $/PB/yr HDD elec, real-year
    bet = cm.beta(years)                       # $/TFLOP GPU elec, real-year

    new_data = (1.0 + g) ** age                # PB ingested each year
    A = cm._replacement_series(new_data, cm.L_HDD)     # HDD manufactured w/ replace
    if abs(g) < 1e-12:
        v_full = (age + 1.0)                   # live volume [PB]
    else:
        v_full = ((1.0 + g) ** (age + 1) - 1.0) / g
    n_cum = v_full / S                         # cumulative object count
    R = cm.REQ_PER_OBJ * n_cum                 # annual request rate
    # GPU carbon per TFLOP: operational (declining) + embodied (constant)
    gpu_kg_tflop = GAMMA_OVER_E * bet + PSI_GPU

    # hot cache (Zipf) provisioned to its yearly increment, with replacement
    frac = np.array([cm.cache_fraction(nn, s) for nn in n_cum])
    v_cache = frac * n_cum * S                 # cache volume [PB]
    dv_cache = np.diff(np.concatenate([[0.0], v_cache]))
    A_cache = cm._replacement_series(np.maximum(dv_cache, 0.0), cm.L_HDD)

    # per-year contributions [kg CO2]
    hdd_y = GAMMA_OVER_E * alph * v_full + PHI_HDD_EMB * A       # full store
    cache_y = GAMMA_OVER_E * alph * v_cache + PHI_HDD_EMB * A_cache
    miss_y = (1.0 - cm.HIT_RATIO) * R * gpu_kg_tflop             # per unit c

    # --- store-everything on HDD ---
    hdd_cum = np.cumsum(hdd_y) / 1e3                             # tonnes

    # --- full prompt regeneration: every request at full cost c ---
    full_regen_cum = c * np.cumsum(R * gpu_kg_tflop) / 1e3

    # --- hot cache + full prompt regeneration on miss ---
    cache_cum = (np.cumsum(cache_y) + c * np.cumsum(miss_y)) / 1e3

    # --- two-tier: k1 IR cold store + hot cache, decode miss at k2*c ---
    two_tier_cum = (k1 * np.cumsum(hdd_y) + np.cumsum(cache_y)
                    + k2 * c * np.cumsum(miss_y)) / 1e3

    return two_tier_cum, hdd_cum, full_regen_cum, cache_cum


def build_carbon_ratios(g, S, c, k1, k2, s=1.0):
    """Carbon analog of ``build_ratios``: return
    ``(ratio_full, ratio_two, ratio_cache)`` NxM arrays indexed
    ``[duration, start]``, each the carbon of store-all-on-HDD over the carbon
    of a regeneration policy."""
    nd, ns = len(DURATIONS), len(START_YEARS)
    ratio_full = np.full((nd, ns), np.nan)
    ratio_two = np.full((nd, ns), np.nan)
    ratio_cache = np.full((nd, ns), np.nan)
    for i, t0 in enumerate(START_YEARS):
        tt, hdd, fr, ca = window_carbon(g, S, c, k1, k2, int(t0), s=s)
        for d in range(nd):
            ratio_full[d, i] = hdd[d] / fr[d]
            ratio_two[d, i] = hdd[d] / tt[d]
            ratio_cache[d, i] = hdd[d] / ca[d]
    return ratio_full, ratio_two, ratio_cache


# ---------------------------------------------------------------------------
# Draw one ratio heatmap
# ---------------------------------------------------------------------------
def draw_ratio(ax, Z, cbar_label, storage_xy=(2031, 6.5),
               regen_xy=(2045.3, 18.2), always_xy=(2042.5, 15.0),
               storage_text="Storage\nis cheaper",
               regen_text="Regeneration\nis cheaper",
               always_text="Regeneration is\nalways cheaper"):
    """Log heatmap of a cost ratio (store-all-on-HDD over a regeneration policy).

    Cells above the break-even ratio 1 are where storing everything costs more
    than regenerating, so regeneration wins. If the swept range straddles 1 we
    use a diverging map centred at 1 (blue below, red above) and trace the
    break-even contour; if the ratio is single-signed we use a sequential map
    fitted to the actual range so the margin variation stays visible (blue side
    when regeneration wins everywhere, red side when storage wins everywhere).

    The region labels default to the dollar-cost wording; carbon panels pass
    ``storage_text``/``regen_text``/``always_text`` overrides.
    """
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    if lo < 1.0 < hi:                           # straddles break-even -> diverging
        M = max(hi, 1.0 / lo)                   # symmetric span in log space
        norm = LogNorm(vmin=1.0 / M, vmax=M)
        # blue = ratio > 1 (regeneration cheaper), red = ratio < 1 (storage cheaper)
        cmap = plt.get_cmap("RdBu").copy()
    else:                                       # single-signed -> sequential, fitted
        norm = LogNorm(vmin=lo, vmax=hi)
        if hi <= 1.0:
            # all ratios < 1 (storage wins everywhere): red side, deeper red
            # the further below break-even
            cmap = plt.get_cmap("Reds_r").copy()
        else:
            # all ratios > 1 here (regeneration cheaper), so use the blue side
            cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")

    # origin='lower': duration increases upward, start year increases rightward.
    im = ax.imshow(np.ma.masked_invalid(Z), origin="lower", cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest",
                   extent=[START_YEARS[0] - 0.5, START_YEARS[-1] + 0.5,
                           DURATIONS[0] - 0.5, DURATIONS[-1] + 0.5])

    # break-even contour: ratio = 1 (log ratio = 0), if it is crossed
    if lo < 1.0 < hi:
        XX, YY = np.meshgrid(START_YEARS.astype(float), DURATIONS.astype(float))
        cs = ax.contour(XX, YY, Z, levels=[1.0], colors="0.15", linewidths=1.6)
        ax.clabel(cs, fmt={1.0: "break-even (ratio = 1)"}, fontsize=14)
        # label the two regions the contour separates: ratio < 1 (blue) means
        # storing everything is cheaper, ratio > 1 (red) means regenerating is.
        region_bbox = dict(boxstyle="round,pad=0.25", fc="white",
                           ec="none", alpha=0.55)
        ax.text(*storage_xy, storage_text, ha="center", va="center",
                fontsize=16, fontweight="bold", color="0.12", bbox=region_bbox)
        ax.text(*regen_xy, regen_text, ha="center",
                va="center", fontsize=16, fontweight="bold", color="0.12",
                bbox=region_bbox)
    else:
        # single-signed panel: one policy wins across the whole sweep; note it
        # in the middle of the panel.
        region_bbox = dict(boxstyle="round,pad=0.25", fc="white",
                           ec="none", alpha=0.55)
        ax.text(*always_xy, always_text, ha="center",
                va="center", fontsize=16, fontweight="bold", color="0.12",
                bbox=region_bbox)

    ax.set_xlabel("Starting Year", fontsize=16)
    ax.set_ylabel("Storage Duration [yr]", fontsize=16)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.5")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    if hi / lo >= 10.0:
        # wide range: decade ticks on the log colorbar
        cb.ax.yaxis.set_major_locator(LogLocator(base=10.0))
        cb.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    else:
        # narrow range: a few log-spaced ratio ticks with plain labels; keep
        # the tick positions inside [lo, hi] (rounding could push a tick to 0,
        # invalid on a log colorbar, or beyond vmin/vmax, stretching the bar)
        cb.ax.minorticks_off()
        ticks = np.geomspace(lo, hi, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.2g}" for t in ticks])
    return im


def make_figure(Z, out_name, cbar_label, **label_kw):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    draw_ratio(ax, Z, cbar_label, **label_kw)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.14)

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / out_name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)     # QA copy
    plt.close(fig)
    print(f"wrote {out}")


def make_cache_vs_storage_figure(ratio_cache,
                                 out_name="image_growing_cache_regen.pdf"):
    """Storage vs (Zipf hot cache + full prompt regeneration on a miss).

    Heatmap of ``store-all-on-HDD / (cache + prompt regen)`` over the same
    (start year, duration) sweep, with the break-even line (ratio = 1) drawn
    by ``draw_ratio``. Caching absorbs the hot Zipf head at HDD cost, so only
    the (1-h) misses pay the full generator cost c; this pulls the break-even
    start year from the mid-2040s of pure prompt regeneration into the
    2032-2038 range, but unlike the IR design it never wins from day one.
    The region labels are positioned for this panel: the break-even contour
    runs near start year 2032-2038 across durations, so the storage label sits
    to its left and the regeneration label to its right.
    """
    lo, hi = float(np.nanmin(ratio_cache)), float(np.nanmax(ratio_cache))
    print(f"cache+prompt-regen ratio range: {lo:.3f} .. {hi:.3f}")
    for d in range(len(DURATIONS)):
        cross = np.where(ratio_cache[d] >= 1.0)[0]
        if len(cross):
            print(f"  duration {DURATIONS[d]:2d}: break-even start year "
                  f"{START_YEARS[cross[0]]}")
    make_figure(
        ratio_cache, out_name,
        "Storage / Cache + Prompt Regen",
        storage_xy=(2028.5, 5.0), regen_xy=(2044.0, 10.0))


def make_carbon_full_regen_figure(ratio_full_carbon,
                                  out_name="image_growing_full_regen_carbon.pdf"):
    """Carbon: store-everything on HDD vs full prompt regeneration.

    Heatmap of ``carbon(store-all) / carbon(prompt regen)`` over the same
    (start year, duration) sweep. Unlike the dollar ratio, which crosses
    break-even in the mid 2040s as GPU price-performance improves 30%/yr, the
    carbon ratio stays two orders of magnitude below 1 at every start year:
    the embodied GPU coefficient is held at its 2025 value, so regenerating a
    2,100-TFLOP image on every request keeps a large constant embodied floor,
    while the storage side's embodied floor of 5,000 kg CO2/PB is small in
    comparison. Storing therefore always emits less than prompt regeneration
    for images, no matter when generation begins.
    """
    lo = float(np.nanmin(ratio_full_carbon))
    hi = float(np.nanmax(ratio_full_carbon))
    print(f"carbon storage/prompt-regen ratio range: {lo:.4f} .. {hi:.4f}")
    make_figure(
        ratio_full_carbon, out_name,
        "Carbon: Storage/Regen From Prompt",
        always_xy=(2037.5, 10.0),
        always_text="Storage always\nemits less")


def make_carbon_cache_regen_figure(ratio_cache_carbon,
                                   out_name="image_growing_cache_regen_carbon.pdf"):
    """Carbon: store-everything on HDD vs a Zipf hot cache whose misses are
    regenerated from the prompt at the full cost c.

    The cache removes 90% of the regenerations and with them 90% of the GPU
    carbon, lifting the ratio about tenfold over the prompt-regeneration
    panel, yet the constant embodied GPU coefficient keeps the ratio below 1
    across the entire sweep: for images the misses alone still embody more
    carbon than manufacturing and powering the full HDD corpus. In dollars
    this policy breaks even for start years past the early 2030s; in carbon
    it never does.
    """
    lo = float(np.nanmin(ratio_cache_carbon))
    hi = float(np.nanmax(ratio_cache_carbon))
    print(f"carbon storage/cache+prompt ratio range: {lo:.4f} .. {hi:.4f}")
    make_figure(
        ratio_cache_carbon, out_name,
        "Carbon: Storage/Cache+Regen",
        always_xy=(2037.5, 10.0),
        always_text="Storage always\nemits less")


def make_carbon_two_tier_figure(ratio_two_carbon,
                                out_name="image_growing_two_tier_carbon.pdf"):
    """Carbon: store-everything on HDD vs the two-tier LazIR design of a
    k1-sized IR cold store plus a Zipf hot cache, decoding misses at k2*c.

    The IR store shrinks both the operational and the embodied HDD carbon by
    k1 = 1/6, and the k2 = 1/200 decode keeps the GPU slice thin, so the ratio
    sits above 1 across the whole sweep: LazIR emits less than storing
    everything at every start year and duration, mirroring the dollar result.
    The ratio is also much flatter along the start-year axis than the dollar
    panel, because the dominant embodied coefficients on both sides are held
    constant while only the operational slices decline with hardware trends.
    """
    lo = float(np.nanmin(ratio_two_carbon))
    hi = float(np.nanmax(ratio_two_carbon))
    print(f"carbon storage/two-tier ratio range: {lo:.4f} .. {hi:.4f}")
    make_figure(
        ratio_two_carbon, out_name,
        "Carbon: Storage/Regen From IR",
        always_xy=(2037.5, 10.0),
        always_text="Regeneration always\nemits less")


def carbon_sanity(g, S, c, k1, k2):
    """Cross-check the 2025-start, 20-year window against the paper's carbon
    table: baseline ~2,083 t, prompt regen ~120,264 t, cache+prompt ~12,198 t,
    LazIR ~579 t for the image modality."""
    tt, hdd, fr, ca = window_carbon(g, S, c, k1, k2, 2025)
    print("carbon sanity, 2025 start kept 20 yr [t CO2]:")
    print(f"  store-all-on-HDD  {hdd[-1]:12,.0f}")
    print(f"  prompt regen      {fr[-1]:12,.0f}")
    print(f"  cache + prompt    {ca[-1]:12,.0f}")
    print(f"  two-tier LazIR    {tt[-1]:12,.0f}")


def sanity_check(ratio_full, ratio_two, ratio_cache):
    """Report the ratio at the four corners of the swept range."""
    print("sanity (ratio = store-all-on-HDD / regeneration):")
    for lbl, d in [("dur 0", 0), ("dur 20", DURATION_MAX)]:
        for lbl2, i in [("start 2025", 0), ("start 2050", len(START_YEARS) - 1)]:
            print(f"  {lbl:7s} {lbl2:11s}  vs full-regen={ratio_full[d, i]:7.3f}  "
                  f"vs two-tier={ratio_two[d, i]:7.3f}  "
                  f"vs cache={ratio_cache[d, i]:7.3f}")


if __name__ == "__main__":
    name, model, c, S, k1, k2 = IMAGE
    ratio_full, ratio_two, ratio_cache = build_ratios(G, S, c, k1, k2)
    sanity_check(ratio_full, ratio_two, ratio_cache)

    make_figure(
        ratio_full, "image_growing_full_regen.pdf",
        "Storage/Regeneration From Prompt")
    make_figure(
        ratio_two, "image_growing_two_tier.pdf",
        "Storage/Regeneration From IR")
    # Section 4.2: Zipf hot cache backed by full prompt regeneration on a miss
    make_cache_vs_storage_figure(ratio_cache)

    # carbon versions of the same three panels (paper's carbon section)
    carbon_sanity(G, S, c, k1, k2)
    carbon_full, carbon_two, carbon_cache = build_carbon_ratios(G, S, c, k1, k2)
    make_carbon_full_regen_figure(carbon_full)
    make_carbon_cache_regen_figure(carbon_cache)
    make_carbon_two_tier_figure(carbon_two)
