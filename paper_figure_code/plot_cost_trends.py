#!/usr/bin/env python3
"""
Plot historical and projected HDD $/TB and GPU $/TFLOP trends.

Three public functions:
    plot_HDD_price()          – HDD cost per TB, 2010-2035
    plot_compute_cost()       – Consumer GPU cost per TFLOP (FP32), 2015-2035
    plot_epoch_gpu_trend()    – Epoch AI ML-hardware FP32 FLOP/s per $, 2008-2024

All source data are listed in the docstrings / comments of each function
with verifiable URLs.

Usage:
    python plot_cost_trends.py          # saves all figures as PDF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import os

# ── research-paper style (large fonts, embedded TrueType) ───────────
plt.rcParams.update({
    # fonts
    'font.family':         'sans-serif',
    'font.sans-serif':     ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size':           20,
    'axes.titlesize':      22,
    'axes.labelsize':      21,
    'xtick.labelsize':     19,
    'ytick.labelsize':     19,
    'legend.fontsize':     17,
    # lines / markers
    'lines.linewidth':     2.2,
    'lines.markersize':    9,
    # axes / ticks
    'axes.linewidth':      1.4,
    'xtick.major.width':   1.4,
    'ytick.major.width':   1.4,
    'xtick.major.size':    6,
    'ytick.major.size':    6,
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    # legend
    'legend.frameon':      True,
    'legend.framealpha':   0.9,
    'legend.edgecolor':    '0.7',
    # save
    'figure.dpi':          150,
    'savefig.bbox':        'tight',
    'pdf.fonttype':        42,   # embed TrueType for arXiv / IEEE PDF check
    'ps.fonttype':         42,
})

# ── output directory ────────────────────────────────────────────────
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ====================================================================
# HDD $/TB
# ====================================================================
def plot_HDD_price():
    """Plot historical HDD $/TB from 2010 to 2025 and project to 2035.

    Each yearly point is a representative high capacity consumer HDD,
    priced at launch MSRP or an earliest dated retail observation in the
    same year, sourced from a reputable launch review or news article.
    Prices are nominal USD, not inflation adjusted. The original jcmit.net
    source URL has lapsed and now redirects to an unrelated site, so each
    drive below is sourced individually from a currently loading page.

    ┌──────┬─────────────────────────────────┬──────┬────────┬──────────┬────────────┐
    │ Year │ Drive                           │ Cap. │ Price  │ $/TB     │ Date       │
    ├──────┼─────────────────────────────────┼──────┼────────┼──────────┼────────────┤
    │ 2010 │ Hitachi Deskstar 7K3000         │  3TB │ 249.99 │  83.33   │ 2010-12-14 │
    │ 2011 │ Hitachi Deskstar 3TB, pre flood │  3TB │ 130.00 │  43.33   │ 2011-09    │
    │ 2012 │ WD Red WD30EFRX                 │  3TB │ 180.00 │  60.00   │ 2012-08-09 │
    │ 2013 │ WD Red WD40EFRX                 │  4TB │ 200.00 │  50.00   │ 2013-11-13 │
    │ 2014 │ WD Red WD60EFRX                 │  6TB │ 299.99 │  50.00   │ 2014-07-21 │
    │ 2015 │ Seagate Archive ST8000AS0002    │  8TB │ 260.00 │  32.50   │ 2014-12-09 │
    │ 2016 │ Seagate BarraCuda Pro 10TB      │ 10TB │ 534.99 │  53.50   │ 2016-07-19 │
    │ 2017 │ Seagate IronWolf ST12000VN0007  │ 12TB │ 475.00 │  39.58   │ 2017-10-25 │
    │ 2018 │ Seagate IronWolf ST14000VN0008  │ 14TB │ 569.99 │  40.71   │ 2018-10-03 │
    │ 2019 │ WD Red WD140EFFX                │ 14TB │ 450.00 │  32.14   │ 2019-10-25 │
    │ 2020 │ Seagate IronWolf Pro 18TB       │ 18TB │ 609.00 │  33.83   │ 2020-09-04 │
    │ 2021 │ WD Gold WD201KRYZ               │ 20TB │ 679.99 │  34.00   │ 2021-11-17 │
    │ 2022 │ WD Red Pro WD201KFGX            │ 20TB │ 499.99 │  25.00   │ 2022-02-28 │
    │ 2023 │ WD Red Pro WD221KFGX            │ 22TB │ 499.00 │  22.68   │ 2023-02-02 │
    │ 2024 │ Seagate IronWolf Pro 24TB       │ 24TB │ 649.00 │  27.04   │ 2024-01-25 │
    │ 2025 │ Seagate IronWolf Pro 30TB, HAMR │ 30TB │ 599.99 │  20.00   │ 2025-07-17 │
    └──────┴─────────────────────────────────┴──────┴────────┴──────────┴────────────┘

    Source URLs, one per row above, all verified to currently load and
    contain the cited price and date:
        2010 https://www.storagereview.com/news/hitachi-deskstar-7k3000-deskstar-5k3000-and-xl-external-hard-drives-released
        2011 https://www.backblaze.com/blog/farming-hard-drives-2-years-and-1m-later/
        2012 https://silentpcreview.com/western-digital-red-3tb-1tb-hard-drives/
        2013 https://techgage.com/article/wd-red-4tb-nas-hard-drive-review/
        2014 https://www.servethehome.com/western-digital-red-5tb-6tb-red-pro-drives-released/
        2015 https://www.digitaltrends.com/computing/archive-everything-seagates-new-affordable-8tb-hard-disk/
        2016 https://www.storagereview.com/review/seagate-barracuda-pro-10tb-hdd-review
        2017 https://www.storagereview.com/review/seagate-ironwolf-review-12tb
        2018 https://www.storagereview.com/review/seagate-ironwolf-14tb-nas-hdd-review
        2019 https://www.storagereview.com/news/western-digital-announces-new-wd-red-nas-drives
        2020 https://www.servethehome.com/seagate-ironwolf-pro-18tb-hdd-and-4tb-ironwolf-nas-ssds-arrive/
        2021 https://www.techtimes.com/articles/268108/20211117/western-digital-20tb-hdds-spotted-selling-679-99-worth.htm
        2022 https://www.storagereview.com/news/wd-red-pro-20tb-now-available
        2023 https://www.storagereview.com/review/wd-red-pro-22tb-review
        2024 https://www.storagereview.com/news/seagate-launches-ironwolf-pro-24tb-nas-hdd
        2025 https://www.storagenewsletter.com/2025/07/17/seagate-availability-of-exos-m-and-ironwolf-pro-up-to-30tb-3-5-inch-sata-6gb-hdds/

    Notes on selection:
    - 2011: the Thailand floods destroyed roughly 30% of global HDD
      production. The pre flood September 2011 Hitachi Deskstar 3TB at
      $130 is used, taken from the Backblaze "Farming Hard Drives" post
      which also records the post flood spike to $249 in November 2011.
      See https://en.wikipedia.org/wiki/Hard_disk_drive for context.
    - 2016 sits above the trend because the BarraCuda Pro 10TB is a
      premium 7,200 rpm CMR drive, while 2015 uses the slower 5,900 rpm
      SMR Archive drive. Both were the newest large capacity consumer
      HDDs at their respective launches.
    - 2025: the IronWolf Pro 30TB is the first HAMR based consumer NAS
      drive to reach broad channel and retail availability, announced at
      $599.99 MSRP on 2025-07-17. The small year over year decline from
      2024 to 2025 is consistent with industry reports of HDD supply
      tightening: Q4 2025 contract prices rose roughly 4 percent quarter
      over quarter, the largest jump in eight quarters, driven by AI
      infrastructure demand and a slow HAMR ramp. See:
        https://www.tomshardware.com/pc-components/hdds/hdd-prices-spike-as-ai-infrastructure-and-chinas-pc-push-collide-hard-drives-record-biggest-price-increase-in-eight-quarters-suppliers-warn-pressure-will-continue
        https://www.techradar.com/pro/surging-ai-demand-for-hdd-means-that-you-may-have-to-wait-up-to-a-year-for-32-tb-hard-disk-drives-warns-research-and-yes-prices-are-also-going-up

    Projection model anchored at 2025 with a 7.7% annual decline:
        P_HDD(t) = 20.00 * 0.923^(t - 2025)    [nominal $/TB]
    Rationale: a log linear fit of 2010 to 2024, excluding the 2011 flood
    point, gives roughly 7 to 8% per year decline. HAMR roadmaps from
    Seagate forecast 20% per year areal density growth through 2034,
    which historically translates into 7 to 12% per year price decline.
    """

    # ── historical data, nominal $/TB ────────────────────────────────
    years_hist = np.array([
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
        2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
    ])
    dollar_per_tb = np.array([
        83.33, 43.33, 60.00, 50.00, 50.00, 32.50, 53.50, 39.58,
        40.71, 32.14, 33.83, 34.00, 25.00, 22.68, 27.04, 20.00,
    ])

    # ── projection model ─────────────────────────────────────────────
    anchor_year, anchor_price, annual_decline = 2025, 20.00, 0.077
    years_proj = np.arange(2025, 2036)
    price_proj = anchor_price * (1 - annual_decline) ** (years_proj - anchor_year)

    # ── fitted exponential through 2010-2024, excluding 2011 flood ───
    mask_fit = ~np.isin(years_hist, [2011, 2025])
    yf = years_hist[mask_fit]
    pf = dollar_per_tb[mask_fit]
    # log-linear fit: log(P) = a + b*(t)
    coeffs = np.polyfit(yf, np.log(pf), 1)
    years_fit = np.linspace(2010, 2035, 200)
    price_fit = np.exp(np.polyval(coeffs, years_fit))

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.scatter(years_hist, dollar_per_tb, zorder=5, s=80, color="#1f77b4",
               edgecolors="white", linewidths=1.0, label="Historical")

    ax.plot(years_fit, price_fit, "--", color="#1f77b4", alpha=0.55,
            linewidth=1.6, label=f"Fit 2010–2025 (-{-coeffs[0]*100:.1f}%/yr)")

    ax.plot(years_proj, price_proj, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label="Projection: 7.7%/yr decline")

    ax.set_xlabel("Year")
    ax.set_ylabel("HDD Price  [$/TB]", fontsize=19)
    ax.set_xlim(2009, 2036)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hdd_price_trend.pdf"), bbox_inches="tight")
    print(f"Saved hdd_price_trend.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Enterprise-only HDD $/TB  (Seagate Exos nearline line)
# ====================================================================
def plot_HDD_price_enterprise():
    """Plot enterprise-only HDD $/TB, 2016-2025, using the Seagate Exos line.

    Companion to plot_HDD_price(): that panel tracks the newest high capacity
    CONSUMER / NAS drive each year (Deskstar, WD Red, IronWolf), while this one
    tracks the newest high capacity 7,200 rpm ENTERPRISE nearline drive, all
    from a single family so the series is on a consistent basis. The family is
    Seagate Exos, which before the 2017 Exos rebrand was sold as "Enterprise
    Capacity 3.5 HDD" (Constellation ES lineage); the 2016 point is that direct
    predecessor. Every drive is a 7,200 rpm helium (or HAMR, for 2025) 3.5-inch
    SATA nearline drive with a 5 year warranty and 550 TB/yr workload rating,
    priced at launch MSRP or an earliest dated retail observation in the same
    year. Prices are nominal USD, not inflation adjusted.

    ┌──────┬─────────────────────────────────────┬──────┬────────┬──────────┬──────┐
    │ Year │ Drive                               │ Cap. │ Price  │ $/TB     │ Src  │
    ├──────┼─────────────────────────────────────┼──────┼────────┼──────────┼──────┤
    │ 2016 │ Ent. Capacity 3.5 v5 ST8000NM0055   │  8TB │ 550.00 │  68.75   │ [1]  │
    │ 2017 │ Exos X12 ST12000NM0007              │ 12TB │ 499.00 │  41.58   │ [2]* │
    │ 2018 │ Exos X14 ST14000NM0018              │ 14TB │ 580.00 │  41.43   │ [3]* │
    │ 2019 │ Exos X16 ST16000NM001G              │ 16TB │ 629.00 │  39.31   │ [4]  │
    │ 2020 │ Exos X18 ST18000NM000J              │ 18TB │ 561.75 │  31.21   │ [5]  │
    │ 2022 │ Exos X20 ST20000NM007D              │ 20TB │ 669.00 │  33.45   │ [6]  │
    │ 2024 │ Exos X24 ST24000NM002H              │ 24TB │ 480.00 │  20.00   │ [7]  │
    │ 2025 │ Exos M 30TB, HAMR Mozaic 3+         │ 30TB │ 599.99 │  20.00   │ [8]  │
    └──────┴─────────────────────────────────────┴──────┴────────┴──────────┴──────┘

    Source URLs:
      [1] 2016 Ent. Capacity 8TB, ~$550 at review time:
          https://camelcamelcamel.com/Seagate-256MB-3-5-Inch-Enterprise-ST8000NM0055/product/B016AG0ITU
      [2] 2017 Exos X12 12TB (datasheet 2017, ST12000NM0007):
          https://www.seagate.com/www-content/datasheets/pdfs/exos-x-12-DS1946-1-1709US-en_US.pdf
      [3] 2018 Exos X14 14TB (launch review Dec 2018):
          https://www.kitguru.net/components/hard-drives/simon-crisp/seagate-exos-x14-14tb-hdd-review/
      [4] 2019 Exos X16 16TB, MSRP $629:
          https://www.storagereview.com/review/seagate-exos-x16-16tb-hdd-review
      [5] 2020 Exos X18 18TB, MSRP $561.75:
          https://www.storagereview.com/news/seagate-exos-x18-released
      [6] 2022 Exos X20 20TB, MSRP $669:
          https://www.storagereview.com/review/seagate-exos-x20-20tb-enterprise-hdd-review
      [7] 2024 Exos X24 24TB, MSRP $480:
          https://www.storagereview.com/review/seagate-exos-x24-24tb-hdd-review
      [8] 2025 Exos M 30TB HAMR, $599.99:
          https://www.tomshardware.com/pc-components/hdds/seagate-exos-m-30tb-hdd-review

    * 2017 and 2018 launch MSRPs were not recoverable from a single dated
      source; the values shown are best estimates from launch-era enterprise
      channel pricing and are the two least certain points in the series.

    Styling and normalization match plot_HDD_price_normalized() exactly: same
    blue/orange colours, log y-axis, and every value divided by the last
    historical point (2025 = $20.00 per TB, Exos M 30TB) so the curve passes
    through 1.0 at 2025. The projection uses the same 7.7% annual decline used
    in plot_HDD_price(), so the enterprise and consumer panels sit on an equal
    basis.
    """

    # ── historical data, nominal $/TB (Seagate Exos nearline line) ────
    years_hist = np.array([2016, 2017, 2018, 2019, 2020, 2022, 2024, 2025])
    dollar_per_tb = np.array([
        68.75, 41.58, 41.43, 39.31, 31.21, 33.45, 20.00, 20.00,
    ])

    anchor_year = years_hist[-1]            # 2025
    anchor_val  = dollar_per_tb[-1]         # 20.00 $/TB (Exos M 30TB)
    norm_y = dollar_per_tb / anchor_val

    # ── projection: 7.7% per year decline, anchored at 1.0 in 2025 ───
    annual_decline = 0.077
    years_proj = np.arange(2025, 2036)
    proj_y = (1 - annual_decline) ** (years_proj - anchor_year)

    # ── fit through 2016-2024 in normalized units, excluding 2025 ─────
    mask_fit = ~np.isin(years_hist, [2025])
    coeffs = np.polyfit(years_hist[mask_fit], np.log(norm_y[mask_fit]), 1)
    fit_pct = -coeffs[0] * 100
    years_fit = np.linspace(2016, 2035, 200)
    fit_y = np.exp(np.polyval(coeffs, years_fit))

    # ── plot (identical styling to plot_HDD_price_normalized) ─────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.axhline(1.0, color="0.5", linewidth=0.9, linestyle=":", alpha=0.7)

    ax.scatter(years_hist, norm_y, zorder=5, s=80, color="#1f77b4",
               edgecolors="white", linewidths=1.0, label="Historical")
    ax.plot(years_fit, fit_y, "--", color="#1f77b4", alpha=0.55,
            linewidth=1.6,
            label=f"Fit 2016–2025 (-{fit_pct:.1f}%/yr)")
    ax.plot(years_proj, proj_y, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label="Projection: -7.7%/yr")

    ax.set_xlabel("Year")
    ax.set_ylabel("Enterprise HDD Price ($/TB) \n (Normalized to 2025)", fontsize=19)
    ax.set_xlim(2009, 2036)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([0.02, 0.1, 0.5, 1, 5, 10, 50])
    ax.set_ylim(0.018, 60)
    ax.legend(loc="lower left", fontsize=17)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hdd_price_trend_enterprise.pdf"),
                bbox_inches="tight")
    print(f"Saved hdd_price_trend_enterprise.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Tape (LTO) $/TB
# ====================================================================
def plot_tape_price():
    """Plot LTO tape native $/TB across recent generations, fit a log-linear
    (exponential) trend, and project it forward to 2035 (Section 3.3).

    Each point is a single-cartridge launch-era RETAIL price divided by the
    generation's NATIVE (uncompressed) capacity, so it is a $/TB dated to the
    launch year rather than a present-day price. Prices are read from dated
    Wayback Machine snapshots of tapeandmedia.com product pages captured near
    each generation's launch, so the whole series comes from one retailer on a
    consistent basis (HPE cartridges except where noted). Vendors headline the
    2.5:1 "compressed" capacity (6.25/15/30/45/75 TB); we use native.

    ┌───────┬──────┬───────────┬───────────────┬─────────┐
    │ Gen   │ Year │ Native    │ Cartridge ($) │ $/TB    │
    ├───────┼──────┼───────────┼───────────────┼─────────┤
    │ LTO-6 │ 2013 │  2.5 TB   │    117.32     │  46.93  │
    │ LTO-7 │ 2015 │  6   TB   │    186.68     │  31.11  │
    │ LTO-8 │ 2017 │ 12   TB   │    188.00     │  15.67  │
    │ LTO-9 │ 2021 │ 18   TB   │    202.50     │  11.25  │
    │ LTO-10│ 2025 │ 30   TB   │    312.16     │  10.41  │
    └───────┴──────┴───────────┴───────────────┴─────────┘

    Dated launch-era Wayback snapshots of tapeandmedia.com (price source):
        LTO-6  http://web.archive.org/web/20130111023406/http://www.tapeandmedia.com:80/hp-lto-6-tape-ultrium-tapes.asp
        LTO-7  http://web.archive.org/web/20151213020546/http://www.tapeandmedia.com:80/hp-lto-7-tape-ultrium-tapes-c7977a.asp
        LTO-8  http://web.archive.org/web/20171226234539/https://www.tapeandmedia.com/lto-8-tape-media-tapes.asp
        LTO-9  http://web.archive.org/web/20210924065625/https://tapeandmedia.com/hpe-lto-9-tape-with-barium-ferrite-bafe-q2079a/
        LTO-10 http://web.archive.org/web/20250621164207/https://tapeandmedia.com/hpe-lto-10-ultrium-tape-30tb-75tb-compressed-rw-data-cartridge-q2080a/
    Release years and native capacities:
        https://en.wikipedia.org/wiki/Linear_Tape-Open
        https://www.lto.org/lto-8/
    LTO-10 launch (30 TB native, June 2025):
        https://www.fujifilm.com/us/en/news/data-storage/fujifilm-launches-lto-ultrium-10-data-cartridge

    Modelling note: a log-linear fit over 2013-2025 gives a ~11.8%/yr decline
    in native $/TB (R^2 ~ 0.84), close to HDD's ~10%/yr rather than flat. The
    decline is steep across LTO-6 to LTO-8 and flattens toward ~$10/TB at
    LTO-9/LTO-10. Section 3.3 anchors at the LTO-10 native price of ~$10/TB in
    2025 and applies this fitted decline going forward.
    """

    # ── native $/TB by generation, launch-era retail (Wayback) ────────
    gen_names  = ["LTO-6", "LTO-7", "LTO-8", "LTO-9", "LTO-10"]
    years_hist = np.array([2013, 2015, 2017, 2021, 2025])
    cap_tb     = np.array([2.5, 6.0, 12.0, 18.0, 30.0])
    cart_usd   = np.array([117.32, 186.68, 188.00, 202.50, 312.16])
    dollar_per_tb = cart_usd / cap_tb

    # ── log-linear (exponential) fit, as for HDD and GPU ──────────────
    coeffs = np.polyfit(years_hist, np.log(dollar_per_tb), 1)
    tape_decline = 1.0 - np.exp(coeffs[0])          # annual $/TB decline rate
    years_fit = np.linspace(2013, 2025, 200)
    price_fit = np.exp(np.polyval(coeffs, years_fit))

    # ── projection: anchor at observed LTO-10 2025 $/TB, apply the fit ─
    anchor_year, anchor_price = 2025, float(dollar_per_tb[-1])
    years_proj = np.arange(2025, 2036)
    price_proj = anchor_price * (1.0 - tape_decline) ** (years_proj - anchor_year)
    print(f"Tape $/TB log-linear fit: {tape_decline*100:.1f}%/yr decline, "
          f"anchor ${anchor_price:.2f}/TB (2025)")

    # Same wide aspect and log y-axis as the HDD and GPU price panels, so the
    # three panels of Fig. 3 sit at equal 0.33\textwidth width without the tape
    # panel being visually squeezed.
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.scatter(years_hist, dollar_per_tb, zorder=5, s=80, color="#1f77b4",
               edgecolors="white", linewidths=1.0, label="LTO generations")

    ax.plot(years_fit, price_fit, "--", color="#1f77b4", alpha=0.55,
            linewidth=1.6, label=f"Fit 2013–2025 (-{tape_decline*100:.1f}%/yr)")

    ax.plot(years_proj, price_proj, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label="Projection: 10%/yr decline")

    # per-generation labels
    gen_offsets = {
        "LTO-6":  (6, 5),
        "LTO-7":  (6, 5),
        "LTO-8":  (6, 5),
        "LTO-9":  (6, 6),
        "LTO-10": (-8, 10),
    }
    for name, x, y in zip(gen_names, years_hist, dollar_per_tb):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=gen_offsets[name], fontsize=16, color="0.25")

    ax.set_xlabel("Year")
    ax.set_ylabel("Tape Price ($/TB)", fontsize=19)
    ax.set_xlim(2011, 2036)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.set_yscale("log")
    ax.set_ylim(0.8, 70)
    ax.set_yticks([1, 10, 50])
    ax.set_yticklabels(["1", "10", "50"])
    ax.minorticks_off()
    ax.legend(loc="lower left", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "tape_price_trend.pdf"), bbox_inches="tight")
    print(f"Saved tape_price_trend.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# GPU $/TFLOP
# ====================================================================
def plot_compute_cost():
    """Plot historical GPU $/TFLOP (FP32) and project to 2035.

    FP32 TFLOPS computed as: 2 × CUDA_cores × boost_clock_GHz.
    All data sourced from the Wikipedia article for each GeForce generation.

    ┌─────────────┬──────┬────────┬──────────────┬───────┬──────────┬────────────────────────────────────────────────────────────────┐
    │ GPU         │ Year │ CUDA   │ Boost (MHz)  │ FP32  │ MSRP ($) │ Source URL                                                     │
    │             │      │ cores  │              │ TFLOPS│          │                                                                │
    ├─────────────┼──────┼────────┼──────────────┼───────┼──────────┼────────────────────────────────────────────────────────────────┤
    │ GTX 980 Ti  │ 2015 │ 2816   │ 1075         │  6.1  │   649    │ https://en.wikipedia.org/wiki/GeForce_900_series               │
    │ GTX 1080 Ti │ 2017 │ 3584   │ 1582         │ 11.3  │   699    │ https://en.wikipedia.org/wiki/GeForce_10_series                │
    │ RTX 2080 Ti │ 2018 │ 4352   │ 1545         │ 13.4  │   999    │ https://en.wikipedia.org/wiki/GeForce_RTX_20_series            │
    │ RTX 3090    │ 2020 │ 10496  │ 1695         │ 35.6  │  1499    │ https://en.wikipedia.org/wiki/GeForce_30_series                │
    │ RTX 4090    │ 2022 │ 16384  │ 2520         │ 82.6  │  1599    │ https://en.wikipedia.org/wiki/GeForce_RTX_40_series            │
    │ RTX 5090    │ 2025 │ 21760  │ 2407         │104.8  │  1999    │ https://en.wikipedia.org/wiki/GeForce_50_series                │
    ├─────────────┼──────┼────────┼──────────────┼───────┼──────────┼────────────────────────────────────────────────────────────────┤
    │ Derived $/TFLOP:                                                                                                               │
    │ GTX 980 Ti : 649 /  6.1 = $106/TFLOP                                                                                          │
    │ GTX 1080 Ti: 699 / 11.3 =  $62/TFLOP                                                                                          │
    │ RTX 2080 Ti: 999 / 13.4 =  $75/TFLOP                                                                                          │
    │ RTX 3090  : 1499 / 35.6 =  $42/TFLOP                                                                                          │
    │ RTX 4090  : 1599 / 82.6 =  $19/TFLOP                                                                                          │
    │ RTX 5090  : 1999 /104.8 =  $19/TFLOP                                                                                          │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    Additional context:
        Epoch AI GPU price-performance dataset (FP32 FLOP/s per constant-2024 $,
        2006-2024) reports a long-run doubling time of ~2.5 years → ~25%/yr.
        Source: https://ourworldindata.org/grapher/gpu-price-performance

    Projection model (anchored at 2025, 30%/yr decline):
        P_GPU(t) = 19 * 0.70^(t - 2025)    [$/TFLOP]
    Rationale: Full 2015-2025 CAGR ~15.7%; faster 2017-2022 window ~21%;
    Epoch long-run ~30%. We adopt 30% in line with the Epoch AI aggregate
    estimate, despite the 2022-2025 stagnation (RTX 4090 ≈ RTX 5090 at $19/TFLOP).
    """

    # ── historical data ──────────────────────────────────────────────
    gpu_names = [
        "GTX 980 Ti", "GTX 1080 Ti", "RTX 2080 Ti",
        "RTX 3090", "RTX 4090", "RTX 5090",
    ]
    years_hist = np.array([2015, 2017, 2018, 2020, 2022, 2025])
    tflops     = np.array([6.1, 11.3, 13.4, 35.6, 82.6, 104.8])
    msrp       = np.array([649, 699, 999, 1499, 1599, 1999])
    dpt        = msrp / tflops  # $/TFLOP

    # ── projection model ─────────────────────────────────────────────
    anchor_year, anchor_price, annual_decline = 2025, 19.0, 0.30
    years_proj = np.arange(2025, 2036)
    price_proj = anchor_price * (1 - annual_decline) ** (years_proj - anchor_year)

    # ── log-linear fit through all historical points ─────────────────
    coeffs = np.polyfit(years_hist, np.log(dpt), 1)
    years_fit = np.linspace(2014, 2036, 200)
    price_fit = np.exp(np.polyval(coeffs, years_fit))

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.scatter(years_hist, dpt, zorder=5, s=90, color="#2ca02c",
               edgecolors="white", linewidths=1.0, label="Historical")

    # per-GPU label offsets (in points) chosen to avoid overlap with neighbours
    name_offsets = {
        "GTX 980 Ti":  (8, 8),
        "GTX 1080 Ti": (-90, -8),
        "RTX 2080 Ti": (8, 6),
        "RTX 3090":    (-95,  4),
        "RTX 4090":    (-90, -22),
        "RTX 5090":    (-95, 10),
    }
    for name, x, y in zip(gpu_names, years_hist, dpt):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=name_offsets[name], fontsize=14, color="0.25")

    ax.plot(years_fit, price_fit, "--", color="#2ca02c", alpha=0.55,
            linewidth=1.6, label=f"Exp. fit 2015–2025 ({-coeffs[0]*100:.1f}%/yr)")

    ax.plot(years_proj, price_proj, "-o", color="#d62728", markersize=8,
            linewidth=2.2, label="Projection: 30%/yr decline")

    ax.set_xlabel("Year")
    ax.set_ylabel("GPU Price  [$/TFLOP]")
    ax.set_xlim(2014, 2036)
    ax.set_xticks(np.arange(2015, 2036, 3))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "gpu_price_trend.pdf"), bbox_inches="tight")
    print(f"Saved gpu_price_trend.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Epoch AI – FP32 FLOP/s per dollar (from CSV)
# ====================================================================
def plot_epoch_gpu_trend():
    """Plot FP32 FLOP/s per dollar for ML hardware from Epoch AI dataset.

    Data source:
        Epoch AI, "Data on Machine Learning Hardware", 2024.
        CSV: https://epoch.ai/data/ml_hardware.csv
        Web: https://epoch.ai/data/machine-learning-hardware
        License: CC-BY 4.0

    Key findings from Epoch AI (https://epoch.ai/blog/trends-in-machine-learning-hardware):
        - FP32 computational performance doubles every 2.3 years (~36%/yr)
        - FP32 price-performance (FLOP/$) doubles every:
              * 2.1 years for ML GPUs  (~39%/yr improvement → ~28%/yr $/TFLOP decline)
              * 2.5 years for general GPUs (~32%/yr improvement → ~24%/yr $/TFLOP decline)
        - "Performance per dollar improves around 30% each year"
          (https://epoch.ai/data-insights/price-performance-hardware)

    These rates strongly support our projected 25%/yr decline in consumer $/TFLOP,
    which sits between the general-GPU rate (~24%) and the ML-GPU rate (~28%).

    Data points extracted from ml_hardware.csv (all with FP32 perf + release price):
    ┌──────────────────────────────────────────┬────────────┬──────────┬──────────┬──────────────┐
    │ Hardware                                 │ Date       │ FP32     │ Price($) │ GFLOP/s/$    │
    │                                          │            │ (TFLOP/s)│          │              │
    ├──────────────────────────────────────────┼────────────┼──────────┼──────────┼──────────────┤
    │ NVIDIA GeForce GTX 280                   │ 2008-06-16 │    0.6   │     649  │     0.96     │
    │ NVIDIA GeForce GTX 580                   │ 2010-11-09 │    1.6   │     499  │     3.17     │
    │ NVIDIA GeForce GTX TITAN                 │ 2013-02-19 │    4.7   │     999  │     4.71     │
    │ NVIDIA GeForce GTX 780                   │ 2013-05-23 │    4.2   │     649  │     6.40     │
    │ NVIDIA GeForce GTX 980                   │ 2014-09-19 │    5.0   │     549  │     9.07     │
    │ NVIDIA Tesla K80                         │ 2014-11-17 │    8.1   │    5000  │     1.63     │
    │ NVIDIA GeForce GTX TITAN X               │ 2015-03-17 │    6.7   │     999  │     6.70     │
    │ NVIDIA P100                              │ 2016-04-05 │    9.3   │    5699  │     1.63     │
    │ NVIDIA GeForce GTX 1080                  │ 2016-05-27 │    8.9   │     599  │    14.83     │
    │ NVIDIA GeForce GTX 1080 Ti               │ 2017-03-10 │   11.3   │     699  │    16.20     │
    │ NVIDIA Tesla V100 SXM2                   │ 2017-07-01 │   15.7   │   11500  │     1.36     │
    │ NVIDIA GeForce RTX 2080 Ti 11GB          │ 2018-09-20 │   13.4   │     999  │    13.45     │
    │ NVIDIA A100 SXM4 40 GB                   │ 2020-05-14 │   19.5   │   15000  │     1.30     │
    │ NVIDIA GeForce RTX 3080                  │ 2020-09-01 │   29.8   │     699  │    42.60     │
    │ NVIDIA GeForce RTX 3090                  │ 2020-09-01 │   35.6   │    1499  │    23.70     │
    │ NVIDIA GeForce RTX 4090                  │ 2022-09-20 │   82.6   │    1599  │    51.66     │
    │ NVIDIA H100 SXM5 80GB                    │ 2022-09-20 │   66.9   │   33600  │     1.99     │
    │ AMD Radeon RX 7900 XTX                   │ 2022-11-03 │   61.4   │     999  │    61.46     │
    │ AMD Instinct MI300X                      │ 2023-12-06 │  163.4   │   15000  │    10.89     │
    └──────────────────────────────────────────┴────────────┴──────────┴──────────┴──────────────┘
    """

    # ── Data from Epoch AI ml_hardware.csv (all entries with FP32 + price) ──
    # Format: (name, date_str, fp32_tflops, price_usd, type)
    #   type: "consumer" or "datacenter"
    data = [
        ("GTX 280",       "2008-06-16",   0.622,    649, "consumer"),
        ("GTX 580",       "2010-11-09",   1.581,    499, "consumer"),
        ("Tesla K20X",    "2012-11-12",   3.935,   7699, "datacenter"),
        ("GTX TITAN",     "2013-02-19",   4.709,    999, "consumer"),
        ("GTX 780",       "2013-05-23",   4.155,    649, "consumer"),
        ("Quadro K6000",  "2013-07-23",   5.197,   5265, "datacenter"),
        ("GTX 980",       "2014-09-19",   4.981,    549, "consumer"),
        ("Tesla K80",     "2014-11-17",   8.137,   5000, "datacenter"),
        ("GTX TITAN X",   "2015-03-17",   6.695,    999, "consumer"),
        ("P100",          "2016-04-05",   9.300,   5699, "datacenter"),
        ("GTX 1080",      "2016-05-27",   8.873,    599, "consumer"),
        ("GTX 1080 Ti",   "2017-03-10",  11.340,    699, "consumer"),
        ("V100 SXM2",     "2017-07-01",  15.667,  11500, "datacenter"),
        ("RTX 2080 Ti",   "2018-09-20",  13.448,    999, "consumer"),
        ("A100 SXM4",     "2020-05-14",  19.500,  15000, "datacenter"),
        ("RTX 3080",      "2020-09-01",  29.768,    699, "consumer"),
        ("RTX 3090",      "2020-09-01",  35.581,   1499, "consumer"),
        ("RTX A6000",     "2020-10-05",  38.710,   4649, "datacenter"),
        ("RTX 4090",      "2022-09-20",  82.575,   1599, "consumer"),
        ("H100 SXM5",     "2022-09-20",  66.910,  33600, "datacenter"),
        ("RTX 4080",      "2022-09-20",  48.745,   1199, "consumer"),
        ("RX 7900 XTX",   "2022-11-03",  61.420,    999, "consumer"),
        ("MI300X",        "2023-12-06", 163.379,  15000, "datacenter"),
    ]

    # Parse dates to fractional years
    def to_year(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    consumer_years, consumer_gflops_per_dollar = [], []
    dc_years, dc_gflops_per_dollar = [], []
    consumer_names, dc_names = [], []

    for name, date_str, fp32_tflops, price, typ in data:
        yr = to_year(date_str)
        gflops_per_dollar = fp32_tflops * 1000 / price  # GFLOP/s per $
        if typ == "consumer":
            consumer_years.append(yr)
            consumer_gflops_per_dollar.append(gflops_per_dollar)
            consumer_names.append(name)
        else:
            dc_years.append(yr)
            dc_gflops_per_dollar.append(gflops_per_dollar)
            dc_names.append(name)

    consumer_years = np.array(consumer_years)
    consumer_gflops_per_dollar = np.array(consumer_gflops_per_dollar)
    dc_years = np.array(dc_years)
    dc_gflops_per_dollar = np.array(dc_gflops_per_dollar)

    # ── Log-linear fits ──────────────────────────────────────────────
    # Consumer GPUs
    c_coeffs = np.polyfit(consumer_years, np.log(consumer_gflops_per_dollar), 1)
    c_doubling = np.log(2) / c_coeffs[0]
    c_annual_pct = (np.exp(c_coeffs[0]) - 1) * 100

    # Datacenter GPUs
    dc_coeffs = np.polyfit(dc_years, np.log(dc_gflops_per_dollar), 1)
    dc_doubling = np.log(2) / dc_coeffs[0]
    dc_annual_pct = (np.exp(dc_coeffs[0]) - 1) * 100

    years_line = np.linspace(2008, 2025, 200)
    c_fit = np.exp(np.polyval(c_coeffs, years_line))
    dc_fit = np.exp(np.polyval(dc_coeffs, years_line))

    print(f"[FP32 Consumer GPUs]   doubling time: {c_doubling:.2f} yr  |  "
          f"annual improvement: +{c_annual_pct:.1f}%/yr  "
          f"(≈ {100 - 100/(1 + c_annual_pct/100):.1f}%/yr $/TFLOP decline)")
    print(f"[FP32 Datacenter GPUs] doubling time: {dc_doubling:.2f} yr  |  "
          f"annual improvement: +{dc_annual_pct:.1f}%/yr  "
          f"(≈ {100 - 100/(1 + dc_annual_pct/100):.1f}%/yr $/TFLOP decline)")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(consumer_years, consumer_gflops_per_dollar, zorder=5, s=90,
               color="#1f77b4", edgecolors="white", linewidths=1.0,
               label="Consumer GPUs")
    ax.scatter(dc_years, dc_gflops_per_dollar, zorder=5, s=90, marker="^",
               color="#ff7f0e", edgecolors="white", linewidths=1.0,
               label="Datacenter GPUs")

    ax.plot(years_line, c_fit, "--", color="#1f77b4", alpha=0.55, linewidth=1.6)
    ax.plot(years_line, dc_fit, "--", color="#ff7f0e", alpha=0.55, linewidth=1.6)

    # per-GPU label offsets in points (avoid neighbour overlap)
    consumer_offsets = {
        "GTX 280":     (8,  -4),
        "GTX 580":     (8,   6),
        "GTX 980":     (-55, 10),
        "GTX 1080 Ti": (-90,-18),
        "RTX 2080 Ti": (8,   6),
        "RTX 3090":    (8,  -4),
        "RTX 4090":    (-90, 10),
    }
    dc_offsets = {
        "P100":      (-40,-20),
        "V100 SXM2": (8,  -16),
        "A100 SXM4": (-65, 8),
        "H100 SXM5": (-90,-18),
    }
    for name, yr, val in zip(consumer_names, consumer_years, consumer_gflops_per_dollar):
        if name in consumer_offsets:
            ax.annotate(name, (yr, val), textcoords="offset points",
                        xytext=consumer_offsets[name], fontsize=14,
                        color="#1f77b4")
    for name, yr, val in zip(dc_names, dc_years, dc_gflops_per_dollar):
        if name in dc_offsets:
            ax.annotate(name, (yr, val), textcoords="offset points",
                        xytext=dc_offsets[name], fontsize=14,
                        color="#ff7f0e")

    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("FP32 GFLOP/s per Dollar")
    ax.set_xlim(2007, 2025)
    ax.set_xticks(np.arange(2008, 2025, 3))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "epoch_gpu_price_perf.pdf"), bbox_inches="tight")
    print(f"Saved epoch_gpu_price_perf.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Epoch AI – FP16 TFLOP/s per dollar (from CSV)
# ====================================================================
def plot_epoch_fp16_trend():
    """Plot FP16 TFLOP/s per dollar for ML hardware from Epoch AI dataset.

    Data source:
        Epoch AI, "Data on Machine Learning Hardware", 2024.
        CSV: https://epoch.ai/data/ml_hardware.csv
        Web: https://epoch.ai/data/machine-learning-hardware
        License: CC-BY 4.0

    All entries in ml_hardware.csv with both FP16 (half precision) performance
    and a release price, sorted chronologically:

    ┌──────────────────────────────────────────┬────────────┬──────────┬──────────────┬──────────────┬────────────┐
    │ Hardware                                 │ Date       │ Price($) │ FP16 (TFLOPS)│ GFLOP/s/$    │ Type       │
    ├──────────────────────────────────────────┼────────────┼──────────┼──────────────┼──────────────┼────────────┤
    │ NVIDIA P100                              │ 2016-04-05 │    5699  │    18.70     │     3.28     │ datacenter │
    │ NVIDIA GeForce GTX 1080                  │ 2016-05-27 │     599  │     0.14     │     0.23     │ consumer   │
    │ NVIDIA P40                               │ 2016-09-13 │    5699  │     0.18     │     0.03     │ datacenter │
    │ NVIDIA Quadro P5000                      │ 2016-10-01 │    2499  │     0.14     │     0.06     │ datacenter │
    │ NVIDIA Quadro P6000                      │ 2016-10-01 │    5999  │     0.20     │     0.03     │ datacenter │
    │ NVIDIA Quadro P4000                      │ 2017-02-06 │     815  │     0.08     │     0.10     │ datacenter │
    │ NVIDIA Quadro P600                       │ 2017-02-07 │     178  │     0.02     │     0.10     │ datacenter │
    │ NVIDIA GeForce GTX 1080 Ti               │ 2017-03-10 │     699  │     0.18     │     0.25     │ consumer   │
    │ NVIDIA TITAN Xp                          │ 2017-04-06 │    1199  │     0.19     │     0.16     │ consumer   │
    │ NVIDIA Tesla V100 PCIe 16 GB             │ 2017-06-21 │   10664  │    28.26     │     2.65     │ datacenter │
    │ NVIDIA Tesla V100 SXM2                   │ 2017-07-01 │   11500  │    31.33     │     2.72     │ datacenter │
    │ NVIDIA Titan V                           │ 2017-12-01 │    2999  │    29.80     │     9.94     │ datacenter │
    │ NVIDIA Tesla V100 PCIe 32 GB             │ 2018-03-27 │   11458  │    28.26     │     2.47     │ datacenter │
    │ NVIDIA Tesla V100 SXM2 32 GB             │ 2018-03-27 │   11458  │    31.33     │     2.73     │ datacenter │
    │ NVIDIA Quadro RTX 6000                   │ 2018-08-13 │    6300  │    32.62     │     5.18     │ datacenter │
    │ NVIDIA Quadro RTX 5000                   │ 2018-08-13 │    2300  │    22.30     │     9.70     │ datacenter │
    │ NVIDIA GeForce RTX 2080 Ti 11GB          │ 2018-09-20 │     999  │    26.90     │    26.93     │ consumer   │
    │ NVIDIA Quadro RTX 4000                   │ 2018-11-13 │     899  │    14.24     │    15.84     │ datacenter │
    │ NVIDIA A100 SXM4 40 GB                   │ 2020-05-14 │   15000  │    77.97     │     5.20     │ datacenter │
    │ NVIDIA GeForce RTX 3080                  │ 2020-09-01 │     699  │    29.77     │    42.59     │ consumer   │
    │ NVIDIA GeForce RTX 3090                  │ 2020-09-01 │    1499  │    35.58     │    23.74     │ consumer   │
    │ NVIDIA RTX A6000                         │ 2020-10-05 │    4649  │    38.70     │     8.32     │ datacenter │
    │ NVIDIA RTX A4000                         │ 2021-04-12 │    1000  │    19.17     │    19.17     │ datacenter │
    │ NVIDIA RTX A5000                         │ 2021-04-12 │    2250  │    27.77     │    12.34     │ datacenter │
    │ NVIDIA A10 PCIe                          │ 2021-04-12 │    2800  │    31.24     │    11.16     │ datacenter │
    │ AMD Radeon Instinct MI250X               │ 2021-11-08 │   14500  │   383.00     │    26.41     │ datacenter │
    │ Huawei Ascend 910B                       │ 2022-03-17 │   16854  │   376.00     │    22.31     │ datacenter │
    │ NVIDIA GeForce RTX 3090 Ti               │ 2022-03-29 │    1999  │    40.00     │    20.01     │ consumer   │
    │ NVIDIA GeForce RTX 4090                  │ 2022-09-20 │    1599  │    82.60     │    51.66     │ consumer   │
    │ NVIDIA H100 SXM5 80GB                    │ 2022-09-20 │   33600  │   133.80     │     3.98     │ datacenter │
    │ NVIDIA GeForce RTX 4080                  │ 2022-09-20 │    1199  │    48.74     │    40.65     │ consumer   │
    │ AMD Radeon RX 7900 XTX                   │ 2022-11-03 │     999  │   122.80     │   122.92     │ consumer   │
    │ NVIDIA RTX 6000 Ada Generation           │ 2022-12-03 │    6799  │    91.06     │    13.39     │ datacenter │
    │ NVIDIA L20 PCIe                          │ 2023-09-16 │    4299  │    59.35     │    13.81     │ datacenter │
    │ AMD Instinct MI300X                      │ 2023-12-06 │   15000  │  1307.40     │    87.16     │ datacenter │
    │ NVIDIA B100 SXM (Blackwell) [a]          │ 2024-11-15 │   35000  │  1750.00     │    50.00     │ datacenter │
    └──────────────────────────────────────────┴────────────┴──────────┴──────────────┴──────────────┴────────────┘

    [a] Blackwell does not separately publish a non-tensor FP16 rate. The Epoch
        CSV lists vector FP16 = Tensor-FP16 = 2250 TFLOPS for B200 and 2500 for
        GB200 (same value in both columns), so we apply the same convention to
        B100 SXM (1750 TFLOPS dense, source: Epoch CSV Tensor-FP16/BF16 column).

    Note: Pascal-generation consumer GPUs (GTX 1080, GTX 1080 Ti, TITAN Xp) report
    very low FP16 rates (~0.14–0.19 TFLOPS) because Pascal lacks native FP16 tensor
    acceleration (half-rate FP16 = FP32/2).  Volta (V100, Titan V) and Turing (RTX
    2080 Ti) introduced hardware FP16 tensor cores, causing the large jump visible
    in the consumer curve around 2018.

    ── Plotted subset ────────────────────────────────────────────────
    To keep the figure focused, only the user-requested NVIDIA flagships
    plus the AMD Instinct lineage are actually plotted. Three older NVIDIA
    datacenter cards (K80, M60, P100) are listed below for context but
    K80 and M60 have no FP16 hardware so they appear in this summary
    table only and are not plotted.

    The "Tensor FP16" column shows the matrix / tensor engine FP16 rate
    when the GPU has one; entries with no tensor or matrix engine fall
    back to the FP16 (vector) value in the previous column, which is
    what would be used as a "tensor-like" comparison point for that GPU.

    ┌──────────────────────────────────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┬─────────┐
    │ Hardware                                 │ Date       │ Price($) │ FP16 (TFLOPS)│ Tensor FP16  │ GFLOP/s/$    │ Plotted │
    │                                          │            │          │              │ (TFLOPS)     │ (FP16 col)   │         │
    ├──────────────────────────────────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┼─────────┤
    │ NVIDIA Tesla K80    [no FP16 hw]         │ 2014-11-17 │    5000  │    N/A       │    N/A       │    N/A       │   no    │
    │ NVIDIA Tesla M60    [no FP16 hw]         │ 2015-08-30 │    2700  │    N/A       │    N/A       │    N/A       │   no    │
    │ NVIDIA Tesla P100   [no tensor → vec]    │ 2016-04-05 │    5699  │    18.70     │    18.70     │     3.28     │   yes   │
    │ AMD Radeon Instinct MI25 [no matrix→vec] │ 2017-06-20 │   16000  │    24.60     │    24.60     │     1.54     │   yes   │
    │ NVIDIA Tesla V100 SXM2 32 GB             │ 2018-03-27 │   11458  │    31.33     │   125.00     │     2.73     │   yes   │
    │ AMD Radeon Instinct MI60 [no matrix→vec] │ 2018-11-18 │    7000  │    29.50     │    29.50     │     4.21     │   yes   │
    │ NVIDIA A100 SXM4 40 GB                   │ 2020-05-14 │   15000  │    77.97     │   312.00     │     5.20     │   yes   │
    │ AMD Instinct MI100         [matrix=vec]  │ 2020-11-16 │    6400  │   184.60     │   184.60     │    28.84     │   yes   │
    │ AMD Radeon Instinct MI250X [matrix=vec]  │ 2021-11-08 │   14500  │   383.00     │   383.00     │    26.41     │   yes   │
    │ NVIDIA H100 SXM5 80GB                    │ 2022-09-20 │   33600  │   133.80     │   989.40     │     3.98     │   yes   │
    │ AMD Instinct MI300X        [matrix=vec]  │ 2023-12-06 │   15000  │  1307.40     │  1307.40     │    87.16     │   yes   │
    │ NVIDIA B100 SXM (Blackwell)              │ 2024-11-15 │   35000  │  1750.00     │  1750.00     │    50.00     │   yes   │
    │ AMD Instinct MI350X        [matrix=vec]  │ 2025-06-12 │   25000  │  2306.90     │  2306.90     │    92.28     │   yes   │
    └──────────────────────────────────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┴─────────┘

    Pricing / convention notes for the plotted subset:
        - K80, M60: NVIDIA list prices at launch.
        - MI25: AMD launch list price reported as $16,000.
        - MI60: AMD did not publish an MSRP at launch; $7,000 used as a
          best estimate based on retail/OEM listings 2018-2019.
        - MI350X: launched 2025-06-12 at $25,000 (post AMD price revision
          from initial $15,000 expectation, per HSBC research).
        - AMD MI100/MI250X/MI300X/MI350X publish "matrix" FP16 rates
          that are reported in the Epoch CSV's FP16 column, while NVIDIA
          cards list vector FP16. We follow Epoch's convention here.
    """

    # Format: (name, date_str, fp16_tflops, price_usd, type)
    # Restricted to the flagship V100/A100/H100/B100 NVIDIA generations plus
    # AMD MI25/MI60/MI100/MI250X/MI300X/MI350X, and P100. K80, M60 are in
    # the table above but have no FP16 hardware so are not plotted.
    data = [
        ("P100",              "2016-04-05",   18.700,  5699, "datacenter"),
        ("MI25",              "2017-06-20",   24.600, 16000, "datacenter"),
        ("V100 SXM2 32GB",    "2018-03-27",   31.330, 11458, "datacenter"),
        ("MI60",              "2018-11-18",   29.500,  7000, "datacenter"),
        ("A100 SXM4 40GB",    "2020-05-14",   77.970, 15000, "datacenter"),
        ("MI100",             "2020-11-16",  184.600,  6400, "datacenter"),
        ("MI250X",            "2021-11-08",  383.000, 14500, "datacenter"),
        ("H100 SXM5 80GB",    "2022-09-20",  133.800, 33600, "datacenter"),
        ("MI300X",            "2023-12-06", 1307.400, 15000, "datacenter"),
        # B100 vector FP16 = Tensor-FP16 (Epoch CSV convention for Blackwell).
        ("B100 SXM",          "2024-11-15", 1750.000, 35000, "datacenter"),
        ("MI350X",            "2025-06-12", 2306.900, 25000, "datacenter"),
    ]

    # Parse dates to fractional years
    def to_year(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    consumer_years, consumer_gflops_per_dollar = [], []
    dc_years, dc_gflops_per_dollar = [], []
    consumer_names, dc_names = [], []

    for name, date_str, fp16_tflops, price, typ in data:
        yr = to_year(date_str)
        gflops_per_dollar = fp16_tflops * 1000 / price  # GFLOP/s per $
        if typ == "consumer":
            consumer_years.append(yr)
            consumer_gflops_per_dollar.append(gflops_per_dollar)
            consumer_names.append(name)
        else:
            dc_years.append(yr)
            dc_gflops_per_dollar.append(gflops_per_dollar)
            dc_names.append(name)

    consumer_years = np.array(consumer_years)
    consumer_gflops_per_dollar = np.array(consumer_gflops_per_dollar)
    dc_years = np.array(dc_years)
    dc_gflops_per_dollar = np.array(dc_gflops_per_dollar)

    # ── Log-linear fits ──────────────────────────────────────────────
    has_consumer = len(consumer_years) > 0
    if has_consumer:
        c_coeffs = np.polyfit(consumer_years, np.log(consumer_gflops_per_dollar), 1)
        c_doubling_yrs = np.log(2) / c_coeffs[0]
        c_annual_pct = (np.exp(c_coeffs[0]) - 1) * 100

    dc_coeffs = np.polyfit(dc_years, np.log(dc_gflops_per_dollar), 1)
    dc_doubling_yrs = np.log(2) / dc_coeffs[0]
    dc_annual_pct = (np.exp(dc_coeffs[0]) - 1) * 100

    if has_consumer:
        print(f"[FP16 Consumer GPUs]  doubling time: {c_doubling_yrs:.2f} yr  |  "
              f"annual improvement: +{c_annual_pct:.1f}%/yr  "
              f"(≈ {100 - 100/(1 + c_annual_pct/100):.1f}%/yr $/TFLOP decline)")
    print(f"[FP16 Datacenter GPUs] doubling time: {dc_doubling_yrs:.2f} yr  |  "
          f"annual improvement: +{dc_annual_pct:.1f}%/yr  "
          f"(≈ {100 - 100/(1 + dc_annual_pct/100):.1f}%/yr $/TFLOP decline)")

    years_line = np.linspace(2016, 2026, 200)
    if has_consumer:
        c_fit  = np.exp(np.polyval(c_coeffs,  years_line))
    dc_fit = np.exp(np.polyval(dc_coeffs, years_line))

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # Consumer GPUs commented out — datacenter only, in blue.
    # ax.scatter(consumer_years, consumer_gflops_per_dollar, zorder=5, s=90,
    #            color="#1f77b4", edgecolors="white", linewidths=1.0,
    #            label=f"Consumer (+{c_annual_pct:.0f}%/yr, 2$\\times$/{c_doubling_yrs:.1f}yr)")
    ax.scatter(dc_years, dc_gflops_per_dollar, zorder=5, s=90, marker="^",
               color="#1f77b4", edgecolors="white", linewidths=1.0,
               label=f"Datacenter GPUs\n+{dc_annual_pct:.0f}%/yr, 2$\\times$/{dc_doubling_yrs:.1f}yr")

    # ax.plot(years_line, c_fit,  "--", color="#1f77b4", alpha=0.55, linewidth=1.6)
    ax.plot(years_line, dc_fit, "--", color="#1f77b4", alpha=0.55, linewidth=1.6)

    # per-GPU label offsets (avoid overlap with neighbours)
    consumer_offsets = {
        "RTX 2080 Ti": (8,  6),
        "RTX 3090":    (-90, 6),
        "RTX 4090":    (8,  6),
        "RX 7900 XTX": (-100, 8),
    }
    dc_offsets = {
        "V100 SXM2 32GB": (8,  0),
        "A100 SXM4 40GB": (-20, 6),
        "H100 SXM5 80GB": (-20,  -18),
        # "MI250X":         (-50, 10),
        # "MI300X":         (-65, 10),
        "B100 SXM":       (-25, 10),
    }
    # for name, yr, val in zip(consumer_names, consumer_years, consumer_gflops_per_dollar):
    #     if name in consumer_offsets:
    #         ax.annotate(name, (yr, val), textcoords="offset points",
    #                     xytext=consumer_offsets[name], fontsize=14,
    #                     color="#1f77b4")
    for name, yr, val in zip(dc_names, dc_years, dc_gflops_per_dollar):
        if name in dc_offsets:
            ax.annotate(name, (yr, val), textcoords="offset points",
                        xytext=dc_offsets[name], fontsize=14,
                        color="#1f77b4")

    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("FP16 GFLOP/s per Dollar",fontsize=19)
    ax.set_xlim(2015, 2026)
    ax.set_xticks(np.arange(2016, 2026, 2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper left", fontsize=17)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "epoch_fp16_price_perf.pdf"), bbox_inches="tight")
    print(f"Saved epoch_fp16_price_perf.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Lowest-bit TFLOP/s per dollar (per-GPU lowest supported precision)
# ====================================================================
def plot_epoch_fp16_lowbit_trend():
    """Augmented version of plot_epoch_fp16_trend(): instead of pricing every
    GPU at FP16, each GPU is priced at the *lowest numeric precision its
    tensor / matrix engine natively supports*. This rewards the newer
    accelerators that added FP8 (Hopper, CDNA3) and FP4 (Blackwell, CDNA4),
    where the cheap-precision throughput per dollar is several times the FP16
    rate, so the price-performance curve steepens versus the FP16-only view.

    Convention (dense, no 2x structured-sparsity bonus), matching the rest of
    this file:
        - GPUs with no tensor / matrix engine (P100, MI25, MI60) fall back to
          their vector FP16 rate, same value as plot_epoch_fp16_trend().
        - GPUs whose lowest float precision is FP16/BF16 use the dense tensor
          (matrix) FP16 rate.
        - Hopper (H100) and CDNA3 (MI300X) drop to FP8.
        - Blackwell (B100) and CDNA4 (MI350X) drop to FP4.
    INT8/INT4 integer rates are deliberately excluded so the y-axis stays a
    single floating-point throughput-per-dollar metric.

    Datacenter / accelerator subset (plotted):
    
    ┌──────────────────────┬────────────┬──────────┬────────┬───────────┬───────────┬──────────────────────────────────────┐
    │ Hardware             │ Date       │ Price($) │ Prec.  │ TFLOP/s   │ GFLOP/s/$ │ Lowest-bit throughput source          │
    │                      │            │          │ (low)  │ (dense)   │           │                                       │
    ├──────────────────────┼────────────┼──────────┼────────┼───────────┼───────────┼──────────────────────────────────────┤
    │ P100                 │ 2016-04-05 │    5699  │ FP16   │    18.70  │    3.28   │ vector FP16, no tensor engine        │
    │ MI25                 │ 2017-06-20 │   16000  │ FP16   │    24.60  │    1.54   │ vector FP16, no matrix engine        │
    │ V100 SXM2 32GB       │ 2018-03-27 │   11458  │ FP16   │   125.00  │   10.91   │ tensor FP16 (Volta)                  │
    │ MI60                 │ 2018-11-18 │    7000  │ FP16   │    29.50  │    4.21   │ vector FP16, no matrix engine        │
    │ A100 SXM4 40GB       │ 2020-05-14 │   15000  │ FP16   │   312.00  │   20.80   │ tensor FP16/BF16 (no FP8)            │
    │ MI100                │ 2020-11-16 │    6400  │ FP16   │   184.60  │   28.84   │ matrix FP16 (no FP8)                 │
    │ MI250X               │ 2021-11-08 │   14500  │ FP16   │   383.00  │   26.41   │ matrix FP16/BF16 (no FP8)            │
    │ H100 SXM5 80GB       │ 2022-09-20 │   33600  │ FP8    │  1978.90  │   58.90   │ NVIDIA H100 datasheet (FP8 dense)    │
    │ MI300X               │ 2023-12-06 │   15000  │ FP8    │  2614.90  │  174.33   │ AMD MI300X datasheet (FP8 dense)     │
    │ B100 SXM (Blackwell) │ 2024-11-15 │   35000  │ FP4    │  7000.00  │  200.00   │ NVIDIA Blackwell brief (FP4 dense)   │
    │ MI350X (CDNA4)       │ 2025-06-12 │   25000  │ FP4    │  9216.00  │  368.64   │ AMD MI350 series (FP4 dense est.)    │
    └──────────────────────┴────────────┴──────────┴────────┴───────────┴───────────┴──────────────────────────────────────┘

    Notes on the FP4 dense figures:
        - B100 SXM: NVIDIA quotes 14 PFLOPS FP4 with 2:4 sparsity, i.e.
          ~7.0 PFLOPS dense.
        - MI350X: AMD quotes ~18.45 PFLOPS FP4 with sparsity, i.e.
          ~9.2 PFLOPS dense. Treated as a best estimate, revisit against the
          final datasheet before publication.

    ── Non-datacenter (consumer) low-bit reference, NOT plotted here ─────
    Listed for a future consumer-only variant of this figure. Same dense,
    lowest-float-precision convention; launch MSRP.
    ┌──────────────────────┬────────────┬──────────┬────────┬───────────┬───────────┐
    │ Hardware             │ Date       │ Price($) │ Prec.  │ TFLOP/s   │ GFLOP/s/$ │
    ├──────────────────────┼────────────┼──────────┼────────┼───────────┼───────────┤
    │ RTX 2080 Ti          │ 2018-09-20 │     999  │ FP16   │   113.80  │  113.91   │
    │ RTX 3090             │ 2020-09-01 │    1499  │ FP16   │   142.00  │   94.73   │
    │ RTX 3090 Ti          │ 2022-03-29 │    1999  │ FP16   │   160.00  │   80.04   │
    │ RTX 4090             │ 2022-09-20 │    1599  │ FP8    │   660.60  │  413.13   │
    │ RTX 5090 (Blackwell) │ 2025-01-30 │    1999  │ FP4    │  1676.00  │  838.42   │
    └──────────────────────┴────────────┴──────────┴────────┴───────────┴───────────┘
        RTX 4090 FP8 = 660.6 TFLOPS dense (1321 sparse). RTX 5090 FP4 =
        ~1676 TFLOPS dense (3352 AI TOPS quoted with sparsity).
    """

    # Format: (name, date_str, lowbit_tflops, price_usd, precision, fp16_tflops)
    #   fp16_tflops = dense tensor/matrix FP16 rate (vector FP16 for the
    #   pre-tensor P100/MI60), i.e. the value used in the FP16 figure. For
    #   FP16-precision GPUs lowbit_tflops == fp16_tflops so the low-bit and
    #   FP16 points coincide and are drawn as one half-blue/half-green marker.
    data = [
        ("P100",              "2016-04-05",   18.700,  5699, "FP16",   18.700),
        ("MI25",              "2017-06-20",   24.600, 16000, "FP16",   24.600),
        ("V100 SXM2 32GB",    "2018-03-27",  125.000, 11458, "FP16",  125.000),
        ("MI60",              "2018-11-18",   29.500,  7000, "FP16",   29.500),
        ("A100 SXM4 40GB",    "2020-05-14",  312.000, 15000, "FP16",  312.000),
        ("MI100",             "2020-11-16",  184.600,  6400, "FP16",  184.600),
        ("MI250X",            "2021-11-08",  383.000, 14500, "FP16",  383.000),
        ("H100 SXM5 80GB",    "2022-09-20", 1978.900, 33600, "FP8",   989.400),
        ("MI300X",            "2023-12-06", 2614.900, 15000, "FP8",  1307.400),
        # B200 price $40,000: top of Jensen Huang's stated $30k-$40k Blackwell
        # per-unit range (no official standalone list price). Source:
        # https://www.tomshardware.com/pc-components/gpus/nvidias-jensen-huang-says-blackwell-gpu-to-cost-dollar30000-dollar40000-later-clarifies-that-pricing-will-vary-as-they-wont-sell-just-the-chip
        ("B200 SXM",          "2024-11-15", 9000.000, 40000, "FP4",  2250.000),
        ("MI350X",            "2025-06-12", 9216.000, 25000, "FP4",  2306.900),
    ]

    # ── Presentation: normalized $/(GFLOP/s), matching
    #    plot_tensor_fp16_price_normalized(). The metric declines over time
    #    and every value is divided by the last historical data point
    #    (MI350X, 2025-06-12) so the curve passes through 1.0 at that release.
    #    MI25 is dropped as an outlier (high list price for a vector-only
    #    Vega 10 card), consistent with the FP16 normalized figure.
    #
    #    Both the FP16 and the lowest-bit series are drawn on one axis (both
    #    normalized by the same anchor, the last low-bit point), so the FP16
    #    projection can be shown alongside the low-bit projection.
    #    Colour convention requested:
    #      - blue triangle        : FP16 (tensor) point of a sub-FP16 GPU.
    #      - dark-green triangle   : GPU whose lowest precision drops below
    #        FP16 (FP8 / FP4); a thin connector links it to the blue FP16 point.
    #      - half-blue/half-green : GPU whose lowest precision is still FP16,
    #        so its FP16 and low-bit points coincide (no low-bit speed-up).
    #      - dark-brown projection : lowest-bit projection.
    #      - orange projection     : FP16 projection.
    DARK_GREEN = "#006d2c"
    FP16_BLUE  = "#1f77b4"
    DARK_BROWN = "#654321"

    # Parse dates to fractional years
    def to_year(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    FP16_ORANGE = "#ff7f0e"

    plotted = [r for r in data if r[0] != "MI25"]
    dc_names = [r[0] for r in plotted]
    dc_years = np.array([to_year(r[1]) for r in plotted])
    lowbit_dpg = np.array([r[3] / (r[2] * 1000) for r in plotted])  # $/(GFLOP/s) low bit
    fp16_dpg   = np.array([r[3] / (r[5] * 1000) for r in plotted])  # $/(GFLOP/s) FP16
    dc_prec  = [r[4] for r in plotted]
    # FP16-precision GPUs: low-bit point == FP16 point, so the two coincide.
    shared   = np.array([p == "FP16" for p in dc_prec])

    # ── normalize BOTH series by the last low-bit data point (MI350X) so the
    #    two curves share one axis; the FP16-precision points then coincide. ──
    anchor_idx  = int(np.argmax(dc_years))
    anchor_year = dc_years[anchor_idx]
    anchor_val  = lowbit_dpg[anchor_idx]
    lowbit_y = lowbit_dpg / anchor_val
    fp16_y   = fp16_dpg   / anchor_val

    # ── low-bit fit + projection (anchored at MI350X = 1.0) ──────────
    lb_coeffs = np.polyfit(dc_years, np.log(lowbit_y), 1)
    lb_decline = (1 - np.exp(lb_coeffs[0])) * 100
    years_fit = np.linspace(2016, anchor_year, 200)
    lb_fit_y = np.exp(np.polyval(lb_coeffs, years_fit))
    years_proj = np.arange(2026, 2036)
    lb_proj_y = np.exp(lb_coeffs[0] * (years_proj - anchor_year))  # = 1 at anchor

    # ── FP16 fit + projection (anchored at FP16 MI350X) ──────────────
    fp16_coeffs = np.polyfit(dc_years, np.log(fp16_y), 1)
    fp16_decline = (1 - np.exp(fp16_coeffs[0])) * 100
    fp16_anchor_y = fp16_y[anchor_idx]
    fp16_fit_y = np.exp(np.polyval(fp16_coeffs, years_fit))
    fp16_proj_y = fp16_anchor_y * np.exp(fp16_coeffs[0] * (years_proj - anchor_year))

    print(f"[Low-bit Datacenter GPUs, normalized] low-bit fit -{lb_decline:.1f}%/yr, "
          f"FP16 fit -{fp16_decline:.1f}%/yr  (anchor MI350X {anchor_year:.1f})")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(1.0, color="0.5", linewidth=0.9, linestyle=":", alpha=0.7)

    sub = ~shared  # GPUs that drop below FP16 (have a distinct low-bit point)

    # Annotated A/H/B(/V100) flagships stay opaque; the unlabelled rest
    # (P100, MI family) are dimmed, as in plot_tensor_fp16_price_normalized.
    ANN = {"V100 SXM2 32GB", "A100 SXM4 40GB", "H100 SXM5 80GB", "B200 SXM"}
    ann = np.array([n in ANN for n in dc_names])
    A_HI, A_LO = 1.0, 0.45

    # FP16 (tensor) points for the sub-FP16 GPUs: blue triangles.
    ax.scatter(dc_years[sub & ann], fp16_y[sub & ann], zorder=5, s=130, marker="^",
               color=FP16_BLUE, edgecolors="white", linewidths=1.0,
               alpha=A_HI, label="FP16")
    ax.scatter(dc_years[sub & ~ann], fp16_y[sub & ~ann], zorder=5, s=130, marker="^",
               color=FP16_BLUE, edgecolors="white", linewidths=1.0,
               alpha=A_LO, label="_nolegend_")
    # FP8/FP4 low-bit points: solid dark-green triangles.
    ax.scatter(dc_years[sub & ann], lowbit_y[sub & ann], zorder=6, s=150, marker="^",
               color=DARK_GREEN, edgecolors="white", linewidths=1.0,
               alpha=A_HI, label="low precision")
    ax.scatter(dc_years[sub & ~ann], lowbit_y[sub & ~ann], zorder=6, s=150, marker="^",
               color=DARK_GREEN, edgecolors="white", linewidths=1.0,
               alpha=A_LO, label="_nolegend_")
    # FP16-is-lowest points: half blue / half green triangles (coincide on
    # both series). fillstyle is a Line2D feature, so use ax.plot. No legend
    # entry (covered by the FP16 / lowest-bit entries above).
    for m, a in ((shared & ann, A_HI), (shared & ~ann, A_LO)):
        ax.plot(dc_years[m], lowbit_y[m], linestyle="none", marker="^",
                markersize=12, fillstyle="left",
                markerfacecolor=FP16_BLUE, markerfacecoloralt=DARK_GREEN,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6, alpha=a)

    # dashed historical fits (blue = FP16, green = lowest bit); no legend entry.
    ax.plot(years_fit, fp16_fit_y, "--", color=FP16_BLUE, alpha=0.8,
            linewidth=1.8, label="_nolegend_")
    ax.plot(years_fit, lb_fit_y, "--", color=DARK_GREEN, alpha=0.8,
            linewidth=1.8, label="_nolegend_")

    # FP16 proj. listed above low-precision proj. in legend
    ax.plot(years_proj, fp16_proj_y, "-o", color=FP16_ORANGE, markersize=8,
            linewidth=2.2, label=f"FP16 proj. (-{fp16_decline:.0f}%/yr)")
    ax.plot(years_proj, lb_proj_y, "-o", color=DARK_BROWN, markersize=8,
            linewidth=2.2, label=f"Low-precision proj. (-{lb_decline:.0f}%/yr)")

    # Annotate only the A/H/B NVIDIA flagships (A100, H100, B100); never the
    # MI family. For H100/B100 label both their FP16 and low-bit points.
    # (name, series_y_array, text, (dx,dy) offset in points, colour)
    idx = {n: i for i, n in enumerate(dc_names)}
    annotations = [
        ("V100 SXM2 32GB", fp16_y,   "V100",        (-15,    -25),  FP16_BLUE),
        ("A100 SXM4 40GB", fp16_y,   "A100",        (-18,  14), FP16_BLUE),
        ("H100 SXM5 80GB", fp16_y,   "H100 (FP16)", (-20,  13), FP16_BLUE),
        ("H100 SXM5 80GB", lowbit_y, "H100 (FP8)",  (-80,  -25), DARK_GREEN),
        ("B200 SXM",       fp16_y,   "B200 (FP16)", (8,   -5), FP16_BLUE),
        ("B200 SXM",       lowbit_y, "B200 (FP4)",  (-95,   -25), DARK_GREEN),
    ]
    for name, yarr, text, off, color in annotations:
        i = idx[name]
        ax.annotate(text, (dc_years[i], yarr[i]), textcoords="offset points",
                    xytext=off, fontsize=17, color=color)

    ax.set_xlabel("Year")
    ax.set_ylabel("GPU Price \n (Normalized to 2025)", fontsize=19)
    ax.set_xlim(2015.5, 2035.5)
    ax.set_xticks(np.arange(2017, 2036, 3))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([0.01, 0.1, 1, 10, 50])
    ax.legend(loc="lower left", fontsize=15, ncol=1)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "epoch_fp16_lowbit_price_perf_normalized.pdf"),
                bbox_inches="tight")
    print(f"Saved epoch_fp16_lowbit_price_perf_normalized.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Tensor FP16 / BF16 TFLOP/s per dollar (matrix throughput)
# ====================================================================
def plot_epoch_tensor_fp16_trend(inverted=True):
    """Plot Tensor-FP16/BF16 TFLOP/s per dollar for ML hardware.

    Set ``inverted=True`` to plot $/(GFLOP/s) (cost per unit throughput,
    which declines over time). Set ``inverted=False`` to plot the
    original GFLOP/s per $ (throughput per dollar, which grows over time).

    Tensor-FP16 (matrix-multiply / tensor-core) throughput is the metric that
    actually governs ML training and inference performance on tensor-equipped
    accelerators. It is typically 4 to 7x higher than the vector ("CUDA core")
    FP16 rate plotted in plot_epoch_fp16_trend(); e.g. NVIDIA H100 SXM5 has
    133.8 TFLOPS vector FP16 vs. 989.4 TFLOPS Tensor-FP16 (dense).

    Coverage starts at Volta (2017), the first NVIDIA architecture with tensor
    cores. Pascal and earlier have no tensor units and are excluded.

    Primary source:
        Epoch AI, "Data on Machine Learning Hardware", 2024.
        CSV: https://epoch.ai/data/ml_hardware.csv  (column "Tensor-FP16/BF16
        performance (FLOP/s)"). License: CC-BY 4.0.

    The Epoch CSV provides Tensor-FP16 values for 14 entries. For workstation
    and consumer cards that have tensor cores but a blank Tensor-FP16 cell in
    the CSV, the value is filled in from the vendor datasheet/whitepaper:

        - NVIDIA Volta/Turing/Ampere/Ada whitepapers and product datasheets
          (nvidia.com); Titan V product page.
        - AMD RDNA 3 architecture whitepaper for RX 7900 XTX (WMMA FP16).
        - Huawei Atlas/Ascend product brief for Ascend 910B (BF16 tensor).

    All values are dense (no 2x sparsity speed-up), matching the convention
    used by the Epoch CSV (e.g. A100 = 312 TFLOPS dense, not 624 sparse).

    ┌──────────────────────────────────────┬────────────┬──────────┬──────────────┬───────────┬─────────────┬────────────┬───────────────────────────────────────────┐
    │ Hardware                             │ Date       │ Price($) │ Tensor FP16  │ GFLOP/s/$ │ $/(GFLOP/s) │ Type       │ Tensor-FP16 source                        │
    │                                      │            │          │ (TFLOPS)     │           │             │            │                                           │
    ├──────────────────────────────────────┼────────────┼──────────┼──────────────┼───────────┼─────────────┼────────────┼───────────────────────────────────────────┤
    │ NVIDIA Tesla V100 PCIe 16 GB         │ 2017-06-21 │   10664  │     112.0    │   10.50   │   0.09521   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA Tesla V100 SXM2               │ 2017-07-01 │   11500  │     125.0    │   10.87   │   0.09200   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA Titan V                       │ 2017-12-01 │    2999  │     110.0    │   36.68   │   0.02726   │ datacenter │ NVIDIA Titan V product page (110 TFLOPS DL) │
    │ NVIDIA Tesla V100 PCIe 32 GB         │ 2018-03-27 │   11458  │     112.0    │    9.77   │   0.10230   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA Tesla V100 SXM2 32 GB         │ 2018-03-27 │   11458  │     125.0    │   10.91   │   0.09166   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA Quadro RTX 6000               │ 2018-08-13 │    6300  │     130.5    │   20.71   │   0.04828   │ datacenter │ NVIDIA Quadro RTX 6000 datasheet          │
    │ NVIDIA Quadro RTX 5000               │ 2018-08-13 │    2300  │      89.2    │   38.78   │   0.02578   │ datacenter │ NVIDIA Quadro RTX 5000 datasheet          │
    │ NVIDIA GeForce RTX 2080 Ti 11GB      │ 2018-09-20 │     999  │     113.8    │  113.91   │   0.00878   │ consumer   │ Epoch CSV                                 │
    │ NVIDIA Quadro RTX 4000               │ 2018-11-13 │     899  │      57.0    │   63.40   │   0.01577   │ datacenter │ NVIDIA Quadro RTX 4000 datasheet          │
    │ NVIDIA A100 SXM4 40 GB               │ 2020-05-14 │   15000  │     312.0    │   20.80   │   0.04808   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA GeForce RTX 3080              │ 2020-09-01 │     699  │     119.0    │  170.24   │   0.00587   │ consumer   │ NVIDIA Ampere GA102 whitepaper            │
    │ NVIDIA GeForce RTX 3090              │ 2020-09-01 │    1499  │     142.0    │   94.73   │   0.01056   │ consumer   │ NVIDIA Ampere GA102 whitepaper            │
    │ NVIDIA RTX A6000                     │ 2020-10-05 │    4649  │     154.8    │   33.30   │   0.03003   │ datacenter │ NVIDIA RTX A6000 datasheet                │
    │ NVIDIA RTX A4000                     │ 2021-04-12 │    1000  │      76.7    │   76.70   │   0.01304   │ datacenter │ NVIDIA RTX A4000 datasheet                │
    │ NVIDIA RTX A5000                     │ 2021-04-12 │    2250  │     111.1    │   49.38   │   0.02025   │ datacenter │ NVIDIA RTX A5000 datasheet                │
    │ NVIDIA A10 PCIe                      │ 2021-04-12 │    2800  │     125.0    │   44.64   │   0.02240   │ datacenter │ NVIDIA A10 datasheet                      │
    │ AMD Radeon Instinct MI250X           │ 2021-11-08 │   14500  │     383.0    │   26.41   │   0.03786   │ datacenter │ Epoch CSV                                 │
    │ Huawei Ascend 910B                   │ 2022-03-17 │   16854  │     376.0    │   22.31   │   0.04482   │ datacenter │ Huawei Atlas 800T A2 product brief        │
    │ NVIDIA GeForce RTX 3090 Ti           │ 2022-03-29 │    1999  │     160.0    │   80.04   │   0.01249   │ consumer   │ Epoch CSV                                 │
    │ NVIDIA GeForce RTX 4090              │ 2022-09-20 │    1599  │     330.0    │  206.38   │   0.00485   │ consumer   │ Epoch CSV                                 │
    │ NVIDIA H100 SXM5 80GB                │ 2022-09-20 │   33600  │     989.4    │   29.45   │   0.03396   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA GeForce RTX 4080              │ 2022-09-20 │    1199  │     195.0    │  162.64   │   0.00615   │ consumer   │ NVIDIA Ada AD103 whitepaper               │
    │ AMD Radeon RX 7900 XTX               │ 2022-11-03 │     999  │     122.8    │  122.92   │   0.00814   │ consumer   │ AMD RDNA 3 whitepaper (WMMA FP16)         │
    │ NVIDIA RTX 6000 Ada Generation       │ 2022-12-03 │    6799  │     364.2    │   53.57   │   0.01867   │ datacenter │ NVIDIA RTX 6000 Ada datasheet             │
    │ NVIDIA L20 PCIe                      │ 2023-09-16 │    4299  │     119.5    │   27.80   │   0.03597   │ datacenter │ Epoch CSV                                 │
    │ AMD Instinct MI300X                  │ 2023-12-06 │   15000  │    1307.4    │   87.16   │   0.01147   │ datacenter │ Epoch CSV                                 │
    │ NVIDIA B100 SXM (Blackwell)          │ 2024-11-15 │   35000  │    1750.0    │   50.00   │   0.02000   │ datacenter │ Epoch CSV                                 │
    └──────────────────────────────────────┴────────────┴──────────┴──────────────┴───────────┴─────────────┴────────────┴───────────────────────────────────────────┘

    Note on "datacenter" label: includes true datacenter accelerators (V100,
    A100, H100, B100, MI250X, MI300X, L20, A10, Ascend 910B) as well as workstation
    cards (Quadro RTX, RTX A-series, RTX 6000 Ada, Titan V) which are not sold
    in the GeForce/Radeon retail channel. Strict consumer cards are GeForce
    GTX/RTX and Radeon RX.

    ── Plotted subset ────────────────────────────────────────────────
    The figure restricts the points to the user-requested NVIDIA flagships
    and AMD Instinct lineage, with five older / matrix-engine-less cards
    listed in this summary for context but excluded from the plot
    (no tensor / matrix unit, so no Tensor-FP16 number).

    Both an FP16 (vector) column and a Tensor FP16 column are shown so
    the two precision rates can be compared on the same row. The plot
    in this function uses the Tensor FP16 column when present, and falls
    back to the FP16 (vector) value when the GPU has no tensor or matrix
    engine (P100, MI25, MI60). K80 and M60 have no FP16 hardware at all
    so are listed in the table only and excluded from the plot.

    ┌──────────────────────────────────────────┬────────────┬──────────┬──────────────┬──────────────┬───────────┬─────────────┬─────────┐
    │ Hardware                                 │ Date       │ Price($) │ FP16 (TFLOPS)│ Tensor FP16  │ GFLOP/s/$ │ $/(GFLOP/s) │ Plotted │
    │                                          │            │          │  (vector)    │ (TFLOPS)     │ (plotted) │ (plotted)   │         │
    ├──────────────────────────────────────────┼────────────┼──────────┼──────────────┼──────────────┼───────────┼─────────────┼─────────┤
    │ NVIDIA Tesla K80    [no FP16 hw]         │ 2014-11-17 │    5000  │    N/A       │    N/A       │   N/A     │   N/A       │   no    │
    │ NVIDIA Tesla M60    [no FP16 hw]         │ 2015-08-30 │    2700  │    N/A       │    N/A       │   N/A     │   N/A       │   no    │
    │ NVIDIA Tesla P100   [no tensor → vec]    │ 2016-04-05 │    5699  │    18.70     │    18.70     │   3.28    │   0.30476   │   yes   │
    │ AMD Radeon Instinct MI25 [outlier]       │ 2017-06-20 │   16000  │    24.60     │    24.60     │   N/A     │   N/A       │   no    │
    │ NVIDIA Tesla V100 SXM2 32 GB             │ 2018-03-27 │   11458  │    31.33     │   125.00     │  10.91    │   0.09166   │   yes   │
    │ AMD Radeon Instinct MI60 [no matrix→vec] │ 2018-11-18 │    7000  │    29.50     │    29.50     │   4.21    │   0.23729   │   yes   │
    │ NVIDIA A100 SXM4 40 GB                   │ 2020-05-14 │   15000  │    77.97     │   312.00     │  20.80    │   0.04808   │   yes   │
    │ AMD Instinct MI100         [matrix=vec]  │ 2020-11-16 │    6400  │   184.60     │   184.60     │  28.84    │   0.03467   │   yes   │
    │ AMD Radeon Instinct MI250X [matrix=vec]  │ 2021-11-08 │   14500  │   383.00     │   383.00     │  26.41    │   0.03786   │   yes   │
    │ NVIDIA H100 SXM5 80GB                    │ 2022-09-20 │   33600  │   133.80     │   989.40     │  29.45    │   0.03396   │   yes   │
    │ AMD Instinct MI300X        [matrix=vec]  │ 2023-12-06 │   15000  │  1307.40     │  1307.40     │  87.16    │   0.01147   │   yes   │
    │ NVIDIA B100 SXM (Blackwell)              │ 2024-11-15 │   35000  │  1750.00     │  1750.00     │  50.00    │   0.02000   │   yes   │
    │ AMD Instinct MI350X        [matrix=vec]  │ 2025-06-12 │   25000  │  2306.90     │  2306.90     │  92.28    │   0.01084   │   yes   │
    └──────────────────────────────────────────┴────────────┴──────────┴──────────────┴──────────────┴───────────┴─────────────┴─────────┘

    Architectural notes on the plotted subset:
        - K80 (Kepler), M60 (Maxwell): no FP16 hardware at all.
        - P100 (Pascal): full-rate vector FP16 but no tensor cores.
        - MI25 (Vega 10), MI60 (Vega 20): vector FP16 only, no matrix engine.
        - V100 (Volta) and later NVIDIA: FP16 tensor cores from Volta forward.
        - MI100 onward (CDNA 1 to CDNA 4): matrix engine providing FP16 MMA.

    Pricing notes for the plotted subset entries that are not in the main
    table above:
        - MI25: AMD launch list price reported as $16,000.
        - MI60: AMD did not publish an MSRP at launch; $7,000 used as a
          best estimate based on retail/OEM listings 2018-2019.
        - MI100: AMD launch MSRP $6,400.
        - MI350X: launched 2025-06-12 at $25,000 (AMD revised price,
          per HSBC research; FP16 matrix peak 2306.9 TFLOPS per GPU,
          AMD MI350X datasheet).
    """

    # Format: (name, date_str, tensor_fp16_tflops, price_usd, type)
    # Plot includes V100/A100/H100/B100 NVIDIA flagships and AMD MI100/
    # MI250X/MI300X/MI350X (with their matrix-engine values), plus the
    # pre-tensor cards P100 and MI60 carrying their vector FP16 rate
    # as a fall-back. K80, M60, MI25 are in the table only: K80/M60 have
    # no FP16 hardware at all, and MI25 is excluded as an outlier (its
    # very high $16k list price relative to its 24.6 TFLOPS vector FP16
    # would dominate the fit even though it has no matrix engine).
    data = [
        ("P100",              "2016-04-05",   18.7,   5699, "datacenter"),
        ("V100 SXM2 32GB",    "2018-03-27",  125.0,  11458, "datacenter"),
        ("MI60",              "2018-11-18",   29.5,   7000, "datacenter"),
        ("A100 SXM4 40GB",    "2020-05-14",  312.0,  15000, "datacenter"),
        ("MI100",             "2020-11-16",  184.6,   6400, "datacenter"),
        ("MI250X",            "2021-11-08",  383.0,  14500, "datacenter"),
        ("H100 SXM5 80GB",    "2022-09-20",  989.4,  33600, "datacenter"),
        ("MI300X",            "2023-12-06", 1307.4,  15000, "datacenter"),
        ("B100 SXM",          "2024-11-15", 1750.0,  35000, "datacenter"),
        ("MI350X",            "2025-06-12", 2306.9,  25000, "datacenter"),
    ]

    def to_year(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    consumer_years, consumer_gflops_per_dollar = [], []
    dc_years, dc_gflops_per_dollar = [], []
    consumer_names, dc_names = [], []

    for name, date_str, tfp16_tflops, price, typ in data:
        yr = to_year(date_str)
        gflops_per_dollar = tfp16_tflops * 1000 / price  # GFLOP/s per $
        if typ == "consumer":
            consumer_years.append(yr)
            consumer_gflops_per_dollar.append(gflops_per_dollar)
            consumer_names.append(name)
        else:
            dc_years.append(yr)
            dc_gflops_per_dollar.append(gflops_per_dollar)
            dc_names.append(name)

    consumer_years = np.array(consumer_years)
    consumer_gflops_per_dollar = np.array(consumer_gflops_per_dollar)
    dc_years = np.array(dc_years)
    dc_gflops_per_dollar = np.array(dc_gflops_per_dollar)

    # ── Log-linear fits ──────────────────────────────────────────────
    has_consumer = len(consumer_years) > 0
    if has_consumer:
        c_coeffs = np.polyfit(consumer_years, np.log(consumer_gflops_per_dollar), 1)
        c_doubling_yrs = np.log(2) / c_coeffs[0]
        c_annual_pct = (np.exp(c_coeffs[0]) - 1) * 100

    dc_coeffs = np.polyfit(dc_years, np.log(dc_gflops_per_dollar), 1)
    dc_doubling_yrs = np.log(2) / dc_coeffs[0]
    dc_annual_pct = (np.exp(dc_coeffs[0]) - 1) * 100

    if has_consumer:
        print(f"[Tensor-FP16 Consumer GPUs]   doubling time: {c_doubling_yrs:.2f} yr  |  "
              f"annual improvement: +{c_annual_pct:.1f}%/yr  "
              f"(≈ {100 - 100/(1 + c_annual_pct/100):.1f}%/yr $/TFLOP decline)")
    print(f"[Tensor-FP16 Datacenter GPUs] doubling time: {dc_doubling_yrs:.2f} yr  |  "
          f"annual improvement: +{dc_annual_pct:.1f}%/yr  "
          f"(≈ {100 - 100/(1 + dc_annual_pct/100):.1f}%/yr $/TFLOP decline)")

    years_line = np.linspace(2016, 2026, 200)
    if has_consumer:
        c_fit  = np.exp(np.polyval(c_coeffs,  years_line))
    dc_fit = np.exp(np.polyval(dc_coeffs, years_line))

    # ── Projection 2026-2035 ─────────────────────────────────────────
    # Anchor the projection at the latest datacenter point (now MI350X,
    # mid 2025) and extend it to 2035 using the same log-linear growth
    # rate measured from the historical fit.
    anchor_idx = int(np.argmax(dc_years))
    anchor_year = dc_years[anchor_idx]
    anchor_val  = dc_gflops_per_dollar[anchor_idx]
    growth_rate = np.exp(dc_coeffs[0]) - 1   # annual fractional growth
    years_proj = np.arange(2026, 2036)
    proj_vals  = anchor_val * (1 + growth_rate) ** (years_proj - anchor_year)

    # ── Series for plotting: invert metric if requested ──────────────
    if inverted:
        dc_y      = 1.0 / dc_gflops_per_dollar
        dc_fit_y  = 1.0 / dc_fit
        proj_y    = 1.0 / proj_vals
        rate_pct  = (1 - 1.0 / (1 + growth_rate)) * 100
        sign      = "-"
        ylabel    = "Tensor FP16 $/(GFLOP/s)"
        legend_loc = "upper right"
        legend_anchor = (1.0, 1.0)
        inset_pos = [0.30, 0.45, 0.42, 0.50]
        # Offsets (in points) for the family labels drawn on the main axes.
        # Comment shows year and $/(GFLOP/s) at that family's anchor SKU
        # so the values can be cross-checked against the data table above.
        family_offsets = {
            "V100": (-15, -20),  # 2018.23, $0.0917/(GFLOP/s)  (V100 SXM2 32GB)
            "A100": (-5,   8),   # 2020.37, $0.0481/(GFLOP/s)  (A100 SXM4 40GB)
            "H100": (-8,   8),   # 2022.72, $0.0340/(GFLOP/s)  (H100 SXM5 80GB)
            "B100": (3,    5),   # 2024.87, $0.0200/(GFLOP/s)  (B100 SXM)
        }
        out_name = "epoch_tensor_fp16_price_perf.pdf"
    else:
        dc_y      = dc_gflops_per_dollar
        dc_fit_y  = dc_fit
        proj_y    = proj_vals
        rate_pct  = growth_rate * 100
        sign      = "+"
        ylabel    = "Tensor FP16 (GFLOP/s/$)"
        legend_loc = "lower right"
        legend_anchor = None
        inset_pos = [0.10, 0.45, 0.42, 0.50]
        # Offsets (in points) for the family labels drawn inside the zoom
        # inset. Comment shows year and GFLOP/s/$ at that family's anchor
        # SKU so the values can be cross-checked against the data table.
        family_offsets = {
            "V100": (3,  -2),    # 2018.23, 10.91 GFLOP/s/$  (V100 SXM2 32GB)
            "A100": (-5,  4),    # 2020.37, 20.80 GFLOP/s/$  (A100 SXM4 40GB)
            "H100": (-8,  6),    # 2022.72, 29.45 GFLOP/s/$  (H100 SXM5 80GB)
            "B100": (-20, -15),  # 2024.87, 50.00 GFLOP/s/$  (B100 SXM)
        }
        out_name = "epoch_tensor_fp16_throughput_per_dollar.pdf"

    # ── Plot ─────────────────────────────────────────────────────────
    # Single panel with a linear-scale zoom inset (rectangle box) showing
    # the 2017-2025 historical view, while the main panel runs linearly
    # across 2017-2035 with the fit and projection lines.
    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Identify the four NVIDIA flagship datacenter generations
    # (V100/A100/H100/B100). When there is no inset (inverted=True),
    # highlight them in a darker blue on the main panel. When the
    # inset is present (inverted=False), keep the main panel uniform
    # and let the inset do the highlighting.
    anchor_skus = {"V100 SXM2 32GB", "A100 SXM4 40GB",
                   "H100 SXM5 80GB", "B100 SXM"}
    anchor_mask = np.array([n in anchor_skus for n in dc_names])

    if inverted:
        ax.scatter(dc_years[~anchor_mask], dc_y[~anchor_mask],
                   zorder=5, s=60, marker="^",
                   color="#9ecae1", edgecolors="white", linewidths=1.0,
                   label="Historical", alpha=0.85)
        ax.scatter(dc_years[anchor_mask], dc_y[anchor_mask],
                   zorder=6, s=90, marker="^",
                   color="#08519c", edgecolors="white", linewidths=1.2,
                   alpha=0.95)
    else:
        ax.scatter(dc_years, dc_y, zorder=5, s=60, marker="^",
                   color="#9ecae1", edgecolors="white", linewidths=1.0,
                   label="Historical", alpha=0.85)
    ax.plot(years_line, dc_fit_y, "--", color="#1f77b4", alpha=0.7,
            linewidth=2.0,
            label=f"Fit 2016-2025 ({sign}{rate_pct:.1f}%/yr)")
    ax.plot(years_proj, proj_y, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2,
            label=f"Projection: {sign}{rate_pct:.1f}%/yr")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_xlim(2015.5, 2035.5)
    ax.set_xticks(np.arange(2017, 2036, 3))
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    legend_kwargs = dict(loc=legend_loc, fontsize=14)
    if legend_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_anchor
    ax.legend(**legend_kwargs)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Family annotations / zoom inset ─────────────────────────────
    family_anchor = {
        "V100": "V100 SXM2 32GB",
        "A100": "A100 SXM4 40GB",
        "H100": "H100 SXM5 80GB",
        "B100": "B100 SXM",
    }
    name_to_point = {n: (y, (1.0 / v) if inverted else v) for n, y, v in
                     zip(dc_names, dc_years, dc_gflops_per_dollar)}

    if inverted:
        # No inset: mark the family labels directly on the main axes.
        for fam, sku in family_anchor.items():
            if sku in name_to_point:
                yr, val = name_to_point[sku]
                ax.annotate(fam, (yr, val), textcoords="offset points",
                            xytext=family_offsets[fam], fontsize=14,
                            color="#1f77b4")
    else:
        # Linear-scale 2017-2025 historical zoom inset with family labels.
        ax_zoom = ax.inset_axes(inset_pos)
        ax_zoom.scatter(dc_years[~anchor_mask], dc_y[~anchor_mask],
                        zorder=5, s=40, marker="^", color="#6baed6",
                        edgecolors="white", linewidths=0.8)
        ax_zoom.scatter(dc_years[anchor_mask], dc_y[anchor_mask],
                        zorder=6, s=60, marker="^", color="#08519c",
                        edgecolors="white", linewidths=1.0)
        ax_zoom.plot(years_line, dc_fit_y, "--", color="#1f77b4",
                     alpha=0.55, linewidth=1.4)

        for fam, sku in family_anchor.items():
            if sku in name_to_point:
                yr, val = name_to_point[sku]
                ax_zoom.annotate(fam, (yr, val), textcoords="offset points",
                                 xytext=family_offsets[fam], fontsize=12,
                                 color="#1f77b4")

        ax_zoom.set_xlim(2016.5, 2025.5)
        ax_zoom.set_xticks(np.arange(2017, 2026, 2))
        ax_zoom.set_ylim(0, None)
        ax_zoom.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_zoom.tick_params(axis="both", which="major", labelsize=10)
        ax_zoom.grid(True, alpha=0.3, linestyle=":")
        for spine in ax_zoom.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("0.3")
        ax_zoom.set_facecolor("white")
        ax_zoom.patch.set_alpha(0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, out_name), bbox_inches="tight")
    print(f"Saved {out_name} → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Largest mass production HDD capacity per year
# ====================================================================
def plot_HDD_capacity():
    """Plot the largest mass production HDD capacity per year, 2010 to
    2025, and project to 2035.

    Each yearly point is the highest capacity 3.5 inch HDD that reached
    volume production in that year. Lab demonstrations, sample units,
    and announcements without shipments are not counted. Hyperscaler
    only volume shipments do count, since those move at million unit
    scale per quarter.

    ┌──────┬──────┬───────────────────────────────────┬────────────┐
    │ Year │ TB   │ Drive                             │ Volume date│
    ├──────┼──────┼───────────────────────────────────┼────────────┤
    │ 2010 │   3  │ Hitachi Deskstar 7K3000           │ 2010-12    │
    │ 2011 │   3  │ Hitachi Deskstar 7K3000, no new   │ 2011       │
    │ 2012 │   4  │ Hitachi Deskstar 5K4000           │ 2011-12-12 │
    │ 2013 │   6  │ HGST Ultrastar He6, first helium  │ 2013-11-04 │
    │ 2014 │   8  │ HGST Ultrastar He8                │ 2014-Q4    │
    │ 2015 │  10  │ HGST Ultrastar He10               │ 2015-12-02 │
    │ 2016 │  10  │ He10 / Seagate Enterprise Cap 10TB│ 2016       │
    │ 2017 │  12  │ HGST Ultrastar He12               │ 2017-Q3    │
    │ 2018 │  14  │ WD HGST Ultrastar Hs14, SMR       │ 2018       │
    │ 2019 │  16  │ Seagate Exos X16                  │ 2019-06-04 │
    │ 2020 │  18  │ Seagate Exos X18 / WD HC550       │ 2020-08    │
    │ 2021 │  20  │ WD Gold 20TB / Seagate Exos X20   │ 2021-09    │
    │ 2022 │  22  │ WD Gold and Red Pro 22TB          │ 2022-07-19 │
    │ 2023 │  24  │ Seagate Exos X24                  │ 2023-10-18 │
    │ 2024 │  30  │ Seagate Mozaic 3+ Exos 30TB, HAMR │ 2024-Q1    │
    │ 2025 │  32  │ Seagate Exos M 32TB, HAMR         │ 2024-12-17 │
    └──────┴──────┴───────────────────────────────────┴────────────┘

    Source URLs, all verified to currently load and contain the cited
    capacity and date:
        2010 https://www.storagereview.com/news/hitachi-deskstar-7k3000-deskstar-5k3000-and-xl-external-hard-drives-released
        2012 https://www.businesswire.com/news/home/20111212005377/en/Hitachi-GST-Ships-Two-New-4TB-Deskstar-Based-Solutions
        2013 https://www.prnewswire.com/news-releases/hgst-ships-6tb-ultrastar-he6-helium-filled-drives-for-high-density-massive-scale-out-data-center-environments-230452991.html
        2014 https://www.storagereview.com/review/hgst-ultrastar-helium-he8-8tb-enterprise-hard-drive-review
        2015 https://www.anandtech.com/show/9830/western-digital-expands-hgst-helium-drive-lineup-with-10tb-ultrastar-he10
        2017 https://www.techpowerup.com/232748/hgst-ships-ultrastar-he12-12tb-enterprise-hard-drives
        2018 https://www.anandtech.com/show/11901/western-digital-now-shipping-14tb-hgst-ultrastar-hs14
        2019 https://investors.seagate.com/news/news-details/2019/Seagate-Delivers-Industrys-First-Enterprise-Ready-Exos-16TB-Hard-Drive-And-Raises-The-Bar-With-IronWolf-16TB-For-NAS/default.aspx
        2020 https://blocksandfiles.com/2020/08/17/wd-seagate-18tb-disk-drive-race/
        2021 https://www.servethehome.com/seagate-exos-x20-and-ironwolf-pro-20tb-hdds-shipping/
        2022 https://www.westerndigital.com/company/newsroom/press-releases/2022/2022-07-19-western-digital-extends-hdd-technology-and-areal-density-leadership
        2023 https://investors.seagate.com/news/news-details/2023/Seagates-New-Exos-24TB-Hard-Drives-Deliver-Market-Leading-Capacity-for-Hyperscalers-and-Enterprise-Data-Centers/default.aspx
        2024 https://www.storagenewsletter.com/2024/01/18/seagate-mozaic-3-ww-highest-capacity-hdd-at-30tb-and-shipping-in-1q24/
        2025 https://www.tomshardware.com/pc-components/hdds/seagate-launches-32tb-exos-m-hard-drive-based-on-hamr-technology-mozaic-3-drives-are-the-worlds-first-generally-available-hamr-hdds

    Notes on selection:
    - 2011 and 2016 stayed flat: no new capacity record entered volume
      production those years. The Thailand floods in late 2011 also
      depressed new product introductions through 2012.
    - 2013: HGST Ultrastar He6 was the first hermetically sealed helium
      filled HDD, marking the start of the helium platform era.
    - 2018: the first 14TB drives shipped were SMR, the WD Ultrastar
      Hs14. Conventional CMR 14TB followed later, but the SMR Hs14 had
      already entered volume production.
    - 2024: Seagate's Mozaic 3+ 30TB Exos shipped in volume to hyperscale
      cloud customers in Q1 2024. By mid 2025 over a million Mozaic
      drives had been shipped, qualifying as mass production.
    - 2025: the 32TB Exos M HAMR variant became the first generally
      available HAMR HDD on 2024-12-17 and ramped through 2025. Broad
      retail and channel availability for the 30TB IronWolf Pro arrived
      on 2025-07-17. 36TB samples shipped in 2025 but did not reach
      volume production.

    Projection model anchored at 2025 with 15% annual capacity growth:
        C_HDD(t) = 32 * 1.15^(t - 2025)    [TB]
    Rationale: the 2010 to 2025 fit gives roughly 17% per year. Seagate's
    HAMR Mozaic roadmap targets 50TB by 2028 and 100TB+ by mid 2030s, in
    line with 13 to 15% per year. We adopt 15% as a balanced anchor.
    """

    years_hist = np.array([
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
        2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
    ])
    capacity_tb = np.array([
        3,  3,  4,  6,  8, 10, 10, 12,
        14, 16, 18, 20, 22, 24, 30, 32,
    ], dtype=float)

    # ── projection model ─────────────────────────────────────────────
    anchor_year, anchor_cap, annual_growth = 2025, 32.0, 0.15
    years_proj = np.arange(2025, 2036)
    cap_proj = anchor_cap * (1 + annual_growth) ** (years_proj - anchor_year)

    # ── log linear fit through all 2010-2025 points ──────────────────
    coeffs = np.polyfit(years_hist, np.log(capacity_tb), 1)
    fit_pct = (np.exp(coeffs[0]) - 1) * 100
    years_fit = np.linspace(2010, 2035, 200)
    cap_fit = np.exp(np.polyval(coeffs, years_fit))

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.scatter(years_hist, capacity_tb, zorder=5, s=80, color="#1f77b4",
               edgecolors="white", linewidths=1.0, label="Historical")

    ax.plot(years_fit, cap_fit, "--", color="#1f77b4", alpha=0.55,
            linewidth=1.6, label=f"Exp. fit 2010–2025 (+{fit_pct:.1f}%/yr)")

    ax.plot(years_proj, cap_proj, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label="Projection: +15%/yr growth")

    ax.set_xlabel("Year")
    ax.set_ylabel("Largest HDD \n Capacity  [TB]", fontsize=19)
    ax.set_xlim(2009, 2036)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([10, 50, 150])
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hdd_capacity_trend.pdf"),
                bbox_inches="tight")
    print(f"Saved hdd_capacity_trend.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# HDD price normalized to last historical data point (2025)
# ====================================================================
def plot_HDD_price_normalized():
    """Plot HDD $/TB normalized to the 2025 value, 2010 to 2035.

    Uses the same nominal $/TB historical series as plot_HDD_price().
    Every value, including the 2010 to 2025 historical points and the
    2025 to 2035 projection, is divided by the last historical data
    point (2025 = $20.00 per TB, Seagate IronWolf Pro 30TB), giving a
    dimensionless ratio. By construction the curve passes through 1.0
    at year 2025.
    """

    # ── identical to plot_HDD_price() data ───────────────────────────
    years_hist = np.array([
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
        2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
    ])
    dollar_per_tb = np.array([
        83.33, 43.33, 60.00, 50.00, 50.00, 32.50, 53.50, 39.58,
        40.71, 32.14, 33.83, 34.00, 25.00, 22.68, 27.04, 20.00,
    ])

    anchor_year = years_hist[-1]            # 2025
    anchor_val  = dollar_per_tb[-1]         # 20.00 $/TB
    norm_y = dollar_per_tb / anchor_val

    # ── projection: 7.7% per year decline, anchored at 1.0 in 2025 ───
    annual_decline = 0.077
    years_proj = np.arange(2025, 2036)
    proj_y = (1 - annual_decline) ** (years_proj - anchor_year)

    # ── fit through 2010-2024 in normalized units, excluding 2011 ────
    mask_fit = ~np.isin(years_hist, [2011, 2025])
    coeffs = np.polyfit(years_hist[mask_fit], np.log(norm_y[mask_fit]), 1)
    fit_pct = -coeffs[0] * 100
    years_fit = np.linspace(2010, 2035, 200)
    fit_y = np.exp(np.polyval(coeffs, years_fit))

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.axhline(1.0, color="0.5", linewidth=0.9, linestyle=":", alpha=0.7)

    ax.scatter(years_hist, norm_y, zorder=5, s=80, color="#1f77b4",
               edgecolors="white", linewidths=1.0, label="Historical")
    ax.plot(years_fit, fit_y, "--", color="#1f77b4", alpha=0.55,
            linewidth=1.6,
            label=f"Fit 2010–2025 (-{fit_pct:.1f}%/yr)")
    ax.plot(years_proj, proj_y, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label="Projection: -7.7%/yr")

    ax.set_xlabel("Year")
    ax.set_ylabel("HDD Price \n (Normalized to 2025)", fontsize=19)
    ax.set_xlim(2009, 2036)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([0.02, 0.1, 0.5, 1, 5, 10, 50])
    ax.set_ylim(0.018, 60)
    ax.legend(loc="lower left", fontsize=17)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hdd_price_trend_normalized.pdf"),
                bbox_inches="tight")
    print(f"Saved hdd_price_trend_normalized.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Tape price normalized to last historical data point (LTO-10, 2025)
# ====================================================================
def plot_tape_price_normalized():
    """Plot LTO tape native $/TB normalized to the 2025 value, 2013 to 2035.

    Uses the same native $/TB series as plot_tape_price(). Every value,
    including the LTO-6 through LTO-10 historical points and the 2025 to
    2035 projection, is divided by the last historical data point
    (LTO-10 2025 = $10.41 per TB native), giving a dimensionless ratio.
    By construction the curve passes through 1.0 at year 2025, matching
    the normalization of plot_HDD_price_normalized() for Fig. price-trends.
    """

    # ── identical to plot_tape_price() data ──────────────────────────
    gen_names  = ["LTO-6", "LTO-7", "LTO-8", "LTO-9", "LTO-10"]
    years_hist = np.array([2013, 2015, 2017, 2021, 2025])
    cap_tb     = np.array([2.5, 6.0, 12.0, 18.0, 30.0])
    cart_usd   = np.array([117.32, 186.68, 188.00, 202.50, 312.16])
    dollar_per_tb = cart_usd / cap_tb

    anchor_year = years_hist[-1]            # 2025
    anchor_val  = dollar_per_tb[-1]         # 10.41 $/TB (LTO-10 native)
    norm_y = dollar_per_tb / anchor_val

    # ── log-linear fit over all points, in normalized units ──────────
    coeffs = np.polyfit(years_hist, np.log(norm_y), 1)
    tape_decline = 1.0 - np.exp(coeffs[0])          # annual $/TB decline rate
    fit_pct = tape_decline * 100
    years_fit = np.linspace(2013, 2025, 200)
    fit_y = np.exp(np.polyval(coeffs, years_fit))

    # ── projection: anchored at 1.0 in 2025, apply the fitted decline ─
    years_proj = np.arange(2025, 2036)
    proj_y = (1.0 - tape_decline) ** (years_proj - anchor_year)

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.axhline(1.0, color="0.5", linewidth=0.9, linestyle=":", alpha=0.7)

    ax.scatter(years_hist, norm_y, zorder=5, s=80, color="#1f77b4",
               edgecolors="white", linewidths=1.0, label="LTO generations")
    ax.plot(years_fit, fit_y, "--", color="#1f77b4", alpha=0.55,
            linewidth=1.6,
            label=f"Fit 2013–2025 (-{fit_pct:.1f}%/yr)")
    ax.plot(years_proj, proj_y, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label=f"Projection: -{fit_pct:.0f}%/yr")

    # per-generation labels
    gen_offsets = {
        "LTO-6":  (6, 5),
        "LTO-7":  (6, 5),
        "LTO-8":  (6, 5),
        "LTO-9":  (6, 6),
        "LTO-10": (-8, 10),
    }
    for name, x, y in zip(gen_names, years_hist, norm_y):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=gen_offsets[name], fontsize=16, color="0.25")

    ax.set_xlabel("Year")
    ax.set_ylabel("Tape Price \n (Normalized to 2025)", fontsize=19)
    ax.set_xlim(2011, 2036)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([0.2, 0.5, 1, 2, 5])
    ax.set_ylim(0.2, 6)
    ax.minorticks_off()
    ax.legend(loc="lower left", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "tape_price_trend_normalized.pdf"),
                bbox_inches="tight")
    print(f"Saved tape_price_trend_normalized.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Tensor FP16 $/(GFLOP/s) normalized to last data point (B100 SXM, 2024)
# ====================================================================
def plot_tensor_fp16_price_normalized():
    """Plot datacenter tensor FP16 $/(GFLOP/s) normalized to the last
    data point, 2018 to 2035.

    Uses the same restricted datacenter tensor FP16 series as
    plot_epoch_tensor_fp16_trend(inverted=True): V100 / A100 / H100 /
    B100 NVIDIA flagships and AMD MI100 / MI250X / MI300X / MI350X.
    Every historical value and projection value is divided by the last
    historical data point, AMD MI350X at 2025-06-12, $0.01084 per
    (GFLOP/s), so the curve passes through 1.0 at the MI350X release.
    """

    # ── same 10-point restricted set as plot_epoch_tensor_fp16_trend()
    # P100 and MI60 carry their vector FP16 rate as a fall-back since
    # their architectures have no tensor/matrix engine. MI25 is excluded
    # as an outlier (high list price for a vector-only Vega 10 card).
    data = [
        ("P100",              "2016-04-05",   18.7,   5699),
        ("V100 SXM2 32GB",    "2018-03-27",  125.0,  11458),
        ("MI60",              "2018-11-18",   29.5,   7000),
        ("A100 SXM4 40GB",    "2020-05-14",  312.0,  15000),
        ("MI100",             "2020-11-16",  184.6,   6400),
        ("MI250X",            "2021-11-08",  383.0,  14500),
        ("H100 SXM5 80GB",    "2022-09-20",  989.4,  33600),
        ("MI300X",            "2023-12-06", 1307.4,  15000),
        ("B100 SXM",          "2024-11-15", 1750.0,  35000),
        ("MI350X",            "2025-06-12", 2306.9,  25000),
    ]

    def to_year(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    dc_names  = [r[0] for r in data]
    dc_years  = np.array([to_year(r[1]) for r in data])
    dc_dpg    = np.array([r[3] / (r[2] * 1000) for r in data])  # $/(GFLOP/s)

    # ── normalize to last data point (MI350X, 2025-06-12) ────────────
    anchor_idx  = int(np.argmax(dc_years))
    anchor_year = dc_years[anchor_idx]
    anchor_val  = dc_dpg[anchor_idx]
    norm_y = dc_dpg / anchor_val

    # ── log linear fit through all historical points ────────────────
    coeffs = np.polyfit(dc_years, np.log(norm_y), 1)
    decline_pct = (1 - np.exp(coeffs[0])) * 100
    years_fit = np.linspace(2016, anchor_year, 200)
    fit_y = np.exp(np.polyval(coeffs, years_fit))

    # ── projection from anchor to 2035 at the fitted rate ────────────
    years_proj = np.arange(2026, 2036)
    proj_y = np.exp(coeffs[0] * (years_proj - anchor_year))  # = 1 at anchor

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.axhline(1.0, color="0.5", linewidth=0.9, linestyle=":", alpha=0.7)

    # Highlight the V100/A100/H100/B100 anchors in darker blue.
    anchor_skus = {"V100 SXM2 32GB", "A100 SXM4 40GB",
                   "H100 SXM5 80GB", "B100 SXM"}
    anchor_mask = np.array([n in anchor_skus for n in dc_names])

    ax.scatter(dc_years[~anchor_mask], norm_y[~anchor_mask],
               zorder=6, s=130, marker="^",
               color="#9ecae1", edgecolors="white", linewidths=1.0,
               label="Historical", alpha=0.85)
    ax.scatter(dc_years[anchor_mask], norm_y[anchor_mask],
               zorder=6, s=130, marker="^",
               color="#08519c", edgecolors="white", linewidths=1.0,
               alpha=0.95)

    ax.plot(years_fit, fit_y, "--", color="#1f77b4", alpha=0.7,
            linewidth=2.0,
            label=f"Fit 2016-2025 (-{decline_pct:.1f}%/yr)")
    ax.plot(years_proj, proj_y, "-o", color="#ff7f0e", markersize=8,
            linewidth=2.2, label=f"Projection: -{decline_pct:.1f}%/yr")

    # Family annotations on the main axes (MI350X sits at 1.0 by construction).
    family_anchor = {
        "V100": "V100 SXM2 32GB",
        "A100": "A100 SXM4 40GB",
        "H100": "H100 SXM5 80GB",
        "B100": "B100 SXM",
    }
    family_offsets = {
        "V100": (-15, -20),
        "A100": (-35,  -20),
        "H100": (-8,   8),
        "B100": (3,    5),
    }
    name_to_point = {n: (y, v) for n, y, v in zip(dc_names, dc_years, norm_y)}
    for fam, sku in family_anchor.items():
        if sku in name_to_point:
            yr, val = name_to_point[sku]
            ax.annotate(fam, (yr, val), textcoords="offset points",
                        xytext=family_offsets[fam], fontsize=14,
                        color="#08519c")

    ax.set_xlabel("Year")
    ax.set_ylabel("Tensor FP16 $/(GFLOP/s)\n(Normalized to 2025)", fontsize=19)
    ax.set_xlim(2015.5, 2035.5)
    ax.set_xticks(np.arange(2017, 2036, 3))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([0.02, 0.1, 0.5, 1, 5, 10, 50])
    ax.set_ylim(0.018, 60)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR,
                             "epoch_tensor_fp16_price_perf_normalized.pdf"),
                bbox_inches="tight")
    print(f"Saved epoch_tensor_fp16_price_perf_normalized.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
if __name__ == "__main__":
    plot_HDD_price()
    plot_HDD_price_enterprise()
    plot_HDD_price_normalized()
    plot_HDD_capacity()
    plot_tape_price()
    plot_tape_price_normalized()
    plot_compute_cost()
    plot_epoch_gpu_trend()
    plot_epoch_fp16_trend()
    plot_epoch_fp16_lowbit_trend()
    plot_epoch_tensor_fp16_trend(inverted=True)
    plot_epoch_tensor_fp16_trend(inverted=False)
    plot_tensor_fp16_price_normalized()
    print("Done.")