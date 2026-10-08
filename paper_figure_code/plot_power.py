#!/usr/bin/env python3
"""Plot GPU energy per TFLOP (J/TFLOP) and revisit the flat 1 J/TFLOP assumption.

The cost model of the paper fixes a single GPU energy coefficient
    epsilon_GPU = 1 J/TFLOP
cited to the NVIDIA A100 datasheet (about 312 TFLOP/s dense tensor-FP16 at a
400 W TDP, so 400 / 312 = 1.28 J/TFLOP, rounded to 1). See
sections/3_storage_model.tex and the commented derivation in
sections/5_working_examples.tex.

That constant is an A100-era snapshot. This script recomputes it from the
same datacentre accelerators that populate Figure 2b (the GPU price-performance
panel, epoch_fp16_lowbit_price_perf_normalized.pdf, produced by
plot_epoch_fp16_lowbit_trend() in plot_cost_trends.py). For every GPU in that
figure we take its board TDP and its dense throughput and form

    epsilon_GPU = TDP [W] / throughput [TFLOP/s]   [J/TFLOP]

using two throughput conventions that match Figure 2b exactly:
    - FP16 series:      dense tensor / matrix FP16 rate (blue).
    - Lowest-bit series: the lowest native float precision the tensor engine
      supports (FP8 on Hopper / CDNA3, FP4 on Blackwell / CDNA4) (green).

Energy per TFLOP falls sharply across generations, so the flat 1 J/TFLOP is now
conservative (high). The figure fits a log-linear decline and projects it to
2035, and prints the 2025 anchor values that should replace the 1 J/TFLOP
constant going forward.

GPU set, TDP, dense throughput, and derived J/TFLOP (MI25 dropped as an
outlier, matching Figure 2b). Throughput columns are copied verbatim from
plot_epoch_fp16_lowbit_trend() so the two figures stay consistent.

┌──────────────────────┬────────────┬────────┬───────────┬───────────┬──────────┬──────────┬──────────┐
│ Hardware             │ Date       │ TDP(W) │ FP16 dense│ low-bit   │ Prec.    │ J/TFLOP  │ J/TFLOP  │
│                      │            │        │ (TFLOP/s) │ (TFLOP/s) │ (low)    │ FP16     │ low-bit  │
├──────────────────────┼────────────┼────────┼───────────┼───────────┼──────────┼──────────┼──────────┤
│ P100 SXM2            │ 2016-04-05 │   300  │    18.70  │    18.70  │ FP16     │  16.04   │  16.04   │
│ V100 SXM2 32GB       │ 2018-03-27 │   300  │   125.00  │   125.00  │ FP16     │   2.40   │   2.40   │
│ MI60                 │ 2018-11-18 │   300  │    29.50  │    29.50  │ FP16     │  10.17   │  10.17   │
│ A100 SXM4 40GB       │ 2020-05-14 │   400  │   312.00  │   312.00  │ FP16     │   1.28   │   1.28   │  <- old anchor
│ MI100                │ 2020-11-16 │   300  │   184.60  │   184.60  │ FP16     │   1.63   │   1.63   │
│ MI250X               │ 2021-11-08 │   560  │   383.00  │   383.00  │ FP16     │   1.46   │   1.46   │
│ H100 SXM5 80GB       │ 2022-09-20 │   700  │   989.40  │  1978.90  │ FP8      │   0.71   │   0.35   │
│ MI300X               │ 2023-12-06 │   750  │  1307.40  │  2614.90  │ FP8      │   0.57   │   0.29   │
│ B200 SXM             │ 2024-11-15 │  1000  │  2250.00  │  9000.00  │ FP4      │   0.44   │   0.11   │
│ MI350X (CDNA4)       │ 2025-06-12 │  1000  │  2306.90  │  9216.00  │ FP4      │   0.43   │   0.11   │
└──────────────────────┴────────────┴────────┴───────────┴───────────┴──────────┴──────────┴──────────┘

TDP source URLs, one per GPU, each verified to state the board power below:
    P100 SXM2  300 W   https://www.techpowerup.com/gpu-specs/tesla-p100-sxm2.c2797
    V100 SXM2  300 W   https://www.techpowerup.com/gpu-specs/tesla-v100-sxm2-32-gb.c3185
    MI60       300 W   https://www.techpowerup.com/gpu-specs/radeon-instinct-mi60.c3233
    A100 SXM4  400 W   https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    MI100      300 W   https://www.techpowerup.com/gpu-specs/radeon-instinct-mi100.c3496
    MI250X     560 W   https://www.techpowerup.com/gpu-specs/radeon-instinct-mi250x.c3837
    H100 SXM5  700 W   https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    MI300X     750 W   https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    B200 SXM  1000 W   https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu
    MI350X    1000 W   https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf

Throughput provenance: identical to the Figure 2b table in
plot_epoch_fp16_lowbit_trend() (Epoch AI ml_hardware.csv for FP16, vendor
datasheets for the FP8 / FP4 dense low-bit rates, dense with no 2:4 sparsity
bonus). See that docstring for the per-rate sources.

Notes:
    - SXM/OAM board TDP is used throughout so every point is a full-module
      figure comparable to the A100's 400 W anchor, not a chip-only number.
    - MI350X quoted at the 1000 W air-cooled OAM envelope; the liquid-cooled
      MI355X sibling runs up to 1400 W, which would raise its J/TFLOP by 40%.
    - The A100 FP16 point reproduces the paper's 1.28 J/TFLOP by construction,
      so the figure is anchored to the same reference the old constant used.

Usage:
    python plot_power.py          # saves gpu_energy_per_tflop.pdf
"""

import os
import sys
import pathlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── shared research-paper style (per project rule) ──────────────────
sys.path.insert(0, "/home/yunjia/Desktop/scripts")
try:
    from plot_style import apply_style
    apply_style()
except Exception:
    pass

# ── override with the exact rcParams of plot_cost_trends.py so this panel
#    is visually identical to Figure 2b when placed beside it ──────────
plt.rcParams.update({
    'font.family':         'sans-serif',
    'font.sans-serif':     ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size':           20,
    'axes.titlesize':      22,
    'axes.labelsize':      21,
    'xtick.labelsize':     19,
    'ytick.labelsize':     19,
    'legend.fontsize':     17,
    'lines.linewidth':     2.2,
    'lines.markersize':    9,
    'axes.linewidth':      1.4,
    'xtick.major.width':   1.4,
    'ytick.major.width':   1.4,
    'xtick.major.size':    6,
    'ytick.major.size':    6,
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    'legend.frameon':      True,
    'legend.framealpha':   0.9,
    'legend.edgecolor':    '0.7',
    'figure.dpi':          150,
    'savefig.bbox':        'tight',
    'pdf.fonttype':        42,
    'ps.fonttype':         42,
})

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# The old flat assumption we are revisiting.
OLD_ASSUMPTION_J_PER_TFLOP = 1.0


def plot_gpu_energy_per_tflop():
    """Compute and plot J/TFLOP = TDP / dense-throughput for the Figure 2b GPUs.

    Format: (name, date_str, tdp_w, fp16_tflops, lowbit_tflops, precision)
        fp16_tflops   : dense tensor/matrix FP16 rate (Figure 2b blue series).
        lowbit_tflops : lowest native float precision rate (Figure 2b green
                        series); equals fp16_tflops for FP16-only GPUs.
    """
    # TDP from the URLs in the module docstring; throughput matches Figure 2b.
    data = [
        ("P100 SXM2",       "2016-04-05",  300,   18.700,   18.700, "FP16"),
        ("V100 SXM2 32GB",  "2018-03-27",  300,  125.000,  125.000, "FP16"),
        ("MI60",            "2018-11-18",  300,   29.500,   29.500, "FP16"),
        ("A100 SXM4 40GB",  "2020-05-14",  400,  312.000,  312.000, "FP16"),
        ("MI100",           "2020-11-16",  300,  184.600,  184.600, "FP16"),
        ("MI250X",          "2021-11-08",  560,  383.000,  383.000, "FP16"),
        ("H100 SXM5 80GB",  "2022-09-20",  700,  989.400, 1978.900, "FP8"),
        ("MI300X",          "2023-12-06",  750, 1307.400, 2614.900, "FP8"),
        ("B200 SXM",        "2024-11-15", 1000, 2250.000, 9000.000, "FP4"),
        ("MI350X",          "2025-06-12", 1000, 2306.900, 9216.000, "FP4"),
    ]

    DARK_GREEN  = "#006d2c"
    FP16_BLUE   = "#1f77b4"
    DARK_BROWN  = "#654321"
    FP16_ORANGE = "#ff7f0e"

    def to_year(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    names   = [r[0] for r in data]
    years   = np.array([to_year(r[1]) for r in data])
    tdp     = np.array([r[2] for r in data], dtype=float)
    fp16_t  = np.array([r[3] for r in data], dtype=float)
    lowbit_t = np.array([r[4] for r in data], dtype=float)
    prec    = [r[5] for r in data]

    # ── energy per TFLOP  [J/TFLOP] = TDP[W] / throughput[TFLOP/s] ────
    fp16_energy   = tdp / fp16_t
    lowbit_energy = tdp / lowbit_t
    # FP16-precision GPUs have coincident FP16 and low-bit points.
    shared = np.array([p == "FP16" for p in prec])

    # ── log-linear fits (energy declines with time) + projection ─────
    fp16_coeffs = np.polyfit(years, np.log(fp16_energy), 1)
    lb_coeffs   = np.polyfit(years, np.log(lowbit_energy), 1)
    fp16_decline = (1 - np.exp(fp16_coeffs[0])) * 100
    lb_decline   = (1 - np.exp(lb_coeffs[0])) * 100

    years_fit  = np.linspace(years.min(), years.max(), 200)
    fp16_fit   = np.exp(np.polyval(fp16_coeffs, years_fit))
    lb_fit     = np.exp(np.polyval(lb_coeffs, years_fit))

    anchor_year = 2025.0
    years_proj  = np.arange(2025, 2036)
    fp16_anchor = float(np.exp(np.polyval(fp16_coeffs, anchor_year)))
    lb_anchor   = float(np.exp(np.polyval(lb_coeffs, anchor_year)))
    fp16_proj   = fp16_anchor * np.exp(fp16_coeffs[0] * (years_proj - anchor_year))
    lb_proj     = lb_anchor   * np.exp(lb_coeffs[0]   * (years_proj - anchor_year))

    print(f"[GPU energy per TFLOP]  FP16 fit -{fp16_decline:.1f}%/yr, "
          f"low-bit fit -{lb_decline:.1f}%/yr")
    print(f"  2025 anchor (fitted):  FP16 = {fp16_anchor:.2f} J/TFLOP, "
          f"low-bit = {lb_anchor:.2f} J/TFLOP")
    print(f"  old flat assumption:   {OLD_ASSUMPTION_J_PER_TFLOP:.2f} J/TFLOP "
          f"(A100-era); FP16 anchor is {OLD_ASSUMPTION_J_PER_TFLOP/fp16_anchor:.1f}x lower now")

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    sub = ~shared  # GPUs whose low-bit point drops below FP16

    ANN = {"V100 SXM2 32GB", "A100 SXM4 40GB", "H100 SXM5 80GB", "B200 SXM"}
    ann = np.array([n in ANN for n in names])
    A_HI, A_LO = 1.0, 0.45

    # FP16 (blue triangles), split by annotated / dimmed
    ax.scatter(years[sub & ann], fp16_energy[sub & ann], zorder=5, s=130,
               marker="^", color=FP16_BLUE, edgecolors="white", linewidths=1.0,
               alpha=A_HI, label="FP16")
    ax.scatter(years[sub & ~ann], fp16_energy[sub & ~ann], zorder=5, s=130,
               marker="^", color=FP16_BLUE, edgecolors="white", linewidths=1.0,
               alpha=A_LO, label="_nolegend_")
    # low-bit (dark-green triangles)
    ax.scatter(years[sub & ann], lowbit_energy[sub & ann], zorder=6, s=150,
               marker="^", color=DARK_GREEN, edgecolors="white", linewidths=1.0,
               alpha=A_HI, label="low precision")
    ax.scatter(years[sub & ~ann], lowbit_energy[sub & ~ann], zorder=6, s=150,
               marker="^", color=DARK_GREEN, edgecolors="white", linewidths=1.0,
               alpha=A_LO, label="_nolegend_")
    # FP16-is-lowest points: half-blue / half-green triangles (coincident)
    for m, a in ((shared & ann, A_HI), (shared & ~ann, A_LO)):
        ax.plot(years[m], lowbit_energy[m], linestyle="none", marker="^",
                markersize=12, fillstyle="left",
                markerfacecolor=FP16_BLUE, markerfacecoloralt=DARK_GREEN,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6, alpha=a)

    # dashed historical fits (no legend entry)
    ax.plot(years_fit, fp16_fit, "--", color=FP16_BLUE, alpha=0.8,
            linewidth=1.8, label="_nolegend_")
    ax.plot(years_fit, lb_fit, "--", color=DARK_GREEN, alpha=0.8,
            linewidth=1.8, label="_nolegend_")

    # forward projections (FP16 proj. listed above low-precision proj. in legend)
    ax.plot(years_proj, fp16_proj, "-o", color=FP16_ORANGE, markersize=8,
            linewidth=2.2, label=f"FP16 proj. (-{fp16_decline:.0f}%/yr)")
    ax.plot(years_proj, lb_proj, "-o", color=DARK_BROWN, markersize=8,
            linewidth=2.2, label=f"Low-precision proj. (-{lb_decline:.0f}%/yr)")

    # annotate the A/H/B flagships (and V100), FP16 and low-bit where they split
    idx = {n: i for i, n in enumerate(names)}
    annotations = [
        ("V100 SXM2 32GB", fp16_energy,   "V100",        (-18,  12), FP16_BLUE),
        ("A100 SXM4 40GB", fp16_energy,   "A100",        (6,     8), FP16_BLUE),
        ("H100 SXM5 80GB", fp16_energy,   "H100 (FP16)", (6,    10), FP16_BLUE),
        ("H100 SXM5 80GB", lowbit_energy, "H100 (FP8)",  (-92, -22), DARK_GREEN),
        ("B200 SXM",       fp16_energy,   "B200 (FP16)", (8,     8), FP16_BLUE),
        ("B200 SXM",       lowbit_energy, "B200 (FP4)",  (-92, -22), DARK_GREEN),
    ]
    for name, yarr, text, off, color in annotations:
        i = idx[name]
        ax.annotate(text, (years[i], yarr[i]), textcoords="offset points",
                    xytext=off, fontsize=17, color=color)

    ax.set_xlabel("Year")
    ax.set_ylabel("GPU Energy [J/TFLOP]", fontsize=19)
    ax.set_xlim(2015.5, 2035.5)
    ax.set_xticks(np.arange(2017, 2036, 3))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.legend(loc="lower left", fontsize=15, ncol=1)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "gpu_energy_per_tflop.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved gpu_energy_per_tflop.pdf → {FIGURE_DIR}")
    plt.close(fig)


def plot_hdd_power_density():
    """Compute and plot enterprise HDD power density (W/TB) across the Seagate
    nearline lineage that became Exos, from as early as possible (2010) to 2025.

    Datacentre corpora sit on enterprise nearline drives, not the NAS/consumer
    drives of the Figure 2a price series, so this uses the single-vendor Seagate
    enterprise 3.5" 7200 RPM CMR family for a consistent basis:
        Constellation ES / ES.2 / ES.3  (2010-2013)
        -> Enterprise Capacity 3.5 HDD   (2014-2016)
        -> Exos X12 / X14 / X16 / X18 / X20 / X22 / X24 / Exos M  (2017-2025).

    The paper fixes a system-level HDD power density of
        P_HDD^sys = 1 W/TB
    (sections/3_storage_model.tex), derived from a bare-drive figure of about
    0.4 W/TB rounded up for PUE, controllers, and networking. This function
    recomputes the bare-drive number per generation as

        power density = drive power [W] / capacity [TB]   [W/TB]

    with two drive-power conventions:
        - Idle series      : datasheet "Idle A average" power (blue). The
          relevant figure for a mostly-at-rest powered corpus and the closest
          match to the paper's ~0.4 W/TB bare-drive anchor.
        - Max-operating series : datasheet "Max operating power" (orange), the
          busy-drive worst-case upper bound. Where a sheet has no dedicated Max
          line, its highest operating figure is used (see notes).

    All SATA 6 Gb/s, 7200 RPM variants (SAS draws ~0.5 to 1.5 W more; do not
    mix). Per-drive datasheet power, one source URL per drive:

    ┌──────┬──────────────────────────────────────┬──────┬───────┬───────┬──────────┬──────────┐
    │ Year │ Model                                │ Cap. │ Idle  │ MaxOp │ W/TB idle│ W/TB max │
    ├──────┼──────────────────────────────────────┼──────┼───────┼───────┼──────────┼──────────┤
    │ 2010 │ Constellation ES   ST2000NM0011      │  2TB │ 7.26  │ 10.43 │  3.63    │  5.22    │
    │ 2011 │ Constellation ES.2 ST33000650NS      │  3TB │ 7.70  │ 10.70 │  2.57    │  3.57    │
    │ 2013 │ Constellation ES.3 ST4000NM0033      │  4TB │ 6.73  │ 11.27 │  1.68    │  2.82    │
    │ 2014 │ Ent. Capacity 3.5 v4 ST6000NM0024    │  6TB │ 6.90  │ 11.27 │  1.15    │  1.88    │
    │ 2016 │ Ent. Capacity 3.5 v5 ST8000NM0055    │  8TB │ 7.60  │ 11.00 │  0.95    │  1.38    │
    │ 2016 │ Ent. Capacity 3.5 10TB(He)ST10000NM0016│10TB│ 4.50 │  8.00 │  0.45    │  0.80    │
    │ 2017 │ Exos X12           ST12000NM0007     │ 12TB │ 5.00  │  7.80 │  0.42    │  0.65    │
    │ 2018 │ Exos X14           ST14000NM0018     │ 14TB │ 5.00  │ 10.00 │  0.36    │  0.71    │
    │ 2019 │ Exos X16           ST16000NM001G     │ 16TB │ 5.00  │ 10.00 │  0.31    │  0.63    │
    │ 2020 │ Exos X18           ST18000NM000J     │ 18TB │ 5.30  │  9.40 │  0.29    │  0.52    │
    │ 2022 │ Exos X20           ST20000NM007D     │ 20TB │ 5.40  │  9.40 │  0.27    │  0.47    │
    │ 2023 │ Exos X22           ST22000NM001E     │ 22TB │ 5.74  │  9.40 │  0.26    │  0.43    │
    │ 2024 │ Exos X24           ST24000NM002H     │ 24TB │ 6.30  │  8.90 │  0.26    │  0.37    │
    │ 2025 │ Exos M (Mozaic 3+ HAMR) ST30000NM004K│ 30TB │ 6.90  │  9.50 │  0.23    │  0.32    │
    └──────┴──────────────────────────────────────┴──────┴───────┴───────┴──────────┴──────────┘
    All idle and MaxOp values are PDF-verified verbatim from the datasheets.
    MaxOp = datasheet "Max operating power" (highest operating figure on the
    sheet); for the older Constellation/Ent.Cap sheets that have no dedicated
    "Max" line, it is the sheet's highest operating figure (typical/average
    random read), see notes.

    Datasheet source URLs, one per drive, idle/operating watts read from each:
        2010 https://www.seagate.com/docs/pdf/datasheet/disc/ds_constellation_es.pdf  (PDF-verified: idle 7.26, random-read 10.43)
        2011 https://www.seagate.com/www-content/product-content/constellation-fam/constellation-es/constellation-es-2/en-us/docs/constellation-es2-fips-ds1725-5-1207us.pdf  (PDF-verified: idle 7.7, op 10.7)
        2013 https://www.seagate.com/www-content/product-content/constellation-fam/constellation-es/constellation-es-3/en-us/docs/constellation-es-3-data-sheet-ds1769-1-1210us.pdf  (PDF-verified: idle 6.73, op 11.27)
        2014 https://www.seagate.com/www-content/product-content/enterprise-hdd-fam/enterprise-capacity-3-5-hdd/constellation-es-4/en-us/docs/enterprise-capacity-3-5-hdd-v4-ds1791-3-1403us.pdf
        2016 https://www.seagate.com/www-content/datasheets/pdfs/ent-cap-3-5-hdd-data-sheetDS1882-3-1610US-en_US.pdf  (8TB v5)
        2016 https://www.seagate.com/www-content/datasheets/pdfs/ent-cap-3-5-hdd-10tbDS1863-3-1603US-en_US.pdf  (10TB helium)
        2017 https://www.seagate.com/www-content/datasheets/pdfs/exos-x-12-DS1946-1-1709US-en_US.pdf
        2018 https://www.seagate.com/www-content/datasheets/pdfs/exos-x-14-channel-DS1974-4-1812US-en_US.pdf
        2019 https://www.seagate.com/www-content/datasheets/pdfs/exos-x16-DS2011-1-1904US-en_US.pdf
        2020 https://www.seagate.com/files/www-content/datasheets/pdfs/exos-x18-channel-DS2045-4-2106US-en_US.pdf
        2022 https://www.seagate.com/files/www-content/datasheets/pdfs/exos-x20-channel-DS2080-2111GB-en_EM.pdf
        2023 https://www.seagate.com/content/dam/seagate/en/content-fragments/products/datasheets/exos-x22/exos-x22-DS2080-2209US-en_US.pdf
        2024 https://www.seagate.com/content/dam/seagate/en/content-fragments/products/datasheets/exos-x24/exos-x24-DS2080-2307US-en_US.pdf
        2025 https://www.seagate.com/content/dam/seagate/en/content-fragments/products/datasheets/exos-m-v2/exos-m-v2-DS2045-3-2506-en_US.pdf

    Notes / caveats:
        - All 14 rows (idle and MaxOp) were fetched and read verbatim from the
          Seagate datasheet PDFs at the URLs below.
        - "MaxOp" semantics per sheet:
            * Sheets with an explicit "Max Operating, Random Read/Write 4K/16Q"
              line (10TB He, Exos X14/X16/X18/X20/X22/X24, Exos M): the random-
              read value is reported (the larger of the read/write pair).
              Exos X14 = 10.0 (read) / 6.0 (write); X16 = 10.0 / 6.3;
              X18/X20/X22 = 9.4 / 6.4; X24 = 8.9 / 7.1; Exos M = 9.5.
            * Sheets with no dedicated Max line (older single-row format): the
              only / highest operating figure on the sheet is used instead, so
              these points are conservative relative to a true worst-case max.
              Constellation ES/ES.2/ES.3 and Ent.Cap v4 give "Typical Operating,
              Random Read" (10.43 / 10.7 / 11.27 / 11.27); Ent.Cap 8TB v5 gives
              "Average Operating Power" 11.0.
            * Exos X12 (DS1946, 2017) predates the two-row avg/max format: it
              has a SINGLE "Operating Power" row = 7.8 W for the SATA
              ST12000NM0007 (footnote 4: Random 8K1Q 50/50 W/R @ 40 IO/s, a
              light workload), with NO separate max line (re-verified verbatim).
              7.8 W is therefore the only operating figure for the SATA model,
              not a max, so this point sits low versus the X14+ max points; the
              SAS/dual-port ST12000NM0008 on the same sheet is 9.3 W but is a
              different interface and is not used.
        - The IDLE series is a single consistent metric across all 14 sheets and
          is the one to cite for the coefficient.
        - 2016 has two points: the 8TB air-filled v5 and the first 10TB helium
          drive. Helium is the step change that roughly halves idle W/TB
          (0.95 -> 0.45).
        - X24 row is the single-port ISE model ST24000NM002H (idle 6.3, max 8.9);
          the dual-port SAS block on the same sheet is idle 6.5 / max 9.8.
        - 2025 Exos M is the Mozaic 3+ HAMR drive (ST30000NM004K).
        - All values are SATA 6 Gb/s, 7200 RPM; SAS variants draw ~0.5 to 1.5 W
          more, do not mix interfaces.

    Trend observation: the decline has two regimes split by the introduction of
    helium at 10 TB in 2016, so it is fit as two separate exponentials. The pre-
    2016 segment (air-filled, 2 to 8 TB) falls steeply as capacity outruns a
    roughly constant per-drive power. The post-2016 segment (helium, 10 to 30 TB)
    is nearly flat: helium drives keep adding platters and heads, so per-drive
    idle power (~5 to 7 W) rises roughly in step with capacity and W/TB barely
    moves. That near-flat post-helium regime is why the paper treats HDD power
    density as a fixed 1 W/TB rather than a declining coefficient.
    """

    # Format: (year, name, capacity_tb, idle_w, max_operating_w)
    # Seagate enterprise nearline lineage (SATA, 7200 RPM); idle = datasheet
    # "Idle A average", max_op = datasheet "Max operating power" (highest
    # operating figure on the sheet). All PDF-verified; see docstring.
    data = [
        (2010, "Constellation ES",  2,  7.26, 10.43),
        (2011, "Constellation ES.2",3,  7.70, 10.70),
        (2013, "Constellation ES.3",4,  6.73, 11.27),
        (2014, "Ent. Cap. 6TB",     6,  6.90, 11.27),
        (2016, "Ent. Cap. 8TB",     8,  7.60, 11.00),
        (2016, "Ent. Cap. 10TB He",10,  4.50,  8.00),
        (2017, "Exos X12",         12,  5.00,  7.80),
        (2018, "Exos X14",         14,  5.00, 10.00),
        (2019, "Exos X16",         16,  5.00, 10.00),
        (2020, "Exos X18",         18,  5.30,  9.40),
        (2022, "Exos X20",         20,  5.40,  9.40),
        (2023, "Exos X22",         22,  5.74,  9.40),
        (2024, "Exos X24",         24,  6.30,  8.90),
        (2025, "Exos M 30TB",      30,  6.90,  9.50),
    ]

    years   = np.array([r[0] for r in data], dtype=float)
    cap     = np.array([r[2] for r in data], dtype=float)
    idle_w  = np.array([r[3] for r in data], dtype=float)
    oper_w  = np.array([r[4] for r in data], dtype=float)

    # ── power density  [W/TB] = drive power [W] / capacity [TB] ──────
    idle_wpb = 1000.0 * idle_w / cap
    oper_wpb = 1000.0 * oper_w / cap

    IDLE_BLUE   = "#1f77b4"
    OPER_ORANGE = "#ff7f0e"

    # ── two-segment (piecewise) exponential fits, split at the 2016
    #    helium drop. The pre segment runs 2010 -> 8TB air-filled v5, the post
    #    segment runs 10TB helium -> Exos M. The two 2016 points share a year,
    #    so the split is by index (SPLIT = first helium row) not by year. ─────
    SPLIT = 5                                   # index of the 10TB helium row
    pre  = np.arange(len(years)) < SPLIT        # 2010 -> 8TB air (steep)
    post = ~pre                                 # 10TB He -> Exos M (flat)

    def fit_series(y, mask):
        c = np.polyfit(years[mask], np.log(y[mask]), 1)
        decline = (1 - np.exp(c[0])) * 100
        xs = np.linspace(years[mask].min(), years[mask].max(), 100)
        return c, decline, xs, np.exp(np.polyval(c, xs))

    idle_pre_c,  idle_pre_d,  idle_pre_x,  idle_pre_y  = fit_series(idle_wpb, pre)
    idle_post_c, idle_post_d, idle_post_x, idle_post_y = fit_series(idle_wpb, post)
    act_pre_c,   act_pre_d,   act_pre_x,   act_pre_y   = fit_series(oper_wpb, pre)
    act_post_c,  act_post_d,  act_post_x,  act_post_y  = fit_series(oper_wpb, post)

    print(f"[HDD power density] piecewise fits, split at 2016 helium drop:")
    print(f"  idle  : pre-2016 -{idle_pre_d:.1f}%/yr  |  post-2016 {-idle_post_d:+.1f}%/yr")
    print(f"  active: pre-2016 -{act_pre_d:.1f}%/yr  |  post-2016 {-act_post_d:+.1f}%/yr")
    print(f"  (post-2016 near-flat; paper anchor 0.4 W/TB bare -> 1 W/TB system)")

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.scatter(years, oper_wpb, zorder=5, s=90, marker="^", color=OPER_ORANGE,
               edgecolors="white", linewidths=1.0, label="Active")
    ax.scatter(years, idle_wpb, zorder=6, s=90, color=IDLE_BLUE,
               edgecolors="white", linewidths=1.0, label="Idle")

    # piecewise exponential fits: pre (steep) and post (flat) per series
    ax.plot(act_pre_x,  act_pre_y,  "--", color=OPER_ORANGE, alpha=0.7, linewidth=1.8,
            label=f"Active fit ($-${act_pre_d:.0f}%/yr $\\to$ $-${act_post_d:.0f}%/yr)")
    ax.plot(act_post_x, act_post_y, "--", color=OPER_ORANGE, alpha=0.7, linewidth=1.8)
    ax.plot(idle_pre_x,  idle_pre_y,  "--", color=IDLE_BLUE, alpha=0.7, linewidth=1.8,
            label=f"Idle fit ($-${idle_pre_d:.0f}%/yr $\\to$ $-${idle_post_d:.0f}%/yr)")
    ax.plot(idle_post_x, idle_post_y, "--", color=IDLE_BLUE, alpha=0.7, linewidth=1.8)

    # mark the 2016 helium transition (8TB air -> 10TB He)
    ax.annotate("introducing\nHelium", xy=(2016, idle_wpb[SPLIT]),
                xytext=(2011.6, 300), fontsize=14, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.4))

    ax.set_xlabel("Year")
    ax.set_ylabel("HDD Power Density [W/PB]", fontsize=19)
    ax.set_xlim(2009, 2026)
    ax.set_xticks(np.arange(2010, 2026, 3))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([200, 500, 1000, 2000, 5000])
    ax.set_ylim(180, 6500)
    ax.legend(loc="upper right", fontsize=15, ncol=1)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "hdd_power_density.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved hdd_power_density.pdf → {FIGURE_DIR}")
    plt.close(fig)


if __name__ == "__main__":
    plot_gpu_energy_per_tflop()
    plot_hdd_power_density()
