#!/usr/bin/env python3
"""
Plot growth in the amount of generated data over time.

Public functions:
    plot_adobe_firefly_growth()  – Cumulative Adobe Firefly generated images

All source data are listed in the docstrings / comments of each function
with verifiable URLs.

Usage:
    python plot_data_amount.py          # saves all figures as PDF
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
# Adobe Firefly cumulative generated images
# ====================================================================
def plot_adobe_firefly_growth():
    """Plot cumulative Adobe Firefly generated images over time.

    Source: adobe.csv (date, count_numeric_min). The y axis is the
    "count_numeric_min" column, the lower bound parsed from each
    Adobe public statement. The x axis spans the months covered by
    the data points.
    """
    import csv
    import matplotlib.dates as mdates

    csv_path = os.path.join(os.path.dirname(__file__), "adobe.csv")
    dates, counts = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            counts.append(float(row["count_numeric_min"]))

    order = np.argsort(dates)
    dates  = [dates[i]  for i in order]
    counts = np.array([counts[i] for i in order])

    counts_billion = counts / 1e9

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(dates, counts_billion, "-o", color="#1f77b4", markersize=8,
            linewidth=2.2, markeredgecolor="white", markeredgewidth=1.0,
            label="Cumulative images generated")

    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Amount\n (billions)", fontsize=19)
    ax.set_ylim(0, None)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 8]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "adobe_firefly_growth.pdf"),
                bbox_inches="tight")
    print(f"Saved adobe_firefly_growth.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# Cumulative AI-generated outputs across modalities
# ====================================================================
def plot_modality_growth():
    """Plot cumulative generated-content growth over time for two modalities,
    in billions of artifacts on a log y axis, using the same blue/orange
    marker style as the HDD and GPU panels of Figure 1.

    ── Images: Adobe Firefly cumulative generated images (adobe.csv) ──
      2023-05: 0.07 B   2023-10: 3.0 B   2024-03: 6.5 B   2024-04: 7.0 B
      2024-10: 13.0 B   2025-02: 18.0 B  2025-03: 20.0 B  2025-04: 22.0 B
      Source: Adobe newsroom / blog announcements, see adobe.csv.

    ── Molecules: Enamine REAL Space make-on-demand library ───────────
      Algorithmically generated, synthesizable small-molecule space whose
      enumerated size has grown explosively:
      2020-11: 15.5 B   2022-08: 31 B   2024-02: 48.1 B   2025-09: 83 B
      Sources:
        https://enamine.net/compound-collections/real-compounds/real-space-navigator
        https://www.biosolveit.de/2024/04/15/behind-the-scenes-of-enamines-real-space/
        https://www.biosolveit.de/2025/09/23/enamines-real-space-september-2025-update-now-83-billion/

    Each modality is drawn on its own linear y axis, images on the left
    (blue) and molecules on the right (orange), so both growth curves are
    legible despite their different absolute counts.
    """
    import matplotlib.dates as mdates

    series = [
        {
            "label": "Images (Adobe Firefly)",
            "color": "#1f77b4",      # same blue as Fig. 1(b)/(c)
            "marker": "o",
            "dates":  ["2023-05-10", "2023-10-11", "2024-03-26", "2024-04-23",
                       "2024-10-14", "2025-02-12", "2025-03-18", "2025-04-24"],
            "counts_billion": [0.07, 3.0, 6.5, 7.0, 13.0, 18.0, 20.0, 22.0],
        },
        {
            "label": "Molecules (Enamine REAL)",
            "color": "#ff7f0e",      # same orange as the Fig. 1 projections
            "marker": "s",
            "dates":  ["2020-11-01", "2022-08-01", "2024-02-01", "2025-09-01"],
            "counts_billion": [15.5, 31.0, 48.1, 83.0],
        },
    ]

    img = series[0]

    def to_dates(s):
        return [datetime.strptime(d, "%Y-%m-%d") for d in s["dates"]]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))   # match Fig. 2b/c/d normalized panels

    ax1.plot(to_dates(img), img["counts_billion"], "-",
             marker=img["marker"], color=img["color"], markersize=13,
             linewidth=3.2, markeredgecolor="white", markeredgewidth=1.6,
             label=img["label"])

    ax1.set_xlabel("Year", fontsize=19)
    ax1.set_ylabel("Firefly Images (billion)", fontsize=19,
                   color=img["color"])
    ax1.tick_params(axis="y", labelcolor=img["color"], labelsize=18)
    ax1.tick_params(axis="x", labelsize=18)
    ax1.set_ylim(0, 24)
    ax1.set_yticks([0, 8, 16, 24])

    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(img["color"])
    ax1.spines["left"].set_linewidth(1.6)
    ax1.spines["bottom"].set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "modality_growth.pdf"),
                bbox_inches="tight")
    print(f"Saved modality_growth.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
# AI-image demand, request side: image-generation requests per day
# ====================================================================
def plot_aigc_access():
    """Plot the request-side demand for AI images: the aggregate rate of image
    generation requests served per day across all platforms, rather than the
    cumulative object count of Figure fig:imggen. This flow is what the serving
    infrastructure must sustain, and it is rising steeply.

    AI image generation requests per day (millions), all platforms:
      2022-10: 5    take-off after the mid-2022 launches of DALL-E 2,
                    Midjourney, Stable Diffusion (launch-era estimate)
      2023-08: 34   Everypixel: >15 billion images since 2022, ~34 M/day
                    (https://journal.everypixel.com/ai-image-statistics)  [anchor]
      2025-09: 60   surge after Google's Nano Banana launch (Aug 2025), which
                    alone ran ~36 M/day in its first weeks (estimate)
      2026-01: 80   aggregate 2026 estimate across platforms
                    (https://imagera.ai/blog/ai-image-generation-statistics-2026) [anchor]
    Anchored on the two Everypixel/2026 figures; 2022 and 2025 bracket the
    take-off and the Nano-Banana surge.
    """
    import matplotlib.dates as mdates

    dates = ["2022-10-01", "2023-08-01", "2025-09-01", "2026-01-01"]
    perday_m = [5, 34, 60, 80]
    d = [datetime.strptime(x, "%Y-%m-%d") for x in dates]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(d, perday_m, "-", color="#1f77b4", linewidth=2.2, marker="o",
            markersize=8, markeredgecolor="white", markeredgewidth=1.0)

    ax.set_xlabel("Year")
    ax.set_ylabel("AI Image Requests\nper Day (millions)", fontsize=18)
    ax.set_ylim(0, 90)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "aigc_access.pdf"),
                bbox_inches="tight")
    print(f"Saved aigc_access.pdf → {FIGURE_DIR}")
    plt.close(fig)


# ====================================================================
if __name__ == "__main__":
    plot_adobe_firefly_growth()
    plot_modality_growth()
    plot_aigc_access()
    print("Done.")