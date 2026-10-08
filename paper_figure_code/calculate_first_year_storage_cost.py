#!/usr/bin/env python3
"""
calculate_first_year_storage_cost.py

Projects the cost to:
  (1) store 1 PB of data on HDD, from 2025 to 2045
  (2) generate 1 PB worth of images with five diffusion models, 2025 to 2045

Assumptions
-----------
HDD storage
  - Price anchor      : $20 / TB in 2025
  - Annual decline    : 10 %/year
  - Cost model        : raw hardware only (no power, cooling, or opex)

GPU computation
  - Throughput-capacity anchor : $20 / (TFLOP/s) in 2025
                                 (e.g. RTX 5090: ~$2,000 / 100 TFLOP/s FP32)
  - Annual decline    : 30 %/year
  - GPU lifetime      : 3 years  (= 9.46 × 10^7 seconds)
  - GPU utilization   : 40 %  (accounts for idle time and non-ML workloads)
  => Amortised cost per TFLOP of actual computation in year t:
       p_TFLOP(t) = P_GPU(t) / (GPU_LIFETIME_S * GPU_UTILIZATION)
                  ≈ $5.3 × 10^-7 / TFLOP  in 2025

Image parameters
  - Image size        : 3 MB / image  (1024×1024 × 3 bytes, uncompressed 8-bit RGB)
  - 1 PB (SI)         : 10^15 bytes  =>  ~3.33 × 10^8 images per 1 PB

Per-image generation FLOPs (denoising + VAE decode, from Table 1 in the paper):
  SD 1.5:      ~50    TFLOP
  SDXL:        ~180   TFLOP
  PixArt-α:    ~240   TFLOP
  SD 3 Med:    ~950   TFLOP
  FLUX.1-dev:  ~3024  TFLOP  (108 TF/pass × 28 steps × CFG=1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── global font settings  (consistent with plot_cost_trends.py) ──────────────
plt.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  11,
})

# ── output directory ──────────────────────────────────────────────────────────
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── years ─────────────────────────────────────────────────────────────────────
YEARS = np.arange(2025, 2046)


# =============================================================================
# Cost-model parameters
# =============================================================================

# HDD
HDD_PRICE_2025  = 20.0    # $/TB in 2025  (diskprices.com, 2025)
HDD_DECLINE     = 0.10    # 10 %/yr

# GPU  (capacity metric: $/[TFLOP/s sustained FP32 throughput])
GPU_PRICE_2025  = 20.0    # $/[TFLOP/s]  (RTX 5090: $2,000 / 104.8 TFLOP/s ≈ $19)
GPU_DECLINE     = 0.30    # 30 %/yr  (Epoch AI long-run estimate)
GPU_LIFETIME_YR = 3.0     # assumed service life in years
GPU_UTILIZATION = 0.40    # fraction of time doing useful ML work

GPU_LIFETIME_S  = GPU_LIFETIME_YR * 365.25 * 24 * 3600   # ≈ 9.46 × 10^7 s

# Storage: 1 PB in SI units
PB_IN_TB = 1_000          # 1 PB (SI) = 1,000 TB  (1 TB = 10^12 bytes)

# Image/generation parameters
IMAGE_SIZE_MB = 3.0        # MB per image  (1024^2 × 3 bytes, uncompressed 8-bit RGB)
PB_IN_MB      = 1e9        # 1 PB (SI) = 10^15 bytes = 10^9 MB  (1 MB = 10^6 bytes)
N_IMAGES      = PB_IN_MB / IMAGE_SIZE_MB    # ≈ 3.33 × 10^8

# Per-image total TFLOPs (denoising Gen + VAE Dec) from the paper's table
MODELS = {
    "SD 1.5":        50.02,
    "SDXL":         180.08,
    "PixArt-\u03b1":    240.08,   # α rendered via Unicode
    "SD 3 Med":     950.15,
    "FLUX.1-dev":  3024.15,
    "FLUX.2-dev":  7900.15,  # 280 TF/pass × 28 steps × CFG=1 + VAE dec
}

MODEL_COLORS = {
    "SD 1.5":            "#1f77b4",
    "SDXL":              "#ff7f0e",
    "PixArt-\u03b1":         "#2ca02c",
    "SD 3 Med":          "#d62728",
    "FLUX.1-dev":        "#9467bd",
    "FLUX.2-dev":        "#8c564b",
}

MODEL_MARKERS = {
    "SD 1.5":            "o",
    "SDXL":              "s",
    "PixArt-\u03b1":         "^",
    "SD 3 Med":          "D",
    "FLUX.1-dev":        "P",
    "FLUX.2-dev":        "X",
}


# =============================================================================
# Core cost functions
# =============================================================================

def hdd_price_per_tb(year):
    """Return HDD price in $/TB for the given year.

    Model: P_HDD(t) = 20 × 0.90^(t-2025)  [$/TB]
    """
    return HDD_PRICE_2025 * (1.0 - HDD_DECLINE) ** (year - 2025)


def gpu_capacity_price(year):
    """Return GPU throughput-capacity cost in $/[TFLOP/s] for the given year.

    Model: P_GPU(t) = 20 × 0.70^(t-2025)  [$/TFLOP/s]
    """
    return GPU_PRICE_2025 * (1.0 - GPU_DECLINE) ** (year - 2025)


def amortised_cost_per_tflop(year):
    """Return amortised cost per TFLOP of actual computation in the given year.

    Derivation:
        p_TFLOP(t) = P_GPU(t) / (GPU_LIFETIME_S × GPU_UTILIZATION)

    Numerically in 2025:
        p_TFLOP = $20 / (9.46×10^7 s × 0.40) ≈ $5.3 × 10^-7 / TFLOP
    """
    return gpu_capacity_price(year) / (GPU_LIFETIME_S * GPU_UTILIZATION)


def storage_cost_1pb(year):
    """Hardware cost ($) to provision 1 PB of HDD capacity in the given year.

    C_store(t) = 1,000 [TB] × P_HDD(t)  [$]
    """
    return PB_IN_TB * hdd_price_per_tb(year)


def generation_cost_1pb(year, model_name):
    """Cost ($) to generate 1 PB worth of images with the given model.

    C_gen(t, m) = N_images × F_m [TFLOP/img] × p_TFLOP(t) [$/TFLOP]

    where N_images = 10^9 MB / 3 MB ≈ 3.33 × 10^8  for 1 PB (SI).
    """
    tflop_per_image = MODELS[model_name]
    total_tflop     = N_IMAGES * tflop_per_image
    return total_tflop * amortised_cost_per_tflop(year)


# =============================================================================
# Bar-chart helpers and constants
# =============================================================================

BAR_YEARS     = np.array([2025, 2030, 2035, 2040, 2045])
BAR_MODELS    = ["SD 1.5", "FLUX.1-dev", "FLUX.2-dev"]
STORAGE_COLOR = "#aec7e8"          # light blue for the storage bar

GEN_COLORS = {
    "SD 1.5":     "#2ca02c",
    "FLUX.1-dev": "#9467bd",
    "FLUX.2-dev": "#8c564b",
}
GEN_LABELS = {
    "SD 1.5":     "SD 1.5",
    "FLUX.1-dev": "FLUX.1-dev (28 steps)",
    "FLUX.2-dev": "FLUX.2-dev (28 steps)",
}

# Design at PANEL_W inches wide; LaTeX scales to \columnwidth (~3.33 in).
# Scale factor ≈ 0.476  →  18pt titles → ~8.6pt, 13pt ticks → ~6.2pt  (legible).
PANEL_W      = 7.0
PANEL_H      = 3.2
PANEL_TITLE  = 18
PANEL_LABEL  = 15
PANEL_TICK   = 13


def _dollar_log_fmt(ax):
    """Human-readable $ formatter for a log-scale y-axis."""
    def fmt(v, _):
        if v >= 1e9:  return f"${v/1e9:.1f}B"
        if v >= 1e6:  return f"${v/1e6:.1f}M"
        if v >= 1e3:  return f"${v/1e3:.0f}K"
        return f"${v:.0f}"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _draw_stacked_panel(ax, store_arr, gen_arr, model, ylabel=True):
    """Draw one stacked-bar panel (log y-scale, no legend)."""
    x = np.arange(len(BAR_YEARS))
    w = 0.55

    ax.bar(x, store_arr, w,
           color=STORAGE_COLOR, edgecolor="white", linewidth=0.8,
           zorder=3)
    ax.bar(x, gen_arr, w, bottom=store_arr,
           color=GEN_COLORS[model], edgecolor="white", linewidth=0.8,
           zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BAR_YEARS, fontsize=PANEL_TICK, rotation=30, ha="right")
    ax.set_xlabel("Year", fontsize=PANEL_LABEL)
    ax.set_title(model, fontsize=PANEL_TITLE, pad=4)
    ax.tick_params(axis="y", labelsize=PANEL_TICK)
    ax.grid(True, axis="y", alpha=0.3, zorder=0, which="both")
    _dollar_log_fmt(ax)
    if ylabel:
        ax.set_ylabel("Cost [$]", fontsize=PANEL_LABEL)


# =============================================================================
# Plot A – Stacked bars, fixed 1-PB corpus  (no embedded legend)
# =============================================================================

def plot_stacked_fixed():
    """3-panel log-scale stacked bar: fixed 1-PB corpus."""
    store_arr = np.array([storage_cost_1pb(y)           for y in BAR_YEARS])
    gen_dict  = {m: np.array([generation_cost_1pb(y, m) for y in BAR_YEARS])
                 for m in BAR_MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(PANEL_W, PANEL_H))
    for i, (ax, model) in enumerate(zip(axes, BAR_MODELS)):
        _draw_stacked_panel(ax, store_arr, gen_dict[model], model,
                            ylabel=(i == 0))
    fig.tight_layout(pad=0.8, w_pad=1.2)
    out = os.path.join(FIGURE_DIR, "stacked_cost_fixed.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# Plot B – Stacked bars, corpus growing 20 %/year  (no embedded legend)
# =============================================================================

def plot_stacked_growing():
    """3-panel log-scale stacked bar: corpus grows 20%/year."""
    vols      = np.array([1.20 ** (y - 2025) for y in BAR_YEARS])
    store_arr = np.array([storage_cost_1pb(y)           for y in BAR_YEARS]) * vols
    gen_dict  = {m: np.array([generation_cost_1pb(y, m) for y in BAR_YEARS]) * vols
                 for m in BAR_MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(PANEL_W, PANEL_H))
    for i, (ax, model) in enumerate(zip(axes, BAR_MODELS)):
        _draw_stacked_panel(ax, store_arr, gen_dict[model], model,
                            ylabel=(i == 0))
    fig.tight_layout(pad=0.8, w_pad=1.2)
    out = os.path.join(FIGURE_DIR, "stacked_cost_growing.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# Plot C – Standalone legend
# =============================================================================

def plot_legend():
    """Generate a standalone legend strip: stacked_legend.pdf."""
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=STORAGE_COLOR,           label="HDD storage"),
        mpatches.Patch(color=GEN_COLORS["SD 1.5"],     label="SD 1.5"),
        mpatches.Patch(color=GEN_COLORS["FLUX.1-dev"], label="FLUX.1-dev (28 steps)"),
        mpatches.Patch(color=GEN_COLORS["FLUX.2-dev"], label="FLUX.2-dev (28 steps)"),
    ]
    fig, ax = plt.subplots(figsize=(PANEL_W, 0.55))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=4,
              fontsize=PANEL_TICK, framealpha=0.9,
              handlelength=1.5, borderaxespad=0)
    fig.tight_layout(pad=0.1)
    out = os.path.join(FIGURE_DIR, "stacked_legend.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# Summary printout
# =============================================================================

def print_summary():
    check_years = [2025, 2030, 2035, 2040, 2045]

    print("=" * 65)
    print(f"HDD price ($/TB):")
    for y in check_years:
        print(f"  {y}:  ${hdd_price_per_tb(y):>8.2f}")

    print()
    print("Cost to store 1 PB (raw HDD hardware):")
    for y in check_years:
        print(f"  {y}:  ${storage_cost_1pb(y):>10,.0f}")

    print()
    print(f"Amortised GPU cost per TFLOP  "
          f"[lifetime={GPU_LIFETIME_YR} yr, util={GPU_UTILIZATION:.0%}]:")
    for y in check_years:
        print(f"  {y}:  ${amortised_cost_per_tflop(y):.3e} / TFLOP")

    print()
    print("Cost to generate 1 PB of 3 MB images:")
    for model in MODELS:
        print(f"  {model}  ({MODELS[model]:.0f} TFLOP/img):")
        for y in check_years:
            print(f"    {y}: ${generation_cost_1pb(y, model):>12,.0f}")

    print("=" * 65)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print_summary()
    plot_stacked_fixed()
    plot_stacked_growing()
    plot_legend()
    print("\nDone.")

