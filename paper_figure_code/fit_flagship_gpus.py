#!/usr/bin/env python3
"""Exponential fit of $/(GFLOP/s) for the four NVIDIA flagship datacenter
generations: V100 (SXM2 32GB), A100 (SXM4 40GB), H100 (SXM5 80GB), B100 (SXM).

Compares the resulting annual decline against the broader fit reported in
plot_epoch_tensor_fp16_trend(), which uses all datacenter SKUs and gives
roughly 14% per year.
"""

from datetime import datetime
import numpy as np


# (name, release_date, dense Tensor-FP16 TFLOPS, USD price at release)
ROWS = [
    ("V100 SXM2 32GB", "2018-03-27",  125.0,  11458),
    ("A100 SXM4 40GB", "2020-05-14",  312.0,  15000),
    ("H100 SXM5 80GB", "2022-09-20",  989.4,  33600),
    ("B100 SXM",       "2024-11-15", 1750.0,  35000),
]


def to_year(d):
    dt = datetime.strptime(d, "%Y-%m-%d")
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def main():
    years = np.array([to_year(r[1]) for r in ROWS])
    dollar_per_gflops = np.array([r[3] / (r[2] * 1000) for r in ROWS])
    gflops_per_dollar = 1.0 / dollar_per_gflops

    # log linear fit: log(y) = a + b * t
    coeffs = np.polyfit(years, np.log(dollar_per_gflops), 1)
    slope, intercept = coeffs

    annual_factor = np.exp(slope)              # per year multiplier of $/(GFLOP/s)
    annual_decline_pct = (1 - annual_factor) * 100
    halving_years = -np.log(2) / slope         # years for $/(GFLOP/s) to halve

    # Same fit but on GFLOP/s per $ for reference (sign flipped).
    growth_pct = (1 / annual_factor - 1) * 100
    doubling_years = np.log(2) / -slope

    print("Flagship NVIDIA datacenter GPUs (V100, A100, H100, B100):")
    print(f"{'Hardware':<18s} {'year':>8s}  {'$/(GFLOP/s)':>12s}  {'GFLOP/s/$':>10s}")
    for (name, _, _, _), yr, dpg, gpd in zip(ROWS, years, dollar_per_gflops, gflops_per_dollar):
        print(f"{name:<18s} {yr:>8.2f}  {dpg:>12.5f}  {gpd:>10.2f}")

    print()
    print(f"Log linear fit slope (log $/(GFLOP/s) per year): {slope:.4f}")
    print(f"Annual change in $/(GFLOP/s):  x{annual_factor:.4f} per year"
          f"  → {annual_decline_pct:.2f}%/yr decline")
    print(f"Annual change in GFLOP/s per $: +{growth_pct:.2f}%/yr growth")
    print(f"Halving time of $/(GFLOP/s):   {halving_years:.2f} years")
    print(f"Doubling time of GFLOP/s/$:    {doubling_years:.2f} years")
    print()
    print("Reference, full datacenter fit in plot_epoch_tensor_fp16_trend():")
    print("  ~14%/yr decline in $/(GFLOP/s).")


if __name__ == "__main__":
    main()
