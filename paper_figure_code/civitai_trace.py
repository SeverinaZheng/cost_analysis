#!/usr/bin/env python3
"""Figures for Section 5.2 (Civitai image-retrieval replay).

The replay drives a plain LRU simulator over the Civitai image-request trace
(`trace_analysis/civitai/merged.csv`, ~2.06 B requests, object key = image_id,
all object sizes = 1).  This script produces three groups of figures:

  1) plot_popularity()   object popularity, showing the trace is heavily skewed
                         (a small fraction of objects serve most requests).
  2) plot_miss_ratio()   LRU miss ratio vs cache size: a bar chart of the final
                         whole-trace miss ratio per cache size, and a line plot
                         of the per-interval miss ratio over time, one line per
                         cache size.
  3) plot_trace_shape()  other trace characteristics worth citing in 5.2: daily
                         request volume and cumulative catalogue growth.

Popularity and trace-shape figures read compact cache files produced by a
single pass over the 115 GB trace (see civitai_cache/scan_trace.sh); run that
script once before plotting.  The miss-ratio figures read the simulator output
in `libCacheSim_primitive/_build/result/*.cachesim`.

Usage:
    python civitai_trace.py        # regenerates every figure into ./figures
"""

import os
import re
import sys
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# ── shared paper style (required) ───────────────────────────────────
sys.path.insert(0, "/home/yunjia/Desktop/scripts")
from plot_style import apply_style, PALETTE, HATCHES

apply_style()

# ── paths ───────────────────────────────────────────────────────────
HERE        = pathlib.Path(__file__).parent
CACHE_DIR   = HERE / "civitai_cache"
RESULT_DIR  = pathlib.Path("/scratch/yunjia/libCacheSim_primitive/_build/result")
FIGURE_DIR  = HERE / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def _save(fig, name):
    out = FIGURE_DIR / name
    fig.savefig(out, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


def _need(path):
    if not pathlib.Path(path).exists():
        raise SystemExit(
            f"missing cache file {path}\n"
            f"run the one-time scan first:\n"
            f"    bash {CACHE_DIR / 'scan_trace.sh'}"
        )


# ====================================================================
# 1) Object popularity  (zipf-like skew)
# ====================================================================
def plot_popularity():
    """Show that a small fraction of objects serve most of the requests.

    Two views built from the frequency-of-frequency histogram
    (civitai_cache/freq_hist.csv: count,num_objects):

      popularity_zipf.pdf  rank vs request count on log-log axes; a roughly
                           straight, steep line is the signature of a
                           Zipf-like popularity distribution.
      popularity_cdf.pdf   cumulative share of requests served by the most
                           popular fraction of objects (a concentration /
                           Lorenz curve), annotated with the request share of
                           the top 1% and top 10% of objects.
    """
    fh = CACHE_DIR / "freq_hist.csv"
    _need(fh)
    h = pd.read_csv(fh)                              # count, num_objects
    h = h.sort_values("count", ascending=False).reset_index(drop=True)

    counts   = h["count"].to_numpy(dtype=np.float64)
    nobj     = h["num_objects"].to_numpy(dtype=np.float64)
    n_unique = nobj.sum()
    n_req    = (counts * nobj).sum()
    print(f"popularity: {n_unique:,.0f} unique objects, {n_req:,.0f} requests")

    # --- rank-frequency (Zipf) --------------------------------------
    # objects are already in descending popularity order by group; the first
    # rank of each group is 1 + (#objects more popular than it).
    rank_start = 1.0 + np.concatenate(([0.0], np.cumsum(nobj)[:-1]))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(rank_start, counts, color=PALETTE[3], linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Object Rank (Most Popular First)")
    ax.set_ylabel("Number of Requests")
    ax.grid(True, which="both", axis="both", alpha=0.35, linestyle=":")
    _save(fig, "popularity_zipf.pdf")

    # --- cumulative concentration (Lorenz) --------------------------
    cum_obj_frac = np.cumsum(nobj) / n_unique
    cum_req_frac = np.cumsum(counts * nobj) / n_req

    # request share captured by the top 1% / 10% of objects
    def share_at(obj_frac):
        i = np.searchsorted(cum_obj_frac, obj_frac)
        i = min(i, len(cum_req_frac) - 1)
        return cum_req_frac[i]

    top1, top10 = share_at(0.01), share_at(0.10)
    print(f"popularity: top 1% objects -> {top1:.1%} of requests; "
          f"top 10% -> {top10:.1%}")

    fig, ax = plt.subplots(figsize=(3.4, 2.84))
    # prepend origin so the curve starts at (0,0)
    x = np.concatenate(([0.0], cum_obj_frac)) * 100.0
    y = np.concatenate(([0.0], cum_req_frac)) * 100.0
    ax.plot(x, y, color=PALETTE[3], linewidth=2.2)
    ax.plot([0, 100], [0, 100], color="0.6", linewidth=1.2, linestyle="--",
            label="uniform popularity")

    for frac, share, txt in [(1.0, top1, "top 1%"), (10.0, top10, "top 10%")]:
        ax.scatter([frac], [share * 100], color=PALETTE[4], zorder=5, s=40)
        ax.annotate(f"{txt}\n{share:.0%} of req.",
                    xy=(frac, share * 100),
                    xytext=(frac + 12, share * 100 - 12),
                    fontsize=11,
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))

    ax.set_xlabel("Most Popular Objects (%)")
    ax.set_ylabel("Requests Served (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "popularity_cdf.pdf")


# ====================================================================
# 2) LRU miss ratio vs cache size
# ====================================================================
_CACHESIM_RE = re.compile(r"civitai_lru_([0-9.]+)\.cachesim$")


def _load_cachesim():
    """Return {fraction: dataframe} for each civitai_lru_<frac>.cachesim run.

    The superseded `*_before_*` files (3-column format) are ignored.
    """
    runs = {}
    for p in sorted(RESULT_DIR.glob("civitai_lru_*.cachesim")):
        m = _CACHESIM_RE.search(p.name)
        if not m:                                   # skips *_before_* etc.
            continue
        df = pd.read_csv(p)
        runs[float(m.group(1))] = df
    if not runs:
        raise SystemExit(f"no civitai_lru_<frac>.cachesim files in {RESULT_DIR}")
    return dict(sorted(runs.items()))


def plot_miss_ratio():
    """Bar chart of final whole-trace miss ratio and per-interval miss-ratio
    lines, one series per cache size.

      miss_ratio_bar.pdf    final cumulative miss ratio at end of trace, per
                            cache size (fraction of the working set).
      miss_ratio_interval.pdf  per-interval (daily) miss ratio over trace time.

    Cache size is the fraction of distinct objects the cache may hold (see
    lru_sim.cpp).  A run that did not reach the end of the trace is flagged as
    partial so its final bar is not silently compared against full runs.
    """
    runs = _load_cachesim()
    fracs = list(runs.keys())

    # detect partial runs (did not process the full request stream)
    max_req = max(df["total_req"].iloc[-1] for df in runs.values())
    partial = {f: df["total_req"].iloc[-1] < 0.95 * max_req
               for f, df in runs.items()}

    # --- final whole-trace miss ratio: bar chart --------------------
    finals = [runs[f]["total_miss_ratio"].iloc[-1] for f in fracs]
    labels = [f"{f:g}" + ("\n(partial)" if partial[f] else "") for f in fracs]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(fracs))]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    bars = ax.bar(range(len(fracs)), finals, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.6)
    for f, bar, val in zip(fracs, bars, finals):
        if partial[f]:
            bar.set_hatch("//")
        ax.annotate(f"{val:.1%}", xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cache Size (Fraction of Distinct Objects)")
    ax.set_ylabel("Whole-trace Miss Ratio")
    ax.set_ylim(0, max(finals) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig, "miss_ratio_bar.pdf")

    if any(partial.values()):
        bad = [f"{f:g}" for f, p in partial.items() if p]
        print(f"  NOTE: partial run(s) {bad}: did not reach end of trace; "
              f"their final miss ratio is not comparable to full runs.")

    # --- per-interval miss ratio over time: line plot ---------------
    # figsize chosen so that, placed at 0.58\columnwidth against the 0.40-width
    # popularity_cdf panel (Fig. 7a), the two render at equal font size and
    # equal height with x-axes aligned.
    fig, ax = plt.subplots(figsize=(5.58, 2.60))
    for i, f in enumerate(fracs):
        df = runs[f]
        days = df["timestamp"].to_numpy() / 24.0     # trace hours -> days
        color = PALETTE[i % len(PALETTE)]
        lab = f"{f:g}" + (" (partial)" if partial[f] else "")
        ax.plot(days, df["interval_miss_ratio"],
                color=color, linewidth=1.6, label=lab)
        # whole-trace (final cumulative) miss ratio as a horizontal line,
        # labelled with its value just below the line at the left edge
        final = df["total_miss_ratio"].iloc[-1]
        ax.axhline(final, color=color, linewidth=1.4, linestyle="--",
                   alpha=0.9)
        ax.annotate(f"{final:.1%}", xy=(days[0], final),
                    xytext=(2, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=11, color=color,
                    fontweight="bold")
    ax.set_xlabel("Trace Time (Days)")
    ax.set_ylabel("Interval Miss Ratio")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    # proxy entry so the dashed lines are explained in the legend
    ax.plot([], [], color="0.4", linewidth=1.4, linestyle="--",
            label="average")
    # put the "cache size" label inline, on the same row as the sizes, by
    # prepending it as a legend entry with an invisible handle
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    blank = Line2D([], [], linestyle="none", marker="")
    handles = [blank] + handles
    labels = ["cache size:"] + labels
    leg = ax.legend(handles, labels, loc="lower right",
                    bbox_to_anchor=(1.0, 1.01), ncol=len(handles),
                    columnspacing=0.9, handlelength=1.6, handletextpad=0.5,
                    borderaxespad=0.0, frameon=False, fontsize=9)
    ax.grid(True, axis="both", alpha=0.35, linestyle=":")
    _save(fig, "miss_ratio_interval.pdf")


# ====================================================================
# 3) Other trace characteristics for Section 5.2
# ====================================================================
def plot_trace_shape():
    """Daily request volume and cumulative catalogue growth.

      daily_requests.pdf   requests per calendar day over the trace span; shows
                           the sustained load the replay subjects the cache to.
      catalog_growth.pdf   cumulative distinct objects ever requested vs daily
                           newly-seen objects; the ever-growing catalogue is the
                           cost driver Section 5.2 reasons about.
    """
    rp, npd, sm = (CACHE_DIR / "req_per_day.csv",
                   CACHE_DIR / "new_per_day.csv",
                   CACHE_DIR / "summary.csv")
    for p in (rp, npd, sm):
        _need(p)

    req = pd.read_csv(rp).sort_values("day").reset_index(drop=True)
    new = pd.read_csv(npd).sort_values("day").reset_index(drop=True)
    summary = dict(zip(pd.read_csv(sm)["key"], pd.read_csv(sm)["value"]))

    def to_date(day):                                # epoch-day -> datetime
        return datetime.fromtimestamp(day * 86400, tz=timezone.utc)

    req_dates = [to_date(d) for d in req["day"]]
    new_dates = [to_date(d) for d in new["day"]]

    with_model = summary.get("requests_with_model", 0)
    total      = summary.get("total_requests", 1)
    print(f"trace: {total:,.0f} requests, "
          f"{summary.get('unique_objects', 0):,.0f} unique objects, "
          f"{with_model / total:.1%} of requests carry a model_id")

    # --- daily request volume ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(req_dates, req["num_requests"] / 1e6,
            color=PALETTE[3], linewidth=1.6)
    ax.fill_between(req_dates, req["num_requests"] / 1e6,
                    color=PALETTE[0], alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Requests per Day\n(millions)")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _save(fig, "daily_requests.pdf")

    # --- catalogue growth -------------------------------------------
    cum_unique = np.cumsum(new["num_new_objects"].to_numpy())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(new_dates, cum_unique / 1e6,
            color=PALETTE[3], linewidth=2.0,
            label="cumulative distinct objects")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Distinct\nObjects (millions)")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2 = ax.twinx()
    ax2.plot(new_dates, new["num_new_objects"] / 1e6,
             color=PALETTE[4], linewidth=1.3, alpha=0.8,
             label="new objects per day")
    ax2.set_ylabel("New Objects per\nDay (millions)")
    ax2.set_ylim(0, None)
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    fig.autofmt_xdate()
    _save(fig, "catalog_growth.pdf")


# ====================================================================
if __name__ == "__main__":
    print("[1/3] object popularity")
    plot_popularity()
    print("[2/3] LRU miss ratio")
    plot_miss_ratio()
    print("[3/3] trace shape")
    plot_trace_shape()
    print("Done.")
