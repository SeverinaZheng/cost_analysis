#!/usr/bin/env bash
# One-time single-pass scan over the 115 GB civitai merged trace.
# Reads the compressed copy (faster than the raw 115 GB) and emits compact
# cache files consumed by civitai_trace.py.  All object sizes are 1, the
# object key is image_id (field 2), the timestamp is unix epoch seconds.
#
# Outputs (in this directory):
#   freq_hist.csv   count,num_objects        popularity (frequency-of-frequency)
#   req_per_day.csv day,num_requests         request volume per calendar day
#   new_per_day.csv day,num_new_objects      first-seen objects per calendar day
#   summary.csv     key,value                totals (requests, unique, with_model, ts range)
set -euo pipefail

SRC=/scratch/yunjia/cost_analysis/trace_analysis/civitai/merged.csv.zst
OUT=/scratch/yunjia/cost_analysis/paper_figure_code/civitai_cache

zstd -dc -T0 "$SRC" | awk -F, -v OUT="$OUT" '
NR == 1 { next }                       # skip header
{
    id = $2
    if (!(id in c)) {                  # never-seen object -> first-seen day
        newday[int($1 / 86400)]++
    }
    c[id]++
    day = int($1 / 86400)
    reqday[day]++
    if ($3 != "") withmodel++
    if (tot == 0 || $1 < mints) mints = $1
    if (tot == 0 || $1 > maxts) maxts = $1
    tot++
}
END {
    # frequency-of-frequency histogram
    fh = OUT "/freq_hist.csv"
    print "count,num_objects" > fh
    for (id in c) h[c[id]]++
    n = 0
    for (k in h) { print k "," h[k] > fh; n++ }

    rp = OUT "/req_per_day.csv"
    print "day,num_requests" > rp
    for (d in reqday) print d "," reqday[d] > rp

    np = OUT "/new_per_day.csv"
    print "day,num_new_objects" > np
    for (d in newday) print d "," newday[d] > np

    sm = OUT "/summary.csv"
    print "key,value" > sm
    print "total_requests," tot > sm
    print "unique_objects," length(c) > sm
    print "requests_with_model," withmodel > sm
    print "min_ts," mints > sm
    print "max_ts," maxts > sm
}
'
echo "SCAN_DONE"
