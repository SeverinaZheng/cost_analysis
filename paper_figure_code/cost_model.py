#!/usr/bin/env python3
"""Shared break-even cost model for the two-tier (cache + IR cold store)
storage-vs-regeneration analysis in Section 2 of the paper.

The model follows the equations in
``ComputeCostTradeoff/sections/future_analysis.tex``:
HDD provisioning with replacement, a Zipfian hot cache, an
intermediate-representation (IR) cold store covering all objects, plus
operational electricity and full lifecycle carbon for both the full-HDD
baseline and the two-tier system.

The central output is the break-even per-object generation cost
``c*`` [TFLOP]: objects whose true generation cost lies below ``c*`` are
cheaper to keep as a small hot cache plus IR cold store and regenerate on
miss, than to persist every object on HDD.  ``c*`` is obtained by setting
the cumulative two-tier cost equal to the cumulative full-HDD cost at the
horizon and solving the (linear in c) balance.

Only numpy is required; the generalized-harmonic / Zipf cache-fraction
computation uses an exact partial sum plus an integral tail so it stays
accurate for the astronomically large object counts in this study.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Exogenous, empirically grounded constants (Table "facts" in the paper)
# ---------------------------------------------------------------------------
YEAR0 = 2025                 # anchor year
HORIZON = 2045               # end of analysis horizon

P_HDD0 = 20.0                # $/TB  HDD price anchor (2025)
HDD_DECLINE = 0.90           # 10%/yr decline
P_GPU0 = 18.0                # $/(TFLOP/s) GPU price anchor (2025), B200 FP16
GPU_DECLINE = 0.70           # 30%/yr decline

L_GPU = 3.0                  # GPU lifetime [yr]
U_GPU = 0.40                 # GPU utilisation
DELTA_YR = 3.156e7           # seconds per year
LUD = L_GPU * U_GPU * DELTA_YR   # = 3.78e7 s

L_HDD = 5                    # HDD service life [yr]

ALPHA = 315.0                # $/PB/yr   HDD electricity coefficient (2025 anchor)
                             #   = 0.45 W/TB (0.32 bare Exos M x 1.4 PUE) * 8760 * 0.08
BETA = 9.78e-9               # $/TFLOP   GPU electricity coefficient (2025 anchor)
                             #   = 0.44 J/TFLOP (B200) * 0.08 / 3.6e6
# Operational electricity per unit declines at the SAME rate as the matching
# capital price: the areal-density / throughput growth that cuts $/TB and
# $/TFLOP also cuts W/TB and J/TFLOP (see Fig. power-trends, fitted at ~9%/yr
# for HDD power density and ~32%/yr for GPU energy per TFLOP). The electricity
# PRICE E_RATE is held at its 2025 value throughout; only the device energy
# intensity declines.
ALPHA_DECLINE = HDD_DECLINE  # 10%/yr, matching the HDD capital-price decline
BETA_DECLINE  = GPU_DECLINE  # 30%/yr, matching the GPU capital-price decline

GAMMA_E = 0.30               # kg CO2/kWh grid carbon intensity
E_RATE = 0.08                # $/kWh electricity rate
P_C = 50.0                   # $/tonne CO2 carbon price

PHI_HDD = 2628.0             # kg CO2/PB/yr HDD operational carbon
PHI_GPU = 8.33e-8            # kg CO2/TFLOP GPU operational carbon
CAP_PHI_HDD = 5000.0         # kg CO2/PB     HDD embodied carbon
PSI_GPU = 1.32e-8            # kg CO2/TFLOP  GPU embodied carbon

HIT_RATIO = 0.90             # target cache hit ratio
REQ_PER_OBJ = 10.0           # annual request rate = 10x cumulative object count


# ---------------------------------------------------------------------------
# Price trajectories
# ---------------------------------------------------------------------------
def p_hdd(year):
    """HDD price [$/TB] in a given year."""
    return P_HDD0 * HDD_DECLINE ** (year - YEAR0)


def p_tflop(year):
    """Amortised GPU compute price [$/TFLOP] in a given year."""
    return (P_GPU0 * GPU_DECLINE ** (year - YEAR0)) / LUD


def alpha(year):
    """HDD electricity coefficient [$/PB/yr] in a given year.

    Declines from the 2025 anchor at the same 10%/yr as the HDD capital price,
    because rising areal density spreads a roughly constant per-drive wattage
    over more terabytes. The live corpus in year ``year`` is billed at that
    year's coefficient, since dead drives are replaced by more efficient ones.
    """
    return ALPHA * ALPHA_DECLINE ** (year - YEAR0)


def beta(year):
    """GPU electricity coefficient [$/TFLOP] in a given year.

    Declines from the 2025 anchor at the same 30%/yr as the GPU capital price,
    tracking the fall in energy-per-TFLOP as throughput grows.
    """
    return BETA * BETA_DECLINE ** (year - YEAR0)


# ---------------------------------------------------------------------------
# Tape (LTO) price trajectory and cumulative cost  (Section 3.3)
# ---------------------------------------------------------------------------
# Tape native $/TB is fit from launch-era single-cartridge RETAIL prices across
# LTO-6 to LTO-10, read from dated Wayback Machine snapshots of tapeandmedia.com
# product pages captured near each generation's launch (one retailer, consistent
# basis; prices divided by NATIVE capacity, not the 2.5:1 compressed headline).
#
#   Gen     Year  Native   Cartridge $   $/TB    Wayback snapshot (price source)
#   LTO-6   2013   2.5 TB     117.32      46.93   http://web.archive.org/web/20130111023406/http://www.tapeandmedia.com:80/hp-lto-6-tape-ultrium-tapes.asp
#   LTO-7   2015   6.0 TB     186.68      31.11   http://web.archive.org/web/20151213020546/http://www.tapeandmedia.com:80/hp-lto-7-tape-ultrium-tapes-c7977a.asp
#   LTO-8   2017  12.0 TB     188.00      15.67   http://web.archive.org/web/20171226234539/https://www.tapeandmedia.com/lto-8-tape-media-tapes.asp
#   LTO-9   2021  18.0 TB     202.50      11.25   http://web.archive.org/web/20210924065625/https://tapeandmedia.com/hpe-lto-9-tape-with-barium-ferrite-bafe-q2079a/
#   LTO-10  2025  30.0 TB     312.16      10.41   http://web.archive.org/web/20250621164207/https://tapeandmedia.com/hpe-lto-10-ultrium-tape-30tb-75tb-compressed-rw-data-cartridge-q2080a/
#
# A log-linear fit ln($/TB) = a + b*year over these five points gives an annual
# decline of 1 - exp(b) ~ 11.8%/yr (R^2 ~ 0.84 in log space), close to HDD's
# ~10%/yr rather than the flat trend previously assumed. The decline is steep
# over LTO-6 to LTO-8 and flattens toward ~$10/TB at LTO-9/LTO-10. We anchor at
# the LTO-10 native price of ~$10/TB in 2025 and, to stay on the same basis as
# HDD, adopt a rounded 10%/yr decline rather than the raw 11.8% fit.
_TAPE_FIT_YEARS = np.array([2013, 2015, 2017, 2021, 2025])
_TAPE_FIT_PTB   = np.array([117.32, 186.68, 188.00, 202.50, 312.16]) \
                  / np.array([2.5, 6.0, 12.0, 18.0, 30.0])
_TAPE_FIT_B, _TAPE_FIT_A = np.polyfit(_TAPE_FIT_YEARS, np.log(_TAPE_FIT_PTB), 1)
_TAPE_FIT_DECLINE = float(1.0 - np.exp(_TAPE_FIT_B))   # ~0.118 (11.8%/yr), reference

P_TAPE0 = 10.0               # $/TB  LTO-10 native anchor (2025): 30 TB at ~$300/cartridge
TAPE_DECLINE = 0.10          # 10%/yr, rounded from the ~11.8% fit to match HDD
L_TAPE = 30                  # tape media life [yr] >> horizon, so NO replacement


def p_tape(year):
    """Tape (LTO) price [$/TB] in a given year.

    Anchored at the LTO-10 native price of $10/TB in 2025 and declined at
    10%/yr, a rounded form of the ~11.8% log-linear fit over LTO-6..LTO-10
    launch-era retail prices, matching the HDD decline basis.
    """
    return P_TAPE0 * (1.0 - TAPE_DECLINE) ** (year - YEAR0)


def tape_storage_cost(g, horizon=HORIZON):
    """Cumulative tape *media* cost [$] from YEAR0 to ``horizon`` for a corpus
    growing at rate ``g`` (``g=0`` is the steady 1 PB/yr baseline).

    This is the cartridge capital only. The media never needs replacement
    within the horizon (media life L_TAPE ~30 yr), so each petabyte is bought
    exactly once at that year's flat tape price. Drive, library, and power
    costs are added separately in ``tape_storage_cost_full``.
    """
    years = np.arange(YEAR0, horizon + 1)
    age = years - YEAR0
    new_data = (1.0 + g) ** age           # PB ingested each year, bought once
    ptape = p_tape(years)                 # $/TB (flat)
    return float(np.sum(1000.0 * new_data * ptape))   # 1000 TB per PB


# Tape infrastructure beyond the media itself (Section 3.3).  These are the
# costs HDD has no separate analogue for, because an HDD bundles the read/write
# mechanism and interface into each unit.
P_DRIVE = 10_000.0           # $/drive  LTO-10 tape drive (over $10k)
DRIVE_RATE_MBPS = 400.0      # MB/s     LTO-10 native transfer rate
DRIVE_DUTY = 0.50            # write-side duty cycle (ingest + verify)
# PB/yr a single drive can write: 400 MB/s * 3.156e7 s/yr * duty, /1e9 MB/PB
DRIVE_TPUT_PB_YR = DRIVE_RATE_MBPS * DELTA_YR * DRIVE_DUTY / 1.0e9  # ~6.3 PB/yr
L_DRIVE = 5                  # drive service life [yr], re-bought
N_DRIVE_MIN = 2              # minimum drives provisioned (redundancy)

TAPE_CART_TB = 30.0          # TB native per LTO-10 cartridge (30 TB at launch)
C_SLOT = 100.0               # $/cartridge slot  library frame + robotics
                             # (enterprise estimate; durable, one-time)

# Tape operational power: idle cartridges draw zero, only mounted drives and
# the library controller/robotics consume power. We model tape electricity as a
# fixed 10% of the HDD coefficient at every year, so the tape coefficient shares
# the HDD 10%/yr decline and stays a constant 90% below HDD.
TAPE_ELEC_FRAC = 0.10        # tape $/PB/yr is always 10% of the HDD coefficient
ALPHA_TAPE0 = TAPE_ELEC_FRAC * ALPHA   # = 31.5 $/PB/yr, tape electricity anchor (2025)


def alpha_tape(year):
    """Tape electricity coefficient [$/PB/yr] in a given year.

    Held at a fixed 10% of the HDD coefficient alpha(year), so it is anchored at
    31.5 $/PB/yr in 2025 and declines at the same 10%/yr as HDD.
    """
    return TAPE_ELEC_FRAC * alpha(year)


def tape_storage_cost_full(g, horizon=HORIZON):
    """Full tape storage cost [$] over [YEAR0, horizon] for a corpus growing at
    rate ``g``, returned as a breakdown dict with keys ``media``, ``drives``,
    ``library``, ``power``, and ``total``.

    Beyond the cartridge ``media`` capital, this adds the components HDD does
    not carry separately:
      * ``drives``  : LTO-10 drives sized to the annual ingest throughput
                      (with a minimum for redundancy), re-bought every L_DRIVE
                      years as they wear out.
      * ``library`` : frame slots plus robotics, one per cartridge of live
                      capacity, provisioned as the corpus grows (durable, no
                      replacement within the horizon).
      * ``power``   : mounted-drive and controller electricity at ALPHA_TAPE,
                      far below HDD because idle cartridges draw nothing.
    """
    years = np.arange(YEAR0, horizon + 1)
    age = years - YEAR0
    new_data = (1.0 + g) ** age                       # PB ingested each year

    # live (powered/housed) volume in PB
    if abs(g) < 1e-12:
        v_full = (age + 1).astype(np.float64)
    else:
        v_full = ((1.0 + g) ** (age + 1) - 1.0) / g

    # --- media (cartridges, bought once each) ---
    media = tape_storage_cost(g, horizon)

    # --- drives: size to annual ingest, re-buy at end of life ---
    n_req = np.maximum(N_DRIVE_MIN, np.ceil(new_data / DRIVE_TPUT_PB_YR))
    dN = np.diff(np.concatenate([[0.0], n_req]))      # net new drives needed
    dN = np.maximum(dN, 0.0)
    A_drive = _replacement_series(dN, L_DRIVE)        # add end-of-life re-buys
    drives = float(np.sum(A_drive) * P_DRIVE)

    # --- library slots + robotics: one slot per cartridge of live capacity ---
    slots_peak = np.ceil(v_full[-1] * 1000.0 / TAPE_CART_TB)
    library = float(slots_peak * C_SLOT)

    # --- power: mounted drives + controller, idle cartridges draw zero;
    #     tape electricity is a fixed 10% of the HDD coefficient each year ---
    power = float(np.sum(alpha_tape(years) * v_full))

    total = media + drives + library + power
    return {"media": media, "drives": drives, "library": library,
            "power": power, "total": total}


# ---------------------------------------------------------------------------
# Zipf hot-cache fraction
# ---------------------------------------------------------------------------
_CUTOFF = 20_000             # exact terms before switching to integral tail


def _harmonic_factory(s):
    """Return H(n, s) = sum_{i=1}^{n} i^{-s} as a fast closure.

    The exact prefix sum over the first ``_CUTOFF`` terms is computed once;
    larger n use that prefix plus an integral tail, accurate for any s > 0 and
    for the astronomically large object counts in this study.
    """
    i = np.arange(1, _CUTOFF + 1, dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(i ** (-s))])   # prefix[k]=H(k,s)
    s1 = abs(s - 1.0) < 1e-9

    def H(n):
        n = float(n)
        if n <= _CUTOFF:
            return prefix[int(round(n))]
        lo, hi = _CUTOFF + 0.5, n + 0.5
        tail = np.log(hi / lo) if s1 else (hi ** (1 - s) - lo ** (1 - s)) / (1 - s)
        return prefix[_CUTOFF] + tail

    return H


def cache_fraction(n_objects, s=1.0, hit=HIT_RATIO):
    """Fraction M/N of a Zipf(s) catalogue of N objects that must be cached
    to serve ``hit`` of all requests.

    For s = 1 this reproduces the paper's closed form M/N = 0.944 N^{-0.10}
    at hit = 0.9; for general s it inverts H(M,s)/H(N,s) = hit by bisection.
    """
    n = float(n_objects)
    if n < 1.5:
        return 1.0
    H = _harmonic_factory(s)
    target = hit * H(n)
    lo, hi = 1.0, n
    for _ in range(100):
        mid = np.sqrt(lo * hi)          # geometric midpoint (M spans orders)
        if H(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-6:
            break
    return min(1.0, 0.5 * (lo + hi) / n)


# ---------------------------------------------------------------------------
# Core break-even computation
# ---------------------------------------------------------------------------
def _replacement_series(new_data, l_hdd=L_HDD):
    """Annual provisioning with end-of-life replacement:
    A(y) = new_data(y) + A(y - l_hdd)."""
    a = np.zeros_like(new_data)
    for j in range(len(new_data)):
        a[j] = new_data[j] + (a[j - l_hdd] if j - l_hdd >= 0 else 0.0)
    return a


def breakeven_c(g, S, k1, k2, s=1.0,
                horizon=HORIZON, include_power=True, include_carbon=False):
    """Break-even per-object generation cost c* [TFLOP].

    This matches the two-tier break-even of Equation eq:cachethreshirpow:
    capital (HDD with replacement) plus operational electricity on both sides,
    with the numerator subtracting the latent store (k1 fraction) plus the hot
    cache's HDD capital and its electricity, and the denominator scaling with
    k2 (1-h) R (p_TFLOP + beta).  The project reports carbon as emissions only,
    not as a dollar cost, so ``include_carbon`` defaults to False; the break-even
    is capital + electricity.

    Parameters
    ----------
    g       : corpus growth rate per year (e.g. 0.20)
    S       : object size [PB]
    k1      : IR-store size fraction of full object (0 < k1 <= 1)
    k2      : IR-decode compute fraction of full generation (0 < k2 <= 1)
    s       : Zipf skew of the request distribution
    """
    years = np.arange(YEAR0, horizon + 1)
    nyr = len(years)
    age = years - YEAR0                       # 0, 1, 2, ...

    phdd = p_hdd(years)
    ptf = p_tflop(years)
    alph = alpha(years)                       # $/PB/yr, declining 10%/yr
    bet = beta(years)                         # $/TFLOP, declining 30%/yr

    # new data ingested each year [PB]
    new_data = (1.0 + g) ** age
    A = _replacement_series(new_data)         # manufactured (with replacement)

    # cumulative live object count and request rate
    if abs(g) < 1e-12:
        n_cum = (age + 1) / S
        v_full = (age + 1).astype(np.float64)            # live volume [PB]
    else:
        n_cum = ((1.0 + g) ** (age + 1) - 1.0) / (g * S)
        v_full = ((1.0 + g) ** (age + 1) - 1.0) / g
    R = REQ_PER_OBJ * n_cum

    # hot-cache volume per year [PB]
    frac = np.array([cache_fraction(n, s) for n in n_cum])
    m_count = frac * n_cum
    v_cache = m_count * S
    dv_cache = np.diff(np.concatenate([[0.0], v_cache]))   # yearly increment
    A_cache = _replacement_series(np.maximum(dv_cache, 0.0))

    # --- full-HDD baseline ---
    C_hdd_cum = np.sum(1000.0 * A * phdd)
    total_full = C_hdd_cum
    if include_power:
        E_full = np.sum(alph * v_full)
        total_full += E_full
    if include_carbon:
        E_full_for_carbon = np.sum(alph * v_full)
        F_full = (GAMMA_E / E_RATE) * E_full_for_carbon + CAP_PHI_HDD * np.sum(A)
        total_full += P_C * F_full / 1e3

    # --- two-tier fixed (c-independent) part ---
    C_ir_cum = k1 * C_hdd_cum
    C_cache_cum = np.sum(1000.0 * A_cache * phdd)
    total_2t_fixed = C_ir_cum + C_cache_cum
    powered_2t = v_cache + k1 * v_full        # IR store + hot cache stay powered
    if include_power:
        total_2t_fixed += np.sum(alph * powered_2t)
    if include_carbon:
        E_2t = np.sum(alph * powered_2t)
        F_2t_fixed = ((GAMMA_E / E_RATE) * E_2t
                      + CAP_PHI_HDD * (k1 * np.sum(A) + np.sum(A_cache)))
        total_2t_fixed += P_C * F_2t_fixed / 1e3

    # --- two-tier miss-regeneration part (linear in c) ---
    miss = 0.10 * R                            # regenerations per year
    coef_c = k2 * np.sum(miss * ptf)           # GPU capital
    if include_power:
        coef_c += k2 * np.sum(bet * miss)
    if include_carbon:
        coef_c += k2 * (PHI_GPU + PSI_GPU) * np.sum(miss) * P_C / 1e3

    # break-even: total_2t_fixed + coef_c * c = total_full
    return (total_full - total_2t_fixed) / coef_c


def cost_breakdown(g, S, k1, k2, c, year, s=1.0):
    """Cumulative cost [$] through ``year`` broken into components for both
    strategies, given the object's actual generation cost ``c`` [TFLOP].

    Returns ``(full, two_tier)`` where each is a dict with keys
    ``hdd``, ``gpu``, ``elec``, ``carbon`` (capital HDD, amortised GPU compute,
    electricity, and carbon cost respectively).
    """
    years = np.arange(YEAR0, year + 1)
    age = years - YEAR0
    phdd = p_hdd(years)
    ptf = p_tflop(years)
    alph = alpha(years)                       # $/PB/yr, declining 10%/yr
    bet = beta(years)                         # $/TFLOP, declining 30%/yr

    new_data = (1.0 + g) ** age
    A = _replacement_series(new_data)
    if abs(g) < 1e-12:
        n_cum = (age + 1) / S
        v_full = (age + 1).astype(np.float64)
    else:
        n_cum = ((1.0 + g) ** (age + 1) - 1.0) / (g * S)
        v_full = ((1.0 + g) ** (age + 1) - 1.0) / g
    R = REQ_PER_OBJ * n_cum

    frac = np.array([cache_fraction(n, s) for n in n_cum])
    v_cache = frac * n_cum * S
    dv_cache = np.diff(np.concatenate([[0.0], v_cache]))
    A_cache = _replacement_series(np.maximum(dv_cache, 0.0))

    miss = 0.10 * R
    sum_miss = np.sum(miss)
    gamma_over_e = GAMMA_E / E_RATE

    # --- full-HDD baseline: store everything, no regeneration ---
    hdd_cap = np.sum(1000.0 * A * phdd)
    elec_full = np.sum(alph * v_full)
    carbon_kg_full = gamma_over_e * elec_full + CAP_PHI_HDD * np.sum(A)
    full = {
        "hdd": hdd_cap,
        "gpu": 0.0,
        "elec": elec_full,
        "carbon": P_C * carbon_kg_full / 1e3,
    }

    # --- two-tier: hot cache + IR cold store, decode on miss at k2*c ---
    hdd_cap_2t = np.sum(1000.0 * A_cache * phdd) + k1 * hdd_cap
    gpu_cap_2t = k2 * c * np.sum(miss * ptf)
    elec_hdd_2t = np.sum(alph * (v_cache + k1 * v_full))
    elec_gpu_2t = k2 * c * np.sum(bet * miss)
    elec_2t = elec_hdd_2t + elec_gpu_2t
    carbon_kg_2t = (gamma_over_e * elec_2t
                    + CAP_PHI_HDD * (k1 * np.sum(A) + np.sum(A_cache))
                    + PSI_GPU * k2 * c * sum_miss)
    two_tier = {
        "hdd": hdd_cap_2t,
        "gpu": gpu_cap_2t,
        "elec": elec_2t,
        "carbon": P_C * carbon_kg_2t / 1e3,
    }
    return full, two_tier


# ---------------------------------------------------------------------------
# Representative parameter set, one per modality (see Table in ablation.tex)
# ---------------------------------------------------------------------------
MODALITIES = [
    # name, model, c_gen [TFLOP], S [PB], k1, k2
    ("Image",    "FLUX.1-dev",     2100.0, 3.0e-9,  1.0 / 6,   1.0 / 200),
    ("Audio",    "MusicGen Large",   21.0, 5.1e-9,  1.0 / 212, 1.0 / 470),
    ("Molecule", "Uni-Mol",           5.0, 5.0e-12, 1.0 / 80,  1.0 / 7000),
]

# default sweep settings
G_DEFAULT = 0.20
S_DEFAULT = 1.0


if __name__ == "__main__":
    # Full tape cost over 2025-2045 for the steady and growing corpus
    # (Section 3.3): media + drives + library + power.
    print(f"Tape: P_TAPE0=${P_TAPE0:.0f}/TB flat, drive=${P_DRIVE:,.0f} "
          f"(~{DRIVE_TPUT_PB_YR:.1f} PB/yr, life {L_DRIVE}yr), "
          f"slot=${C_SLOT:.0f}/{TAPE_CART_TB:.0f}TB, alpha_tape(2025)={ALPHA_TAPE0:.1f} $/PB/yr (10% of HDD)")
    for label, g in [("steady  (g= 0%)", 0.0), ("growing (g=20%)", G_DEFAULT)]:
        b = tape_storage_cost_full(g)
        print(f"  {label}: media ${b['media']:,.0f} | drives ${b['drives']:,.0f} "
              f"| library ${b['library']:,.0f} | power ${b['power']:,.0f} "
              f"=> total ${b['total']:,.0f}")
