#!/usr/bin/env python3
r"""Step-by-step reproduction of every number reported in Sections 3 and 4.

Section 3  Cost Model 1: Materialized-Artifact Storage
    3.1  Parameter Anchors                     -> section_3_1_parameter_anchors()
    3.2  Materialized-Artifact Storage on HDD  -> section_3_2_hdd_storage()
    3.3  Materialized-Artifact Storage on Tape -> section_3_3_tape_storage()
    3.4  Cost Over Next 20 years (HDD vs tape) -> section_3_4_horizon_cost()

Section 4  Cost Model 2: Prompt-Based Regeneration
    4.0  Access model / image / compute price  -> section_4_0_regen_anchors()
    4.1  Prompt-Based Regeneration             -> section_4_1_prompt_regeneration()
    4.2  Caching Hot Objects                   -> section_4_2_caching()
    4.3  Cost Over Next 20 years (cache)       -> section_4_3_cache_horizon()

Section 5  Cost Model 3: LazIR, an Intermediate-Representation Store
    5.0  IR anchors, break-even, 20-year cost  -> section_5_ir_store()

The file is deliberately self-contained: every constant is spelled out with its
units at the top, and every reported figure is produced by an explicit function
call in ``main()``.  The numbers agree with cost_model.py, which is used by the
plotting scripts; this file is the readable, auditable derivation of them.

Run:  python calculation.py
"""

import numpy as np

# ===========================================================================
# CONSTANTS  (all anchors used in Sections 3 and 4, with units)
# ===========================================================================

# ---- Horizon and workload (Section 3.1, "Workload assumptions") -----------
YEAR0        = 2025            # anchor year
HORIZON      = 2045            # end of the 20-year analysis horizon
S_OBJ        = 3.0e-9          # PB per object   (3 MB image = 3e-9 PB)
G            = 0.20            # exponential annual ingest growth rate (20%/yr)
L_HDD        = 5              # HDD service life [yr] (enterprise refresh interval)

# ---- HDD price (Section 3.1, "Disk price") --------------------------------
C_HDD_2025   = 20_000.0       # $/PB HDD price anchor in 2025 (= 20 $/TB)
DROP_HDD     = 0.10           # HDD $/PB annual decline (10%/yr)

# ---- Electricity price and HDD power (Section 3.1, "Electricity price") ----
E_RATE       = 0.08           # $/kWh data-center electricity tariff
HOURS_PER_YR = 8_760.0        # h/yr
P_HDD_SYS    = 450.0          # W/PB HDD system power density in 2025
#   = 320 W/PB bare Seagate Exos M 30 TB (9.5 W/drive) x 1.4 PUE overhead
#   c_elec,HDD(2025) = P_HDD_SYS * 8760 h/yr * E_RATE / 1000  (see eq. alpha)

# ---- Tape price and infrastructure (Section 3.3) --------------------------
C_TAPE_2025  = 10_000.0       # $/PB LTO-10 native anchor 2025 (30 TB at ~$300 -> 10 $/TB)
DROP_TAPE    = 0.10           # tape $/PB annual decline (rounded from ~12% fit)
L_TAPE       = 30             # tape media life [yr] >> horizon -> bought once, no replace
P_DRIVE      = 10_000.0       # $/drive LTO-10 tape drive (over $10k each)
DRIVE_RATE_MBPS = 400.0       # MB/s LTO-10 native transfer rate
DRIVE_DUTY   = 0.50           # write-side duty cycle
L_DRIVE      = 5              # drive service life [yr], re-bought as they wear
N_DRIVE_MIN  = 2              # minimum drives (redundancy)
TAPE_CART_TB = 30.0           # TB native per LTO-10 cartridge
C_SLOT       = 100.0          # $/cartridge slot (library frame + robotics)
TAPE_ELEC_FRAC = 0.10         # tape electricity held at a fixed 10% of HDD

# ---- GPU / regeneration (Section 4.0, "Compute price" & "AI-generated image")
C_GPU_2025   = 18.0           # $/(TFLOP/s) GPU capacity-price anchor 2025 (B200 FP16)
DROP_GPU     = 0.30           # GPU $/(TFLOP/s) annual decline (30%/yr)
L_GPU        = 3.0            # GPU service life [yr]
U_GPU        = 0.40           # GPU effective utilisation
DELTA_YR     = 3.156e7        # seconds per year
FLOP_REGEN   = 2_100.0        # TFLOP to regenerate one 1024x1024 FLUX.1-dev image
REQ_PER_OBJ  = 10.0           # average requests per object per year
C_ELEC_GPU_2025 = 9.78e-9     # $/TFLOP GPU electricity anchor 2025
#   = 0.44 J/TFLOP (B200) * 0.08 $/kWh / 3.6e6 J/kWh
HIT_RATIO    = 0.90           # target hot-cache hit ratio h

# ---- LazIR intermediate-representation store (Section 5) ------------------
K_SIZE       = 1.0 / 6.0      # IR store fraction k_size for FLUX.1-dev (f=8,c=16 -> 2c/(3f^2)=1/6)
K_GEN        = 1.0 / 200.0    # IR decode compute fraction k_gen (VAE decode vs full denoise)
FLOP_IR      = K_GEN * FLOP_REGEN  # TFLOP per miss decoded from the stored latent (~10.5, quoted ~10.3)

# ---- Derived seconds-of-service normaliser for GPU amortisation -----------
LUD = L_GPU * U_GPU * DELTA_YR   # L * u * Delta_yr  [s]  (~3.79e7)


# ===========================================================================
# PRICE / COEFFICIENT TRAJECTORIES  (all decline geometrically from 2025)
# ===========================================================================
def c_hdd(year):
    """HDD price [$/PB], eq:phdd:  20,000 * 0.90^(t-2025)."""
    return C_HDD_2025 * (1.0 - DROP_HDD) ** (year - YEAR0)


def c_tape(year):
    """Tape price [$/PB]:  10,000 * 0.90^(t-2025)."""
    return C_TAPE_2025 * (1.0 - DROP_TAPE) ** (year - YEAR0)


def c_elec_hdd(year):
    """HDD electricity coefficient [$/PB/yr], eq:alphahdd.

    Anchored at c_elec_hdd(2025) ~ 315 $/PB/yr and declining at the same
    10%/yr as the HDD price (denser drives spread a fixed wattage over more TB).
    """
    return c_elec_hdd_2025() * (1.0 - DROP_HDD) ** (year - YEAR0)


def c_elec_hdd_2025():
    """2025 HDD electricity anchor [$/PB/yr] from the power density, eq:alpha."""
    return P_HDD_SYS * HOURS_PER_YR * E_RATE / 1.0e3


def c_elec_tape(year):
    """Tape electricity coefficient [$/PB/yr], eq:alphatape: 10% of HDD."""
    return TAPE_ELEC_FRAC * c_elec_hdd(year)


def c_gpu(year):
    """GPU capacity price [$/(TFLOP/s)], eq:pgpu:  18 * 0.70^(t-2025)."""
    return C_GPU_2025 * (1.0 - DROP_GPU) ** (year - YEAR0)


def c_tflop(year):
    """Amortised GPU serving price [$/TFLOP], eq:ptflop: c_gpu(y)/(L*u*Delta_yr)."""
    return c_gpu(year) / LUD


def c_elec_gpu(year):
    """GPU electricity coefficient [$/TFLOP], declining at 30%/yr."""
    return C_ELEC_GPU_2025 * (1.0 - DROP_GPU) ** (year - YEAR0)


# ===========================================================================
# SHARED VOLUME / COUNT HELPERS
# ===========================================================================
YEARS = np.arange(YEAR0, HORIZON + 1)     # 2025..2045 inclusive (21 years)
AGE   = YEARS - YEAR0                      # 0..20


def new_data(g):
    """New data ingested in each year [PB]:  S_data(tau) = (1+g)^(tau-2025)."""
    return (1.0 + g) ** AGE


def live_volume(g):
    """Cumulative (powered) corpus volume in each year [PB], S_full_data(y).

    Sum of all ingest to date: [(1+g)^(y-2024)-1]/g, with the g->0 limit age+1.
    """
    if abs(g) < 1e-12:
        return (AGE + 1).astype(float)
    return ((1.0 + g) ** (AGE + 1) - 1.0) / g


def object_count(g):
    """Cumulative object count N(y) = live_volume / s_obj."""
    return live_volume(g) / S_OBJ


def request_rate(g):
    """Annual request rate R^(g)(y) = 10 * cumulative object count, eq:nobjg."""
    return REQ_PER_OBJ * object_count(g)


def replacement_series(purchase_new, life):
    """Annual provisioning with end-of-life replacement, eq:provision:
        A(y) = purchase_new(y) + A(y - life),   A(y)=0 for y<2025.
    Returns the manufactured PB (or drive count) purchased each year.
    """
    a = np.zeros_like(purchase_new, dtype=float)
    for j in range(len(purchase_new)):
        a[j] = purchase_new[j] + (a[j - life] if j - life >= 0 else 0.0)
    return a


# ---- Zipf(s=1) hot-cache fraction (Appendix A closed form + exact inversion)
# The paper reports the closed form F_cached ~ 0.944 * N^-0.10 for h=0.90.
# For the per-year cache cost we invert the generalized-harmonic ratio exactly,
# which matches cost_model.cache_fraction and the closed form for large N.
_CUTOFF = 20_000


def _harmonic(s=1.0):
    i = np.arange(1, _CUTOFF + 1, dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(i ** (-s))])
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
    """Fraction M/N of a Zipf(s) catalogue that must be cached for hit ratio."""
    n = float(n_objects)
    if n < 1.5:
        return 1.0
    H = _harmonic(s)
    target = hit * H(n)
    lo, hi = 1.0, n
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if H(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-6:
            break
    return min(1.0, 0.5 * (lo + hi) / n)


def cache_fraction_closedform(n, hit=HIT_RATIO):
    """Paper closed form for Zipf(1), h=0.90:  F_cached ~ 0.944 * N^-0.10."""
    return 0.944 * n ** -0.10


# ===========================================================================
# STORAGE COST BUILDING BLOCKS  (Section 3.2)
# ===========================================================================
def hdd_capital_no_replace(g):
    """Cumulative HDD capital w/o replacement, eq:chddcum / eq:chddcumg [$].
    Each year buy S_data(tau) PB at that year's price (1000 TB per PB baked in
    through the $/PB price)."""
    return float(np.sum(new_data(g) * c_hdd(YEARS)))


def hdd_capital_with_replace(g):
    """Cumulative HDD capital w/ replacement, eq:chddcumrepl [$].
    Manufactured PB each year (new + end-of-life re-buys) at year price."""
    manufactured = replacement_series(new_data(g), L_HDD)
    return float(np.sum(manufactured * c_hdd(YEARS)))


def hdd_electricity(g):
    """Cumulative HDD electricity, eq:celecfullcum [$].
    Every powered PB (live volume) billed at that year's declining coefficient."""
    return float(np.sum(c_elec_hdd(YEARS) * live_volume(g)))


def hdd_total(g):
    """Total materialized-artifact storage on HDD, eq:chddtotal [$]."""
    return hdd_capital_with_replace(g) + hdd_electricity(g)


# ===========================================================================
# SECTION 3.1  Parameter Anchors
# ===========================================================================
def section_3_1_parameter_anchors():
    print("=" * 74)
    print("SECTION 3.1  Parameter Anchors")
    print("=" * 74)
    print(f"  Object size            s_obj            = {S_OBJ:g} PB (3 MB image)")
    print(f"  Ingest growth          g                = {G:.0%}/yr")
    print(f"  HDD service life        L_HDD           = {L_HDD} yr")
    print(f"  Horizon                                  = {YEAR0}-{HORIZON}")
    print()
    print(f"  HDD price anchor        C_HDD(2025)     = ${C_HDD_2025:,.0f}/PB "
          f"(= {C_HDD_2025/1000:.0f} $/TB)")
    print(f"  HDD price decline       drop_HDD        = {DROP_HDD:.0%}/yr")
    print(f"    => C_HDD(t) = {C_HDD_2025:,.0f} x {1-DROP_HDD:.2f}^(t-2025) $/PB")
    print(f"       check C_HDD(2035)                  = ${c_hdd(2035):,.0f}/PB")
    print()
    print(f"  Electricity rate        e               = ${E_RATE}/kWh")
    print(f"  HDD system power        P_HDD^sys       = {P_HDD_SYS:.0f} W/PB "
          f"(320 W/PB bare x 1.4 PUE)")
    alpha0 = c_elec_hdd_2025()
    print(f"    c_elec,HDD(2025) = {P_HDD_SYS:.0f} x {HOURS_PER_YR:.0f} x "
          f"{E_RATE}/1000")
    print(f"    => c_elec,HDD(2025)                   = {alpha0:.1f} $/PB/yr  (~315)")
    print(f"       c_elec,tape(2025) = 0.1 x {alpha0:.1f} = "
          f"{c_elec_tape(2025):.1f} $/PB/yr  (~31.5)")


# ===========================================================================
# SECTION 3.2  Materialized-Artifact Storage on HDD
# ===========================================================================
def section_3_2_hdd_storage():
    print("\n" + "=" * 74)
    print("SECTION 3.2  Materialized-Artifact Storage on HDD")
    print("=" * 74)

    # 3.2.1 Cumulative capital cost, no replacement
    cap_lin = hdd_capital_no_replace(0.0)
    cap_exp = hdd_capital_no_replace(G)
    print("  3.2.1 Cumulative capital cost (no replacement), eq:chddcum/chddcumg")
    print(f"    Linear      (g=0)   C_HDD,cum(2045)   = ${cap_lin:,.0f}   (~$178,000)")
    print(f"    Exponential (g=20%) C_HDD,cum(2045)   = ${cap_exp:,.0f}   (~$1,008,000)")

    # 3.2.2 Replacement
    cap_lin_r = hdd_capital_with_replace(0.0)
    cap_exp_r = hdd_capital_with_replace(G)
    # how many times each 2025 PB is manufactured across the horizon
    times_bought = len(range(0, len(AGE), L_HDD))
    print("  3.2.2 With HDD replacement (L_HDD=5 -> each PB bought 3-4x), "
          "eq:chddcumrepl")
    print(f"    a 2025 PB is re-bought at 2025,2030,2035,2040,2045 = "
          f"{times_bought} times")
    print(f"    Linear      (g=0)   C_HDD,cum(2045)   = ${cap_lin_r:,.0f}   (~$344,000)")
    print(f"    Exponential (g=20%) C_HDD,cum(2045)   = ${cap_exp_r:,.0f}   (~$1,515,300)")

    # 3.2.3 Electricity
    e_lin = hdd_electricity(0.0)
    e_exp = hdd_electricity(G)
    print("  3.2.3 Operational electricity, eq:celecfullcum")
    print(f"    Linear      (g=0)   E_full(2045)      = ${e_lin:,.0f}   (~$21,000)")
    print(f"    Exponential (g=20%) E_full(2045)      = ${e_exp:,.0f}   (~$81,000)")


# ===========================================================================
# SECTION 3.3  Materialized-Artifact Storage on Tape
# ===========================================================================
def tape_media(g):
    """Cumulative tape media cost, eq:ctapecumg [$] (one purchase per PB)."""
    return float(np.sum(new_data(g) * c_tape(YEARS)))


def tape_drives(g):
    """Tape-drive capital [$]: sized to annual ingest throughput, re-bought."""
    thru_pb_yr = DRIVE_RATE_MBPS * DELTA_YR * DRIVE_DUTY / 1.0e9   # ~6.3 PB/yr
    n_req = np.maximum(N_DRIVE_MIN, np.ceil(new_data(g) / thru_pb_yr))
    dN = np.maximum(np.diff(np.concatenate([[0.0], n_req])), 0.0)
    return float(np.sum(replacement_series(dN, L_DRIVE)) * P_DRIVE)


def tape_library(g):
    """Robotic-library slot cost [$]: one slot per cartridge of peak live volume."""
    slots = np.ceil(live_volume(g)[-1] * 1000.0 / TAPE_CART_TB)
    return float(slots * C_SLOT)


def tape_power(g):
    """Tape electricity [$]: mounted drives + controller only, at c_elec,tape."""
    return float(np.sum(c_elec_tape(YEARS) * live_volume(g)))


def section_3_3_tape_storage():
    print("\n" + "=" * 74)
    print("SECTION 3.3  Materialized-Artifact Storage on Tape")
    print("=" * 74)
    thru = DRIVE_RATE_MBPS * DELTA_YR * DRIVE_DUTY / 1.0e9
    print("  3.3.1 Tape price anchor")
    print(f"    C_tape(2025)                          = ${C_TAPE_2025:,.0f}/PB "
          f"(= {C_TAPE_2025/1000:.0f} $/TB), drop_tape = {DROP_TAPE:.0%}/yr")
    print(f"    one LTO-10 drive absorbs ~{thru:.1f} PB/yr at {DRIVE_DUTY:.0%} duty")
    print("  3.3.2 Media cost, eq:ctapecumg")
    print(f"    Linear      (g=0)   media(2045)       = ${tape_media(0.0):,.0f}   (~$89,000)")
    print(f"    Exponential (g=20%) media(2045)       = ${tape_media(G):,.0f}   (~$504,000)")
    print("  3.3.3 Drives, library, power")
    for label, g in [("Linear      (g=0)  ", 0.0), ("Exponential (g=20%)", G)]:
        print(f"    {label}  drives=${tape_drives(g):,.0f}  "
              f"library=${tape_library(g):,.0f}  power=${tape_power(g):,.0f}")


# ===========================================================================
# SECTION 3.4  Cost Over Next 20 years  (Table: horizon-cost)
# ===========================================================================
def section_3_4_horizon_cost():
    print("\n" + "=" * 74)
    print("SECTION 3.4  Cost Over Next 20 years  (Table 2)")
    print("=" * 74)
    for label, g in [("Linear (g=0)", 0.0), ("Exponential (g=20%)", G)]:
        cap_nr = hdd_capital_no_replace(g)
        cap_wr = hdd_capital_with_replace(g)
        elec   = hdd_electricity(g)
        total  = cap_wr + elec
        print(f"  HDD  {label}")
        print(f"    Capital w/o replace  = ${cap_nr:,.0f}")
        print(f"    Capital w/ replace   = ${cap_wr:,.0f}")
        print(f"    Electricity          = ${elec:,.0f}")
        print(f"    TOTAL                = ${total:,.0f}")

    hdd_lin = hdd_total(0.0)
    hdd_exp = hdd_total(G)
    print(f"  HDD exponential / linear ratio          = {hdd_exp/hdd_lin:.1f}x  (~4.4x)")

    for label, g in [("Linear (g=0)", 0.0), ("Exponential (g=20%)", G)]:
        m, d, li, p = tape_media(g), tape_drives(g), tape_library(g), tape_power(g)
        total = m + d + li + p
        hdd_t = hdd_total(g)
        print(f"  Tape {label}")
        print(f"    Media=${m:,.0f}  Drives=${d:,.0f}  Library=${li:,.0f}  "
              f"Power=${p:,.0f}")
        print(f"    TOTAL                = ${total:,.0f}   "
              f"(HDD is {hdd_t/total:.1f}x tape)")


# ===========================================================================
# SECTION 4.0  Access model / image / compute price anchors
# ===========================================================================
def section_4_0_regen_anchors():
    print("\n" + "=" * 74)
    print("SECTION 4.0  Regeneration anchors (access, image, compute price)")
    print("=" * 74)
    print(f"  Requests per object per year            = {REQ_PER_OBJ:.0f}")
    print(f"  FLUX.1-dev generation cost flop_regen   = {FLOP_REGEN:,.0f} TFLOP")
    print(f"  Image size s_obj                        = {S_OBJ:g} PB (3 MB)")
    print(f"  GPU price anchor C_GPU(2025)            = ${C_GPU_2025:.0f}/(TFLOP/s), "
          f"drop_GPU = {DROP_GPU:.0%}/yr")
    print(f"  GPU amortiser L*u*Delta_yr              = {LUD:.3e} s "
          f"(L={L_GPU:.0f}yr, u={U_GPU:.2f})")
    print(f"    => c_TFLOP(2025) = {c_gpu(2025):.1f}/{LUD:.3e} "
          f"= {c_tflop(2025):.3e} $/TFLOP")
    print(f"  GPU electricity c_elec,GPU(2025)        = {C_ELEC_GPU_2025:.3e} $/TFLOP")


# ===========================================================================
# SECTION 4.1  Prompt-Based Regeneration
# ===========================================================================
def regen_total_cost(g, flop=FLOP_REGEN):
    """Cumulative prompt-based regeneration cost [$]: compute + GPU electricity.
    Sum over years of R(y) * flop * (c_TFLOP(y) + c_elec,GPU(y))."""
    R = request_rate(g)
    per_flop = c_tflop(YEARS) + c_elec_gpu(YEARS)
    return float(np.sum(R * flop * per_flop))


def regen_breakeven_flop(g):
    """Break-even generation cost [TFLOP], eq:cthreshpow:
        flop* = C_total_HDD(y) / sum_tau R(tau) (c_TFLOP(tau) + c_elec,GPU(tau))."""
    R = request_rate(g)
    denom = float(np.sum(R * (c_tflop(YEARS) + c_elec_gpu(YEARS))))
    return hdd_total(g) / denom


def section_4_1_prompt_regeneration():
    print("\n" + "=" * 74)
    print("SECTION 4.1  Prompt-Based Regeneration")
    print("=" * 74)
    for label, g, approx in [("Linear      (g=0)  ", 0.0, "~20"),
                             ("Exponential (g=20%)", G, "~50")]:
        be = regen_breakeven_flop(g)
        print(f"    {label}  break-even flop*      = {be:,.1f} TFLOP   ({approx})")
    for label, g, approx in [("Linear      (g=0)  ", 0.0, "~$38M"),
                             ("Exponential (g=20%)", G, "~$68M")]:
        tot = regen_total_cost(g)
        print(f"    {label}  regen @2100 TFLOP     = ${tot:,.0f}   ({approx})")
    # Takeaway 1: storage is Nx cheaper than regeneration
    for label, g in [("Linear      (g=0)  ", 0.0), ("Exponential (g=20%)", G)]:
        ratio = regen_total_cost(g) / hdd_total(g)
        print(f"    {label}  regen / storage      = {ratio:.0f}x "
              f"(Takeaway 1: ~100x linear, ~40x exp)")


# ===========================================================================
# SECTION 4.2  Caching Hot Objects
# ===========================================================================
def section_4_2_caching():
    print("\n" + "=" * 74)
    print("SECTION 4.2  Caching Hot Objects")
    print("=" * 74)
    print(f"  Target hit ratio h                      = {HIT_RATIO:.0%}")
    print("  Closed form F_cached ~ 0.944 * N^-0.10 for Zipf(1), h=0.90:")
    for n in (1e8, 1e10):
        f_cf = cache_fraction_closedform(n)
        f_ex = cache_fraction(n)
        print(f"    N = {n:.0e}   F_cached(closed form) = {f_cf:.3f} "
              f"({f_cf:.1%})   exact = {f_ex:.3f}")
    print("  (paper: ~15% at N=1e8, ~9.4% at N=1e10)")


# ===========================================================================
# SECTION 4.3  Cost Over Next 20 years  (cache policy)
# ===========================================================================
def cache_hdd_and_elec(g):
    """Cache HDD capital (with replacement) + cache electricity [$]."""
    n_cum = object_count(g)
    frac = np.array([cache_fraction(n) for n in n_cum])
    v_cache = frac * n_cum * S_OBJ                    # cache volume [PB], eq:vcache
    dv = np.maximum(np.diff(np.concatenate([[0.0], v_cache])), 0.0)
    A_cache = replacement_series(dv, L_HDD)
    cap = float(np.sum(A_cache * c_hdd(YEARS)))       # $/PB price -> $
    elec = float(np.sum(c_elec_hdd(YEARS) * v_cache)) # eq:celecgpu cache term
    return cap + elec, cap, elec, v_cache


def cache_breakeven_flop(g):
    """Cache break-even generation cost [TFLOP], eq:cachethreshpow:
        flop* = (C_total_HDD - sum[C_cache,HDD + C_elec,cache])
                / sum (1-h) R (c_TFLOP + c_elec,GPU)."""
    total_cache, _, _, _ = cache_hdd_and_elec(g)
    R = request_rate(g)
    denom = float(np.sum((1.0 - HIT_RATIO) * R * (c_tflop(YEARS) + c_elec_gpu(YEARS))))
    return (hdd_total(g) - total_cache) / denom


def section_4_3_cache_horizon():
    print("\n" + "=" * 74)
    print("SECTION 4.3  Cost Over Next 20 years  (cache policy)")
    print("=" * 74)
    for label, g, approx in [("Linear      (g=0)  ", 0.0, "~$38,000"),
                             ("Exponential (g=20%)", G, "~$137,000")]:
        total_cache, cap, elec, _ = cache_hdd_and_elec(g)
        print(f"    {label}  cache HDD+elec        = ${total_cache:,.0f}   "
              f"({approx})  [cap ${cap:,.0f} + elec ${elec:,.0f}]")
    for label, g, approx in [("Linear      (g=0)  ", 0.0, "~182"),
                             ("Exponential (g=20%)", G, "~454")]:
        be = cache_breakeven_flop(g)
        print(f"    {label}  cache break-even flop*= {be:,.1f} TFLOP   ({approx})")
    # roughly tenfold lift over prompt-based regeneration
    for label, g in [("Linear      (g=0)  ", 0.0), ("Exponential (g=20%)", G)]:
        lift = cache_breakeven_flop(g) / regen_breakeven_flop(g)
        print(f"    {label}  lift over prompt-only = {lift:.1f}x  (~10x)")


# ===========================================================================
# SECTION 5  Cost Model 3: LazIR, an Intermediate-Representation Store
# ===========================================================================
def ir_store_capital(g):
    """Cumulative HDD capital for the latent (IR) store, eq:cirhddsumapprox [$].
    Every object is persisted as a k_size-sized latent, provisioned and replaced
    on the same L_HDD cycle, so the bill is exactly k_size of the full-corpus
    materialized-artifact capital (with replacement)."""
    return K_SIZE * hdd_capital_with_replace(g)


def ir_store_electricity(g):
    """Cumulative HDD electricity for the IR store [$], the k_size*E_full term of
    eq:celec2tiercum (powered latent volume k_size * S_full_data)."""
    return K_SIZE * hdd_electricity(g)


def ir_miss_cost(g):
    """Cumulative cache-miss regeneration from the stored latent, compute plus GPU
    electricity [$]: sum over years of (1-h) R(y) * k_gen*flop_regen *
    (c_TFLOP(y) + c_elec,GPU(y)), the C_miss,IR + C_elec,GPU^IR terms."""
    R = request_rate(g)
    per_flop = c_tflop(YEARS) + c_elec_gpu(YEARS)
    return float(np.sum((1.0 - HIT_RATIO) * R * FLOP_IR * per_flop))


def lazir_total(g):
    """Total LazIR cost over the horizon [$], eq:cmodir summed with electricity:
    hot cache (HDD capital + electricity) + IR store (capital + electricity)
    + cache-miss decode (compute + GPU electricity)."""
    cache_tot, _, _, _ = cache_hdd_and_elec(g)
    return cache_tot + ir_store_capital(g) + ir_store_electricity(g) + ir_miss_cost(g)


def lazir_breakeven_flop(g):
    """LazIR break-even full-generation cost [TFLOP], eq:cachethreshirpow:
        flop* = [(1-k_size) C_total_HDD - sum(C_cache,HDD + C_elec,cache)]
                / [k_gen sum (1-h) R (c_TFLOP + c_elec,GPU)]."""
    cache_tot, _, _, _ = cache_hdd_and_elec(g)
    R = request_rate(g)
    denom = K_GEN * float(np.sum((1.0 - HIT_RATIO) * R
                                 * (c_tflop(YEARS) + c_elec_gpu(YEARS))))
    return ((1.0 - K_SIZE) * hdd_total(g) - cache_tot) / denom


def section_5_ir_store():
    print("\n" + "=" * 74)
    print("SECTION 5  Cost Model 3: LazIR, an Intermediate-Representation Store")
    print("=" * 74)
    print(f"  IR store fraction       k_size          = 1/6  = {K_SIZE:.4f} "
          f"(FLUX.1-dev f=8, c=16)")
    print(f"  IR decode fraction      k_gen           = 1/200 = {K_GEN:.4f}")
    print(f"    => miss decode = k_gen x {FLOP_REGEN:,.0f} = {FLOP_IR:.1f} TFLOP "
          f"(quoted ~10.3)")
    print(f"  break-even lift ~ (1-k_size)/k_gen        = "
          f"{(1.0 - K_SIZE) / K_GEN:.0f}x over the plain cache")
    for label, g, approx in [("Linear      (g=0)  ", 0.0, "~29,700"),
                             ("Exponential (g=20%)", G, "~74,200")]:
        be = lazir_breakeven_flop(g)
        print(f"    {label}  break-even flop*      = {be:,.0f} TFLOP   ({approx})")
    for label, g, approx in [("Linear      (g=0)  ", 0.0, "~$118,000"),
                             ("Exponential (g=20%)", G, "~$437,000")]:
        cache_tot, _, _, _ = cache_hdd_and_elec(g)
        tot = lazir_total(g)
        print(f"    {label}  LazIR total          = ${tot:,.0f}   ({approx})")
        print(f"        [cache ${cache_tot:,.0f} + IR cap ${ir_store_capital(g):,.0f}"
              f" + IR elec ${ir_store_electricity(g):,.0f}"
              f" + miss ${ir_miss_cost(g):,.0f}]")
    # Takeaway 3: 3.1x-3.7x cheaper than materialized-artifact storage on HDD
    for label, g in [("Linear      (g=0)  ", 0.0), ("Exponential (g=20%)", G)]:
        ratio = hdd_total(g) / lazir_total(g)
        print(f"    {label}  HDD storage / LazIR  = {ratio:.1f}x  "
              f"(Takeaway 3: 3.1x-3.7x)")


# ===========================================================================
def main():
    section_3_1_parameter_anchors()
    section_3_2_hdd_storage()
    section_3_3_tape_storage()
    section_3_4_horizon_cost()
    section_4_0_regen_anchors()
    section_4_1_prompt_regeneration()
    section_4_2_caching()
    section_4_3_cache_horizon()
    section_5_ir_store()


if __name__ == "__main__":
    main()
