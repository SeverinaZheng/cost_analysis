"""
calculate_cache_threshold.py
-----------------------------
Computes the break-even generation cost c* for the caching model in §1.3.

The caching strategy (cache top-M Zipfian hot objects, regenerate 10% misses)
beats full HDD storage when:

    sum_{tau} C_mod(tau)  <  sum_{tau} C_HDD_ann(tau)

where:
    C_HDD_ann(tau)   = annual HDD cost to store ALL new data in year tau
                     = 20,000 * [(1+g)*0.90]^(tau-2025)  [$]
    C_cache_HDD(tau) = HDD cost for the cache increment only  [$]
    C_miss(tau)      = 0.10 * R^(g)(tau) * c * p_TFLOP(tau)  [$]
    C_mod(tau)       = C_cache_HDD(tau) + C_miss(tau)

Rearranging gives a threshold on c:
    c  <  [sum C_HDD_ann - sum C_cache_HDD] / sum_miss_unit

where sum_miss_unit = sum 0.10 * 100 * N_obj_cum * p_TFLOP  [$/TFLOP].

Parameters (edit as needed):
    g   : annual corpus growth fraction  (0.20 = 20%)
    y   : horizon year
    S   : per-object size in PB          (3e-9 = 3 MB)
    h   : target Zipf hit ratio          (0.90 = 90%)
"""

import math

# ── configurable parameters ──────────────────────────────────────────────────
g   = 0.20          # annual growth rate
y   = 2045          # horizon year
S   = 3e-9          # object size [PB]   (3 MB)
h   = 0.90          # target hit ratio
# ─────────────────────────────────────────────────────────────────────────────

# cost model constants (anchored at 2025)
HDD_2025    = 20.0       # $/TB in 2025
HDD_decay   = 0.90       # annual decline factor
GPU_2025    = 20.0       # $/(TFLOP/s) in 2025
GPU_decay   = 0.70       # annual decline factor
GPU_life    = 9.46e7     # GPU lifetime [s]  (≈ 3 yr)
GPU_util    = 0.40       # utilisation fraction
EULER_GAMMA = 0.5772156649

def hdd_price(tau):
    """$/TB in year tau."""
    return HDD_2025 * HDD_decay**(tau - 2025)

def p_tflop(tau):
    """Amortised GPU cost [$/TFLOP] in year tau."""
    return GPU_2025 * GPU_decay**(tau - 2025) / (GPU_life * GPU_util)

def N_obj_cum(tau):
    """Cumulative object count at end of year tau (growing at g/yr)."""
    return (1 + g)**(tau - 2024) / (g * S) - 1 / (g * S)
    # = [(1+g)^(tau-2024) - 1] / (g*S)

def V_cache(tau):
    """
    Cache volume [PB] needed for h hit ratio under Zipf s=1.
    h(M,N) = H_M / H_N  =>  ln M ≈ h*ln N - (1-h)*gamma
    """
    N = N_obj_cum(tau)
    if N <= 0:
        return 0.0
    coeff = math.exp(-(1 - h) * EULER_GAMMA)   # ≈ 0.944 for h=0.90
    M = coeff * N**h
    return M * S   # PB

# ── annual quantities ─────────────────────────────────────────────────────────
print(f"Parameters: g={g*100:.0f}%, y={y}, S={S:.1e} PB, hit={h*100:.0f}%")
print(f"{'tau':>4}  {'C_HDD_ann':>12}  {'C_cache_HDD':>13}  {'miss_unit':>12}")
print("-" * 50)

sum_hdd_full   = 0.0   # cumulative full-storage HDD cost
sum_cache_hdd  = 0.0   # cumulative cache HDD cost
sum_miss_unit  = 0.0   # cumulative miss regen coefficient [$/TFLOP]
V_prev = 0.0

for tau in range(2025, y + 1):
    j = tau - 2025

    # full HDD: buy (1+g)^j PB at today's price
    C_hdd_ann = 20_000 * ((1 + g) * HDD_decay)**j

    # cache HDD: only the increment in V_cache
    V_cur = V_cache(tau)
    dV = V_cur - V_prev
    C_chdd = 1_000 * dV * hdd_price(tau)   # $/TB * 1000 TB/PB * PB

    # miss regen unit: 0.10 * R * p_TFLOP  [$/TFLOP]
    N = N_obj_cum(tau)
    miss_u = 0.10 * 100 * N * p_tflop(tau)

    sum_hdd_full  += C_hdd_ann
    sum_cache_hdd += C_chdd
    sum_miss_unit += miss_u
    V_prev = V_cur

    if tau in (2025, 2030, 2035, 2040, 2045):
        print(f"{tau:>4}  ${C_hdd_ann:>11,.0f}  ${C_chdd:>12,.0f}  {miss_u:>12.3e}")

print("-" * 50)
print(f"  CUM  ${sum_hdd_full:>11,.0f}  ${sum_cache_hdd:>12,.0f}  {sum_miss_unit:>12.3e}")

# ── threshold ────────────────────────────────────────────────────────────────
c_thresh = (sum_hdd_full - sum_cache_hdd) / sum_miss_unit

print(f"\n=== Results ===")
print(f"Cumulative full-storage HDD cost  : ${sum_hdd_full:>12,.0f}")
print(f"Cumulative cache HDD cost         : ${sum_cache_hdd:>12,.0f}")
print(f"Cumulative miss-regen coefficient : {sum_miss_unit:.4e} $/TFLOP")
print(f"\nCaching wins when c < {c_thresh:.2f} TFLOP")
print(f"  i.e. c* = {c_thresh:.3e} TFLOP")

# ── cache size at horizon ────────────────────────────────────────────────────
N45 = N_obj_cum(y)
V45 = V_cache(y)
coeff = math.exp(-(1 - h) * EULER_GAMMA)
frac  = coeff * N45**-(1 - h) * 100   # percent of corpus cached

print(f"\nAt y={y}:")
print(f"  Cumulative objects  N = {N45:.3e}")
print(f"  Cache size          M = {V45/S:.3e} objects  ({V45:.2f} PB)")
print(f"  Cache fraction        = {frac:.1f}% of corpus")
