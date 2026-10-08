#!/usr/bin/env python3
"""Externally reported carbon data for the Carbon Emission subsection.

Two independent quantities both rise over the years, so the *dollar* cost of
data-centre carbon is pushed up on two fronts at once:

  (1) AMOUNT  -- the tonnes of CO2 emitted by data-centre compute and storage
      keep growing, tracking the surge in electricity demand.
  (2) PRICE   -- the dollars charged per tonne of CO2 (carbon markets, and the
      regulatory social cost of carbon) also keep rising.

Every series below is an externally reported figure with its source URL given
inline.  Values that we interpolate or derive are flagged as such in comments
so the plotting scripts can mark them distinctly from reported anchors.
"""

# ---------------------------------------------------------------------------
# (A) Data-centre electricity demand [TWh/yr]   -- proxy for compute+storage
#     scale.  IEA "Energy and AI" World Energy Outlook special report (2025),
#     Base Case.  ~12%/yr historically, accelerating to ~15%/yr through 2030.
#       https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai
#       https://www.iea.org/data-and-statistics/charts/
#         global-data-centre-electricity-consumption-by-equipment-base-case-2020-2030
# ---------------------------------------------------------------------------
# (year, TWh, reported?) -- "reported" points are IEA figures: the 2020-2024
# historical annual estimates (built from the IEA reported year-on-year
# increments of +28/+27/+36/+55 TWh, with 2023 matching the IEA central
# estimate of ~360 TWh) plus the 2030 Base Case headline; the 2026 and 2028
# points are our interpolation along the IEA ~15%/yr path to 2030.
DC_ELECTRICITY_TWH = [
    (2020, 270.0, True),   # IEA historical estimate: ~270 TWh
    (2021, 298.0, True),   # IEA historical: +28 TWh year-on-year
    (2022, 325.0, True),   # IEA historical: +27 TWh
    (2023, 360.0, True),   # IEA historical: central estimate ~360 TWh (+36)
    (2024, 415.0, True),   # IEA historical: ~415 TWh (+55); ~1.5% of global demand
    (2026, 545.0, False),  # interpolated 2024->2030 along the Base Case (~14.7%/yr)
    (2028, 718.0, False),  # interpolated
    (2030, 945.0, True),   # IEA Base Case: ~945 TWh by 2030 (more than doubles vs 2024)
]

# ---------------------------------------------------------------------------
# (B) Data-centre CO2 emissions [Mt CO2/yr] from electricity use.
#     IEA "Energy and AI" (2025), Base Case: "from 180 Mt today to 300 Mt by
#     2035", peaking near 320 Mt around 2030.
#       https://www.iea.org/data-and-statistics/charts/
#         global-data-centre-co2-emissions-base-case-2020-2035
#       https://www.iea.org/reports/energy-and-ai/energy-supply-for-ai
#     The 2020 point is derived from the 270 TWh demand at a ~0.55 kg/kWh
#     global grid intensity; emissions grow more slowly than electricity
#     because the grid keeps decarbonising.
# ---------------------------------------------------------------------------
# (year, Mt CO2, reported?)
DC_CO2_MT = [
    (2020, 150.0, False),  # derived: 270 TWh x ~0.55 kg/kWh
    (2024, 180.0, True),   # IEA: ~180 Mt "today"
    (2030, 320.0, True),   # IEA: peak ~320 Mt by 2030
    (2035, 300.0, True),   # IEA: ~300 Mt by 2035 (shallow decline)
]

# ---------------------------------------------------------------------------
# (C) Carbon PRICE per tonne over time -- two independent valuations.
# ---------------------------------------------------------------------------
# (C1) EU ETS allowance, annual average [EUR/tonne CO2].  Traded market price.
#      EU Commission / EEA / Statista.  Rose ~13x from 2017 to its 2023 peak.
#        https://climate.ec.europa.eu/eu-action/carbon-markets/about-eu-ets_en
#        https://www.statista.com/statistics/1322214/
#          carbon-prices-european-union-emission-trading-scheme/
#        https://tradingeconomics.com/commodity/carbon
# (year, EUR/tonne)
EU_ETS_EUR = [
    (2017,  5.8),
    (2018, 16.0),
    (2019, 25.0),
    (2020, 25.0),   # EU Commission: annual avg ~EUR 25 in 2020
    (2021, 54.0),
    (2022, 81.0),
    (2023, 84.0),   # 2022-23 peak ~EUR 80 (energy crisis + tighter cap)
    (2024, 65.0),   # EU Commission: annual avg EUR 65 in 2024
]
EUR_TO_USD = 1.08   # approximate 2024 conversion for the USD axis

# (C2) US social cost of carbon, central estimate [USD/tonne CO2].
#      The regulatory dollar value assigned to one tonne of CO2.  Rose ~9x
#      from the first 2010 IWG estimate to the 2023 EPA update.
#        https://www.epa.gov/system/files/documents/2023-12/
#          epa_scghg_2023_report_final.pdf
#        https://www.rff.org/publications/explainers/social-cost-carbon-101/
#        https://en.wikipedia.org/wiki/Social_cost_of_carbon
# (year_of_estimate, USD/tonne, label)
SCC_USD = [
    (2010,  21.0, "IWG 2010 (3% disc.)"),
    (2016,  42.0, "IWG 2016 (3% disc.)"),
    (2023, 190.0, "EPA 2023 (2% disc.)"),
]

# Voluntary-market carbon price used as the model anchor in the paper.
#   https://ecosystemmarketplace.com  (see \cite{ecosystemmarketplace2023})
VOLUNTARY_USD = 50.0


def _interp(series_xy, year):
    """Piecewise-linear interpolation/extrapolation over (x, y) pairs."""
    xs = [p[0] for p in series_xy]
    ys = [p[1] for p in series_xy]
    if year <= xs[0]:
        return ys[0]
    if year >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= year <= xs[i + 1]:
            t = (year - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def co2_mt(year):
    """IEA Base-Case data-centre CO2 emissions [Mt] at a given year."""
    return _interp([(y, v) for y, v, _ in DC_CO2_MT], year)


def eu_ets_usd(year):
    """EU ETS annual-average allowance price converted to USD/tonne."""
    return _interp(EU_ETS_EUR, year) * EUR_TO_USD


def scc_usd(year):
    """US social cost of carbon central estimate [USD/tonne]."""
    return _interp([(y, v) for y, v, _ in SCC_USD], year)
