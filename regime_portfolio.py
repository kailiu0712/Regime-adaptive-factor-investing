"""
regime_portfolio.py
===================
Derives factor portfolios from regime-split IC analysis and provides
economic interpretation of each factor's regime sensitivity.

Three portfolios
----------------
benchmark  – factors with strong, consistent IC across all market states.
             These work well in general and form the "default" portfolio.

normal     – factors with significantly better IC during normal (low-vol) regimes.
             Economic rationale: stable macro environment lets fundamental signals
             (value, quality, growth) play out over a typical holding horizon.

abnormal   – factors with significantly better IC during high-volatility episodes.
             Economic rationale: during market stress, risk-aversion and liquidity
             scrambles dominate; sentiment, ownership concentration, and
             microstructure signals (ILLIQ, shareholder count) often outperform.

Regime sensitivity score
------------------------
  sens = ICIR_abnormal - ICIR_normal

  High positive sens → abnormal factor (outperforms in stress).
  High negative sens → normal factor (outperforms in calm).
  Near zero         → regime-agnostic (benchmark candidate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ICIR_THRESHOLD, TSTAT_THRESHOLD, ALL_FACTORS, RES_DIR, REGIME_SENSITIVITY_THRESHOLD, TOP_K_RFF_FACTORS, TOP_K_DYNAMIC

# ── Qualitative descriptions for economic interpretation ─────────────────────
# These are filled based on the factor origin; used in the report.
FACTOR_DESCRIPTIONS = {
    # Valuation
    "EP_TTM":               "Earnings yield (TTM). Cheap stocks tend to outperform in calm markets.",
    "BP":                   "Book-to-price. Deep value factor; may lag in stress when fundamentals gap.",
    "EBIT2MV":              "EBIT/Market value. Enterprise value measure of cheapness.",
    "SP1_TTM":              "Sales yield (TTM). Revenue-based value; less distorted by leverage.",
    "REP_TTM":              "Cross-sectional rank of EP_TTM. Robust to outliers.",
    "RBP":                  "Cross-sectional rank of BP. Robust value rank.",
    # Momentum / liquidity
    "RT_2M":                "2-month price return. Medium-term momentum.",
    "RT_3M":                "3-month price return. Classic momentum horizon.",
    "RT_6M":                "6-month price return. Longer-term momentum; prone to reversal in stress.",
    "TO_1M":                "1-month average daily turnover. High turnover ↔ active trading / speculation.",
    "PRank_1M":             "Price level rank (1M). Relative price strength.",
    # Technical / oscillators
    "ACD5":                 "Accumulation/distribution 5-day. Short-term money-flow signal.",
    "AmountIR5":            "Volume information ratio (5d). Abnormal volume relative to baseline.",
    "CCI5":                 "Commodity Channel Index (5d). Overbought/oversold momentum.",
    "StyleBias20":          "Style bias momentum (20d). Measures drift in factor exposures.",
    # High-frequency microstructure
    "ILLIQ":                "Amihud illiquidity. High ILLIQ stocks are harder to trade; liquidity premium.",
    "AptOutFlowRatio":      "Aggressive passive outflow ratio. Selling pressure signal.",
    "BCVP":                 "Buy-side composite volume pressure. Demand-supply imbalance.",
    "M4":                   "4th-order return moment (kurtosis proxy). Tail-risk of the order-flow.",
    # Profitability / quality
    "ROE":                  "Return on equity (latest annual). Core quality factor.",
    "ROATTM":               "Return on assets (TTM). Asset efficiency; less sensitive to leverage.",
    "GrossIncomeRatioTTM":  "Gross margin (TTM). Pricing power; stable across economic cycles.",
    # Growth
    "E_Growth2":            "YoY earnings growth (TTM). Forward-looking quality signal.",
    "OR_Growth2":           "YoY revenue growth (TTM). Top-line growth momentum.",
    # Ownership
    "SHNum":                "Number of shareholders (inverse = concentrated ownership → strong hands).",
    "FundsHoldPropT":       "Fund holding proportion. Institutional demand; smart-money proxy.",
    # Risk / dividend
    "DivR1":                "Dividend yield (1-year). Income floor; valued in risk-off periods.",
    "VOL_orth_ZS":          "Idiosyncratic volatility (orthogonalised). Low-vol anomaly.",
    "Beta_orth_ZS":         "Market beta (orthogonalised). Systematic risk exposure.",
    # Analyst
    "Expect_EP":            "Analyst consensus earnings yield. Forward PE inversion.",
    "Expect_E_Growth":      "Analyst consensus EPS growth. Expectations-driven momentum.",
    "Rating":               "Analyst composite rating score. Sell-side sentiment.",
}


# ── Main classification function ──────────────────────────────────────────────

def classify_factors(
    summary_all: pd.DataFrame,
    summary_normal: pd.DataFrame,
    summary_abnormal: pd.DataFrame,
    icir_threshold: float = ICIR_THRESHOLD,
    tstat_threshold: float = TSTAT_THRESHOLD,
) -> tuple:
    """
    Classify factors into regime-specific pools.

    A factor enters a pool if both:
      |ICIR_{regime}| > icir_threshold  AND
      |t_stat_{regime}| > tstat_threshold

    Additionally computes a 'regime_sensitivity' score
      sens = ICIR_abnormal - ICIR_normal
    to quantify how strongly a factor tilts toward stress vs calm periods.

    Returns
    -------
    pools : dict with keys:
              'benchmark'       – top-K factors by |ICIR_all| (strong everywhere)
              'dynamic_normal'  – factors that weaken in abnormal (regime_sens < -thresh)
              'dynamic_abnormal'– factors that strengthen in abnormal (regime_sens > +thresh)
    signs : dict 'all'/'normal'/'abnormal' → pd.Series(factor → +1 or -1)
            +1 = long high-value stocks, −1 = long low-value stocks.
    """
    def _select(summary: pd.DataFrame) -> list:
        ok_icir  = summary["ICIR"].abs() > icir_threshold
        ok_tstat = summary["t_stat"].abs() > tstat_threshold
        return list(summary.index[ok_icir & ok_tstat])

    def _signs(summary: pd.DataFrame) -> pd.Series:
        return summary["ICIR"].apply(lambda x: 1 if x >= 0 else -1)

    # Broad pools (used for classification table only)
    _broad = {
        "benchmark": _select(summary_all),
        "normal":    _select(summary_normal),
        "abnormal":  _select(summary_abnormal),
    }
    signs = {
        "all":      _signs(summary_all),
        "normal":   _signs(summary_normal),
        "abnormal": _signs(summary_abnormal),
    }

    # ── Build regime-sensitivity table ────────────────────────────────────
    all_factors = list(
        set(summary_all.index)
        | set(summary_normal.index)
        | set(summary_abnormal.index)
    )

    rows = []
    for f in all_factors:
        icir_a = summary_all.loc[f, "ICIR"]        if f in summary_all.index      else np.nan
        t_a    = summary_all.loc[f, "t_stat"]      if f in summary_all.index      else np.nan
        icir_n = summary_normal.loc[f, "ICIR"]     if f in summary_normal.index   else np.nan
        t_n    = summary_normal.loc[f, "t_stat"]   if f in summary_normal.index   else np.nan
        icir_x = summary_abnormal.loc[f, "ICIR"]   if f in summary_abnormal.index else np.nan
        t_x    = summary_abnormal.loc[f, "t_stat"] if f in summary_abnormal.index else np.nan

        sens = (icir_x - icir_n) if (not np.isnan(icir_x) and not np.isnan(icir_n)) else np.nan

        abs_s = (abs(icir_x) - abs(icir_n)) if (not np.isnan(icir_x) and not np.isnan(icir_n)) else np.nan
        rows.append({
            "factor":             f,
            "ICIR_all":           round(icir_a, 4) if not np.isnan(icir_a) else np.nan,
            "t_all":              round(t_a,    4) if not np.isnan(t_a)    else np.nan,
            "ICIR_normal":        round(icir_n, 4) if not np.isnan(icir_n) else np.nan,
            "t_normal":           round(t_n,    4) if not np.isnan(t_n)    else np.nan,
            "ICIR_abnormal":      round(icir_x, 4) if not np.isnan(icir_x) else np.nan,
            "t_abnormal":         round(t_x,    4) if not np.isnan(t_x)    else np.nan,
            "regime_sensitivity": round(sens,   4) if not np.isnan(sens)   else np.nan,
            "abs_sensitivity":    round(abs_s,  4) if not np.isnan(abs_s)  else np.nan,
            "in_benchmark":       f in _broad["benchmark"],
            "in_normal":          f in _broad["normal"],
            "in_abnormal":        f in _broad["abnormal"],
            "description":        FACTOR_DESCRIPTIONS.get(f, ""),
        })

    cls_df = (
        pd.DataFrame(rows)
        .set_index("factor")
        .sort_values("regime_sensitivity", ascending=False)
    )

    # ── Investment pools (for the backtest) ───────────────────────────────
    # benchmark: top-K factors by |ICIR_all|, drawn from the broad benchmark set
    bench_cands = cls_df[cls_df["in_benchmark"]].copy()
    bench_cands["abs_icir_all"] = bench_cands["ICIR_all"].abs()
    k = TOP_K_RFF_FACTORS if TOP_K_RFF_FACTORS else len(bench_cands)
    benchmark_pool = bench_cands.nlargest(k, "abs_icir_all").index.tolist()

    # ── Dynamic pools: top-K by |ICIR_regime| ────────────────────────────
    # Select the best-performing factors *within* each regime rather than the
    # most regime-sensitive ones.  This gives each dynamic pool the same depth
    # as the benchmark (TOP_K_DYNAMIC factors) and includes regime-agnostic
    # strong factors (e.g. ILLIQ, AmountIR5) that also work well in that regime,
    # instead of restricting to the few factors at the sensitivity extremes.
    #
    # Prior approach (sensitivity threshold): picked 4 normal + 5 abnormal factors,
    # all heavily concentrated in value or momentum-reversal.  During the 2022-05
    # crash the thin abnormal pool (3 correlated momentum factors) failed together.

    # dynamic_normal: identical to benchmark — no deviation during calm markets.
    # This ensures Dynamic == Benchmark in normal regimes so the only variable
    # is what happens during abnormal episodes.
    dynamic_normal_pool = benchmark_pool

    # dynamic_abnormal: start from the benchmark sleeve and only replace the
    # weakest benchmark members with train-abnormal factors that are strictly
    # stronger on |ICIR_abnormal|.  This keeps the dynamic sleeve close to the
    # broad benchmark unless the abnormal candidate demonstrably improves the
    # train-period abnormal regime.
    abn_cands = cls_df[cls_df["in_abnormal"]].copy()
    abn_cands["abs_icir_abn"] = abn_cands["ICIR_abnormal"].abs()
    dynamic_abnormal_pool = benchmark_pool.copy()
    candidate_replacements = abn_cands.loc[
        ~abn_cands.index.isin(dynamic_abnormal_pool)
    ].sort_values("abs_icir_abn", ascending=False)
    weakest_benchmark = (
        abn_cands.reindex(dynamic_abnormal_pool)
        .assign(abs_icir_abn=lambda df: df["abs_icir_abn"].fillna(-np.inf))
        .sort_values("abs_icir_abn", ascending=True)
    )

    for new_factor in candidate_replacements.index.tolist():
        if len(dynamic_abnormal_pool) >= TOP_K_DYNAMIC and weakest_benchmark.empty:
            break

        weakest_factor = weakest_benchmark.index[0]
        weakest_score = weakest_benchmark.iloc[0]["abs_icir_abn"]
        new_score = candidate_replacements.loc[new_factor, "abs_icir_abn"]

        if not np.isfinite(new_score) or new_score <= weakest_score:
            continue

        drop_idx = dynamic_abnormal_pool.index(weakest_factor)
        dynamic_abnormal_pool[drop_idx] = new_factor
        weakest_benchmark = weakest_benchmark.drop(index=weakest_factor)

    dynamic_abnormal_pool = dynamic_abnormal_pool[:TOP_K_DYNAMIC]

    # Fallback
    if not dynamic_abnormal_pool:
        dynamic_abnormal_pool = benchmark_pool

    pools = {
        "benchmark":        benchmark_pool,
        "dynamic_normal":   dynamic_normal_pool,
        "dynamic_abnormal": dynamic_abnormal_pool,
    }

    RES_DIR.mkdir(parents=True, exist_ok=True)
    cls_df.to_csv(RES_DIR / "factor_classification.csv")

    # ── Print report ──────────────────────────────────────────────────────
    print("\n── Investment Factor Pools ───────────────────────────")
    for pool, factors in pools.items():
        print(f"  {pool:<20}: ({len(factors)}) {factors}")

    print("\n── Regime Sensitivity (ICIR_abnormal - ICIR_normal) ──")
    disp = cls_df[["ICIR_normal", "ICIR_abnormal", "regime_sensitivity",
                   "in_normal", "in_abnormal"]].round(3)
    print(disp.to_string())

    print("\n── Economic Interpretation ──────────────────────────")
    _print_economic_narrative(cls_df, pools)

    return pools, signs


def _print_economic_narrative(cls_df: pd.DataFrame, pools: dict):
    """
    Print a structured economic narrative about the factor classification.
    """
    abn_factors = cls_df[cls_df["in_abnormal"]].sort_values("ICIR_abnormal", ascending=False)
    nor_factors = cls_df[cls_df["in_normal"] & ~cls_df["in_abnormal"]].sort_values("ICIR_normal", ascending=False)
    bench       = cls_df[cls_df["in_benchmark"]].sort_values("ICIR_all", ascending=False)

    print("\n  BENCHMARK (top-K, RFF input) factors:")
    for f in pools.get("benchmark", []):
        row = cls_df.loc[f] if f in cls_df.index else {}
        icir = row.get("ICIR_all", np.nan) if isinstance(row, pd.Series) else np.nan
        print(f"    {f:25s}  ICIR_all={icir:.3f}  "
              f"| {cls_df.loc[f,'description'][:60] if f in cls_df.index else ''}")

    print("\n  DYNAMIC-NORMAL factors (strong in calm, weakens in stress):")
    for f in pools.get("dynamic_normal", []):
        row = cls_df.loc[f] if f in cls_df.index else {}
        sens = row.get("regime_sensitivity", np.nan) if isinstance(row, pd.Series) else np.nan
        icir_n = row.get("ICIR_normal", np.nan) if isinstance(row, pd.Series) else np.nan
        print(f"    {f:25s}  ICIR_norm={icir_n:.3f}  sens={sens:.2f}")

    print("\n  DYNAMIC-ABNORMAL factors (strengthens/activates in stress):")
    for f in pools.get("dynamic_abnormal", []):
        row = cls_df.loc[f] if f in cls_df.index else {}
        sens = row.get("regime_sensitivity", np.nan) if isinstance(row, pd.Series) else np.nan
        icir_x = row.get("ICIR_abnormal", np.nan) if isinstance(row, pd.Series) else np.nan
        print(f"    {f:25s}  ICIR_abn={icir_x:.3f}  sens={sens:.2f}")

    print("\n  BROAD-NORMAL factors:")
    for f in nor_factors.index:
        print(f"    {f:25s}  ICIR_norm={nor_factors.loc[f,'ICIR_normal']:.3f}  "
              f"| {cls_df.loc[f,'description'][:60]}")

    print("\n  BROAD-ABNORMAL factors:")
    for f in abn_factors.index:
        print(f"    {f:25s}  ICIR_abn={abn_factors.loc[f,'ICIR_abnormal']:.3f}  "
              f"| {cls_df.loc[f,'description'][:70]}")
