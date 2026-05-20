"""
run_pipeline.py
===============
Master orchestrator for the abnormal-regime factor analysis pipeline.

Usage
-----
  cd abnormal_regime_analysis
  python run_pipeline.py

Stages
------
  1. Load all factor data                → panel DataFrame
  2. GARCH model selection               → best (p,1) order
  3. Regime detection                    → daily regime labels (normal/abnormal)
  4. GARCH vs manual comparison          → garch_vs_manual.csv
  5. Factor analysis                     → IC / ICIR / t-stats per regime
  6. Factor classification               → benchmark / normal / abnormal pools
  6b. Regime clustering                  → UMAP + HDBSCAN regime separation plots
  7. Factor coefficient t-stats (FMB)   → fmb_tstats.csv
  8. Rolling backtest (OOS)             → cumulative PnL & performance metrics

All outputs are written to  outputs/results/  and  outputs/plots/.
"""

from __future__ import annotations

import sys
import time
import gc
from pathlib import Path

import pandas as pd

# Add the parent directory to sys.path so this can be run from the project root too
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALL_FACTORS, GARCH_Q,
    TRAIN_START, TRAIN_END,
    OUT_DIR, RES_DIR, PLOT_DIR,
    TARGET_COL,
)
from data_loader     import load_all_factors, compute_market_return
from garch_regime    import (select_garch_order, fit_garch_and_label_regimes,
                              compare_with_manual, plot_garch_regimes)
from factor_analysis import (run_factor_analysis, save_factor_report,
                              plot_quantile_returns, plot_ic_timeseries,
                              factor_coefficient_tstats, summarize_ic_window)
from regime_portfolio   import classify_factors
from regime_clustering  import run_regime_clustering
from backtest           import plot_backtest_results
from icir_predictor     import run_icir_backtest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str, stage: int):
    print(f"\n{'='*60}")
    print(f" STAGE {stage}: {title}")
    print(f"{'='*60}")


def _ensure_dirs():
    for d in (OUT_DIR, RES_DIR, PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    _ensure_dirs()

    # ── Stage 1: Data loading ──────────────────────────────────────────────
    _banner("Load factor data", 1)
    panel = load_all_factors()
    market_ret = compute_market_return(panel)

    avail_factors = [f for f in ALL_FACTORS if f in panel.columns]
    print(f"Active factors: {avail_factors}")

    # ── Stage 2: GARCH model selection ────────────────────────────────────
    _banner("GARCH model selection", 2)

    # Use only the training period for model-order selection
    train_ret = market_ret[
        (market_ret.index >= pd.Timestamp(TRAIN_START)) &
        (market_ret.index <= pd.Timestamp(TRAIN_END))
    ]
    best_p = select_garch_order(train_ret)

    # ── Stage 3: Regime detection (full sample) ────────────────────────────
    _banner("Regime detection (full sample GARCH)", 3)
    regime_df = fit_garch_and_label_regimes(market_ret, p=best_p, q=GARCH_Q)
    regime_df.to_csv(RES_DIR / "regime_labels.csv", index=False)
    print(f"  Saved: regime_labels.csv")

    plot_garch_regimes(market_ret, regime_df)

    # ── Stage 4: GARCH vs manual comparison ───────────────────────────────
    _banner("GARCH vs manually annotated abnormal intervals", 4)
    compare_with_manual(regime_df)

    # ── Stage 5: Factor analysis ──────────────────────────────────────────
    _banner("Cross-sectional factor analysis", 5)
    fa_results = run_factor_analysis(panel, avail_factors, regime_df)
    report_df  = save_factor_report(fa_results)

    # Quantile return plots for all regimes
    plot_quantile_returns(panel, avail_factors, regime_df)

    # IC time-series for top factors
    plot_ic_timeseries(fa_results["ic_all"], regime_df)

    # Fama-MacBeth coefficient t-stats (additional significance evidence)
    _banner("Fama-MacBeth coefficient t-statistics", 5.5)
    factor_coefficient_tstats(panel, avail_factors, regime_df)

    # ── Stage 6: Factor classification ────────────────────────────────────
    _banner("Factor pool classification", 6)
    train_summary_all = summarize_ic_window(
        fa_results["ic_all"], "all", end_date=TRAIN_END
    )
    train_summary_normal = summarize_ic_window(
        fa_results["ic_normal"], "normal", end_date=TRAIN_END
    )
    train_summary_abnormal = summarize_ic_window(
        fa_results["ic_abnormal"], "abnormal", end_date=TRAIN_END
    )
    factor_pools, factor_signs = classify_factors(
        train_summary_all,
        train_summary_normal,
        train_summary_abnormal,
    )

    # Safety fallback: if any investment pool is empty, use benchmark
    for key in ("dynamic_normal", "dynamic_abnormal"):
        if not factor_pools.get(key):
            print(f"  Warning: '{key}' pool is empty – falling back to benchmark factors.")
            factor_pools[key] = factor_pools["benchmark"]

    gc.collect()

    # ── Stage 6b: Regime clustering ────────────────────────────────────────
    _banner("Regime clustering (UMAP + HDBSCAN)", 6)
    run_regime_clustering(
        ic_all           = fa_results["ic_all"],
        regime_df        = regime_df,
        factor_pools     = factor_pools,
        summary_normal   = fa_results["summary_normal"],
        summary_abnormal = fa_results["summary_abnormal"],
    )
    gc.collect()

    # ── Stage 7: ICIR composite backtest — gross AND net ──────────────────
    _banner("ICIR composite backtest (model-free, daily-fresh signal)", 7)

    ic_sums = {
        "all":      fa_results["ic_all"],
        "normal":   fa_results["ic_normal"],
        "abnormal": fa_results["ic_abnormal"],
    }

    # ── 7a: Gross returns (no friction) ────────────────────────────────
    print("\n--- Gross (no transaction cost) ---")
    bt_gross = run_icir_backtest(
        panel        = panel,
        regime_df    = regime_df,
        factor_pools = factor_pools,
        ic_summaries = ic_sums,
        trans_cost   = 0.0,
    )
    pd.DataFrame(bt_gross).to_csv(RES_DIR / "backtest_gross_returns.csv")
    metrics_gross = plot_backtest_results(bt_gross, regime_df, label="gross")

    # ── 7b: Net returns (with friction) ────────────────────────────────
    from config import TRANS_COST as _TC
    print(f"\n--- Net (TC = {_TC:.4f} per side) ---")
    bt_net = run_icir_backtest(
        panel        = panel,
        regime_df    = regime_df,
        factor_pools = factor_pools,
        ic_summaries = ic_sums,
        trans_cost   = _TC,
    )
    pd.DataFrame(bt_net).to_csv(RES_DIR / "backtest_net_returns.csv")
    metrics_net = plot_backtest_results(bt_net, regime_df, label="net")

    # ── Summary comparison ──────────────────────────────────────────────
    print("\n── Gross vs Net Comparison ───────────────────────────────────────")
    comp = pd.concat(
        {"gross": metrics_gross.iloc[:, :2], "net": metrics_net.iloc[:, :2]},
        axis=1,
    )
    print(comp.to_string())

    # Also keep legacy filename pointing to net (default)
    bt_returns = bt_net
    pd.DataFrame(bt_returns).to_csv(RES_DIR / "backtest_daily_returns.csv")

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f" Pipeline complete in {elapsed/60:.1f} minutes")
    print(f" Results → {RES_DIR.resolve()}")
    print(f" Plots   → {PLOT_DIR.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
