"""
ICIR-weighted composite factor backtest.

This path replaces the rolling RFF model with a transparent composite score:
each factor is ranked cross-sectionally on the signal date and combined using
signed ICIR weights estimated on the training sample only.

Key implementation rules
------------------------
1. Factor-pool selection happens upstream on train data only.
2. ICIR weights are fit once on the full train window and frozen in test.
3. Signal date t is traded for holdings from t+1 onward; same-day return is
   never booked from information observed on date t.
4. Benchmark rebalances on the fixed schedule only.
5. Dynamic rebalances on the fixed schedule plus any regime flip.
6. On normal regimes, Dynamic copies the standing Benchmark portfolio by
   construction; on abnormal regimes it switches to the abnormal sleeve.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    REBALANCE_FREQ,
    TOP_QUANTILE,
    TRANS_COST,
    TRAIN_END,
    TEST_START,
    TEST_END,
)
from backtest import one_way_turnover, plot_backtest_results, top_q_weights


def extract_icir_weights(
    ic_series: pd.DataFrame,
    factor_pool: list[str],
    train_end: str = TRAIN_END,
    regime: str = "all",
) -> dict[str, float]:
    """
    Compute signed ICIR weights from the in-sample window only.
    """
    train_ic = ic_series[ic_series.index <= pd.Timestamp(train_end)]
    weights: dict[str, float] = {}

    for factor in factor_pool:
        if factor not in train_ic.columns:
            continue
        col = train_ic[factor].dropna()
        if len(col) < 30 or col.std() < 1e-12:
            continue
        weights[factor] = float(col.mean() / col.std() * np.sqrt(252))

    if weights:
        ranked = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
        print(f"  ICIR weights [{regime}] ({len(weights)} factors):")
        for factor, weight in ranked[:8]:
            print(f"    {factor:25s}: {weight:+.3f}")

    return weights


def icir_score_date(
    date_df: pd.DataFrame,
    factor_cols: list[str],
    icir_weights: dict[str, float],
) -> pd.Series:
    """
    Compute the ICIR-weighted composite score for one cross-section.
    """
    valid = date_df.copy()
    if valid.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(0.0, index=valid.index)
    total_weight = 0.0

    for col in factor_cols:
        weight = icir_weights.get(col, 0.0)
        if abs(weight) < 0.05 or col not in valid.columns:
            continue

        vals = valid[col]
        median = vals.median()
        if pd.isna(median):
            continue

        vals = vals.fillna(median)
        ranks = vals.rank(pct=True)
        if weight < 0:
            ranks = 1.0 - ranks

        scores += abs(weight) * ranks
        total_weight += abs(weight)

    if total_weight < 1e-10:
        return pd.Series(dtype=float)

    scores /= total_weight
    scores.index = valid["SecuCode"].values
    scores.name = "score"
    return scores


def run_icir_backtest(
    panel: pd.DataFrame,
    regime_df: pd.DataFrame,
    factor_pools: dict,
    ic_summaries: dict,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
    rebalance_freq: int = REBALANCE_FREQ,
    top_q: float = TOP_QUANTILE,
    train_end: str = TRAIN_END,
    trans_cost: float = TRANS_COST,
) -> dict:
    """
    Backtest the benchmark and dynamic ICIR strategies.

    Trading timeline:
    - use signal observed on date t
    - trade at the close of t
    - hold new weights from t+1 onward

    This avoids the prior same-day look-ahead bug.
    """
    print("\nExtracting training-period ICIR weights")
    bench_w = extract_icir_weights(
        ic_summaries["all"],
        factor_pools["benchmark"],
        train_end,
        "benchmark",
    )
    _norm_w = extract_icir_weights(
        ic_summaries["normal"],
        factor_pools.get("dynamic_normal", factor_pools["benchmark"]),
        train_end,
        "dynamic_normal",
    )
    abn_w = extract_icir_weights(
        ic_summaries["abnormal"],
        factor_pools.get("dynamic_abnormal", factor_pools["benchmark"]),
        train_end,
        "dynamic_abnormal",
    )

    all_dates = sorted(panel["TradingDay"].unique())
    test_dates = [
        d for d in all_dates
        if pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end)
    ]
    if len(test_dates) < 2:
        raise ValueError("Need at least two test dates to run the next-day backtest.")

    regime_map = dict(zip(regime_df["Date"], regime_df["regime"]))
    ret_panel = panel.set_index(["TradingDay", "SecuCode"])["ret"]

    bench_cols = [c for c in factor_pools["benchmark"] if c in panel.columns]
    norm_cols = [
        c for c in factor_pools.get("dynamic_normal", factor_pools["benchmark"])
        if c in panel.columns
    ]
    abn_cols = [
        c for c in factor_pools.get("dynamic_abnormal", factor_pools["benchmark"])
        if c in panel.columns
    ]

    test_panel = panel[panel["TradingDay"].isin(test_dates)]
    panel_by_date = {
        date: grp.copy()
        for date, grp in test_panel.groupby("TradingDay", sort=False)
    }

    benchmark_signal_idx = set(range(0, len(test_dates) - 1, rebalance_freq))
    dynamic_signal_idx = set(benchmark_signal_idx)
    for idx in range(1, len(test_dates) - 1):
        if regime_map.get(test_dates[idx], 0) != regime_map.get(test_dates[idx - 1], 0):
            dynamic_signal_idx.add(idx)

    print(f"\nICIR Backtest: {test_start} -> {test_end} ({len(test_dates)} days)")
    print(f"  ICIR weights frozen from train sample through {train_end}")
    print(f"  Benchmark rebalance: every {rebalance_freq} days")
    print("  Dynamic rebalance: benchmark schedule + any in-period regime flip")
    print(
        f"  Top {int(top_q*100)}% | TC={trans_cost:.4f} per side "
        f"({'gross' if trans_cost == 0 else 'net'})"
    )
    print(
        f"  Benchmark pool: {len(bench_cols)} factors | "
        f"Normal: {len(norm_cols)} | Abnormal: {len(abn_cols)}"
    )

    daily_rets = {"benchmark": {}, "dynamic": {}}
    curr_w = {"benchmark": pd.Series(dtype=float), "dynamic": pd.Series(dtype=float)}
    turnovers = {"benchmark": [], "dynamic": []}

    for idx, day in enumerate(tqdm(test_dates, desc="ICIR daily loop")):
        day_ret = {"benchmark": 0.0, "dynamic": 0.0}

        try:
            stock_rets = ret_panel.xs(day)
        except KeyError:
            stock_rets = pd.Series(dtype=float)

        for strat in ("benchmark", "dynamic"):
            weights = curr_w[strat]
            if weights.empty or stock_rets.empty:
                continue
            common = weights.index.intersection(stock_rets.index)
            day_ret[strat] = float((weights[common] * stock_rets[common]).sum())

        if idx < len(test_dates) - 1:
            today_df = panel_by_date.get(day, pd.DataFrame())
            next_bench_w: pd.Series | None = None

            if idx in benchmark_signal_idx:
                sigs_b = icir_score_date(today_df, bench_cols, bench_w)
                next_bench_w = (
                    top_q_weights(sigs_b, top_q) if not sigs_b.empty else pd.Series(dtype=float)
                )
                turnover_b = one_way_turnover(curr_w["benchmark"], next_bench_w)
                turnovers["benchmark"].append(turnover_b)
                day_ret["benchmark"] -= turnover_b * trans_cost
                curr_w["benchmark"] = next_bench_w.copy()

            if idx in dynamic_signal_idx:
                regime_now = regime_map.get(day, 0)
                if regime_now == 0:
                    # On normal dates Dynamic must hold the same portfolio as
                    # Benchmark, either the freshly rebalanced one or the
                    # existing benchmark holdings.
                    next_dyn_w = (
                        curr_w["benchmark"].copy()
                        if next_bench_w is None
                        else next_bench_w.copy()
                    )
                else:
                    sigs_d = icir_score_date(today_df, abn_cols, abn_w)
                    next_dyn_w = (
                        top_q_weights(sigs_d, top_q) if not sigs_d.empty else pd.Series(dtype=float)
                    )

                turnover_d = one_way_turnover(curr_w["dynamic"], next_dyn_w)
                turnovers["dynamic"].append(turnover_d)
                day_ret["dynamic"] -= turnover_d * trans_cost
                curr_w["dynamic"] = next_dyn_w.copy()

            del today_df
            gc.collect()

        for strat in ("benchmark", "dynamic"):
            daily_rets[strat][day] = day_ret[strat]

    for strat in ("benchmark", "dynamic"):
        if turnovers[strat]:
            mean_turnover = float(np.mean(turnovers[strat]))
            total_tc = sum(turnovers[strat]) * trans_cost
            print(
                f"\n  {strat.capitalize()} turnover: avg {mean_turnover*100:.1f}% per trade event"
                f" | Total TC drag: {total_tc*100:.2f}% over period"
            )

    out = {}
    for strat in ("benchmark", "dynamic"):
        series = pd.Series(daily_rets[strat]).sort_index()
        series.index.name = "Date"
        series.name = strat
        out[strat] = series

    market = panel.groupby("TradingDay")["ret"].mean()
    market = market[
        [
            d for d in market.index
            if pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end)
        ]
    ]
    market = market.reindex(out["benchmark"].index).fillna(0.0)
    market.name = "market"
    out["market"] = market

    return out
