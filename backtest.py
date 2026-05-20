"""
backtest.py
===========
Rolling cross-sectional backtest engine.

Two strategies compared OOS (2022-2024):
──────────────────────────────────────────────────────────────────────────────
Benchmark  Uses the 'benchmark' factor pool (factors informative across all
           market states) at every rebalance, regardless of regime.

Dynamic    Uses the 'normal' factor pool during calm regimes; automatically
           switches to the 'abnormal' factor pool when the GARCH conditional
           sigma signals an elevated-volatility episode.
           → Embodies the regime-adaptive hypothesis from the factor analysis.
──────────────────────────────────────────────────────────────────────────────

Rolling scheme (no look-ahead)
──────────────────────────────
- Refit every REBALANCE_FREQ trading days (default: 21 ≈ monthly).
- Training window: TRAIN_WINDOW past trading days (default: 252 ≈ 1 year).
- At refit date t:
    a. Determine regime from GARCH sigma at t   (sigma computed on [0, t-1]).
    b. Choose factor pool based on regime.
    c. Train RFF on panel rows with TradingDay ∈ [t-W, t).
    d. Predict scores for all stocks on date t.
    e. Select top-q by score, equal weight.
    f. Hold until next refit (days [t, t+REBALANCE_FREQ)).

Transaction costs
─────────────────
0.1 % one-way applied on the day positions change (refit day).
Cost = one_way_turnover × TRANS_COST
  where one_way_turnover = 0.5 × Σ|w_new - w_old|.

Newly-listed / delisted stocks
────────────────────────────────
At each refit we take the intersection of predicted stocks with those having
valid returns for the current day.  Stocks that drop out mid-hold contribute
their realised return up to delisting; their weight is redistributed on the
next refit.  This is handled implicitly by equal-weighting at each refit.

Performance metrics (annualised, 252 trading days)
───────────────────────────────────────────────────
Total return, annualised return, annualised volatility, Sharpe ratio,
Max drawdown, Calmar ratio, Win rate, Skewness, Kurtosis.
"""

from __future__ import annotations

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config import (
    REBALANCE_FREQ, TRAIN_WINDOW, TOP_QUANTILE,
    TRANS_COST, TEST_START, TEST_END, TARGET_COL,
    PLOT_DIR, RES_DIR,
)
from ml_predictor import RFFRegressor, prepare_training_data, predict_cross_sectional


# ── Position utilities ────────────────────────────────────────────────────────

def top_q_weights(signals: pd.Series, q: float = TOP_QUANTILE) -> pd.Series:
    """
    Select top-q fraction of stocks by signal; equal-weight portfolio.
    Returns weight Series indexed by SecuCode.
    """
    if signals.empty:
        return pd.Series(dtype=float)
    n_top = max(1, int(len(signals) * q))
    top   = signals.nlargest(n_top).index
    return pd.Series(1.0 / n_top, index=top, name="weight")


def one_way_turnover(prev: pd.Series, curr: pd.Series) -> float:
    """
    One-way portfolio turnover = 0.5 × Σ|w_curr - w_prev|.
    All stocks not in prev/curr are treated as zero-weight.
    """
    idx   = prev.index.union(curr.index)
    p, c  = prev.reindex(idx).fillna(0.0), curr.reindex(idx).fillna(0.0)
    return float(0.5 * (c - p).abs().sum())


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    panel: pd.DataFrame,
    regime_df: pd.DataFrame,
    factor_pools: dict,
    test_start: str = TEST_START,
    test_end:   str = TEST_END,
    rebalance_freq: int = REBALANCE_FREQ,
    train_window:   int = TRAIN_WINDOW,
    top_q:          float = TOP_QUANTILE,
    target_col:     str = TARGET_COL,
) -> dict:
    """
    Rolling backtest for Benchmark and Dynamic strategies.

    Parameters
    ----------
    panel          : full factor panel (TradingDay, SecuCode, factors, ret, excess_ret)
    regime_df      : DataFrame with columns [Date, sigma, expand_threshold, regime]
    factor_pools   : dict with keys:
                       'benchmark'        – factors for the benchmark strategy
                       'dynamic_normal'   – factors used when regime==0
                       'dynamic_abnormal' – factors used when regime==1
    test_start/end : OOS evaluation window
    rebalance_freq : days between refits
    train_window   : lookback in trading days
    top_q          : top quantile fraction to hold
    target_col     : training target ('excess_ret' for market-neutral, 'next_ret' for raw)

    Returns
    -------
    dict with keys 'benchmark', 'dynamic', 'market' → pd.Series of daily returns.
      'market' is the equal-weighted cross-sectional return (for alpha computation).
    """
    # ── Prep ──────────────────────────────────────────────────────────────
    all_dates  = sorted(panel["TradingDay"].unique())
    test_dates = [d for d in all_dates
                  if pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end)]

    regime_map = dict(zip(regime_df["Date"], regime_df["regime"]))  # date → 0/1

    bench_cols = factor_pools.get("benchmark",        [])
    norm_cols  = factor_pools.get("dynamic_normal",   []) or bench_cols
    abn_cols   = factor_pools.get("dynamic_abnormal", []) or bench_cols

    # Filter to columns that actually exist in panel
    bench_cols = [c for c in bench_cols if c in panel.columns]
    norm_cols  = [c for c in norm_cols  if c in panel.columns]
    abn_cols   = [c for c in abn_cols   if c in panel.columns]

    if not bench_cols:
        raise ValueError("Benchmark factor pool is empty after filtering for available columns.")

    print(f"\nBacktest window : {test_start} → {test_end}  ({len(test_dates)} days)")
    print(f"Rebalance freq  : every {rebalance_freq} days")
    print(f"Train window    : {train_window} days")
    print(f"Top quantile    : top {int(top_q*100)}% of stocks")
    print(f"Training target : {target_col}")
    print(f"Benchmark factors ({len(bench_cols)}): {bench_cols}")
    print(f"Dynamic-normal  factors ({len(norm_cols)}): {norm_cols}")
    print(f"Dynamic-abnormal factors ({len(abn_cols)}): {abn_cols}")

    # Refit schedule: indices into test_dates
    refit_idx = list(range(0, len(test_dates), rebalance_freq))

    daily_rets   = {"benchmark": {}, "dynamic": {}}
    prev_weights = {"benchmark": pd.Series(dtype=float),
                    "dynamic":   pd.Series(dtype=float)}
    curr_weights = {"benchmark": pd.Series(dtype=float),
                    "dynamic":   pd.Series(dtype=float)}

    # Date-indexed return lookup
    ret_panel = panel.set_index(["TradingDay", "SecuCode"])["ret"]

    # ── Refit loop ────────────────────────────────────────────────────────
    for iter_i, ri in enumerate(tqdm(refit_idx, desc="Refit iterations")):
        pred_date = test_dates[ri]

        # Holding period: this refit → next refit
        next_ri   = refit_idx[iter_i + 1] if iter_i + 1 < len(refit_idx) else len(test_dates)
        hold_days = test_dates[ri: next_ri]

        # ── Regime label at pred_date ─────────────────────────────────
        regime_now = regime_map.get(pred_date, 0)   # 1=abnormal, 0=normal
        dyn_cols   = abn_cols if regime_now == 1 else norm_cols

        # ── Gather training panel ─────────────────────────────────────
        train_dates = [d for d in all_dates if d < pred_date][-train_window:]
        if len(train_dates) < 60:
            continue   # not enough history

        train_panel = panel[panel["TradingDay"].isin(train_dates)]

        # ── Fit benchmark RFF ─────────────────────────────────────────
        Xb, yb, _ = prepare_training_data(train_panel, bench_cols, target_col=target_col)
        if Xb.shape[0] >= 50:
            model_b = RFFRegressor()
            model_b.fit(Xb, yb)
            today_df      = panel[panel["TradingDay"] == pred_date]
            sigs_b        = predict_cross_sectional(model_b, today_df, bench_cols)
            curr_weights["benchmark"] = top_q_weights(sigs_b, top_q)
            del model_b
        del Xb, yb; gc.collect()

        # ── Fit dynamic RFF ───────────────────────────────────────────
        Xd, yd, _ = prepare_training_data(train_panel, dyn_cols, target_col=target_col)
        if Xd.shape[0] >= 50:
            model_d = RFFRegressor()
            model_d.fit(Xd, yd)
            today_df      = panel[panel["TradingDay"] == pred_date]
            sigs_d        = predict_cross_sectional(model_d, today_df, dyn_cols)
            curr_weights["dynamic"] = top_q_weights(sigs_d, top_q)
            del model_d
        del Xd, yd, train_panel; gc.collect()

        # ── Compute portfolio returns over holding period ──────────────
        for h_day in hold_days:
            # Stock-level return on h_day
            try:
                day_rets = ret_panel.xs(h_day)
            except KeyError:
                for strat in ("benchmark", "dynamic"):
                    daily_rets[strat][h_day] = 0.0
                continue

            for strat in ("benchmark", "dynamic"):
                w = curr_weights[strat]
                if w.empty:
                    daily_rets[strat][h_day] = 0.0
                    continue

                common   = w.index.intersection(day_rets.index)
                port_ret = float((w[common] * day_rets[common]).sum())

                # Transaction cost on the refit day only
                if h_day == hold_days[0]:
                    tc = one_way_turnover(prev_weights[strat], w) * TRANS_COST
                    port_ret -= tc
                    prev_weights[strat] = w.copy()

                daily_rets[strat][h_day] = port_ret

    # Convert to sorted Series
    out = {}
    for strat in ("benchmark", "dynamic"):
        s = pd.Series(daily_rets[strat]).sort_index()
        s.index.name = "Date"
        s.name = strat
        out[strat] = s

    # Equal-weighted market return (used to compute alpha)
    mkt_raw = panel.groupby("TradingDay")["ret"].mean()
    mkt_oos = mkt_raw[[d for d in mkt_raw.index
                        if pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end)]]
    mkt_oos = mkt_oos.reindex(out["benchmark"].index).fillna(0.0)
    mkt_oos.name = "market"
    out["market"] = mkt_oos

    return out


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(ret: pd.Series, ann_factor: int = 252) -> dict:
    """
    Standard investment performance metrics (daily return Series as input).

    Metrics
    -------
    Total Return, Ann. Return, Ann. Volatility, Sharpe Ratio,
    Max Drawdown, Calmar Ratio, Win Rate, Skewness, Kurtosis, N Days.
    """
    ret     = ret.dropna()
    n       = len(ret)
    if n == 0:
        return {}

    total_r = float((1 + ret).prod() - 1)
    ann_r   = float((1 + total_r) ** (ann_factor / n) - 1)
    ann_v   = float(ret.std() * np.sqrt(ann_factor))
    sharpe  = ann_r / ann_v if ann_v > 1e-10 else np.nan

    cum     = (1 + ret).cumprod()
    peak    = cum.cummax()
    dd      = (cum - peak) / peak
    max_dd  = float(dd.min())
    calmar  = ann_r / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    return {
        "Total Return":    round(total_r, 4),
        "Ann. Return":     round(ann_r,   4),
        "Ann. Volatility": round(ann_v,   4),
        "Sharpe Ratio":    round(sharpe,  4),
        "Max Drawdown":    round(max_dd,  4),
        "Calmar Ratio":    round(calmar,  4),
        "Win Rate":        round(float((ret > 0).mean()), 4),
        "Skewness":        round(float(ret.skew()),       4),
        "Kurtosis":        round(float(ret.kurt()),       4),
        "N Days":          n,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def compute_alpha_metrics(ret: pd.Series, market: pd.Series, ann: int = 252) -> dict:
    """Alpha and Information Ratio versus the equal-weighted market."""
    common = ret.index.intersection(market.index)
    r, m   = ret[common], market[common]
    alpha  = r - m
    ann_alpha = alpha.mean() * ann
    te        = alpha.std() * np.sqrt(ann)
    ir        = ann_alpha / te if te > 1e-10 else np.nan
    return {
        "Ann. Alpha":  round(float(ann_alpha), 4),
        "Track. Error":round(float(te), 4),
        "Info. Ratio": round(float(ir), 4),
    }


def plot_backtest_results(
    bt_rets: dict,
    regime_df: pd.DataFrame,
    label: str = "",
) -> pd.DataFrame:
    """
    Generate four-panel figure:
      (A) Cumulative return (benchmark vs dynamic), regime shading
      (B) Cumulative alpha vs EW market
      (C) Alpha drawdown + rolling IR
      (D) Metrics table (absolute + alpha)

    label : filename suffix, e.g. "gross" or "net" → backtest_results_gross.png
    Returns metrics DataFrame.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True,  exist_ok=True)

    market  = bt_rets.get("market", None)
    strats  = {k: v for k, v in bt_rets.items() if k != "market"}
    palette = {"benchmark": "steelblue", "dynamic": "darkorange", "market": "grey"}

    import matplotlib.dates as mdates

    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2.5, 1.2, 1.2],
                            hspace=0.45, wspace=0.35)

    ax_cum   = fig.add_subplot(gs[0, :])      # top-full: cumulative PnL
    ax_alpha = fig.add_subplot(gs[1, :])      # mid-full: cumulative alpha vs market
    ax_dd    = fig.add_subplot(gs[2, 0])      # bottom-left: alpha drawdown
    ax_roll  = fig.add_subplot(gs[2, 1])      # bottom-mid: rolling IR
    ax_met   = fig.add_subplot(gs[2, 2])      # bottom-right: metrics

    # ── A: Cumulative PnL ─────────────────────────────────────────────
    if market is not None:
        cum_m = market.cumsum()
        ax_cum.plot(cum_m.index, cum_m.values,
                    label="EW Market", color="grey", linewidth=1.0,
                    linestyle="--", alpha=0.7)
    for strat, rets in strats.items():
        cum = rets.cumsum()
        ax_cum.plot(cum.index, cum.values,
                    label=strat.capitalize(), color=palette[strat], linewidth=1.5)

    # Shade OOS abnormal periods
    t0  = min(v.index.min() for v in strats.values())
    t1  = max(v.index.max() for v in strats.values())
    oos = regime_df[(regime_df["Date"] >= t0) & (regime_df["Date"] <= t1)]
    abn = oos.loc[oos["regime"] == 1, "Date"]

    _shade_spans(ax_cum, abn, color="salmon", alpha=0.18, label="Abnormal regime")

    # Manual intervals (yellow)
    from config import MANUAL_ABNORMAL_INTERVALS
    for i, (s, e) in enumerate(MANUAL_ABNORMAL_INTERVALS):
        s_, e_ = pd.Timestamp(s), pd.Timestamp(e)
        if e_ >= t0 and s_ <= t1:
            ax_cum.axvspan(s_, e_, color="gold", alpha=0.15,
                           label="Manual abnormal" if i == 0 else "")

    tc_note = "Gross (no friction)" if label == "gross" else f"Net (TC={TRANS_COST:.3f}/side)" if label == "net" else ""
    ax_cum.set_title(
        f"Cumulative Return  —  Benchmark vs Dynamic  "
        f"(OOS: {TEST_START[:7]} → {TEST_END[:7]})   {tc_note}",
        fontsize=11, pad=6)
    ax_cum.set_ylabel("Cumulative return (sum of daily returns)")
    ax_cum.legend(fontsize=9)
    ax_cum.grid(alpha=0.25)
    ax_cum.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── A2: Cumulative alpha vs EW market ──────────────────────────────
    for strat, rets in strats.items():
        alp_s = rets - (market if market is not None else pd.Series(0, index=rets.index))
        cum_a  = alp_s.cumsum()
        ax_alpha.plot(cum_a.index, cum_a.values,
                      label=strat.capitalize(), color=palette[strat], linewidth=1.3)
    ax_alpha.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    _shade_spans(ax_alpha, abn, color="salmon", alpha=0.15)
    ax_alpha.set_title("Cumulative Alpha vs EW Market  (isolates pure stock-selection effect)", fontsize=10)
    ax_alpha.set_ylabel("Cumulative excess return (sum)")
    ax_alpha.legend(fontsize=9)
    ax_alpha.grid(alpha=0.25)
    ax_alpha.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_alpha.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_alpha.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── B: Alpha Drawdown vs EW Market ────────────────────────────────
    for strat, rets in strats.items():
        alp  = rets - (market if market is not None else pd.Series(0, index=rets.index))
        cum_a= alp.cumsum()
        dd_a = cum_a - cum_a.cummax()   # absolute drawdown from running peak
        ax_dd.fill_between(dd_a.index, dd_a.values, 0,
                            alpha=0.35, color=palette[strat], label=strat.capitalize())
    ax_dd.set_title("Alpha Drawdown vs EW Market", fontsize=10)
    ax_dd.set_ylabel("Alpha Drawdown")
    ax_dd.legend(fontsize=8)
    ax_dd.grid(alpha=0.25)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    # ── C: Rolling 63-day Information Ratio ───────────────────────────
    win = 63
    for strat, rets in strats.items():
        alp = rets - (market if market is not None else pd.Series(0, index=rets.index))
        ir  = alp.rolling(win).mean() / alp.rolling(win).std() * np.sqrt(252)
        ax_roll.plot(ir.index, ir.values,
                     color=palette[strat], linewidth=0.9, label=strat.capitalize())
    ax_roll.axhline(0, color="black", linewidth=0.6)
    ax_roll.set_title(f"Rolling {win}-day Information Ratio vs Market", fontsize=10)
    ax_roll.legend(fontsize=8)
    ax_roll.grid(alpha=0.25)
    ax_roll.xaxis.set_major_locator(mdates.YearLocator())
    ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax_roll.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    # ── D: Combined metrics table ──────────────────────────────────────
    abs_m   = {s: compute_metrics(r) for s, r in strats.items()}
    alp_m   = {s: compute_alpha_metrics(r, market) for s, r in strats.items()} \
               if market is not None else {}
    combined= {}
    for s in strats:
        m = abs_m[s].copy()
        if s in alp_m:
            m.update(alp_m[s])
        combined[s] = m
    metrics_df = pd.DataFrame(combined)

    ax_met.axis("off")
    col_labels = ["Metric"] + list(metrics_df.columns)
    rows       = [[idx] + [str(v) for v in row] for idx, row in metrics_df.iterrows()]
    tbl        = ax_met.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.4)
    ax_met.set_title("Performance Metrics (absolute + alpha vs EW market)", fontsize=9, pad=2)

    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    out = PLOT_DIR / f"backtest_results{suffix}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    metrics_csv = RES_DIR / f"backtest_metrics{suffix}.csv"
    metrics_df.to_csv(metrics_csv)
    print(f"\n── Backtest Metrics ({label or 'default'}) ────────────────────────────")
    print(metrics_df.to_string())

    return metrics_df


def _shade_spans(ax, date_series: pd.Series, **kwargs):
    """Group consecutive dates into contiguous spans and call axvspan."""
    dates = sorted(date_series.dropna().tolist())
    if not dates:
        return
    spans, start = [], dates[0]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > 5:
            spans.append((start, dates[i - 1]))
            start = dates[i]
    spans.append((start, dates[-1]))
    label = kwargs.pop("label", "")
    for j, (s, e) in enumerate(spans):
        ax.axvspan(s, e, label=label if j == 0 else "", **kwargs)
