"""
factor_analysis.py
==================
Cross-sectional factor performance analysis.

Metrics computed per factor, per regime (all / normal / abnormal):
  - IC          : daily Spearman rank-correlation between factor & next-day return
  - IC mean/std : basic IC statistics
  - ICIR        : annualised IC / std(IC) * sqrt(252)
  - t-statistic : IC_mean / (IC_std / sqrt(n))  – tests H0: ICIR=0
  - % positive IC: fraction of dates with IC > 0

Additional outputs:
  - Quantile return plots (5-bin factor sort → avg next-day return)
  - Combined factor report CSV (all regimes side-by-side)

Economic interpretation guidance (see regime_portfolio.py):
  Factors with high ICIR only in the normal regime → 'value/quality' cycle.
  Factors that outperform during abnormal periods → 'stress/liquidity' plays.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from config import (
    N_QUANTILES, MIN_STOCKS, ICIR_THRESHOLD, TSTAT_THRESHOLD,
    ALL_FACTORS, PLOT_DIR, RES_DIR,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Cross-sectional normalisation ─────────────────────────────────────────────

def cs_rank_normalise(panel: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Within each TradingDay, convert each factor to its cross-sectional
    percentile rank ∈ (0, 1).  NaN stocks are excluded from ranking.

    Operates in-place on a copy; returns the new DataFrame.
    """
    df = panel.copy()

    def _pct_rank(x: pd.Series) -> pd.Series:
        return x.rank(pct=True)

    for col in cols:
        if col in df.columns:
            df[col] = df.groupby("TradingDay")[col].transform(_pct_rank)
    return df


# ── IC computation ────────────────────────────────────────────────────────────

def compute_daily_ic(
    panel: pd.DataFrame,
    factor_cols: list,
    target_col: str = "next_ret",
    date_filter: set | None = None,
) -> pd.DataFrame:
    """
    Compute Spearman IC for each factor on each trading date.

    Key design: dropna is applied PER FACTOR, not across all factors jointly.
    This ensures each factor uses its own maximum valid stock set, so a factor
    with sparse coverage (e.g. analyst data) does not reduce the sample for
    factors that have full coverage.

    Parameters
    ----------
    panel        : full factor panel (must contain TradingDay, factor_cols, target_col)
    factor_cols  : list of factor column names
    target_col   : forward return column (default: next_ret)
    date_filter  : set of dates to restrict computation (None → all dates)

    Returns
    -------
    DataFrame with index=TradingDay, columns=factor_cols, values=IC ∈ [-1, 1].
    NaN in a cell means insufficient valid stocks (<MIN_STOCKS) for that factor/date.
    """
    if date_filter is not None:
        sub = panel[panel["TradingDay"].isin(date_filter)]
    else:
        sub = panel

    # Only keep factors that exist in the panel
    factor_cols = [f for f in factor_cols if f in sub.columns]

    ic_rows = []
    for date, grp in tqdm(sub.groupby("TradingDay"), desc="IC", leave=False):
        row = {"TradingDay": date}
        has_any = False   # track whether at least one factor has valid IC this date

        for col in factor_cols:
            # Per-factor dropna: only require THIS factor and the target to be non-NaN
            valid = grp.dropna(subset=[col, target_col])
            if len(valid) < MIN_STOCKS or valid[col].std() < 1e-10:
                row[col] = np.nan
                continue
            r, _ = stats.spearmanr(valid[col], valid[target_col], nan_policy="omit")
            row[col] = float(r)
            has_any = True

        if has_any:
            ic_rows.append(row)

    if not ic_rows:
        return pd.DataFrame(columns=["TradingDay"] + factor_cols).set_index("TradingDay")

    return pd.DataFrame(ic_rows).set_index("TradingDay")


# ── Summary statistics ────────────────────────────────────────────────────────

def _ic_summary(ic_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Summarise an IC DataFrame into a table of statistics per factor.
    label: 'all', 'normal', or 'abnormal'
    """
    n       = ic_df.notna().sum()
    mean_ic = ic_df.mean()
    std_ic  = ic_df.std()
    icir    = mean_ic / std_ic * np.sqrt(252)
    tstat   = mean_ic / (std_ic / np.sqrt(n))
    pct_pos = (ic_df > 0).mean()

    summary = pd.DataFrame({
        "regime":   label,
        "n_dates":  n,
        "IC_mean":  mean_ic.round(4),
        "IC_std":   std_ic.round(4),
        "ICIR":     icir.round(4),
        "t_stat":   tstat.round(4),
        "pct_pos":  pct_pos.round(4),
    })
    summary.index.name = "factor"
    return summary


def summarize_ic_window(
    ic_df: pd.DataFrame,
    label: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Summarise IC statistics over a restricted date window.

    Used to keep factor-pool selection strictly in-sample while still allowing
    the exploratory report to cover the full history.
    """
    sub = ic_df
    if start_date is not None:
        sub = sub[sub.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        sub = sub[sub.index <= pd.Timestamp(end_date)]
    return _ic_summary(sub, label)


# ── Quantile return analysis ──────────────────────────────────────────────────

def compute_quantile_returns(
    panel: pd.DataFrame,
    factor_col: str,
    target_col: str = "next_ret",
    n_q: int = N_QUANTILES,
    date_filter: set | None = None,
) -> pd.Series:
    """
    Sort stocks into n_q bins by factor_col within each date.
    Return the time-averaged next-day return per quantile bin.

    Used for monotonicity plots: a well-behaved factor shows a
    smooth step pattern from Q1 (worst) to Q5 (best).
    """
    if date_filter is not None:
        sub = panel[panel["TradingDay"].isin(date_filter)]
    else:
        sub = panel

    bucket_rets = []
    for _, grp in sub.groupby("TradingDay"):
        valid = grp.dropna(subset=[factor_col, target_col])
        if len(valid) < n_q * 4:
            continue
        valid = valid.copy()
        valid["_q"] = pd.qcut(valid[factor_col], q=n_q, labels=False, duplicates="drop")
        qr = valid.groupby("_q")[target_col].mean()
        bucket_rets.append(qr)

    if not bucket_rets:
        return pd.Series(dtype=float)

    return pd.concat(bucket_rets, axis=1).mean(axis=1)


# ── Statistical significance: factor-level coefficient t-stats ────────────────

def factor_coefficient_tstats(
    panel: pd.DataFrame,
    factor_cols: list,
    regime_df: pd.DataFrame,
    target_col: str = "next_ret",
) -> pd.DataFrame:
    """
    Cross-sectional Fama-MacBeth style regression.

    For each date t:
      next_ret_i = alpha_t + beta_{j,t} * factor_{j,i,t} + eps_{i,t}

    t-stat for factor j = mean(beta_j) / se(beta_j) across dates.

    This provides coefficient-level significance tests independent of IC.
    Saves fmb_tstats.csv.

    Returns DataFrame with index=factor, columns=[mean_coef, t_stat, p_value] × regime.
    """
    from scipy import stats as scipy_stats

    abnormal_set = set(regime_df.loc[regime_df["regime"] == 1, "Date"])
    normal_set   = set(regime_df["Date"]) - abnormal_set

    factor_cols = [f for f in factor_cols if f in panel.columns]
    results = {}

    for regime_label, date_set in [
        ("all",      None),
        ("normal",   normal_set),
        ("abnormal", abnormal_set),
    ]:
        sub = panel if date_set is None else panel[panel["TradingDay"].isin(date_set)]
        betas = {f: [] for f in factor_cols}

        for _, grp in tqdm(
            sub.groupby("TradingDay"),
            desc=f"FMB [{regime_label}]",
            leave=False,
        ):
            valid = grp.dropna(subset=factor_cols + [target_col])
            if len(valid) < MIN_STOCKS + len(factor_cols):
                continue
            X = valid[factor_cols].values
            y = valid[target_col].values
            # OLS via lstsq (intercept via mean-centring)
            X_demeaned = X - X.mean(axis=0)
            try:
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(y)), X_demeaned]),
                    y,
                    rcond=None,
                )
                for k, f in enumerate(factor_cols):
                    betas[f].append(coef[k + 1])
            except np.linalg.LinAlgError:
                continue

        rows = []
        for f in factor_cols:
            b = np.array(betas[f])
            if len(b) < 5:
                rows.append({"factor": f, "mean_coef": np.nan, "t_stat": np.nan, "p_value": np.nan})
                continue
            t, p = scipy_stats.ttest_1samp(b, 0)
            rows.append({"factor": f, "mean_coef": round(b.mean(), 6),
                         "t_stat": round(float(t), 4), "p_value": round(float(p), 4)})

        df = pd.DataFrame(rows).set_index("factor")
        df.columns = [f"{c}_{regime_label}" for c in df.columns]
        results[regime_label] = df

    combined = pd.concat(results.values(), axis=1)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RES_DIR / "fmb_tstats.csv")
    print("  Saved: fmb_tstats.csv")
    return combined


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_factor_analysis(
    panel: pd.DataFrame,
    factor_cols: list,
    regime_df: pd.DataFrame,
) -> dict:
    """
    Run complete cross-sectional factor analysis split by regime.

    Steps
    -----
    1. Compute daily IC for all / normal / abnormal dates.
    2. Summarise: IC mean, std, ICIR, t-stat, % pos-IC.
    3. Fama-MacBeth coefficient t-stats per regime.

    Returns dict with keys:
      ic_all, ic_normal, ic_abnormal,
      summary_all, summary_normal, summary_abnormal
    """
    factor_cols = [f for f in factor_cols if f in panel.columns]

    all_dates     = set(panel["TradingDay"].unique())
    abnormal_set  = set(regime_df.loc[regime_df["regime"] == 1, "Date"])
    normal_set    = all_dates - abnormal_set

    print(f"  Dates → total: {len(all_dates)}, "
          f"normal: {len(normal_set)}, abnormal: {len(abnormal_set)}")

    print("Computing IC – all dates …")
    ic_all      = compute_daily_ic(panel, factor_cols)
    print("Computing IC – normal dates …")
    ic_normal   = compute_daily_ic(panel, factor_cols, date_filter=normal_set)
    print("Computing IC – abnormal dates …")
    ic_abnormal = compute_daily_ic(panel, factor_cols, date_filter=abnormal_set)

    s_all      = _ic_summary(ic_all,      "all")
    s_normal   = _ic_summary(ic_normal,   "normal")
    s_abnormal = _ic_summary(ic_abnormal, "abnormal")

    return {
        "ic_all": ic_all, "ic_normal": ic_normal, "ic_abnormal": ic_abnormal,
        "summary_all": s_all, "summary_normal": s_normal, "summary_abnormal": s_abnormal,
    }


def save_factor_report(results: dict) -> pd.DataFrame:
    """
    Merge regime summaries into one wide CSV and print a concise overview.
    Saves factor_ic_report.csv and per-regime IC time-series CSVs.
    """
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Wide table: one row per factor, columns prefixed by regime
    frames = []
    for key in ("all", "normal", "abnormal"):
        s = results[f"summary_{key}"].copy()
        s.columns = [f"{c}_{key}" if c != "regime" else c for c in s.columns]
        s = s.drop(columns=["regime"], errors="ignore")
        frames.append(s)

    wide = pd.concat(frames, axis=1)
    wide.index.name = "factor"
    wide.to_csv(RES_DIR / "factor_ic_report.csv")

    for key in ("all", "normal", "abnormal"):
        results[f"ic_{key}"].to_csv(RES_DIR / f"ic_{key}_series.csv")

    print("\n── Factor IC Report (all regimes) ────────────────────")
    display_cols = ["IC_mean_all", "ICIR_all", "t_stat_all",
                    "ICIR_normal", "t_stat_normal",
                    "ICIR_abnormal", "t_stat_abnormal"]
    print(wide[[c for c in display_cols if c in wide.columns]].round(3).to_string())

    return wide


# ── Quantile plots ────────────────────────────────────────────────────────────

def plot_quantile_returns(
    panel: pd.DataFrame,
    factor_cols: list,
    regime_df: pd.DataFrame,
    max_per_regime: int = 12,
):
    """
    For each regime (all / normal / abnormal), plot a grid of factor-sorted
    quantile bar charts showing average next-day return per quantile bin.

    A monotone increasing pattern (Q1 low → Q5 high) indicates a positive-IC factor.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    factor_cols   = [f for f in factor_cols if f in panel.columns]
    abnormal_set  = set(regime_df.loc[regime_df["regime"] == 1, "Date"])
    normal_set    = set(panel["TradingDay"].unique()) - abnormal_set

    regime_map = {
        "all":      None,
        "normal":   normal_set,
        "abnormal": abnormal_set,
    }

    for regime_label, date_filter in regime_map.items():
        n_plots = min(max_per_regime, len(factor_cols))
        ncols   = 3
        nrows   = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else list(axes)

        for i, col in enumerate(factor_cols[:n_plots]):
            qret = compute_quantile_returns(panel, col, date_filter=date_filter)
            ax   = axes[i]
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in qret.values]
            qret.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.7)
            ax.set_title(f"{col}", fontsize=9)
            ax.set_xlabel("Quantile (1=low, 5=high)", fontsize=7)
            ax.set_ylabel("Avg next-day ret", fontsize=7)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.tick_params(axis="x", rotation=0, labelsize=7)

        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Quantile Returns by Factor  ─  {regime_label.upper()} regime",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        out = PLOT_DIR / f"quantile_returns_{regime_label}.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── IC time-series plot ───────────────────────────────────────────────────────

def plot_ic_timeseries(
    ic_all: pd.DataFrame,
    regime_df: pd.DataFrame,
    top_n: int = 8,
):
    """
    Rolling 63-day IC smoothed time-series for the top-n factors (by |ICIR|).
    Shades abnormal periods.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    icir    = (ic_all.mean() / ic_all.std() * np.sqrt(252)).abs()
    top_fac = icir.nlargest(top_n).index.tolist()

    abnormal_set = set(regime_df.loc[regime_df["regime"] == 1, "Date"])

    ncols = 2
    nrows = (len(top_fac) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharex=False)
    axes = axes.flatten()

    for i, col in enumerate(top_fac):
        ax    = axes[i]
        series = ic_all[col].dropna()
        roll   = series.rolling(63, min_periods=20).mean()
        ax.plot(series.index, series.values, alpha=0.25, color="steelblue", linewidth=0.5)
        ax.plot(roll.index,   roll.values,   color="steelblue",  linewidth=1.2, label="63d MA")
        ax.axhline(0, color="black", linewidth=0.6)

        # shade abnormal
        from garch_regime import _shade_regions
        _shade_regions(ax, pd.Series(list(abnormal_set)),
                       color="salmon", alpha=0.20, label="Abnormal")

        ax.set_title(f"{col}  (ICIR={icir[col]:.2f})", fontsize=9)
        ax.set_ylabel("IC", fontsize=7)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(top_fac), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Rolling IC Time Series (top factors by |ICIR|)", fontsize=11)
    plt.tight_layout()
    out = PLOT_DIR / "ic_timeseries.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
