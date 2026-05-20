"""
garch_regime.py
===============
GARCH model selection and market-regime detection.

Workflow (mirrors the spec "practical workflow"):
  1. Fit GARCH(p,1) for p = 1 … MAX_P on the training return series.
  2. Compare AIC / BIC and Ljung-Box on standardised residuals at lags 5/10/15.
  3. Start with p=1 (parsimony).  Accept if LB lag-10 p-value > 0.05.
     Otherwise step up p until diagnostics pass, or fall back to lowest-AIC model.
  4. Fit best-order GARCH on full sample → extract conditional sigma series.
  5. Label each day: regime=1 (abnormal/high-vol) if sigma > expanding-window
     80th-pct threshold, else 0 (normal).  Expanding window = no look-ahead bias.
  6. Compare GARCH abnormal dates vs manually annotated intervals.
  7. Save: garch_order_selection.csv, regime_labels.csv, garch_vs_manual.csv.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from arch import arch_model                          # hard dependency – pip install arch
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import (
    GARCH_MAX_P, GARCH_Q, GARCH_WARMUP,
    GARCH_ABNORM_Q, MANUAL_ABNORMAL_INTERVALS,
    RES_DIR, PLOT_DIR,
)

# Suppress arch/statsmodels convergence noise
warnings.filterwarnings("ignore")


# ── Low-level GARCH fit ───────────────────────────────────────────────────────

def _fit_garch(returns: pd.Series, p: int, q: int):
    """
    Fit GARCH(p,q) with Normal innovations.  Returns arch ModelResult or None.
    Scales returns × 100 for numerical stability (arch convention).
    """
    try:
        am  = arch_model(returns * 100, vol="GARCH", p=p, q=q, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        return res
    except Exception:
        return None


# ── Model-order selection ─────────────────────────────────────────────────────

def select_garch_order(returns: pd.Series) -> int:
    """
    Grid-search GARCH(p,1) for p = 1 .. GARCH_MAX_P.

    Selection logic
    ---------------
    For each p:
      - Fit GARCH(p,1) on `returns`.
      - Extract standardised residuals; run Ljung-Box at lags 5, 10, 15.
      - Record AIC, BIC, and whether residuals are "clean" (LB lag-10 p > 0.05).

    Choice:
      - Among models whose residuals are clean, pick the one with lowest AIC
        (and lowest p as tiebreaker → parsimony).
      - If none is clean, fall back to lowest-AIC model and warn.

    Saves garch_order_selection.csv; prints table; returns best p (int).
    """
    RES_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for p in tqdm(range(1, GARCH_MAX_P + 1), desc="GARCH order selection"):
        res = _fit_garch(returns, p, GARCH_Q)
        if res is None:
            print(f"  GARCH({p},{GARCH_Q}) failed to converge – skipped")
            continue

        std_resid = pd.Series(res.std_resid).dropna()
        lb        = acorr_ljungbox(std_resid, lags=[5, 10, 15], return_df=True)
        lb_p      = lb["lb_pvalue"].values

        records.append(dict(
            p          = p,
            q          = GARCH_Q,
            AIC        = round(res.aic, 2),
            BIC        = round(res.bic, 2),
            LB_lag5_p  = round(float(lb_p[0]), 4),
            LB_lag10_p = round(float(lb_p[1]), 4),
            LB_lag15_p = round(float(lb_p[2]), 4),
            clean_resid= bool(lb_p[1] > 0.05),
        ))

    if not records:
        raise RuntimeError("All GARCH fits failed – check return series.")

    sel_df = pd.DataFrame(records).set_index("p")
    print("\n── GARCH Order Selection ──────────────────────────────")
    print(sel_df.to_string())
    sel_df.to_csv(RES_DIR / "garch_order_selection.csv")

    clean = sel_df[sel_df["clean_resid"]]
    if clean.empty:
        best_p = int(sel_df["AIC"].idxmin())
        print(f"\n  WARN No model passed LB test; using lowest-AIC: p={best_p}")
    else:
        # Among clean models, choose the smallest p with the best AIC
        # (sort by p first → parsimony preference when AICs are close)
        best_p = int(clean.sort_values(["AIC", "p"]).index[0])
        print(f"\n  OK Best GARCH order: p={best_p}  "
              f"(AIC={sel_df.loc[best_p,'AIC']}, LB-lag10 p={sel_df.loc[best_p,'LB_lag10_p']})")

    return best_p


# ── Regime labelling ──────────────────────────────────────────────────────────

def fit_garch_and_label_regimes(
    market_ret: pd.Series,
    p: int,
    q: int = GARCH_Q,
) -> pd.DataFrame:
    """
    Fit GARCH(p,q) on the full market return series and extract the
    conditional volatility path.

    Regime threshold (no look-ahead)
    ---------------------------------
    Uses an expanding-window quantile: at each date t the threshold is the
    GARCH_ABNORM_Q-quantile of sigma values in [0, t].  This means the label
    for date t depends only on history ≤ t.

    Returns
    -------
    DataFrame columns: Date, sigma, expand_threshold, regime
      regime=1  → abnormal (high volatility)
      regime=0  → normal
    """
    print(f"\nFitting GARCH({p},{q}) on full series ({len(market_ret)} obs) …")
    res = _fit_garch(market_ret, p, q)
    if res is None:
        raise RuntimeError(f"GARCH({p},{q}) fit failed on full series.")

    # arch scales by 100 internally; conditional_volatility is already in %-units
    # We divide by 100 to bring back to return units
    cond_vol = np.sqrt(np.maximum(res.conditional_volatility.values, 0)) / 100.0

    sigma = pd.Series(cond_vol, index=market_ret.index, name="sigma")

    # Expanding-window threshold
    thresh = sigma.expanding(min_periods=GARCH_WARMUP).quantile(GARCH_ABNORM_Q)

    regime_df = pd.DataFrame({
        "Date":             sigma.index,
        "sigma":            sigma.values,
        "expand_threshold": thresh.values,
        "regime":           (sigma.values > thresh.values).astype(int),
    }).dropna(subset=["expand_threshold"])

    n_abn = regime_df["regime"].sum()
    n_tot = len(regime_df)
    print(f"  Abnormal days: {n_abn} / {n_tot}  ({n_abn/n_tot*100:.1f}%)")

    return regime_df


# ── GARCH vs manual comparison ────────────────────────────────────────────────

def compare_with_manual(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overlap analysis between GARCH-detected abnormal dates and the
    manually annotated abnormal intervals from the original project.

    Metrics: precision, recall, F1, number of overlapping days.
    Saves garch_vs_manual.csv.
    """
    garch_set = set(regime_df.loc[regime_df["regime"] == 1, "Date"])

    manual_set = set()
    for start, end in MANUAL_ABNORMAL_INTERVALS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        manual_set |= set(
            regime_df.loc[(regime_df["Date"] >= s) & (regime_df["Date"] <= e), "Date"]
        )

    tp = len(garch_set & manual_set)
    fp = len(garch_set - manual_set)
    fn = len(manual_set - garch_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    result = pd.DataFrame([{
        "GARCH_abnormal_days":  len(garch_set),
        "Manual_abnormal_days": len(manual_set),
        "Overlap_days":         tp,
        "False_positive_days":  fp,
        "False_negative_days":  fn,
        "Precision":            round(precision, 4),
        "Recall":               round(recall, 4),
        "F1":                   round(f1, 4),
    }])

    print("\n── GARCH vs Manual Abnormal Dates ────────────────────")
    print(result.to_string(index=False))

    RES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RES_DIR / "garch_vs_manual.csv", index=False)

    return result


# ── Volatility visualisation ──────────────────────────────────────────────────

def plot_garch_regimes(market_ret: pd.Series, regime_df: pd.DataFrame):
    """
    Two-panel plot:
      Top   – equal-weighted market return with GARCH abnormal shading.
      Bottom – GARCH conditional sigma + expanding threshold + manual intervals.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # ── Panel 1: market return ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(market_ret.index, market_ret.values, color="dimgray",
            linewidth=0.6, label="Market return (EW)")
    ax.set_ylabel("Daily return")
    ax.set_title("Market Return with GARCH Regime Shading", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.25)

    # ── Panel 2: conditional sigma ────────────────────────────────────────
    ax = axes[1]
    ax.plot(regime_df["Date"], regime_df["sigma"],
            color="royalblue", linewidth=0.8, label="GARCH σ (cond.)")
    ax.plot(regime_df["Date"], regime_df["expand_threshold"],
            color="crimson", linestyle="--", linewidth=1.0,
            label=f"Threshold (expanding {int(GARCH_ABNORM_Q*100)}th pct)")

    # Shade GARCH-detected abnormal periods
    _shade_regions(
        ax,
        regime_df.loc[regime_df["regime"] == 1, "Date"],
        color="salmon", alpha=0.25, label="GARCH abnormal",
    )
    # Overlay manual intervals in yellow
    for i, (s, e) in enumerate(MANUAL_ABNORMAL_INTERVALS):
        ax.axvspan(
            pd.Timestamp(s), pd.Timestamp(e),
            color="gold", alpha=0.30,
            label="Manual abnormal" if i == 0 else "",
        )

    ax.set_ylabel("Conditional volatility")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out = PLOT_DIR / "garch_volatility.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def _shade_regions(ax, date_series: pd.Series, **kwargs):
    """Group consecutive dates (gap > 5 calendar days) into spans and axvspan."""
    dates = sorted(date_series.tolist())
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
