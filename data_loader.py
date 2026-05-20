"""
data_loader.py
==============
Year-by-year data loading and panel construction.

Key outputs
-----------
- panel DataFrame: (TradingDay, SecuCode, ClosePrice, ret, next_ret, <factors>)
- market_return Series: equal-weighted daily return across all stocks (GARCH input)

Memory strategy
---------------
Load each factor source one year at a time, concat, then left-join to the base panel.
Delete intermediates and gc.collect() after each source to cap peak RAM.
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, YEARS, FACTOR_SOURCES, ALL_FACTORS


# ── Low-level loader ──────────────────────────────────────────────────────────

def _load_csv(year: int, file_tag: str, extra_cols: list) -> pd.DataFrame | None:
    """
    Load one year/tag CSV.  Returns [TradingDay, SecuCode] + extra_cols, or None.
    Silently skips missing files or bad columns.
    """
    path = DATA_DIR / str(year) / f"Factors_{file_tag}_银行&非银_all.csv"
    if not path.exists():
        return None
    try:
        need = ["TradingDay", "SecuCode"] + extra_cols
        df = pd.read_csv(path, usecols=lambda c: c in need, low_memory=False)
        # Some files have an 'Unnamed: 0' index column – drop silently
        df = df[[c for c in need if c in df.columns]]
        df["TradingDay"] = pd.to_datetime(df["TradingDay"])
        df["SecuCode"]   = df["SecuCode"].astype(str).str.zfill(6)
        return df
    except Exception as exc:
        warnings.warn(f"Skipped {path.name}: {exc}")
        return None


# ── Price + return construction ───────────────────────────────────────────────

def load_price_panel() -> pd.DataFrame:
    """
    Load ClosePrice from ValuationNew for all years.
    Computes:
      ret      – daily pct-change per stock (aligned by SecuCode)
      next_ret – 1-trading-day forward return (target for IC & ML)

    Returns sorted DataFrame: [TradingDay, SecuCode, ClosePrice, ret, next_ret]
    """
    frames = []
    for year in tqdm(YEARS, desc="Loading prices"):
        df = _load_csv(year, "ValuationNew", ["ClosePrice"])
        if df is not None:
            frames.append(df)

    price = pd.concat(frames, ignore_index=True)
    price = price.sort_values(["SecuCode", "TradingDay"])

    price["ret"]      = price.groupby("SecuCode")["ClosePrice"].pct_change()
    # shift(-1) within each stock so next_ret at t = ret at t+1
    price["next_ret"] = price.groupby("SecuCode")["ret"].shift(-1)

    # Market-neutral (cross-sectional excess) return: next_ret minus the
    # equal-weighted cross-sectional mean on that date.  This removes the
    # common market factor so an ML model trained on excess_ret learns pure
    # stock-selection alpha rather than macro market-timing signal.
    cs_mean = price.groupby("TradingDay")["next_ret"].transform("mean")
    price["excess_ret"] = price["next_ret"] - cs_mean

    return price[["TradingDay", "SecuCode", "ClosePrice", "ret", "next_ret", "excess_ret"]]


# ── Main panel builder ────────────────────────────────────────────────────────

def load_all_factors() -> pd.DataFrame:
    """
    Build the full factor panel by left-joining each factor source onto the
    price panel.  Year-level loading keeps peak memory low.

    Returns
    -------
    DataFrame with columns: TradingDay, SecuCode, ClosePrice, ret, next_ret,
                             <factor_1>, …, <factor_N>
    Sorted by [TradingDay, SecuCode].
    """
    print("=" * 55)
    print("Loading price panel …")
    base = load_price_panel()

    for file_tag, cols in tqdm(FACTOR_SOURCES, desc="Merging factor sources"):
        # Collect all available years for this source
        yearly = []
        for year in YEARS:
            df = _load_csv(year, file_tag, cols)
            if df is not None:
                yearly.append(df)

        if not yearly:
            print(f"  [skip] {file_tag} – no files found")
            continue

        src = pd.concat(yearly, ignore_index=True)

        # Keep only columns that actually loaded (guards against partial files)
        avail = [c for c in cols if c in src.columns]
        if not avail:
            print(f"  [skip] {file_tag} – target columns absent")
            del src
            continue

        src = src[["TradingDay", "SecuCode"] + avail]
        base = base.merge(src, on=["TradingDay", "SecuCode"], how="left")

        del src, yearly
        gc.collect()

    base = base.sort_values(["TradingDay", "SecuCode"]).reset_index(drop=True)

    n_dates  = base["TradingDay"].nunique()
    n_stocks = base["SecuCode"].nunique()
    avail_f  = [f for f in ALL_FACTORS if f in base.columns]
    missing  = [f for f in ALL_FACTORS if f not in base.columns]

    print(f"\nPanel ready: {len(base):,} rows | {n_dates} dates | {n_stocks} stocks")
    print(f"  Factors loaded  : {len(avail_f)}  {avail_f}")
    if missing:
        print(f"  Factors missing : {missing}")

    return base


# ── Market-level return (GARCH input) ─────────────────────────────────────────

def compute_market_return(panel: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted cross-sectional mean of daily stock returns.
    Used as the univariate input to the GARCH model.

    Returns pd.Series indexed by TradingDay, named 'market_ret'.
    """
    mkt = panel.groupby("TradingDay")["ret"].mean().dropna()
    mkt.name = "market_ret"
    return mkt
