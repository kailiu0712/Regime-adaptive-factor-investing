"""
ml_predictor.py
===============
Random Fourier Features (RFF) cross-sectional return predictor.

Theory
------
RFF (Rahimi & Recht, 2007) approximates an RBF kernel  κ(x,y) = exp(-γ||x-y||²)
via an explicit, low-dimensional random feature map:

    z(x) = √(2/D) · [cos(w₁ᵀx + b₁), …, cos(w_Dᵀx + b_D)]

where  wⱼ ~ N(0, 2γI)  and  bⱼ ~ Uniform(0, 2π).

The kernel regression estimate f(x) = κ(x,·)ᵀα is then approximated by
the linear model f(x) ≈ zᵀθ, with θ estimated by ridge regression:

    θ = (ZᵀZ + αI)⁻¹Zᵀy

This gives a fast, closed-form solution that captures nonlinear interactions
among factors without the memory cost of kernel regression or the training
cost of a neural network.

Cross-sectional workflow
------------------------
At each refit date t, we:
  1. Collect all (stock, date) observations in the training window [t-W, t).
  2. Within each date, rank-normalize each factor to [0, 1]  (handles outliers,
     makes features comparable across stocks and dates).
  3. Pool all observations → fit RFF ridge on (X, y).
  4. At test date t, rank-normalize factors cross-sectionally → predict ŷ.
  5. Rank ŷ, select top-q stocks by predicted return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import RFF_N_COMPONENTS, RFF_GAMMA, RFF_ALPHA


# ── RFF model ─────────────────────────────────────────────────────────────────

class RFFRegressor:
    """
    Random Fourier Features kernel ridge regressor.

    Parameters
    ----------
    n_components : int
        Dimensionality of the RFF feature map D.
        Larger D → better kernel approximation; diminishing returns above ~500.
    gamma : float
        RBF bandwidth.  Larger γ → tighter kernel (more local).
        A practical default of 1.0 works when features are rank-normalised to [0,1].
    alpha : float
        Ridge regularisation λ.  Prevents overfitting on small cross-sections.
    random_state : int
        Seed for the random weight matrix W.
    """

    def __init__(
        self,
        n_components: int = RFF_N_COMPONENTS,
        gamma: float      = RFF_GAMMA,
        alpha: float      = RFF_ALPHA,
        random_state: int = 42,
    ):
        self.n_components  = n_components
        self.gamma         = gamma
        self.alpha         = alpha
        self.random_state  = random_state

        self.W_      = None   # (n_features, n_components)  sampled once at fit
        self.b_      = None   # (n_components,)
        self.coef_   = None   # (n_components,)  ridge solution
        self.scaler_ = None   # StandardScaler fitted on training X

    # ── Feature map ───────────────────────────────────────────────────────

    def _sample_weights(self, n_features: int):
        rng     = np.random.RandomState(self.random_state)
        self.W_ = rng.randn(n_features, self.n_components) * np.sqrt(2.0 * self.gamma)
        self.b_ = rng.uniform(0.0, 2.0 * np.pi, self.n_components)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Map X (N, d) → Z (N, D) via RFF."""
        return np.cos(X @ self.W_ + self.b_) * np.sqrt(2.0 / self.n_components)

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFFRegressor":
        """
        Fit the model.

        1. StandardScaler (mean/std) normalises X across the pooled training set.
        2. W and b are sampled once (fixed across train/predict).
        3. Ridge solution: θ = (ZᵀZ + αI)⁻¹Zᵀy  via np.linalg.solve.

        Parameters
        ----------
        X : (N, d) float32 array, rank-normalised within each cross-section
        y : (N,)   float32 array of 1-day forward returns
        """
        self.scaler_ = StandardScaler()
        Xs           = self.scaler_.fit_transform(X)

        self._sample_weights(X.shape[1])
        Z           = self._transform(Xs)            # (N, D)

        A           = Z.T @ Z + self.alpha * np.eye(self.n_components)
        self.coef_  = np.linalg.solve(A, Z.T @ y)   # (D,)
        return self

    # ── Predict ───────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict return scores for new observations.

        X : (M, d) float32 array, rank-normalised within the current cross-section.
        Returns (M,) array of predicted return scores.
        """
        Xs = self.scaler_.transform(X)
        Z  = self._transform(Xs)
        return Z @ self.coef_


# ── Cross-sectional data preparation ─────────────────────────────────────────

def cs_rank(series: pd.Series) -> pd.Series:
    """Rank-normalise a Series to [0, 1], preserving NaN positions."""
    return series.rank(pct=True)


def _cs_impute_and_rank(df: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
    """
    For each factor column:
      1. Impute NaN with the cross-sectional median for that date
         (stocks with missing factor treated as median quality for that attribute).
      2. If the entire column is NaN (factor not available on this date),
         fill with 0.5 — the neutral rank — so the RFF model sees a zero signal
         rather than a missing value.
      3. Rank-normalise to [0, 1] via cs_rank.

    This preserves every stock that has a valid target, maximising training data.
    """
    df = df.copy()
    for col in factor_cols:
        if col not in df.columns:
            df[col] = 0.5          # factor entirely absent
            continue
        med = df[col].median()     # NaN-safe: pandas skips NaN in median
        if np.isnan(med):
            df[col] = 0.5          # whole column missing → neutral
        else:
            df[col] = df[col].fillna(med)
        df[col] = cs_rank(df[col])
    return df


def prepare_training_data(
    panel: pd.DataFrame,
    factor_cols: list,
    target_col: str = "next_ret",
) -> tuple:
    """
    Build pooled cross-sectional training arrays from a panel subset.

    For each date:
      1. Keep all stocks that have a valid target (next_ret not NaN).
      2. Impute missing factor values with the cross-sectional median
         (see _cs_impute_and_rank). This ensures stocks with partial factor
         coverage still contribute to training rather than being silently dropped.
      3. Rank-normalise each factor within the date's cross-section.
      4. Stack all dates → pooled (X, y) arrays.

    Returns (X, y, meta_df) where:
      X       : (N_total, len(factor_cols)) float32 ndarray
      y       : (N_total,) float32 ndarray
      meta_df : aligned DataFrame with TradingDay, SecuCode for debugging
    """
    factor_cols = [f for f in factor_cols if f in panel.columns]
    daily_dfs   = []

    for _, grp in panel.groupby("TradingDay"):
        # Require only the target to be non-NaN; impute factor NaNs below
        valid = grp.dropna(subset=[target_col]).copy()
        if len(valid) < 10:
            continue
        valid = _cs_impute_and_rank(valid, factor_cols)
        daily_dfs.append(valid)

    if not daily_dfs:
        return np.empty((0, len(factor_cols)), dtype=np.float32), np.empty(0, dtype=np.float32), pd.DataFrame()

    merged   = pd.concat(daily_dfs, ignore_index=True)
    X        = merged[factor_cols].values.astype(np.float32)
    y        = merged[target_col].values.astype(np.float32)
    meta_df  = merged[["TradingDay", "SecuCode"]].reset_index(drop=True)
    return X, y, meta_df


def predict_cross_sectional(
    model: RFFRegressor,
    date_df: pd.DataFrame,
    factor_cols: list,
) -> pd.Series:
    """
    Score all stocks on a single prediction date.

    Uses the same imputation-then-rank approach as prepare_training_data to
    ensure every stock in the universe receives a prediction score, even if
    some of its factor values are missing.

    Parameters
    ----------
    model      : fitted RFFRegressor
    date_df    : rows for the single prediction date (TradingDay, SecuCode, factors)
    factor_cols: factor columns the model was trained on

    Returns
    -------
    pd.Series indexed by SecuCode with predicted return scores.
    """
    factor_cols = [f for f in factor_cols if f in date_df.columns]
    # Keep all stocks with at least one valid factor (or all stocks)
    valid = date_df.copy()
    if valid.empty:
        return pd.Series(dtype=float)

    valid = _cs_impute_and_rank(valid, factor_cols)

    X    = valid[factor_cols].values.astype(np.float32)
    pred = model.predict(X)
    return pd.Series(pred, index=valid["SecuCode"].values, name="pred_ret")
