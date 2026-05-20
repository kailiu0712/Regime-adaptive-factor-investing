"""
regime_clustering.py
====================
Dimensionality reduction + clustering to validate regime separation.

Motivation
----------
After classifying factors into normal/abnormal pools, we want to verify
that these pools genuinely capture distinct market states — not just via
IC statistics, but geometrically: do the two regimes occupy separate
regions of the factor signal space?

Two complementary 2-D views are constructed:

  ┌──────────────────────────────────────────────────────────────────────┐
  │ View A – Factor Composite Space (interpretable)                      │
  │                                                                      │
  │   x-axis = Normal composite score                                    │
  │           = ICIR-weighted mean IC of normal-pool factors on date d   │
  │   y-axis = Abnormal composite score                                  │
  │           = ICIR-weighted mean IC of abnormal-pool factors on date d │
  │                                                                      │
  │   Each point = one trading date.  Good outcome: normal dates cluster │
  │   top-right (strong normal-factor IC), abnormal dates cluster        │
  │   bottom-right or top-left (strong abnormal-factor IC).              │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │ View B – UMAP of full IC space                                       │
  │                                                                      │
  │   Feature vector per date = IC of EVERY factor that day.            │
  │   UMAP(cosine, n_components=2) preserves local + global structure.  │
  │   HDBSCAN on UMAP embedding detects density-based clusters.         │
  │   Good outcome: clusters align well with GARCH regime labels (high  │
  │   ARI), confirming the factor IC surface is regime-dependent.       │
  └──────────────────────────────────────────────────────────────────────┘

Why UMAP + HDBSCAN?
-------------------
- UMAP is substantially faster than t-SNE on >1000 points and preserves
  global topological structure (clusters stay separated), which is
  critical for downstream clustering.
- HDBSCAN is the natural partner to UMAP: it finds clusters of arbitrary
  shape, does not force all points into clusters (noise points flagged),
  and scales well.  Its only hyperparameter (min_cluster_size) is set
  relative to dataset size so no manual tuning is needed.

Outputs
-------
  outputs/plots/regime_clustering_composites.png  – View A (4 panels)
  outputs/plots/regime_clustering_umap.png        – View B (4 panels)
  outputs/results/regime_cluster_summary.csv      – ARI, silhouette, purity

Run as a stage via run_pipeline.py (Stage 5.5) or standalone:
  python regime_clustering.py
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm

from config import PLOT_DIR, RES_DIR

warnings.filterwarnings("ignore")

# ── Colour palette ────────────────────────────────────────────────────────────
_REGIME_COLORS  = {0: "#4878CF", 1: "#D65F5F"}   # blue=normal, red=abnormal
_CLUSTER_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 – Factor composite space (View A)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_regime_composites(
    ic_all: pd.DataFrame,
    normal_pool: list,
    abnormal_pool: list,
    icir_normal: pd.Series,
    icir_abnormal: pd.Series,
) -> pd.DataFrame:
    """
    Build the interpretable 2-D space from the pre-computed IC series.

    For each trading date d:
      normal_score[d]   = Σ_f  |ICIR_normal[f]|  × IC_all[d,f]
                          ─────────────────────────────────────
                          Σ_f  |ICIR_normal[f]|
                          (sum over factors in normal_pool)

      abnormal_score[d] = same, but over abnormal_pool / ICIR_abnormal

    Using |ICIR| as weights ensures factors with stronger regime-specific
    predictive power contribute more to the composite.

    Parameters
    ----------
    ic_all        : date × factor IC DataFrame (from factor_analysis.run_factor_analysis)
    normal_pool   : list of factor names classified as normal-regime
    abnormal_pool : list of factor names classified as abnormal-regime
    icir_normal   : pd.Series (factor → ICIR in normal regime)
    icir_abnormal : pd.Series (factor → ICIR in abnormal regime)

    Returns
    -------
    DataFrame[Date, normal_score, abnormal_score]
    """

    def _weighted_ic(pool: list, icir: pd.Series) -> pd.Series:
        avail = [f for f in pool if f in ic_all.columns]
        if not avail:
            return pd.Series(np.nan, index=ic_all.index)
        w = icir.reindex(avail).fillna(0.0).abs()
        w_sum = w.sum()
        if w_sum < 1e-10:
            w = pd.Series(1.0 / len(avail), index=avail)
        else:
            w = w / w_sum
        ic_sub = ic_all[avail].fillna(0.0)   # impute missing IC days with 0 (neutral)
        return ic_sub @ w

    normal_score   = _weighted_ic(normal_pool,   icir_normal)
    abnormal_score = _weighted_ic(abnormal_pool, icir_abnormal)

    composites = pd.DataFrame({
        "normal_score":   normal_score,
        "abnormal_score": abnormal_score,
    })
    composites.index.name = "Date"
    return composites


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 – UMAP of full IC space (View B)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ic_feature_matrix(ic_all: pd.DataFrame) -> np.ndarray:
    """
    Build date-level feature matrix from the IC series.

    Each row = one trading date.
    Each column = IC of one factor (0-imputed where IC could not be computed).
    Standardised column-wise before embedding.
    """
    mat = ic_all.fillna(0.0).values.astype(np.float32)
    mat = StandardScaler().fit_transform(mat)
    return mat


def run_umap(
    feature_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    UMAP dimensionality reduction to 2-D.

    Hyperparameter guidance:
      n_neighbors : controls local/global balance.  15 is a robust default;
                    increase if data has large-scale manifold structure.
      min_dist    : controls how tightly UMAP packs points in the embedding.
                    0.10 is good for cluster detection.
      metric      : 'cosine' is appropriate for IC vectors (signed correlation
                    values where scale differences across factors are less
                    meaningful than direction).

    Returns (N, 2) float32 embedding.
    """
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components  = 2,
        n_neighbors   = n_neighbors,
        min_dist      = min_dist,
        metric        = metric,
        random_state  = random_state,
        verbose       = False,
    )
    print("  Running UMAP …")
    emb = reducer.fit_transform(feature_matrix)
    return emb.astype(np.float32)


def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int | None = None,
) -> tuple:
    """
    HDBSCAN density-based clustering on the 2-D UMAP embedding.

    min_cluster_size is set to max(10, N//25) if not provided, so the
    algorithm adapts to dataset size without manual tuning.

    Returns (labels, n_clusters_found, noise_fraction).
      labels[i] = -1 means noise (not assigned to any cluster).
    """
    import hdbscan as hdb
    n = len(embedding)
    mcs = min_cluster_size or max(10, n // 25)

    clusterer = hdb.HDBSCAN(
        min_cluster_size = mcs,
        min_samples      = max(3, mcs // 3),
        metric           = "euclidean",
        cluster_selection_epsilon = 0.0,
    )
    labels = clusterer.fit_predict(embedding)

    unique_c    = set(labels) - {-1}
    n_clusters  = len(unique_c)
    noise_frac  = float((labels == -1).mean())
    print(f"  HDBSCAN: {n_clusters} clusters, noise={noise_frac:.1%}")
    return labels, n_clusters, noise_frac


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 – Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    embedding: np.ndarray,
) -> dict:
    """
    Compare predicted cluster labels against ground-truth GARCH regime labels.

    Metrics
    -------
    ARI         : Adjusted Rand Index ∈ [-1, 1].  1 = perfect, 0 = random.
    Silhouette  : Average silhouette score ∈ [-1, 1] on the 2-D embedding.
                  Measures cluster cohesion independent of ground truth.
    Purity      : For each cluster, fraction assigned to majority regime.
                  Mean purity across clusters = overall purity.
    Regime sep. : 1 - |P(abnormal|cluster_0) - P(abnormal|cluster_1)|^{-1}
                  Simplified: how well the dominant cluster matches each regime.
    """
    # ARI (exclude noise points from HDBSCAN)
    mask = pred_labels != -1
    ari  = adjusted_rand_score(true_labels[mask], pred_labels[mask]) if mask.sum() > 0 else np.nan

    # Silhouette (needs ≥2 non-noise clusters)
    unique = set(pred_labels[mask])
    if len(unique) >= 2:
        sil = silhouette_score(embedding[mask], pred_labels[mask])
    else:
        sil = np.nan

    # Per-cluster purity
    purities = []
    cluster_stats = {}
    for c in sorted(unique):
        c_mask    = (pred_labels == c)
        c_true    = true_labels[c_mask]
        majority  = np.bincount(c_true).max() / c_mask.sum()
        pct_abn   = c_true.mean()
        purities.append(majority)
        cluster_stats[int(c)] = {
            "n":          int(c_mask.sum()),
            "pct_abnormal": round(float(pct_abn), 3),
            "purity":     round(float(majority), 3),
        }

    mean_purity = float(np.mean(purities)) if purities else np.nan

    return {
        "ARI":          round(float(ari), 4) if not np.isnan(ari) else np.nan,
        "Silhouette":   round(float(sil), 4) if not np.isnan(sil) else np.nan,
        "Mean_purity":  round(mean_purity,  4),
        "cluster_stats":cluster_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 – Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_composite_space(
    composites: pd.DataFrame,
    regime_labels: np.ndarray,
    cluster_labels_2d: np.ndarray | None = None,
    metrics_2d: dict | None = None,
):
    """
    4-panel figure for View A (factor composite space).

    Panels
    ------
    [0,0] Scatter coloured by GARCH regime (true labels)
    [0,1] Rolling 63-day smoothed trajectory coloured by regime
    [1,0] Scatter coloured by cluster label (HDBSCAN on composites)
    [1,1] Density contour per regime (KDE)
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    x = composites["normal_score"].values
    y = composites["abnormal_score"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel [0,0]: scatter by GARCH regime ─────────────────────────────
    ax = axes[0, 0]
    for regime, label, c in [(0, "Normal", _REGIME_COLORS[0]),
                              (1, "Abnormal", _REGIME_COLORS[1])]:
        m = regime_labels == regime
        ax.scatter(x[m], y[m], c=c, alpha=0.55, s=18, label=label, edgecolors="none")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Normal factor composite IC", fontsize=10)
    ax.set_ylabel("Abnormal factor composite IC", fontsize=10)
    ax.set_title("Factor Composite Space  –  GARCH Regime Labels", fontsize=11)
    ax.legend(fontsize=9, markerscale=1.5)
    _add_quadrant_labels(ax)
    ax.grid(alpha=0.2)

    # ── Panel [0,1]: time trajectory ─────────────────────────────────────
    ax = axes[0, 1]
    dates  = composites.index
    roll_x = pd.Series(x, index=dates).rolling(63, min_periods=15).mean()
    roll_y = pd.Series(y, index=dates).rolling(63, min_periods=15).mean()
    # Colour segments by majority regime in each window
    from matplotlib.collections import LineCollection
    points = np.stack([roll_x.values, roll_y.values], axis=1)
    segs   = np.stack([points[:-1], points[1:]], axis=1)
    seg_r  = regime_labels[1:]
    seg_c  = [_REGIME_COLORS[int(r)] for r in seg_r]
    lc     = LineCollection(segs, colors=seg_c, linewidths=1.2, alpha=0.7)
    ax.add_collection(lc)
    ax.autoscale()
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Normal factor composite IC (63d MA)", fontsize=10)
    ax.set_ylabel("Abnormal factor composite IC (63d MA)", fontsize=10)
    ax.set_title("Time Trajectory in Composite Space", fontsize=11)
    # Legend proxy
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_REGIME_COLORS[0], label="Normal"),
                        Patch(color=_REGIME_COLORS[1], label="Abnormal")],
              fontsize=9)
    ax.grid(alpha=0.2)

    # ── Panel [1,0]: cluster labels ───────────────────────────────────────
    ax = axes[1, 0]
    if cluster_labels_2d is not None:
        unique_c = sorted(set(cluster_labels_2d))
        for i, c_id in enumerate(unique_c):
            m     = cluster_labels_2d == c_id
            color = "lightgrey" if c_id == -1 else _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            label = "Noise" if c_id == -1 else f"Cluster {c_id}"
            ax.scatter(x[m], y[m], c=color, alpha=0.55, s=18,
                       label=label, edgecolors="none")
        if metrics_2d:
            subtitle = (f"ARI={metrics_2d['ARI']:.3f}  "
                        f"Sil={metrics_2d['Silhouette']:.3f}  "
                        f"Purity={metrics_2d['Mean_purity']:.3f}")
            ax.set_title(f"HDBSCAN Clusters on Composite Space\n{subtitle}", fontsize=10)
        ax.legend(fontsize=8, markerscale=1.5)
    else:
        ax.text(0.5, 0.5, "Clustering not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Clusters (composite space)", fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Normal factor composite IC", fontsize=10)
    ax.set_ylabel("Abnormal factor composite IC", fontsize=10)
    ax.grid(alpha=0.2)

    # ── Panel [1,1]: KDE contours per regime ─────────────────────────────
    ax = axes[1, 1]
    from scipy.stats import gaussian_kde
    for regime, label, c in [(0, "Normal", _REGIME_COLORS[0]),
                              (1, "Abnormal", _REGIME_COLORS[1])]:
        m = regime_labels == regime
        if m.sum() < 10:
            continue
        xi, yi = x[m], y[m]
        try:
            kde  = gaussian_kde(np.vstack([xi, yi]))
            xg   = np.linspace(x.min(), x.max(), 60)
            yg   = np.linspace(y.min(), y.max(), 60)
            Xg, Yg = np.meshgrid(xg, yg)
            Z    = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(60, 60)
            ax.contourf(Xg, Yg, Z, levels=5, alpha=0.25,
                        colors=[c] * 5)
            ax.contour(Xg, Yg, Z, levels=5, colors=[c], linewidths=0.8, alpha=0.8)
        except Exception:
            ax.scatter(xi, yi, c=c, alpha=0.3, s=10, edgecolors="none")
        ax.scatter([], [], c=c, label=label)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Normal factor composite IC", fontsize=10)
    ax.set_ylabel("Abnormal factor composite IC", fontsize=10)
    ax.set_title("KDE Density Contours by Regime", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.suptitle(
        "Regime Separation in Factor Composite Space\n"
        "(x = ICIR-weighted normal-pool IC, y = ICIR-weighted abnormal-pool IC)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = PLOT_DIR / "regime_clustering_composites.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_umap_space(
    embedding: np.ndarray,
    regime_labels: np.ndarray,
    cluster_labels: np.ndarray,
    metrics: dict,
    dates: pd.DatetimeIndex,
    method_name: str = "UMAP",
):
    """
    4-panel figure for View B (full IC space embedded to 2-D).

    Panels
    ------
    [0,0] UMAP coloured by GARCH regime  ← primary result
    [0,1] UMAP coloured by cluster label
    [1,0] Cluster overlap heatmap (cluster vs regime, normalised)
    [1,1] Date-coloured embedding (time trajectory in UMAP space)
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    u1, u2 = embedding[:, 0], embedding[:, 1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel [0,0]: regime labels ────────────────────────────────────────
    ax = axes[0, 0]
    for regime, label, c in [(0, "Normal", _REGIME_COLORS[0]),
                              (1, "Abnormal", _REGIME_COLORS[1])]:
        m = regime_labels == regime
        ax.scatter(u1[m], u2[m], c=c, alpha=0.55, s=18, label=label, edgecolors="none")
    ax.set_title(f"{method_name} Embedding  –  GARCH Regime Labels", fontsize=11)
    ax.set_xlabel(f"{method_name}-1", fontsize=10)
    ax.set_ylabel(f"{method_name}-2", fontsize=10)
    ax.legend(fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)

    # ── Panel [0,1]: cluster labels ───────────────────────────────────────
    ax = axes[0, 1]
    unique_c = sorted(set(cluster_labels))
    for i, c_id in enumerate(unique_c):
        m     = cluster_labels == c_id
        color = "lightgrey" if c_id == -1 else _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        label = "Noise" if c_id == -1 else f"Cluster {c_id}"
        ax.scatter(u1[m], u2[m], c=color, alpha=0.55, s=18,
                   label=label, edgecolors="none")
    subtitle = (f"ARI={metrics['ARI']:.3f}   "
                f"Silhouette={metrics['Silhouette']:.3f}   "
                f"Mean purity={metrics['Mean_purity']:.3f}")
    ax.set_title(f"HDBSCAN Clusters  –  {subtitle}", fontsize=9)
    ax.set_xlabel(f"{method_name}-1", fontsize=10)
    ax.set_ylabel(f"{method_name}-2", fontsize=10)
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.2)

    # ── Panel [1,0]: cross-tab heatmap ────────────────────────────────────
    ax = axes[1, 0]
    real_clusters = [c for c in unique_c if c != -1]
    if real_clusters:
        xtab = np.zeros((2, len(real_clusters)))
        for j, c_id in enumerate(real_clusters):
            m = cluster_labels == c_id
            xtab[0, j] = (regime_labels[m] == 0).sum()   # normal
            xtab[1, j] = (regime_labels[m] == 1).sum()   # abnormal
        # Column-normalise to get regime composition per cluster
        col_sums = xtab.sum(axis=0, keepdims=True)
        xtab_n   = xtab / np.where(col_sums > 0, col_sums, 1)
        im = ax.imshow(xtab_n, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(real_clusters)))
        ax.set_xticklabels([f"C{c}" for c in real_clusters], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal", "Abnormal"], fontsize=9)
        ax.set_title("Regime Composition per Cluster (normalised)", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Annotate cells
        for r in range(2):
            for c in range(len(real_clusters)):
                ax.text(c, r, f"{xtab_n[r,c]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if xtab_n[r,c] > 0.6 else "black")
    else:
        ax.text(0.5, 0.5, "No valid clusters", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Regime Composition per Cluster", fontsize=11)

    # ── Panel [1,1]: time-coloured trajectory ─────────────────────────────
    ax = axes[1, 1]
    # Colour by chronological order → detect temporal drift in UMAP space
    time_idx = np.arange(len(dates))
    sc = ax.scatter(u1, u2, c=time_idx, cmap="viridis", alpha=0.6, s=14, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    # Label colorbar with years
    year_ticks = []
    year_labels = []
    for yr in sorted(set(dates.year)):
        idx_yr = np.searchsorted(dates, pd.Timestamp(f"{yr}-01-01"))
        if idx_yr < len(dates):
            year_ticks.append(idx_yr)
            year_labels.append(str(yr))
    cbar.set_ticks([time_idx[i] for i in year_ticks if i < len(time_idx)])
    cbar.set_ticklabels(year_labels[:len([i for i in year_ticks if i < len(time_idx)])])
    ax.set_title(f"{method_name} Space – Chronological Colouring", fontsize=11)
    ax.set_xlabel(f"{method_name}-1", fontsize=10)
    ax.set_ylabel(f"{method_name}-2", fontsize=10)
    ax.grid(alpha=0.2)

    plt.suptitle(
        f"Regime Separation in Full IC Space ({method_name} + HDBSCAN)\n"
        "Each point = one trading date, feature = IC vector across all factors",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = PLOT_DIR / "regime_clustering_umap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5 – Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_regime_clustering(
    ic_all: pd.DataFrame,
    regime_df: pd.DataFrame,
    factor_pools: dict,
    summary_normal: pd.DataFrame,
    summary_abnormal: pd.DataFrame,
) -> dict:
    """
    End-to-end regime clustering pipeline.

    Parameters
    ----------
    ic_all         : date × factor IC DataFrame (from factor_analysis)
    regime_df      : GARCH regime labels (columns: Date, regime)
    factor_pools   : dict 'normal'/'abnormal' → list[str]
    summary_normal : IC summary for normal regime (contains 'ICIR' column)
    summary_abnormal: IC summary for abnormal regime

    Returns
    -------
    dict with keys:
      composites        – pd.DataFrame (dates × 2 composite scores)
      embedding         – (N, 2) UMAP array
      cluster_labels    – HDBSCAN labels on UMAP
      metrics_composite – eval metrics for composite-space clustering
      metrics_umap      – eval metrics for UMAP-space clustering
      summary_csv       – combined metrics DataFrame
    """
    RES_DIR.mkdir(parents=True,  exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Align dates: ic_all ∩ regime_df ──────────────────────────────────
    regime_indexed = regime_df.set_index("Date")
    common_dates   = ic_all.index.intersection(regime_indexed.index).sort_values()

    ic_sub      = ic_all.loc[common_dates]
    true_regime = regime_indexed.loc[common_dates, "regime"].values.astype(int)
    dates_idx   = common_dates

    print(f"\n  Raw IC series: {len(common_dates)} dates "
          f"({true_regime.sum()} abnormal, {(true_regime==0).sum()} normal)")

    # ══ View A: Factor composite space ════════════════════════════════════

    print("\n── View A: Factor composite space ────────────────────")
    icir_normal   = summary_normal["ICIR"]   if "ICIR" in summary_normal.columns   else pd.Series(dtype=float)
    icir_abnormal = summary_abnormal["ICIR"] if "ICIR" in summary_abnormal.columns else pd.Series(dtype=float)

    # Smooth daily IC to reduce noise before computing composites.
    # Rolling 20-day mean removes micro-noise while preserving regime transitions.
    ic_sub_smooth = ic_sub.rolling(window=20, min_periods=5).mean()
    ic_sub_smooth = ic_sub_smooth.dropna(how='all')
    # Re-align regime labels to smoothed dates
    common_smooth  = ic_sub_smooth.index.intersection(regime_indexed.index)
    ic_sub_smooth  = ic_sub_smooth.loc[common_smooth]
    true_regime    = regime_indexed.loc[common_smooth, "regime"].values.astype(int)
    dates_idx      = common_smooth

    # Use new pool key names (dynamic_normal / dynamic_abnormal)
    normal_pool  = factor_pools.get("dynamic_normal",   factor_pools.get("normal",   []))
    abnormal_pool= factor_pools.get("dynamic_abnormal", factor_pools.get("abnormal", []))

    print(f"\n  Smoothed clustering on {len(common_smooth)} dates "
          f"({true_regime.sum()} abnormal, {(true_regime==0).sum()} normal)")

    composites = compute_regime_composites(
        ic_sub_smooth,
        normal_pool,
        abnormal_pool,
        icir_normal,
        icir_abnormal,
    )
    composites.to_csv(RES_DIR / "regime_composites.csv")

    # Cluster in composite 2-D space
    comp_mat = composites[["normal_score", "abnormal_score"]].fillna(0.0).values
    comp_scaled = StandardScaler().fit_transform(comp_mat)
    cluster_comp, n_comp_c, noise_comp = run_hdbscan(comp_scaled)
    metrics_comp = evaluate_clustering(true_regime, cluster_comp,
                                       comp_scaled)
    print(f"  Composite-space → ARI={metrics_comp['ARI']}, "
          f"Sil={metrics_comp['Silhouette']}, Purity={metrics_comp['Mean_purity']}")

    plot_composite_space(
        composites, true_regime, cluster_comp, metrics_comp
    )

    # ══ View B: UMAP of full IC space ═════════════════════════════════════

    print("\n── View B: UMAP of full IC space ─────────────────────")
    feat_mat = build_ic_feature_matrix(ic_sub_smooth)

    embedding = run_umap(feat_mat)
    cluster_umap, n_umap_c, noise_umap = run_hdbscan(embedding)
    metrics_umap = evaluate_clustering(true_regime, cluster_umap, embedding)
    print(f"  UMAP-space → ARI={metrics_umap['ARI']}, "
          f"Sil={metrics_umap['Silhouette']}, Purity={metrics_umap['Mean_purity']}")

    plot_umap_space(
        embedding, true_regime, cluster_umap, metrics_umap, dates_idx
    )

    # ══ Summary CSV ═══════════════════════════════════════════════════════

    summary = pd.DataFrame([
        {
            "view":              "Composite-2D",
            "n_clusters":        n_comp_c,
            "noise_fraction":    round(noise_comp, 3),
            "ARI":               metrics_comp["ARI"],
            "Silhouette":        metrics_comp["Silhouette"],
            "Mean_purity":       metrics_comp["Mean_purity"],
        },
        {
            "view":              "UMAP + HDBSCAN",
            "n_clusters":        n_umap_c,
            "noise_fraction":    round(noise_umap, 3),
            "ARI":               metrics_umap["ARI"],
            "Silhouette":        metrics_umap["Silhouette"],
            "Mean_purity":       metrics_umap["Mean_purity"],
        },
    ])
    summary.to_csv(RES_DIR / "regime_cluster_summary.csv", index=False)
    print("\n── Clustering Summary ────────────────────────────────")
    print(summary.to_string(index=False))

    return {
        "composites":         composites,
        "embedding":          embedding,
        "cluster_labels_umap":cluster_umap,
        "cluster_labels_comp":cluster_comp,
        "metrics_composite":  metrics_comp,
        "metrics_umap":       metrics_umap,
        "summary_csv":        summary,
        "true_regime":        true_regime,
    }


# ── Helper: quadrant annotations ─────────────────────────────────────────────

def _add_quadrant_labels(ax):
    """Add faint diagonal text to label the four quadrants."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # These are placed in relative axes coordinates
    kw = dict(ha="center", va="center", fontsize=7, color="grey",
              alpha=0.6, style="italic", transform=ax.transAxes)
    ax.text(0.15, 0.85, "Abnormal\ndominates", **kw)
    ax.text(0.85, 0.85, "Both\nactive",       **kw)
    ax.text(0.15, 0.15, "Both\nweak",         **kw)
    ax.text(0.85, 0.15, "Normal\ndominates",  **kw)
