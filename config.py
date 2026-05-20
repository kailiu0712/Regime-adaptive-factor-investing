"""
Central configuration for the abnormal-regime analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path


# Directories
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"
RES_DIR = OUT_DIR / "results"


# Data coverage
YEARS = list(range(2017, 2025))


# Factor sources
FACTOR_SOURCES = [
    ("ValuationNew", ["EP_TTM", "BP", "EBIT2MV", "SP1_TTM"]),
    ("ValuationRankNew", ["REP_TTM", "RBP"]),
    ("Technical1New", ["RT_2M", "RT_3M", "RT_6M", "TO_1M", "PRank_1M"]),
    ("Technical2New", ["ACD5", "AmountIR5", "CCI5", "StyleBias20"]),
    # AptOutFlowRatio is excluded because it is effectively missing after 2018.
    ("TechnicalHighFreq1", ["ILLIQ", "BCVP", "M4"]),
    ("FinancialNew", ["ROE", "ROATTM", "GrossIncomeRatioTTM"]),
    ("GrowthNew", ["E_Growth2", "OR_Growth2"]),
    ("ShareholderNew", ["SHNum", "FundsHoldPropT"]),
    ("Risk&Div", ["DivR1", "VOL_orth_ZS", "Beta_orth_ZS"]),
    ("GoGoalNew", ["Expect_EP", "Expect_E_Growth", "Rating"]),
]

ALL_FACTORS: list[str] = [col for _, cols in FACTOR_SOURCES for col in cols]


# GARCH regime detection
GARCH_MAX_P = 4
GARCH_Q = 1
GARCH_WARMUP = 252
GARCH_ABNORM_Q = 0.80


# Factor analysis thresholds
N_QUANTILES = 5
MIN_STOCKS = 20
ICIR_THRESHOLD = 0.20
TSTAT_THRESHOLD = 1.65


# Train / test split
TRAIN_START = "2017-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"


# Backtest configuration
REBALANCE_FREQ = 21
TRAIN_WINDOW = 252
TOP_QUANTILE = 0.20
TRANS_COST = 0.001

# Cross-sectional training target for the RFF path.
TARGET_COL = "excess_ret"


# RFF model
RFF_N_COMPONENTS = 500
RFF_GAMMA = 1.0
RFF_ALPHA = 1e-3


# Factor-pool sizing
TOP_K_RFF_FACTORS = 12
REGIME_SENSITIVITY_THRESHOLD = 1.0
TOP_K_DYNAMIC = 12


# Optional dynamic overlays kept for experimentation.
VOL_TARGET = 0.20
VOL_SCALE_MAX = 2.0
FAST_REGIME_WINDOW = 10
MIN_REGIME_TRAIN = 30


# Manually annotated abnormal intervals from the original project.
MANUAL_ABNORMAL_INTERVALS = [
    ("2018-02-06", "2018-02-26"),
    ("2019-02-01", "2019-02-25"),
    ("2020-01-21", "2020-02-20"),
    ("2020-06-30", "2020-07-09"),
    ("2021-02-08", "2021-02-10"),
    ("2021-07-23", "2021-08-04"),
    ("2022-03-02", "2022-03-18"),
    ("2022-04-20", "2022-04-29"),
    ("2023-10-13", "2023-10-30"),
    ("2024-01-26", "2024-02-08"),
    ("2024-09-24", "2024-11-07"),
]
