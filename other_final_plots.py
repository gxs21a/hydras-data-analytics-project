import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# =====================================================
# CONFIG
# =====================================================
BRANCHES  = ["Pittsburgh", "Albuquerque"] # Sant Antonio, Tampa, Miami, Oklahoma City, Allendale
SUBTYPE   = "Ventilator"   # must match the Subtype column in cleaned data

# Date range mirrors the cleaning pipeline:
#   DATE_START = 2023-01-01  →  SERIES_START
#   ACTIVE_END = 2025-12-31  →  SERIES_END (active rentals end here)
#   DATE_END   = 2026-03-31  →  bootstrapped/known rentals can extend here,
#                               but we cap the series at SERIES_END so the
#                               training window is clean and the test window
#                               has real observable data
SERIES_START   = "2023-01-01"
SERIES_END     = "2026-03-24"   # last W-MON week before the Mar 2026 export date
FORECAST_START = "2025-09-01"   # ~29 weeks of test data; training window ~139 weeks

EWMA_SPAN = 12

# Minimum training weeks needed to support 52-week seasonality.
# SARIMAX needs at least 2 full annual cycles to estimate seasonal params.
MIN_WEEKS_FOR_SEASONAL = 104


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/Imputed_Jan23March26.csv',
    low_memory=False
)

print(f"Rows loaded: {df.shape[0]:,}")

df["Delivery_CallDateTime"] = pd.to_datetime(df["Delivery_CallDateTime"], errors="coerce")
df["EndDateTime"]           = pd.to_datetime(df["EndDateTime"],           errors="coerce")

# ── Subtype column ───────────────────────────────────────────────────────────
# The new cleaning pipeline stores subtype in "Subtype" (mapped from models.csv).
# Drop rows with no subtype assigned.
df = df.dropna(subset=["Subtype"])

print(f"\nTop Subtype values:")
print(df["Subtype"].value_counts().head(10))


# =====================================================
# HELPERS
# =====================================================
def build_time_series(df, branch, subtype):
    """
    Weekly active-rental census for a given branch + subtype.

    A rental is counted as active in week [week_start, week_end] if:
        Delivery_CallDateTime <= week_end  AND  EndDateTime >= week_start

    EndDateTime handling in the new cleaning pipeline:
      - Confirmed active rentals (IsActiveRental == 1): EndDateTime = 2025-12-31
      - Bootstrapped rentals (Delivered / Turn Down / etc.): EndDateTime imputed
        via median bootstrap simulation from known durations of that subtype
      - All other completed rentals: actual EndDateTime, capped at 2026-03-31
    There is no longer a single "censored end" date — the pipeline gives each
    rental a best-estimate EndDateTime, so the census just works correctly.
    """
    mask = (
        (df["Delivery_BranchName"] == branch) &
        (df["Subtype"]             == subtype)
    )
    df_f = df[mask].copy()
    print(f"\n  [{branch} | {subtype}] {df_f.shape[0]:,} rows")

    weeks = pd.date_range(start=SERIES_START, end=SERIES_END, freq="W-MON")
    counts = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        active = (
            (df_f["Delivery_CallDateTime"] <= week_end) &
            (df_f["EndDateTime"]            >= week_start)
        ).sum()
        counts.append(active)

    ts = pd.Series(counts, index=weeks, name="units_on_rent")
    ts.index.name = "week"
    return ts


def apply_ewma(ts, span=EWMA_SPAN):
    return ts.ewm(span=span, adjust=False).mean()


def fit_arima(ts):
    model = ARIMA(ts, order=(1, 1, 1))
    return model.fit()


def fit_sarimax(ts):
    """
    Fit SARIMAX with adaptive seasonal order.
    Falls back to plain ARIMA(1,1,1) if training series is too short
    for 52-week seasonality (avoids zeroed-out seasonal parameters).
    """
    if len(ts) >= MIN_WEEKS_FOR_SEASONAL:
        seasonal_order = (1, 1, 1, 52)
        label = "SARIMAX(1,1,1)(1,1,1,52)"
    else:
        seasonal_order = (0, 0, 0, 0)
        label = "ARIMA(1,1,1) [too few obs for seasonal]"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)

    return result, label


def fit_holtwinters(ts):
    """
    Holt-Winters ETS with additive trend + seasonality.
    Falls back to trend-only (no seasonal) if series is too short.
    """
    if len(ts) >= MIN_WEEKS_FOR_SEASONAL:
        model = ExponentialSmoothing(
            ts, trend="add", seasonal="add",
            seasonal_periods=52, initialization_method="estimated"
        )
    else:
        model = ExponentialSmoothing(
            ts, trend="add", seasonal=None,
            initialization_method="estimated"
        )
    result = model.fit(optimized=True)
    alpha = result.params.get("smoothing_level", float("nan"))
    beta  = result.params.get("smoothing_trend", float("nan"))
    print(f"    HW alpha={alpha:.3f}  beta={beta:.3f}")
    return result


# =====================================================
# SANITY CHECK — plot raw series BEFORE modelling
#
# Expected: a continuous trend that ends near SERIES_END with no
# dip to zero. If you see a dip, check that the cleaning pipeline
# ran successfully and produced Imputed_Jan23March26.csv.
# =====================================================
print("\n" + "="*60)
print("  SANITY CHECK: raw weekly census")
print("="*60)

fig_check, axes_check = plt.subplots(
    nrows=len(BRANCHES), ncols=1,
    figsize=(13, 4 * len(BRANCHES)),
    sharex=False
)
if len(BRANCHES) == 1:
    axes_check = [axes_check]

for i, branch in enumerate(BRANCHES):
    ts_check = build_time_series(df, branch, SUBTYPE)
    ax = axes_check[i]

    ax.plot(ts_check.index, ts_check.values,
            linewidth=1.2, color="steelblue", label="Units on rent")
    ax.axvline(pd.Timestamp(FORECAST_START),
               color="red",  linestyle="--", linewidth=0.9, label="Forecast start")
    ax.axvline(pd.Timestamp(SERIES_END),
               color="gray", linestyle=":",  linewidth=0.8, label="Series end")
    ax.set_title(f"SANITY CHECK — {branch} | {SUBTYPE}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Units on rent")
    ax.set_xlabel("Week")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.legend(fontsize=8)

    print(f"\n  {branch}:")
    print(f"    Series length : {len(ts_check)} weeks")
    print(f"    Min / Max     : {ts_check.min()} / {ts_check.max()}")
    print(f"    Mean          : {ts_check.mean():.1f}")
    print(f"    Last 6 weeks  : {ts_check.iloc[-6:].values}  ← should NOT be near zero")

plt.tight_layout()
plt.suptitle(f"Sanity check — {SUBTYPE} weekly census", fontsize=11, y=1.01)
plt.show()


# =====================================================
# MODEL COMPARISON PLOT
# 5 columns: Baseline ARIMA | Baseline SARIMAX |
#            EWMA-ARIMA | EWMA-SARIMAX | Holt-Winters
# =====================================================
COL_TITLES = [
    "Baseline ARIMA\n(no weighting)",
    "Baseline SARIMAX\n(no weighting)",
    f"EWMA-ARIMA\n(span={EWMA_SPAN})",
    f"EWMA-SARIMAX\n(span={EWMA_SPAN})",
    "Holt-Winters ETS\n(recency built-in)",
]

fig, axes = plt.subplots(
    nrows=len(BRANCHES),
    ncols=5,
    figsize=(22, 5 * len(BRANCHES)),
    sharex=False
)
fig.suptitle(
    f"{SUBTYPE} — Forecast Method Comparison: {' vs '.join(BRANCHES)}",
    fontsize=14, y=1.01
)
if len(BRANCHES) == 1:
    axes = [axes]


# =====================================================
# MAIN LOOP
# =====================================================
for r, branch in enumerate(BRANCHES):

    print(f"\n{'='*60}\n  {branch} | {SUBTYPE}\n{'='*60}")

    ts = build_time_series(df, branch, SUBTYPE)

    ts_train = ts[ts.index <  FORECAST_START]
    ts_test  = ts[ts.index >= FORECAST_START]
    fw       = len(ts_test)

    ts_train_smooth = apply_ewma(ts_train)

    if ts.sum() == 0 or len(ts_train) < 52:
        print("  ⚠ Skipping — no data or insufficient training history.")
        for c in range(5):
            axes[r][c].set_visible(False)
        continue

    # ── Fit models ──────────────────────────────────────────────────
    print("  Fitting Baseline ARIMA...")
    r0_arima = fit_arima(ts_train)

    print("  Fitting Baseline SARIMAX...")
    r1_sarimax, r1_label = fit_sarimax(ts_train)
    print(f"    Model: {r1_label}")

    print("  Fitting EWMA-ARIMA...")
    r2_arima = fit_arima(ts_train_smooth)

    print("  Fitting EWMA-SARIMAX...")
    r3_sarimax, r3_label = fit_sarimax(ts_train_smooth)
    print(f"    Model: {r3_label}")

    print("  Fitting Holt-Winters...")
    r4_hw = fit_holtwinters(ts_train)

    # ── Forecasts ───────────────────────────────────────────────────
    forecasts = [
        r0_arima.forecast(steps=fw),
        r1_sarimax.forecast(steps=fw),
        r2_arima.forecast(steps=fw),
        r3_sarimax.forecast(steps=fw),
        r4_hw.forecast(steps=fw),
    ]
    for fc in forecasts:
        fc.index = ts_test.index

    # ── Plot ────────────────────────────────────────────────────────
    for c, (fc, col_title) in enumerate(zip(forecasts, COL_TITLES)):
        ax = axes[r][c]
        is_ewma_col = c in (2, 3)

        ax.plot(ts_train.index, ts_train.values,
                linewidth=0.8, alpha=0.25, color="steelblue", label="Train (raw)")

        if is_ewma_col:
            ax.plot(ts_train_smooth.index, ts_train_smooth.values,
                    linewidth=1.2, color="steelblue", label="Train (EWMA)")

        ax.plot(ts_test.index, ts_test.values,
                linewidth=1.2, color="green", label="Actual")

        ax.plot(fc.index, fc.values,
                linewidth=1.4, linestyle="--", color="orange", label="Forecast")

        ax.axvline(pd.Timestamp(FORECAST_START),
                   color="gray", linestyle=":", linewidth=0.8, label="Split")

        if c == 0:
            ax.set_ylabel(f"{branch}\nUnits on Rent", fontsize=9)
        else:
            ax.set_ylabel("")

        if r == 0:
            ax.set_title(col_title, fontsize=9, fontweight="bold")

        ax.set_xlabel("Week", fontsize=7)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.legend(fontsize=6)

plt.tight_layout()
plt.show()