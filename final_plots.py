import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# =====================================================
# CONFIG
# =====================================================
BRANCHES  = ["Pittsburgh", "Albuquerque"]
SUBTYPE   = "Ventilator"   # MUST match ModelSubTypeName exactly

SERIES_START   = "2023-01-01"
SERIES_END     = "2026-03-24"
FORECAST_START = "2025-09-01"

# Must match the CENSORED_END used in the cleaning script.
# Any rental whose EndDateTime was null in the raw data was set to this
# date — so the time series will correctly count them as active up to here.
CENSORED_END = "2026-03-31"

EWMA_SPAN = 12


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'
)

print("Number of rows in dataset: ", df.shape[0])

# Convert to datetime
df["Delivery_CallDateTime"] = pd.to_datetime(df["Delivery_CallDateTime"])
df["EndDateTime"]           = pd.to_datetime(df["EndDateTime"])

# =====================================================
# CLEAN SUBTYPE COLUMN
# =====================================================
df = df.dropna(subset=["ModelSubTypeName"])
df = df[df["ModelSubTypeName"] != "Unknown"]

print("\nTop ModelSubTypeName values:")
print(df["ModelSubTypeName"].value_counts().head(10))


# =====================================================
# HELPERS
# =====================================================
def build_time_series(df, branch, subtype, start=SERIES_START, end=SERIES_END):
    """
    Build a weekly census series counting how many units of `subtype`
    were actively on rent in `branch` during each week.

    A rental is active in a given week if:
        Delivery_CallDateTime <= week_end  AND  EndDateTime >= week_start

    Because the cleaning script right-censors still-open rentals at
    CENSORED_END (not a short mean duration), active rentals now remain
    counted all the way to the end of the window instead of dropping off.
    """
    mask = (
        (df["Delivery_BranchName"] == branch) &
        (df["ModelSubTypeName"]    == subtype)
    )
    df_f = df[mask].copy()

    print(f"\nChecking data for {branch} | {subtype}: {df_f.shape[0]} rows")

    # How many are right-censored (still on rent at export)?
    if "EndDateTime_was_censored" in df_f.columns:
        n_censored = df_f["EndDateTime_was_censored"].sum()
        print(f"  Of which {n_censored} are right-censored (still active at export)")

    weeks = pd.date_range(start=start, end=end, freq="W-MON")

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


def fit_arima(ts, order=(1, 1, 1)):
    model = ARIMA(ts, order=order)
    return model.fit()


def fit_sarimax(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def fit_holtwinters(ts):
    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated"
    )
    result = model.fit(optimized=True)
    print(
        f"  HW alpha={result.params['smoothing_level']:.3f}  "
        f"beta={result.params['smoothing_trend']:.3f}  "
        f"gamma={result.params['smoothing_seasonal']:.3f}"
    )
    return result


# =====================================================
# SANITY CHECK — plot raw series BEFORE modelling
#
# Run this block first. You should see a continuous trend
# that cuts off cleanly at SERIES_END with no dip to zero.
# If you still see a dip, the censored rentals are not
# being read correctly from the cleaned CSV.
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

    ax.plot(
        ts_check.index, ts_check.values,
        linewidth=1.2, color="steelblue", label="Units on rent"
    )
    ax.axvline(
        pd.Timestamp(FORECAST_START),
        color="red", linestyle="--", linewidth=0.9, label="Forecast start"
    )
    ax.axvline(
        pd.Timestamp(SERIES_END),
        color="gray", linestyle=":", linewidth=0.8, label="Series end"
    )
    ax.set_title(
        f"SANITY CHECK — {branch} | {SUBTYPE} — raw weekly census",
        fontsize=10, fontweight="bold"
    )
    ax.set_ylabel("Units on rent")
    ax.set_xlabel("Week")
    ax.legend(fontsize=8)

    # Print summary stats to help diagnose
    print(f"\n{branch}:")
    print(f"  Series length : {len(ts_check)} weeks")
    print(f"  Min count     : {ts_check.min()}")
    print(f"  Max count     : {ts_check.max()}")
    print(f"  Mean count    : {ts_check.mean():.1f}")
    last_6 = ts_check.iloc[-6:]
    print(f"  Last 6 weeks  : {last_6.values}  ← should NOT be near zero")

plt.tight_layout()
plt.suptitle(
    f"Sanity check — {SUBTYPE} weekly census (check for false zero dip)",
    fontsize=11, y=1.01
)
plt.show()

# Pause here and inspect the sanity check plot before proceeding.
# Expected: a continuous trend line ending near SERIES_END.
# If still zero at the end: re-run the cleaning script first.


# =====================================================
# PLOT CONFIG for model comparison
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
    fontsize=14,
    y=1.01
)

# Handle single-branch case so axes indexing stays consistent
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

    fw = len(ts_test)

    ts_train_smooth = apply_ewma(ts_train)

    # Skip bad cases
    if ts.sum() == 0 or len(ts_train) < 52:
        print("  ⚠ Skipping — no data or insufficient training history.")
        for c in range(5):
            axes[r][c].set_visible(False)
        continue

    # --- Fit models ---
    print("  Fitting Baseline ARIMA...")
    r0_arima = fit_arima(ts_train)

    print("  Fitting Baseline SARIMAX...")
    r1_sarimax = fit_sarimax(ts_train)

    print("  Fitting EWMA-ARIMA...")
    r2_arima = fit_arima(ts_train_smooth)

    print("  Fitting EWMA-SARIMAX...")
    r3_sarimax = fit_sarimax(ts_train_smooth)

    print("  Fitting Holt-Winters...")
    r4_hw = fit_holtwinters(ts_train)

    # --- Forecasts ---
    forecasts = [
        r0_arima.forecast(steps=fw),
        r1_sarimax.forecast(steps=fw),
        r2_arima.forecast(steps=fw),
        r3_sarimax.forecast(steps=fw),
        r4_hw.forecast(steps=fw),
    ]

    for fc in forecasts:
        fc.index = ts_test.index

    # --- Plot ---
    for c, (fc, col_title) in enumerate(zip(forecasts, COL_TITLES)):

        ax = axes[r][c]
        is_ewma_col = c in (2, 3)

        # Raw train
        ax.plot(
            ts_train.index,
            ts_train.values,
            linewidth=0.8,
            alpha=0.25,
            color="steelblue",
            label="Train (raw)"
        )

        # Smoothed train overlay for EWMA columns
        if is_ewma_col:
            ax.plot(
                ts_train_smooth.index,
                ts_train_smooth.values,
                linewidth=1.2,
                color="steelblue",
                label="Train (EWMA)"
            )

        # Actual test values
        ax.plot(
            ts_test.index,
            ts_test.values,
            linewidth=1.2,
            color="green",
            label="Actual"
        )

        # Forecast
        ax.plot(
            fc.index,
            fc.values,
            linewidth=1.4,
            linestyle="--",
            color="orange",
            label="Forecast"
        )

        # Train/test split line
        ax.axvline(
            pd.Timestamp(FORECAST_START),
            color="gray",
            linestyle=":",
            linewidth=0.8,
            label="Split"
        )

        # Labels
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