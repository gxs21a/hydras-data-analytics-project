import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =====================================================
# CONFIG
# =====================================================
BRANCHES  = ["Dallas", "Pittsburgh"]   # only these two
SUBTYPE   = "Respiratory"              # only Respiratory

SERIES_START   = "2023-01-01"
SERIES_END     = "2025-12-31"
FORECAST_START = "2025-06-01"

EWMA_SPAN = 12

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW.csv')
print("Number of rows in our current dataset: ", df.shape[0])

df["CallDateTime"] = pd.to_datetime(df["CallDateTime"])
df["EndDateTime"]  = pd.to_datetime(df["EndDateTime"])


# =====================================================
# HELPERS
# =====================================================
def build_time_series(df, branch, subtype, start=SERIES_START, end=SERIES_END):
    mask = (
        (df["Delivery_BranchName"] == branch) &
        (df["ModelTypeName"]        == subtype)
    )
    df_f = df[mask].copy()
    weeks = pd.date_range(start=start, end=end, freq="W-MON")

    counts = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        active = (
            (df_f["CallDateTime"] <= week_end) &
            (df_f["EndDateTime"]  >= week_start)
        ).sum()
        counts.append(active)

    ts = pd.Series(counts, index=weeks, name="units_on_rent")
    ts.index.name = "week"
    return ts


def apply_ewma(ts, span=EWMA_SPAN):
    return ts.ewm(span=span, adjust=False).mean()


def fit_arima(ts, order=(1, 1, 1)):
    """Fit ARIMA on whatever series is passed in (raw or smoothed)."""
    model  = ARIMA(ts, order=order)
    return model.fit()


def fit_sarimax(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    """Fit SARIMAX on whatever series is passed in (raw or smoothed)."""
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def fit_holtwinters(ts):
    """Fit Holt-Winters on raw series — recency weighting is internal."""
    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated"
    )
    result = model.fit(optimized=True)
    print(f"  HW alpha={result.params['smoothing_level']:.3f}  "
          f"beta={result.params['smoothing_trend']:.3f}  "
          f"gamma={result.params['smoothing_seasonal']:.3f}")
    return result


# =====================================================
# MAIN COMPARISON PLOT
# 2 rows (cities) x 5 cols (models)
# Col 0: Baseline ARIMA      (raw, no weighting)
# Col 1: Baseline SARIMAX    (raw, no weighting)
# Col 2: EWMA-ARIMA          (EWMA-smoothed input)
# Col 3: EWMA-SARIMAX        (EWMA-smoothed input)
# Col 4: Holt-Winters ETS    (raw, recency baked in)
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
    f"Respiratory — Forecast Method Comparison: Dallas vs Pittsburgh",
    fontsize=14, y=1.01
)

for r, branch in enumerate(BRANCHES):
    print(f"\n{'='*60}\n  {branch} | {SUBTYPE}\n{'='*60}")

    ts       = build_time_series(df, branch, SUBTYPE)
    ts_train = ts[ts.index <  FORECAST_START]
    ts_test  = ts[ts.index >= FORECAST_START]
    fw       = len(ts_test)          # forecast_weeks

    ts_train_smooth = apply_ewma(ts_train)

    if ts.sum() == 0 or len(ts_train) < 52:
        print("  ⚠ Skipping — no data or insufficient training history.")
        for c in range(5):
            axes[r][c].set_visible(False)
        continue

    # --- Fit all 5 models ---
    print("  Fitting Baseline ARIMA...")
    r0_arima   = fit_arima(ts_train)                  # raw

    print("  Fitting Baseline SARIMAX...")
    r1_sarimax = fit_sarimax(ts_train)                # raw

    print("  Fitting EWMA-ARIMA...")
    r2_arima   = fit_arima(ts_train_smooth)           # smoothed

    print("  Fitting EWMA-SARIMAX...")
    r3_sarimax = fit_sarimax(ts_train_smooth)         # smoothed

    print("  Fitting Holt-Winters...")
    r4_hw      = fit_holtwinters(ts_train)            # raw (HW weights internally)

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

    # --- Plot each column ---
    for c, (fc, col_title) in enumerate(zip(forecasts, COL_TITLES)):
        ax = axes[r][c]
        is_ewma_col = c in (2, 3)   # cols where we show smoothed train

        # Raw train — always shown faintly for reference
        ax.plot(ts_train.index, ts_train.values,
                linewidth=0.8, alpha=0.25, color="steelblue", label="Train (raw)")

        # Smoothed train — only on EWMA columns
        if is_ewma_col:
            ax.plot(ts_train_smooth.index, ts_train_smooth.values,
                    linewidth=1.2, color="steelblue", label=f"Train (EWMA)")

        # Actual test period
        ax.plot(ts_test.index, ts_test.values,
                linewidth=1.2, color="green", label="Actual")

        # Forecast
        ax.plot(fc.index, fc.values,
                linewidth=1.4, linestyle="--", color="orange", label="Forecast")

        # Train/test split line
        ax.axvline(pd.Timestamp(FORECAST_START),
                   color="gray", linestyle=":", linewidth=0.8, label="Split")

        # Row label on leftmost column only
        if c == 0:
            ax.set_ylabel(f"{branch}\nUnits on Rent", fontsize=9)
        else:
            ax.set_ylabel("")

        # Column title on top row only
        if r == 0:
            ax.set_title(col_title, fontsize=9, fontweight="bold")

        ax.set_xlabel("Week", fontsize=7)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.legend(fontsize=6)

plt.tight_layout()
plt.show()