import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =====================================================
# CONFIG
# =====================================================
BRANCHES = ["Dallas", "Pittsburgh"]

TOP_N_SUBTYPES = 3

SERIES_START   = "2023-01-01"
SERIES_END     = "2025-12-31"
FORECAST_START = "2025-12-31"

EWMA_SPAN = 12

# =====================================================
# LOAD CLEAN DATA
# =====================================================
df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_forecasting_v1.csv'
)

df["Delivery_CallDateTime"] = pd.to_datetime(df["Delivery_CallDateTime"])
df["EffectiveEndDateTime"] = pd.to_datetime(df["EffectiveEndDateTime"])

# =====================================================
# SELECT TOP SUBTYPES
# =====================================================
top_subtypes = (
    df["ModelSubTypeName"]
    .value_counts()
    .head(TOP_N_SUBTYPES)
    .index
    .tolist()
)

print("Top subtypes:", top_subtypes)

# =====================================================
# BUILD TIME SERIES (FIXED)
# =====================================================
def build_time_series(df, branch, subtype):

    df_f = df[
        (df["Delivery_BranchName"] == branch) &
        (df["ModelSubTypeName"] == subtype)
    ].copy()

    weeks = pd.date_range(start=SERIES_START, end=SERIES_END, freq="W-MON")

    counts = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)

        active = (
            (df_f["Delivery_CallDateTime"] <= week_end) &
            (df_f["EffectiveEndDateTime"] >= week_start)
        ).sum()

        counts.append(active)

    return pd.Series(counts, index=weeks)

# =====================================================
# HELPERS
# =====================================================
def apply_ewma(ts):
    return ts.ewm(span=EWMA_SPAN, adjust=False).mean()

def fit_arima(ts):
    return ARIMA(ts, order=(1,1,1)).fit()

def fit_sarimax(ts):
    return SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,52)).fit(disp=False)

def fit_hw(ts):
    return ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add",
        seasonal_periods=52
    ).fit()

# =====================================================
# PLOTTING
# =====================================================
COL_TITLES = [
    "ARIMA",
    "SARIMAX",
    "EWMA-ARIMA",
    "EWMA-SARIMAX",
    "Holt-Winters"
]

for subtype in top_subtypes:

    fig, axes = plt.subplots(len(BRANCHES), 5, figsize=(22, 5 * len(BRANCHES)))

    for r, branch in enumerate(BRANCHES):

        ts = build_time_series(df, branch, subtype)

        ts_train = ts[ts.index < FORECAST_START]
        ts_test  = ts[ts.index >= FORECAST_START]

        fw = len(ts_test)

        ts_smooth = apply_ewma(ts_train)

        # Fit models
        m0 = fit_arima(ts_train)
        m1 = fit_sarimax(ts_train)
        m2 = fit_arima(ts_smooth)
        m3 = fit_sarimax(ts_smooth)
        m4 = fit_hw(ts_train)

        forecasts = [
            m0.forecast(fw),
            m1.forecast(fw),
            m2.forecast(fw),
            m3.forecast(fw),
            m4.forecast(fw)
        ]

        for fc in forecasts:
            fc.index = ts_test.index

        for c, (fc, title) in enumerate(zip(forecasts, COL_TITLES)):

            ax = axes[r][c]

            ax.plot(ts_train, alpha=0.3, label="Train")

            if c in [2,3]:
                ax.plot(ts_smooth, label="Smoothed")

            ax.plot(ts_test, label="Actual")
            ax.plot(fc, linestyle="--", label="Forecast")

            ax.axvline(pd.Timestamp(FORECAST_START), linestyle=":")

            if c == 0:
                ax.set_ylabel(branch)

            if r == 0:
                ax.set_title(title)

            ax.grid(alpha=0.3)

            if r == 0 and c == 4:
                ax.legend(fontsize=7)

    plt.suptitle(subtype)
    plt.tight_layout()
    plt.show()