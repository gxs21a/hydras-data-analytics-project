
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =====================================================
# CONFIG — hardcode your targets here
# =====================================================
BRANCHES  = ["Pittsburgh", "Dallas"]
SUBTYPES  = ["Respiratory", "Accessories"] #REVISIT THIS PLEASE

SERIES_START = "2023-01-01"
SERIES_END   = "2025-12-31"

FORECAST_START = "2025-06-01"

#read in the data
df = pd.read_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW.csv')
print("Number of rows in our current dataset: ", df.shape[0])

# Already datetime — just ensure correct dtype in case CSV stringified them
df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
df["EndDateTime"]   = pd.to_datetime(df["EndDateTime"])

# =====================================================
# STEP 1 — Build weekly units-on-rent time series
# =====================================================
def build_time_series(df, branch, subtype, start=SERIES_START, end=SERIES_END):
    """
    For a given branch + subtype, returns a weekly Series of units on rent.
    A rental counts in a week if it overlaps that week at all.
    """
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
            (df_f["StartDateTime"] <= week_end) &
            (df_f["EndDateTime"]   >= week_start)
        ).sum()
        counts.append(active)

    ts = pd.Series(counts, index=weeks, name="units_on_rent")
    ts.index.name = "week"
    return ts


# =====================================================
# STEP 2 — Plot all 15 time series (5 cities x 3 subtypes)
# =====================================================
def plot_all_series(df, branches, subtypes):
    """
    Produces a grid of subplots: one row per branch, one column per subtype.
    """
    n_rows = len(branches)
    n_cols = len(subtypes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=False)
    fig.suptitle("Units on Rent Per Week", fontsize=16, y=1.01)

    for r, branch in enumerate(branches):
        for c, subtype in enumerate(subtypes):
            ax  = axes[r][c]
            ts  = build_time_series(df, branch, subtype)

            ax.plot(ts.index, ts.values, linewidth=1.1)
            ax.set_title(f"{branch} | {subtype}", fontsize=9)
            ax.set_xlabel("Week", fontsize=7)
            ax.set_ylabel("Units", fontsize=7)
            ax.tick_params(axis="x", labelsize=6, rotation=30)

    plt.tight_layout()
    plt.show()


# =====================================================
# STEP 3 — Fit ARIMA for one series
# =====================================================
def fit_arima(ts, order=(1, 1, 1)):
    """
    Fits an ARIMA model on a weekly units-on-rent series.
    Returns fitted model result.
    order = (p, d, q) — tune as needed.
    """
    model  = ARIMA(ts, order=order)
    result = model.fit()
    print(result.summary())
    return result

# =====================================================
# STEP 4 — Fit SARIMAX for one series
# =====================================================
def fit_sarimax(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    """
    Fits a SARIMAX model on a weekly units-on-rent series.
    Returns fitted model result.
    seasonal_order = (P, D, Q, s) where s=52 for weekly/annual seasonality.
    """
    model  = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    print(result.summary())
    return result


# =====================================================
# STEP 5 — Run models for all 15 combos + plot forecasts
# =====================================================
def run_all_models(df, branches, subtypes, forecast_start=FORECAST_START):
    for branch in branches:
        for subtype in subtypes:
            print(f"\n{'='*60}")
            print(f"  {branch} | {subtype}")
            print(f"{'='*60}")

            ts = build_time_series(df, branch, subtype)

            if ts.sum() == 0:
                print("  ⚠ No data for this combo — skipping.")
                continue

            # --- Train/test split at July 2025 ---
            ts_train = ts[ts.index <  forecast_start]
            ts_test  = ts[ts.index >= forecast_start]
            forecast_weeks = len(ts_test)

            if len(ts_train) < 52:
                print("  ⚠ Not enough training data — skipping.")
                continue

            # --- Fit models on train only ---
            arima_result   = fit_arima(ts_train)
            sarimax_result = fit_sarimax(ts_train)

            # --- Forecast over the test window ---
            arima_fc   = arima_result.forecast(steps=forecast_weeks)
            sarimax_fc = sarimax_result.forecast(steps=forecast_weeks)

            arima_fc.index   = ts_test.index
            sarimax_fc.index = ts_test.index

            # --- Plot ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            fig.suptitle(f"{branch} | {subtype}", fontsize=12)

            for ax, fc, label in zip(axes, [arima_fc, sarimax_fc], ["ARIMA", "SARIMAX"]):
                ax.plot(ts_train.index, ts_train.values, label="Train",    linewidth=1.2, color="steelblue")
                ax.plot(ts_test.index,  ts_test.values,  label="Actual",   linewidth=1.2, color="green")
                ax.plot(fc.index,       fc.values,       label="Forecast", linewidth=1.2, linestyle="--", color="orange")
                ax.axvline(pd.Timestamp(forecast_start), color="gray", linestyle=":", linewidth=0.8, label="Split")
                ax.set_title(label)
                ax.set_xlabel("Week")
                ax.set_ylabel("Units on Rent")
                ax.legend(fontsize=8)

            plt.tight_layout()
            plt.show()


# =====================================================
# RUN EVERYTHING
# =====================================================
plot_all_series(df, BRANCHES, SUBTYPES)   # Step 2 — all 15 overview plots
run_all_models(df, BRANCHES, SUBTYPES)    # Steps 3–5 — ARIMA + SARIMAX per combo
