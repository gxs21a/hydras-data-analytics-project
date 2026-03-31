import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
BRANCHES  = ["Pittsburgh", "Dallas", "Baltimore"]
SUBTYPES  = ["Respiratory", "Infusion"]


SERIES_START   = "2023-01-01"
SERIES_END     = "2025-12-31"
FORECAST_START = "2025-07-01"


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv')
print("Number of rows in dataset: ", df.shape[0])

# --- Column name consistency check ---
if "CallDateTime" in df.columns:
    CALL_COL = "CallDateTime"
elif "Delivery_CallDateTime" in df.columns:
    CALL_COL = "Delivery_CallDateTime"
    print("⚠ Warning: using 'Delivery_CallDateTime' — consider aligning to 'CallDateTime' in cleaning script.")
else:
    raise ValueError("No CallDateTime column found. Check your cleaned CSV.")


df[CALL_COL]       = pd.to_datetime(df[CALL_COL])
df["EndDateTime"]  = pd.to_datetime(df["EndDateTime"], errors="coerce")




# =====================================================
# STEP 1 — Build daily units-on-rent series
#           filtered by branch + subtype
# =====================================================
def build_daily_series(df, branch, subtype, start=SERIES_START, end=SERIES_END):
    """
    Builds a daily cumulative units-on-rent series for a given branch + subtype.
      +1 on CallDateTime day (rental begins)
      -1 on day after EndDateTime (rental ends)
    """
    mask = (
        (df["Delivery_BranchName"] == branch) &
        (df["ModelTypeName"]        == subtype)
    )
    df_f = df[mask].copy()


    if df_f.empty:
        return None


    start_events = df_f.groupby(df_f[CALL_COL].dt.normalize()).size()
    end_events   = (
        df_f.groupby((df_f["EndDateTime"] + pd.Timedelta(days=1)).dt.normalize()).size() * -1
    )


    events = (
        pd.concat([start_events, end_events])
        .groupby(level=0)
        .sum()
        .sort_index()
    )


    full_index = pd.date_range(start=start, end=end, freq="D")
    y = events.reindex(full_index, fill_value=0).cumsum().astype(float)
    y.name = "items_on_rent"
    return y.loc[start:end]




# =====================================================
# STEP 2 — Fit Random Forest forecaster
# =====================================================
def fit_random_forest(y_train, window_length=60):
    """
    Wraps RandomForestRegressor in sktime's make_reduction.
    window_length: how many past days the model looks back at to predict next step.
    No pre-smoothing — RF handles complexity natively.
    """
    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    forecaster = make_reduction(
        regressor,
        strategy="recursive",
        window_length=window_length
    )
    forecaster.fit(y_train)
    return forecaster




# =====================================================
# STEP 3 — Loop over all combos, fit, plot
# =====================================================
def run_all_models(df, branches, subtypes, forecast_start=FORECAST_START):
    for branch in branches:
        for subtype in subtypes:
            print(f"\n{'='*60}")
            print(f"  {branch} | {subtype}")
            print(f"{'='*60}")


            y = build_daily_series(df, branch, subtype)


            if y is None or y.sum() == 0:
                print("  ⚠ No data for this combo — skipping.")
                continue


            y_train = y.loc[:pd.Timestamp(forecast_start) - pd.Timedelta(days=1)]
            y_test  = y.loc[forecast_start:]


            if len(y_train) < 90:
                print("  ⚠ Not enough training data — skipping.")
                continue


            print("  Fitting Random Forest...")
            forecaster = fit_random_forest(y_train)


            fh     = ForecastingHorizon(y_test.index, is_relative=False)
            y_pred = forecaster.predict(fh)


            # --- Metrics ---
            mae  = np.abs(y_test.values - y_pred.values).mean()
            mask = y_test.values != 0
            mape = np.abs(
                (y_test.values[mask] - y_pred.values[mask]) / y_test.values[mask]
            ).mean() * 100
            print(f"  MAE:  {mae:.1f} units")
            print(f"  MAPE: {mape:.1f}%")


            # --- Plot ---
            plt.figure(figsize=(12, 5))
            plt.plot(y_train.index, y_train.values,
                     linewidth=1.0, color="steelblue", label="Train")
            plt.plot(y_test.index,  y_test.values,
                     linewidth=1.2, color="green",     label="Actual")
            plt.plot(y_pred.index,  y_pred.values,
                     linewidth=1.4, linestyle="--", color="orange", label="Forecast")
            plt.axvline(pd.Timestamp(forecast_start),
                        color="gray", linestyle=":", linewidth=0.8, label="Split")
            plt.title(
                f"Random Forest Forecast — {branch} | {subtype}\n"
                f"MAE: {mae:.1f} units  |  MAPE: {mape:.1f}%",
                fontsize=11
            )
            plt.xlabel("Date")
            plt.ylabel("Items on Rent")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.show()




# =====================================================
# RUN
# =====================================================
run_all_models(df, BRANCHES, SUBTYPES)