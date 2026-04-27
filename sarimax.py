import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =====================================================
# CONFIG
# =====================================================
BRANCHES = ["Pittsburgh", "Miami"]
SUBTYPES = ["Ventilator", "Pump Module"]

SERIES_START   = "2023-01-01"
SERIES_END     = "2026-03-24"
FORECAST_START = "2025-09-01"

HORIZONS = [1, 3, 6, 12]

# Minimum weeks of training data needed to support a 52-week seasonal period.
# SARIMAX needs at least 2 full seasonal cycles to estimate seasonal params.
MIN_WEEKS_FOR_SEASONAL = 104

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'
)

df["Delivery_CallDateTime"] = pd.to_datetime(df["Delivery_CallDateTime"])
df["EndDateTime"]           = pd.to_datetime(df["EndDateTime"])

df = df.dropna(subset=["ModelSubTypeName"])
df = df[df["ModelSubTypeName"] != "Unknown"]

# =====================================================
# HELPERS
# =====================================================
def build_time_series(df, branch, subtype):
    mask = (
        (df["Delivery_BranchName"] == branch) &
        (df["ModelSubTypeName"]    == subtype)
    )
    df_f = df[mask].copy()

    weeks = pd.date_range(start=SERIES_START, end=SERIES_END, freq="W-MON")
    counts = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        active = (
            (df_f["Delivery_CallDateTime"] <= week_end) &
            (df_f["EndDateTime"]            >= week_start)
        ).sum()
        counts.append(active)

    return pd.Series(counts, index=weeks, name="units_on_rent")

def fit_sarimax(ts_train):
    """
    Fit SARIMAX with adaptive seasonal order.
    If the training series is too short for 52-week seasonality,
    fall back to a non-seasonal ARIMA(1,1,1) to avoid zeroed-out
    seasonal parameters and the associated warning.
    """
    if len(ts_train) >= MIN_WEEKS_FOR_SEASONAL:
        seasonal_order = (1, 1, 1, 52)
        model_label = "SARIMAX(1,1,1)(1,1,1,52)"
    else:
        seasonal_order = (0, 0, 0, 0)
        model_label = "ARIMA(1,1,1) — too few obs for seasonal"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            ts_train,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)

    return result, model_label


# =====================================================
# MAIN LOOP
# =====================================================
all_metrics = []

n_combos = len(BRANCHES) * len(SUBTYPES)

# 3 columns: time series | actual vs predicted | error bars
fig, axes = plt.subplots(n_combos, 3, figsize=(18, 5 * n_combos))
axes = axes.reshape(n_combos, 3)

row = 0
for branch in BRANCHES:
    for subtype in SUBTYPES:

        print(f"\n{branch} | {subtype}")

        ts = build_time_series(df, branch, subtype)
        ts_train = ts[ts.index <  FORECAST_START]
        ts_test  = ts[ts.index >= FORECAST_START]

        if ts.sum() == 0 or len(ts_train) < 52:
            print("  Skipping — no data or insufficient history.")
            row += 1
            continue

        result, model_label = fit_sarimax(ts_train)
        print(f"  Model: {model_label}")

        fc = result.forecast(steps=len(ts_test))
        fc.index = ts_test.index

        # ── Metrics ────────────────────────────────────────────
        print(f"  {'Horizon':>10}  {'RMSE':>8}  {'MAE':>8}")
        print(f"  {'─'*30}")

        row_metrics = []
        for h in HORIZONS:
            actual = ts_test.iloc[:h].values
            pred   = fc.iloc[:h].values
            rmse   = np.sqrt(np.mean((actual - pred) ** 2))
            mae    = np.mean(np.abs(actual - pred))
            print(f"  {str(h) + 'w':>10}  {rmse:>8.2f}  {mae:>8.2f}")
            entry = {
                "Branch":  branch,
                "Subtype": subtype,
                "Horizon": f"{h}w",
                "RMSE":    round(rmse, 2),
                "MAE":     round(mae,  2),
            }
            all_metrics.append(entry)
            row_metrics.append(entry)

        # ── Col 0: full time series + forecast ─────────────────
        ax0 = axes[row, 0]
        ax0.plot(ts_train.index, ts_train.values,
                 color="steelblue", linewidth=1.1, label="Train")
        ax0.plot(ts_test.index, ts_test.values,
                 color="green", linewidth=1.1, label="Actual")
        ax0.plot(fc.index, fc.values,
                 color="darkorange", linewidth=1.3,
                 linestyle="--", label="Forecast")
        ax0.axvline(pd.Timestamp(FORECAST_START),
                    color="gray", linestyle=":", linewidth=0.8)
        ax0.set_title(f"{branch}  |  {subtype}\n{model_label}",
                      fontsize=9, fontweight="bold")
        ax0.set_ylabel("Units on rent")
        ax0.set_xlabel("Week")
        ax0.tick_params(axis="x", labelsize=7, rotation=30)
        ax0.legend(fontsize=8)

        # ── Col 1: actual vs predicted scatter ─────────────────
        ax1 = axes[row, 1]

        # Plot one series per horizon so each gets its own colour + label
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
        for h, color in zip(HORIZONS, colors):
            actual_h = ts_test.iloc[:h].values
            pred_h   = fc.iloc[:h].values
            ax1.scatter(actual_h, pred_h, color=color, alpha=0.75,
                        s=55, label=f"{h}w", zorder=3)

        # Perfect-prediction line across the combined range
        all_actual = ts_test.values
        all_pred   = fc.values
        lo = min(all_actual.min(), all_pred.min()) * 0.95
        hi = max(all_actual.max(), all_pred.max()) * 1.05
        ax1.plot([lo, hi], [lo, hi], color="gray", linewidth=0.9,
                 linestyle="--", label="Perfect fit")

        ax1.set_xlabel("Actual (units on rent)")
        ax1.set_ylabel("Predicted (units on rent)")
        ax1.set_title("Actual vs predicted", fontsize=9, fontweight="bold")
        ax1.legend(fontsize=7, title="Horizon", title_fontsize=7)

        # ── Col 2: RMSE & MAE bar chart ────────────────────────
        ax2 = axes[row, 2]
        rmse_vals = [m["RMSE"] for m in row_metrics]
        mae_vals  = [m["MAE"]  for m in row_metrics]

        x     = np.arange(len(HORIZONS))
        bar_w = 0.35
        ax2.bar(x - bar_w / 2, rmse_vals, bar_w,
                color="steelblue",  alpha=0.85, label="RMSE")
        ax2.bar(x + bar_w / 2, mae_vals,  bar_w,
                color="darkorange", alpha=0.85, label="MAE")

        for x_pos, v in zip(np.concatenate([x - bar_w / 2, x + bar_w / 2]),
                            rmse_vals + mae_vals):
            ax2.text(x_pos, v + 0.05, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{h}w" for h in HORIZONS], fontsize=8)
        ax2.set_ylabel("Error (units)")
        ax2.set_title("RMSE & MAE by horizon", fontsize=9, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)

        row += 1


# =====================================================
# SUMMARY TABLE
# =====================================================
metrics_df = pd.DataFrame(all_metrics)

print("\n" + "="*55)
print("  SUMMARY")
print("="*55)

pivot = metrics_df.pivot_table(
    index=["Branch", "Subtype"],
    columns="Horizon",
    values=["RMSE", "MAE"]
).reindex(["1w", "3w", "6w", "12w"], axis=1, level=1)

print(pivot.to_string())

plt.tight_layout()
plt.suptitle("SARIMAX Forecast Evaluation", fontsize=12, y=1.01)
plt.show()