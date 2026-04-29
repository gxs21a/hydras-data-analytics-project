import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BRANCHES       = ["Pittsburgh", "Baltimore"]
SUBTYPES       = ["Ventilator"]
SERIES_START   = "2023-01-01"
SERIES_END     = "2026-03-24"
FORECAST_START = "2025-09-01"
HORIZONS       = [1, 2, 4, 12]
MIN_WEEKS_SEASONAL = 104   # need 2 full 52-week cycles for seasonal SARIMAX

DATA_PATH = (
    "/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/"
    "hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv"
)
PLOTS_DIR = (
    "/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/"
    "hydras-data-analytics-project/plots"
)

# ─────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["Delivery_CallDateTime", "EndDateTime"])
df = df.dropna(subset=["ModelSubTypeName"])
df = df[df["ModelSubTypeName"] != "Unknown"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_series(df, branch, subtype):
    """Weekly units-on-rent for a given branch/subtype combination."""
    sub = df[(df["Delivery_BranchName"] == branch) & (df["ModelSubTypeName"] == subtype)]
    weeks = pd.date_range(SERIES_START, SERIES_END, freq="W-MON")
    counts = [
        ((sub["Delivery_CallDateTime"] <= w + pd.Timedelta(days=6)) &
         (sub["EndDateTime"] >= w)).sum()
        for w in weeks
    ]
    return pd.Series(counts, index=weeks, name="units_on_rent")


def fit_model(ts_train):
    """Fit SARIMAX; fall back to ARIMA if series is too short for seasonality."""
    seasonal = (1, 1, 1, 52) if len(ts_train) >= MIN_WEEKS_SEASONAL else (0, 0, 0, 0)
    label    = "SARIMAX(1,1,1)(1,1,1,52)" if seasonal[3] else "ARIMA(1,1,1)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(ts_train, order=(1, 1, 1), seasonal_order=seasonal,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res, label


def seasonal_naive(ts_train, steps, season=52):
    """Baseline: repeat the last full seasonal cycle."""
    cycle = ts_train.values[-season:]
    reps  = (steps // season) + 2
    return np.tile(cycle, reps)[:steps]


def metrics(actual, pred):
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae  = np.mean(np.abs(actual - pred))
    # Avoid division by zero — mask out zero actuals
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan
    return round(rmse, 2), round(mae, 2), round(mape, 2)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
all_metrics = []
combos = [(b, s) for b in BRANCHES for s in SUBTYPES]

fig, axes = plt.subplots(len(combos), 3, figsize=(18, 5 * len(combos)))
axes = axes.reshape(len(combos), 3)

for row, (branch, subtype) in enumerate(combos):
    print(f"\n{'─'*50}\n{branch} | {subtype}")

    ts       = build_series(df, branch, subtype)
    ts_train = ts[ts.index <  FORECAST_START]
    ts_test  = ts[ts.index >= FORECAST_START]

    if ts.sum() == 0 or len(ts_train) < 52:
        print("  Skipped — no data or insufficient history.")
        continue

    result, label = fit_model(ts_train)
    print(f"  Model : {label}")

    fc       = result.forecast(steps=len(ts_test))
    fc.index = ts_test.index
    baseline = seasonal_naive(ts_train, len(ts_test))

    # ── Metrics table ──────────────────────────────────────────
    print(f"  {'Horizon':>8}  {'SARIMAX RMSE':>13}  {'Base RMSE':>10}  "
          f"{'SARIMAX MAE':>12}  {'Base MAE':>9}  {'SARIMAX MAPE':>13}  {'Base MAPE':>10}")
    print(f"  {'─'*90}")

    row_metrics = []
    for h in HORIZONS:
        act = ts_test.iloc[:h].values
        s_rmse, s_mae, s_mape = metrics(act, fc.iloc[:h].values)
        b_rmse, b_mae, b_mape = metrics(act, baseline[:h])
        print(f"  {str(h)+'w':>8}  {s_rmse:>13.2f}  {b_rmse:>10.2f}  "
              f"{s_mae:>12.2f}  {b_mae:>9.2f}  {s_mape:>12.2f}%  {b_mape:>9.2f}%")
        entry = dict(Branch=branch, Subtype=subtype, Horizon=f"{h}w",
                     SARIMAX_RMSE=s_rmse, Base_RMSE=b_rmse,
                     SARIMAX_MAE=s_mae,   Base_MAE=b_mae,
                     SARIMAX_MAPE=s_mape, Base_MAPE=b_mape)
        all_metrics.append(entry)
        row_metrics.append(entry)

    # ── Col 0: time-series overview ────────────────────────────
    ax = axes[row, 0]
    ax.plot(ts_train.index, ts_train.values, color="steelblue", lw=1.1, label="Train")
    ax.plot(ts_test.index,  ts_test.values,  color="green",     lw=1.1, label="Actual")
    ax.plot(fc.index,       fc.values,       color="darkorange", lw=1.3, ls="--", label="SARIMAX")
    ax.plot(fc.index,       baseline,        color="crimson",    lw=1.0, ls=":",  label="Baseline")
    ax.axvline(pd.Timestamp(FORECAST_START), color="gray", ls=":", lw=0.8)
    ax.set_title(f"{branch}  |  {subtype}\n{label}", fontsize=9, fontweight="bold")
    ax.set_ylabel("Units on rent"); ax.set_xlabel("Week")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.legend(fontsize=7)

    # ── Col 1: actual vs predicted scatter ────────────────────
    ax = axes[row, 1]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for h, c in zip(HORIZONS, colors):
        ax.scatter(ts_test.iloc[:h].values, fc.iloc[:h].values,
                   color=c, alpha=0.75, s=55, label=f"{h}w", zorder=3)
    lo = min(ts_test.values.min(), fc.values.min()) * 0.95
    hi = max(ts_test.values.max(), fc.values.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.9, ls="--", label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, title="Horizon", title_fontsize=7)

    # ── Col 2: RMSE/MAE — SARIMAX vs Baseline, MAPE on twin axis ──
    ax = axes[row, 2]
    x, w = np.arange(len(HORIZONS)), 0.15
    s_rmse = [m["SARIMAX_RMSE"] for m in row_metrics]
    b_rmse = [m["Base_RMSE"]    for m in row_metrics]
    s_mae  = [m["SARIMAX_MAE"]  for m in row_metrics]
    b_mae  = [m["Base_MAE"]     for m in row_metrics]
    s_mape = [m["SARIMAX_MAPE"] for m in row_metrics]
    b_mape = [m["Base_MAPE"]    for m in row_metrics]

    ax.bar(x - 1.5*w, s_rmse, w, color="steelblue",  alpha=0.85, label="SARIMAX RMSE")
    ax.bar(x - 0.5*w, b_rmse, w, color="steelblue",  alpha=0.40, label="Baseline RMSE")
    ax.bar(x + 0.5*w, s_mae,  w, color="darkorange", alpha=0.85, label="SARIMAX MAE")
    ax.bar(x + 1.5*w, b_mae,  w, color="darkorange", alpha=0.40, label="Baseline MAE")

    for vals, offsets in [(s_rmse, x-1.5*w), (b_rmse, x-0.5*w),
                          (s_mae,  x+0.5*w), (b_mae,  x+1.5*w)]:
        for xp, v in zip(offsets, vals):
            ax.text(xp, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=6)

    # MAPE on twin axis (percentage scale separate from unit-based errors)
    ax2 = ax.twinx()
    ax2.plot(x, s_mape, color="green", marker="o", lw=1.5, ms=5, label="SARIMAX MAPE%")
    ax2.plot(x, b_mape, color="green", marker="s", lw=1.0, ms=5, ls="--", label="Baseline MAPE%")
    ax2.set_ylabel("MAPE (%)", color="green", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="green", labelsize=7)

    ax.set_xticks(x); ax.set_xticklabels([f"{h}w" for h in HORIZONS], fontsize=8)
    ax.set_ylabel("Error (units)")
    ax.set_title("SARIMAX vs Baseline — RMSE, MAE & MAPE", fontsize=9, fontweight="bold")

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, ncol=2)
    ax.set_ylim(bottom=0)


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(metrics_df.to_string(index=False))

os.makedirs(PLOTS_DIR, exist_ok=True)
plt.tight_layout()
plt.suptitle("SARIMAX Forecast Evaluation  ·  SARIMAX vs Seasonal-Naive Baseline",
             fontsize=12, y=1.01)
plot_path    = os.path.join(PLOTS_DIR, "sarimax_results.png")
metrics_path = os.path.join(PLOTS_DIR, "sarimax_metrics.csv")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()

metrics_df.to_csv(metrics_path, index=False)
print(f"\nSaved: {plot_path}\n       {metrics_path}")

# MAPE Plot for presentation!!

def plot_mape_executive(all_metrics):
    """
    Executive-facing MAPE chart: AI Model vs Baseline, per city.
    Audience: non-technical stakeholders.
    """
    cities   = ["Pittsburgh", "Baltimore"]
    horizons = ["1 Week", "2 Weeks", "4 Weeks", "12 Weeks"]
    h_keys   = ["1w", "2w", "4w", "12w"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))  # wider + taller
    fig.patch.set_facecolor("white")

    TEAL    = "#1D9E75"
    AMBER   = "#D4A843"
    CARD    = "#F8F9FA"
    LABEL   = "#44448F"
    MUTED   = "#6B7280"

    df = {
        (m["Branch"], m["Horizon"]): m
        for m in all_metrics
        if m["Subtype"] == "Ventilator"
    }

    for ax, city in zip(axes, cities):
        s_mape = [df.get((city, h), {}).get("SARIMAX_MAPE", np.nan) for h in h_keys]
        b_mape = [df.get((city, h), {}).get("Base_MAPE",    np.nan) for h in h_keys]

        x = np.arange(len(horizons))
        w = 0.32

        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=LABEL, labelsize=11)
        ax.xaxis.set_tick_params(bottom=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.8)

        ax.bar(x - w/2, b_mape, w, color=AMBER, alpha=0.85,
               label="Baseline", zorder=3, linewidth=0)
        ax.bar(x + w/2, s_mape, w, color=TEAL, alpha=0.90,
               label="Our Model", zorder=3, linewidth=0)

        # ── Value labels: stacked above bar, no overlap ──────────
        bar_max = max([v for v in b_mape + s_mape if not np.isnan(v)], default=1)
        label_pad   = bar_max * 0.02   # small gap between bar top and number
        improve_pad = bar_max * 0.10   # gap above the taller bar for "↓ Xpp" line

        for xi, (sv, bv) in enumerate(zip(s_mape, b_mape)):
            if np.isnan(sv) or np.isnan(bv):
                continue

            # Baseline value label
            ax.text(xi - w/2, bv + label_pad,
                    f"{bv:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#7A5C1E")

            # AI model value label
            ax.text(xi + w/2, sv + label_pad,
                    f"{sv:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#0F6E56")

            # Improvement callout — always above the taller of the two bars
            improvement = bv - sv
            if improvement > 0:
                top = max(sv, bv) + improve_pad
                ax.text(xi, top,
                        f"↓ {improvement:.1f}pp better",
                        ha="center", va="bottom",
                        fontsize=8, color=TEAL, fontstyle="italic")

        # ── Axis headroom so callouts aren't clipped ─────────────
        ax.set_ylim(0, bar_max * 1.30)

        ax.set_xticks(x)
        ax.set_xticklabels(horizons, color=LABEL, fontsize=11)
        ax.set_ylabel("Forecast Error  (lower = more accurate)",
                      color=MUTED, fontsize=11, labelpad=10)
        ax.set_title(f"{city}", fontsize=16, fontweight="bold",
                     color=LABEL, pad=16)

    # ── Legend ────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=AMBER, alpha=0.85, label="Baseline"),
        mpatches.Patch(color=TEAL,  alpha=0.90, label="Our Model"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               frameon=False, fontsize=12, labelcolor=LABEL,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "How accurately does each approach predict Ventilator demand?",
        fontsize=17, fontweight="bold", color=LABEL, y=1.02
    )
    fig.text(0.5, 0.97,
             "MAPE = % error vs actual units on rent  ·  Lower bars mean fewer costly over- or under-orders",
             ha="center", fontsize=11, color=MUTED)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(os.path.join(PLOTS_DIR, "mape_executive.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("Saved: mape_executive.png")

plot_mape_executive(all_metrics)