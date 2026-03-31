import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv')

df.head()
df["Delivery_CallDateTime"] = pd.to_datetime(df["Delivery_CallDateTime"])
df["EndDateTime"] = pd.to_datetime(df["EndDateTime"], errors="coerce")

start_events = df.groupby("Delivery_CallDateTime").size()

# -1 on day after end day
end_events = df.groupby(df["EndDateTime"] + pd.Timedelta(days=1)).size() * -1
events = pd.concat([start_events, end_events]).groupby(level=0).sum().sort_index()

# full daily series
full_index = pd.date_range(events.index.min(), events.index.max(), freq="D")
y = events.reindex(full_index, fill_value=0).cumsum().astype(float)
y.name = "items_on_rent"

y = y.loc["2023-01-01":"2025-12-31"]

# train/test split
y_train = y.loc["2023-01-01":"2025-06-30"]
y_test = y.loc["2025-07-01":"2025-12-31"]
regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


forecaster = make_reduction(
    regressor,
    strategy="recursive",
    window_length=60
)

# fit
forecaster.fit(y_train)

# predict exact test dates
fh = ForecastingHorizon(y_test.index, is_relative=False)
y_pred = forecaster.predict(fh)

plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_pred.index, y_pred, label="Forecast")
plt.title(f"Forecast of Items on Rent - {Delivery_BranchName}")
plt.xlabel("Date")
plt.ylabel("Items on Rent")
plt.legend()
plt.tight_layout()
plt.show()