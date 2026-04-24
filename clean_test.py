import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/rentals_all.csv',
    low_memory=False
)
print("Number of rows in raw data: ", data.shape[0])


# ModelTypeName -> Drop `Unknown` and missing
data = data[data["ModelTypeName"] != "Unknown"]
data = data.dropna(subset=["ModelTypeName"])


# Removing voided model transactions (where IsVoid == 1)
print("Number of IsVoid == 1: ", data[data["IsVoid"] == 1].shape[0])
data = data[data["IsVoid"] != 1]


# Only use DeliveryStatusDesc Delivered, Completed, Turned-Down, and Partial (drop the rest)
data = data[data["DeliveryStatusDesc"].isin(["Delivered", "Completed", "Turn Down", "Partial"])]


# Convert to datetime format
data["Delivery_CallDateTime"] = pd.to_datetime(data["Delivery_CallDateTime"])
data["EndDateTime"] = pd.to_datetime(data["EndDateTime"])


START_DATE = "2022-07-01"
END_DATE   = "2026-03-31"

# Data export / censoring date — open rentals are right-censored here,
# NOT imputed with a mean duration.
CENSORED_END = pd.Timestamp(END_DATE)


# Only keep Delivery_CallDateTime between start and end date
data = data[
    (data["Delivery_CallDateTime"] >= START_DATE)
    & (data["Delivery_CallDateTime"] <= END_DATE)
]

print("Missing EndDateTime AFTER filtering:", data["EndDateTime"].isna().sum())


# Count rows where Delivery_CallDateTime is after EndDateTime (only for non-null EndDateTime)
count_invalid = (data["Delivery_CallDateTime"] > data["EndDateTime"]).sum()
print(f"Number of rows where Delivery_CallDateTime is after EndDateTime: {count_invalid}")

# Drop rows where Delivery_CallDateTime is after EndDateTime (preserve missing EndDateTime)
data = data[(data["EndDateTime"] > data["Delivery_CallDateTime"]) | (data["EndDateTime"].isna())]

print("Number of rows in cleaned data: ", data.shape[0])


# =====================================================
# BRANCH NAME STANDARDISATION
# =====================================================
CITY_MAP_DICT = {
    "Anaheim-BTS": "Anaheim",
    "Atlanta-Storage": "Atlanta",
    "Baltimore-BTS": "Baltimore",
    "Burbank - ALL": "Burbank",
    "Chicago - Storage": "Chicago",
    "Denver-Storage": "Denver",
    "Detroit-BTS": "Detroit",
    "Houston Distribution Center": "Houston",
    "Houston-BTS": "Houston",
    "Jacksonville Closed": "Jacksonville",
    "Lafayette BTS": "Lafayette",
    "Las Vegas-Closed": "Las Vegas",
    "Memphis - Methodist University": "Memphis",
    "Metro-BTS": "Metro",
    "Mobile-Storage": "Mobile",
    "Monterey-BTS": "Monterey",
    "Nashville-BTS-Closed": "Nashville",
    "Orlando-BTS": "Orlando",
    "Phoenix Support Center": "Phoenix",
    "Richmond Closed": "Richmond",
    "Sacramento-BTS": "Sacramento",
    "San Francisco-BTS-Closed": "San Francisco",
    "St. Louis-BTS": "St. Louis",
    "Z - Dallas Closed": "Dallas",
    "Z Closed - East Baltimore-BTS": "Baltimore",
    "zClosed-Boston": "Boston",
    "zClosed-Columbus": "Columbus",
    "zClosed-Pittsburgh": "Pittsburgh",
    "zClosed-Salt Lake-BTS": "Salt Lake City",
    "zClosed-Seattle": "Seattle",
    "zz Long Island - Do Not Use": "Long Island",
    "zzz-Birmingham Closed": "Birmingham",
    "zzz-Closed-Kansas City-BTS": "Kansas City",
    "zzz-Shreveport-BTS Closed": "Shreveport"
}

data["Delivery_BranchName"] = (
    data["Delivery_BranchName"]
    .map(CITY_MAP_DICT)
    .fillna(data["Delivery_BranchName"])
)


# =====================================================
# MODEL SUBTYPE MERGE
# =====================================================
model_df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/models.csv',
    encoding='latin1'
)


def clean_model_id(col):
    return (
        col.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


data["ModelID"]     = clean_model_id(data["ModelID"])
model_df["ModelID"] = clean_model_id(model_df["ModelID"])


print("\nMODELID DIAGNOSTICS")
data_ids  = set(data["ModelID"].unique())
model_ids = set(model_df["ModelID"].unique())
print("Unique rental ModelIDs:", len(data_ids))
print("Unique model lookup IDs:", len(model_ids))
print("Matching IDs:", len(data_ids & model_ids))
print("Missing in lookup:", len(data_ids - model_ids))


model_lookup = (
    model_df[["ModelID", "ModelSubTypeName"]]
    .drop_duplicates(subset="ModelID")
)

data = data.merge(model_lookup, on="ModelID", how="left")

nulls = data["ModelSubTypeName"].isna().sum()
print("\nPOST-MERGE CHECK")
print("Null ModelSubTypeName:", nulls)
print("Percent missing:", round(nulls / len(data) * 100, 2), "%")


# =====================================================
# IMPUTATION PIPELINE — FIXED
#
# KEY CHANGE: null EndDateTime means the rental is STILL ACTIVE
# at the time of data export. These are right-censored observations.
# We set their EndDateTime to CENSORED_END (the export date) so they
# remain counted as active in the time series all the way to the end
# of the window.
#
# Previously, these were filled with a mean rental duration, which
# made active rentals appear to end a few days/weeks after delivery,
# causing a false drop-to-zero in the forecast window.
# =====================================================

# Flag which rows were originally missing EndDateTime
data["EndDateTime_was_censored"] = data["EndDateTime"].isna()

missing_mask = data["EndDateTime"].isna()
print("\nMissing EndDateTime BEFORE imputation:", missing_mask.sum())
print("These will be treated as right-censored (still on rent at export date).")

# Right-censored rows: rental is still open — use the export date as end
data.loc[missing_mask, "EndDateTime"] = CENSORED_END

print("Missing EndDateTime AFTER imputation:", data["EndDateTime"].isna().sum())
print("Rows after imputation:", data.shape[0])

# Sanity check: how many were censored by branch/subtype
censored_summary = (
    data[data["EndDateTime_was_censored"]]
    .groupby(["Delivery_BranchName", "ModelSubTypeName"])
    .size()
    .reset_index(name="censored_count")
    .sort_values("censored_count", ascending=False)
)
print("\nTop censored (still-active) rentals by branch + subtype:")
print(censored_summary.head(20).to_string(index=False))


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Create `RentalDuration` column (in seconds)
# Note: censored rows will show duration up to export date — expected
data["RentalDuration"] = (
    data["EndDateTime"] - data["Delivery_CallDateTime"]
).dt.total_seconds()

# =====================================================
# SAVE CLEAN DATA
# =====================================================
output_path = (
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/'
    'hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'
)

data.to_csv(output_path, index=False)
print("\nSaved cleaned dataset to:", output_path)