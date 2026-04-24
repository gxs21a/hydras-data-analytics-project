import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/rentals_all.csv', low_memory=False)
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
END_DATE = "2026-03-31"


# Only keep Delivery_CallDateTime between start and end date
data = data[
    (data["Delivery_CallDateTime"] >= START_DATE)
    & (data["Delivery_CallDateTime"] <= END_DATE)
]

print("Missing EndDateTime AFTER filtering:", data["EndDateTime"].isna().sum())

# Cut off EndDateTime in 2026, but preserve rows with missing EndDateTime for imputation REVISIT IF NECESSARY!!!!
# data = data[(data["EndDateTime"] <= END_DATE) | (data["EndDateTime"].isna())]

# Count rows where Delivery_CallDateTime is after EndDateTime
count_invalid = (data["Delivery_CallDateTime"] > data["EndDateTime"]).sum()
print(f"Number of rows where Delivery_CallDateTime is after EndDateTime: {count_invalid}")

# Drop rows where Delivery_CallDateTime is after EndDateTime (preserve missing EndDateTime)
data = data[(data["EndDateTime"] > data["Delivery_CallDateTime"]) | (data["EndDateTime"].isna())]

print("Number of rows in cleaned data: ", data.shape[0])


# Create a Dictionary of all Branches that need name subs
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

# Substitute dictionary of cities - DO NOT touch branches not included in the dictionary
data["Delivery_BranchName"] = data["Delivery_BranchName"].map(CITY_MAP_DICT).fillna(data["Delivery_BranchName"])


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
# IMPUTATION PIPELINE
# =====================================================

# Compute mean duration per ModelSubTypeName (in seconds) from non-missing rows
non_missing = data.dropna(subset=["EndDateTime"]).copy()
non_missing["dur"] = (non_missing["EndDateTime"] - non_missing["Delivery_CallDateTime"]).dt.total_seconds()

duration_by_subtype = non_missing.groupby("ModelSubTypeName")["dur"].mean()

# Global fallback mean
global_duration = non_missing["dur"].mean()

missing_mask = data["EndDateTime"].isna()
print("\nMissing EndDateTime BEFORE imputation:", missing_mask.sum())

impute_values = data.loc[missing_mask, "ModelSubTypeName"].map(duration_by_subtype).fillna(global_duration)
data.loc[missing_mask, "EndDateTime"] = (
    data.loc[missing_mask, "Delivery_CallDateTime"] +
    pd.to_timedelta(impute_values, unit="s")
)

print("Missing EndDateTime AFTER imputation:", data["EndDateTime"].isna().sum())
print("Rows after imputation:", data.shape[0])


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Create `RentalDuration` column (in seconds)
data["RentalDuration"] = (data["EndDateTime"] - data["Delivery_CallDateTime"]).dt.total_seconds()

# Plot of Rental Duration
plt.hist(data["RentalDuration"] / 86400)  # convert seconds to days
plt.xlabel("Rental Duration (Days)")
plt.ylabel("Count (Models on Rent)")
plt.title("Distribution of Rental Duration Across Model Types")
plt.show()


# =====================================================
# SAVE CLEAN DATA
# =====================================================
output_path = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'

data.to_csv(output_path, index=False)

print("\nSaved cleaned dataset to:", output_path)