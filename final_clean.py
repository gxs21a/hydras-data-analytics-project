import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# LOAD RAW DATA
# =====================================================
data = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/rentals.csv',
    low_memory=False
)

print("Raw rows:", data.shape[0])

# =====================================================
# BASIC CLEANING
# =====================================================

# Remove Unknown + null ModelTypeName
data = data[data["ModelTypeName"] != "Unknown"]
data = data.dropna(subset=["ModelTypeName"])

# Remove voided transactions
print("IsVoid == 1:", data[data["IsVoid"] == 1].shape[0])
data = data[data["IsVoid"] != 1]

# =====================================================
# DATE CONVERSION
# =====================================================
data["StartDateTime"]         = pd.to_datetime(data["StartDateTime"])
data["EndDateTime"]           = pd.to_datetime(data["EndDateTime"], errors="coerce")
data["Delivery_CallDateTime"] = pd.to_datetime(data["Delivery_CallDateTime"], errors="coerce")

START_DATE = pd.to_datetime("2022-07-01")
END_DATE   = pd.to_datetime("2025-12-31")

# ⚠️ Do NOT fill missing EndDateTimes here.
# Rows with blank EndDateTime will be imputed later.

# =====================================================
# FILTER VALID DATE LOGIC
# =====================================================
data = data[
    data["EndDateTime"].isna() |
    (
        (data["EndDateTime"] <= END_DATE) &
        (data["EndDateTime"] > data["StartDateTime"]) &
        (data["EndDateTime"] > data["Delivery_CallDateTime"]) &
        (data["Delivery_CallDateTime"] >= data["StartDateTime"])
    )
]

print("Rows after cleaning:", data.shape[0])

# =====================================================
# BRANCH CLEANING
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

data["Delivery_BranchName"] = data["Delivery_BranchName"].map(CITY_MAP_DICT).fillna(
    data["Delivery_BranchName"]
)

# =====================================================
# MODEL SUBTYPE MERGE
# =====================================================

# Load model lookup
model_df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/models.csv',
    encoding='latin1'
)

# Clean ModelID for reliable joining
def clean_model_id(col):
    return (
        col.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

data["ModelID"]     = clean_model_id(data["ModelID"])
model_df["ModelID"] = clean_model_id(model_df["ModelID"])

# Debug: ID overlap
print("\nMODELID DIAGNOSTICS")
data_ids  = set(data["ModelID"].unique())
model_ids = set(model_df["ModelID"].unique())
print("Unique rental ModelIDs:", len(data_ids))
print("Unique model lookup IDs:", len(model_ids))
print("Matching IDs:", len(data_ids & model_ids))
print("Missing in lookup:", len(data_ids - model_ids))

# Deduplicated lookup
model_lookup = (
    model_df[["ModelID", "ModelSubTypeName"]]
    .drop_duplicates(subset="ModelID")
)

# Merge subtype onto main data
data = data.merge(model_lookup, on="ModelID", how="left")

# Post-merge check
nulls = data["ModelSubTypeName"].isna().sum()
print("\nPOST-MERGE CHECK")
print("Null ModelSubTypeName:", nulls)
print("Percent missing:", round(nulls / len(data) * 100, 2), "%")
print("Sample missing ModelIDs:", data[data["ModelSubTypeName"].isna()]["ModelID"].unique()[:10])


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# RentalDuration only for rows that already have an EndDateTime
data["RentalDuration"] = data["EndDateTime"] - data["StartDateTime"]

print("\nRental duration summary (known end dates only):")
print(data["RentalDuration"].describe())


# =====================================================
# QUICK CHECK PLOT
# =====================================================

plt.hist(data["RentalDuration"].dt.days.dropna(), bins=50)
plt.xlabel("Rental Duration (Days)")
plt.ylabel("Count")
plt.title("Distribution of Rental Duration (known end dates)")
plt.show()


# =====================================================
# SAVE CLEAN DATA
# =====================================================
output_path = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'

data.to_csv(output_path, index=False)
print("\nSaved cleaned dataset to:", output_path)
print("Rows with blank EndDateTime (to be imputed):", data["EndDateTime"].isna().sum())