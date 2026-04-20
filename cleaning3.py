import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/rentals.csv', low_memory=False)
print("Number of rows in raw data: ", data.shape[0])

# ModelTypeName -> Drop `Unknown` and missing
data = data[data["ModelTypeName"] != "Unknown"]
data = data.dropna(subset=["ModelTypeName"])

# Removing voided model transactions (where IsVoid == 1)
print("Number of IsVoid == 1: ", data[data["IsVoid"] == 1].shape[0])
data = data[data["IsVoid"] != 1]

# Convert to datetime format
data["StartDateTime"] = pd.to_datetime(data["StartDateTime"])
data["EndDateTime"] = pd.to_datetime(data["EndDateTime"])
data["Delivery_CallDateTime"] = pd.to_datetime(data["Delivery_CallDateTime"])  # NEW

START_DATE = "2022-07-01"
END_DATE = "2025-12-31"

# Only keep StartDateTime between start and end date
data = data[
    (data["StartDateTime"] >= START_DATE)
    & (data["StartDateTime"] <= END_DATE)
]

# Fill missing EndDateTime with end date ADJUST HERE AS NEEDED!!
#data["EndDateTime"] = data["EndDateTime"].fillna(pd.Timestamp(END_DATE))

# Cut off EndDateTime in 2026
data = data[data["EndDateTime"] <= END_DATE]

# Drop rows where StartDateTime OR Delivery_CallDateTime is after EndDateTime
data = data[data["EndDateTime"] > data["StartDateTime"]]
data = data[data["EndDateTime"] > data["Delivery_CallDateTime"]]

print("\nNumber of rows in cleaned data: ", data.shape[0])

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
# MODEL SUBTYPE MERGE (FIXED VERSION)
# =====================================================

# Load model lookup
model_df = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/models.csv',
    encoding='latin1'
)

# ---- CLEAN ModelID (CRITICAL FIX) ----
def clean_model_id(col):
    return (
        col.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

data["ModelID"] = clean_model_id(data["ModelID"])
model_df["ModelID"] = clean_model_id(model_df["ModelID"])

# ---- DEBUG: ID overlap ----
print("\nMODELID DIAGNOSTICS")
data_ids = set(data["ModelID"].unique())
model_ids = set(model_df["ModelID"].unique())

print("Unique rental ModelIDs:", len(data_ids))
print("Unique model lookup IDs:", len(model_ids))
print("Matching IDs:", len(data_ids & model_ids))
print("Missing in lookup:", len(data_ids - model_ids))

# ---- CLEAN LOOKUP ----
model_lookup = (
    model_df[["ModelID", "ModelSubTypeName"]]
    .drop_duplicates(subset="ModelID")
)

# ---- MERGE ----
data = data.merge(
    model_lookup,
    on="ModelID",
    how="left"
)

# ---- POST-MERGE CHECK ----
nulls = data["ModelSubTypeName"].isna().sum()

print("\nPOST-MERGE CHECK")
print("Null ModelSubTypeName:", nulls)
print("Percent missing:", round(nulls / len(data) * 100, 2), "%")

# Show examples of missing matches
missing_examples = (
    data[data["ModelSubTypeName"].isna()]["ModelID"]
    .unique()[:10]
)

print("\nSample missing ModelIDs:", missing_examples)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["RentalDuration"] = data["EndDateTime"] - data["StartDateTime"]

print("\nRental duration summary:")
print(data["RentalDuration"].describe())

# =====================================================
# QUICK CHECK PLOT
# =====================================================
import matplotlib.pyplot as plt

plt.hist(data["RentalDuration"].dt.days, bins=50)
plt.xlabel("Rental Duration (Days)")
plt.ylabel("Count")
plt.title("Distribution of Rental Duration")
plt.show()

print(data[["ModelID", "ModelTypeName", "ModelSubTypeName"]].dropna(subset=["ModelSubTypeName"]).head(20))

# =====================================================
# SAVE CLEAN DATA
# =====================================================
output_path = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'

data.to_csv(output_path, index=False)

print("\nSaved cleaned dataset to:", output_path)