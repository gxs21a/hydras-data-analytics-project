import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD DATA
# =====================================================
data = pd.read_csv(
    '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/rentals.csv',
    low_memory=False
)

print("Number of rows in raw data:", data.shape[0])

# =====================================================
# BASIC CLEANING
# =====================================================
data = data[data["ModelTypeName"] != "Unknown"]
data = data.dropna(subset=["ModelTypeName"])

print("Number of IsVoid == 1:", data[data["IsVoid"] == 1].shape[0])
data = data[data["IsVoid"] != 1]

data = data[data["DeliveryStatusDesc"].isin(["Delivered", "Completed", "Turn Down", "Partial"])]

# Convert datetime
data["Delivery_CallDateTime"] = pd.to_datetime(data["Delivery_CallDateTime"], errors="coerce")
data["EndDateTime"] = pd.to_datetime(data["EndDateTime"], errors="coerce")

# =====================================================
# DATE FILTERING (FIXED)
# =====================================================
START_DATE = pd.Timestamp("2022-07-01")
END_DATE   = pd.Timestamp("2026-03-31")

# Keep based ONLY on delivery date
data = data[
    (data["Delivery_CallDateTime"] >= START_DATE) &
    (data["Delivery_CallDateTime"] <= END_DATE)
].copy()

print("Missing EndDateTime after filtering:", data["EndDateTime"].isna().sum())

# Remove impossible rows only
invalid = (
    data["EndDateTime"].notna() &
    (data["Delivery_CallDateTime"] > data["EndDateTime"])
).sum()
print("Invalid rows:", invalid)

data = data[
    data["EndDateTime"].isna() |
    (data["EndDateTime"] >= data["Delivery_CallDateTime"])
].copy()

# =====================================================
# CRITICAL FIX: PRESERVE ACTIVE RENTALS
# =====================================================
data["EffectiveEndDateTime"] = data["EndDateTime"]

# If still on rent → assume active through END_DATE
data.loc[data["EffectiveEndDateTime"].isna(), "EffectiveEndDateTime"] = END_DATE

# If ends after END_DATE → cap at END_DATE
data["EffectiveEndDateTime"] = data["EffectiveEndDateTime"].clip(upper=END_DATE)

# =====================================================
# BRANCH CLEANUP
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
    "zClosed-Pittsburgh": "Pittsburgh"
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
    return col.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

data["ModelID"] = clean_model_id(data["ModelID"])
model_df["ModelID"] = clean_model_id(model_df["ModelID"])

model_lookup = model_df[["ModelID", "ModelSubTypeName"]].drop_duplicates()
data = data.merge(model_lookup, on="ModelID", how="left")

data = data.dropna(subset=["ModelSubTypeName"])

# =====================================================
# SAVE
# =====================================================
output_path = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_forecasting_v1.csv'
data.to_csv(output_path, index=False)

print("\nSaved cleaned dataset to:", output_path)