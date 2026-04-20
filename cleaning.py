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

# Convert to datetime format
data["StartDateTime"] = pd.to_datetime(data["StartDateTime"])
data["EndDateTime"] = pd.to_datetime(data["EndDateTime"])

START_DATE = "2022-07-01"
END_DATE = "2026-03-31"

# Only keep StartDateTime between start and end date
data = data[
    (data["StartDateTime"] >= START_DATE)
    & (data["StartDateTime"] <= END_DATE)
]

# Fill missing EndDateTime with end date
data["EndDateTime"] = data["EndDateTime"].fillna(pd.Timestamp(END_DATE))

# Cut off EndDateTime in 2026
data = data[data["EndDateTime"] <= END_DATE]

# Drop rows where StartDateTime is after EndDateTime
data = data[data["EndDateTime"] > data["StartDateTime"]]

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

# Substitute dictionary of cities DO NOT touch branches not inluded in the dictionary
data["Delivery_BranchName"] = data["Delivery_BranchName"].map(CITY_MAP_DICT).fillna(data["Delivery_BranchName"])

## Feature Engineering:

# Create `RentalDuration` column
data["RentalDuration"] = data["EndDateTime"] - data["StartDateTime"]

# Plt of Rental Duration
plt.hist((data["RentalDuration"].dt.days))
plt.xlabel("Rental Duration (Days)")
plt.ylabel("Count (Models on Rent)")
plt.title("Distribution of Rental Duration Across Model Types")
plt.show()

data.to_csv('/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW.csv', index=False)