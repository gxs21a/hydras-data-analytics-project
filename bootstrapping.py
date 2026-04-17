# =====================================================
# BOOTSTRAPPING & IMPUTATION
# =====================================================
# Run AFTER cleaning.py.
# Reads cleaned_data_NEW_v2.csv where EndDateTime is
# intentionally blank for rows that need imputation.
#
# Columns used:
#   Start date : Delivery_CallDateTime
#   Subtype    : ModelSubTypeName
#   Status     : DeliveryStatusDesc
# =====================================================

import pandas as pd
import numpy as np

# =====================================================
# CONFIG — update paths as needed
# =====================================================

CLEANED_CSV = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/cleaned_data_NEW_v2.csv'
OUTPUT_CSV  = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/DataWithImputed.csv'

SUBTYPES = [
    "Heated Humidifier",
    "Mattress",
    "Pump Module",
    "Suction Unit",
    "Ventilator",
]

N_SIMULATIONS = 2000

# Statuses with KNOWN end dates → sampling pool for bootstrap
KNOWN_STATUSES = ["Completed", "Active"]

# Statuses with MISSING end dates → need imputation
MISSING_STATUSES = ["Canceled", "Delivered", "Out For Delivery", "Partial", "Turn Down"]


# =====================================================
# LOAD CLEANED DATA
# =====================================================

df = pd.read_csv(CLEANED_CSV, low_memory=False)

df["Delivery_CallDateTime"] = pd.to_datetime(df["Delivery_CallDateTime"], errors="coerce")
df["EndDateTime"]           = pd.to_datetime(df["EndDateTime"],           errors="coerce")

# Compute rental length in days for rows that already have an end date
df["rental_length"] = (
    (df["EndDateTime"] - df["Delivery_CallDateTime"])
    .dt.total_seconds() / (3600 * 24)
)

print(f"Total rows loaded: {len(df):,}")
print(f"Rows with blank EndDateTime: {df['EndDateTime'].isna().sum():,}")


# =====================================================
# IDENTIFY ROWS THAT NEED IMPUTATION
# =====================================================

call_ids_missing = (
    df[
        df["DeliveryStatusDesc"].isin(MISSING_STATUSES) &
        df["EndDateTime"].isna()
    ]["CallLogID"]
    .dropna()
    .unique()
)

print(f"Unique CallLogIDs needing imputation: {len(call_ids_missing):,}\n")


# =====================================================
# BOOTSTRAP SIMULATIONS — one per subtype
# =====================================================

median_rows = {}  # subtype → chosen median simulation row

for subtype in SUBTYPES:

    # Pool: known rental lengths for this subtype
    known_sub = (
        df[
            df["DeliveryStatusDesc"].isin(KNOWN_STATUSES) &
            (df["ModelSubTypeName"] == subtype)
        ]["rental_length"]
        .dropna()
    )

    if known_sub.empty:
        print(f"[WARN] No known durations for '{subtype}' — skipping.")
        continue

    # CallLogIDs needing imputation for this subtype
    boot_ids = (
        df[
            (df["ModelSubTypeName"] == subtype) &
            (df["CallLogID"].isin(call_ids_missing))
        ]["CallLogID"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if boot_ids.empty:
        print(f"[INFO] No missing IDs for '{subtype}' — nothing to impute.")
        continue

    print(f"[{subtype}]  known pool = {len(known_sub):,}  |  IDs to impute = {len(boot_ids):,}")

    # Run simulations
    sim_array = np.random.choice(
        known_sub.values,
        size=(N_SIMULATIONS, len(boot_ids)),
        replace=True,
    )

    sim_df = pd.DataFrame(
        sim_array,
        columns=[f"boot_{cid}" for cid in boot_ids],
    )

    sim_df["sum"]     = sim_df.sum(axis=1)
    sim_df["average"] = sim_df["sum"] / len(boot_ids)
    sim_df = sim_df.sort_values(by="sum").reset_index(drop=True)

    # Optional: save individual bootstrap file per subtype
    # sim_df.to_csv(f"bootstrap_{subtype}.csv", index=False)

    # Store the median row (index 999 out of 0–1999)
    median_rows[subtype] = sim_df.iloc[N_SIMULATIONS // 2 - 1]


# =====================================================
# IMPUTATION — apply median bootstrap row back to df
# =====================================================

for subtype, chosen_row in median_rows.items():

    boot_sub = df[
        (df["ModelSubTypeName"] == subtype) &
        (df["DeliveryStatusDesc"].isin(MISSING_STATUSES)) &
        (df["EndDateTime"].isna())
    ].copy()

    if boot_sub.empty:
        continue

    boot_sub["CallLogID"] = boot_sub["CallLogID"].astype(str)

    # Map simulated durations by CallLogID
    boot_values  = chosen_row.filter(like="boot_")
    boot_ids_str = boot_values.index.str.replace("boot_", "", regex=False).astype(str)
    boot_series  = pd.Series(boot_values.values, index=boot_ids_str)

    boot_sub["imputed_duration"] = boot_sub["CallLogID"].map(boot_series)

    # Convert duration → EndDateTime, floored to seconds
    boot_sub["imputed_EndDateTime"] = (
        boot_sub["Delivery_CallDateTime"] +
        pd.to_timedelta(boot_sub["imputed_duration"], unit="D")
    ).dt.floor("s")

    # Write back into main df
    df.loc[boot_sub.index, "EndDateTime"]   = boot_sub["imputed_EndDateTime"].values
    df.loc[boot_sub.index, "rental_length"] = boot_sub["imputed_duration"].values

    print(f"[{subtype}]  imputed {len(boot_sub):,} rows")


# =====================================================
# SAVE FINAL DATASET
# =====================================================

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved imputed dataset → {OUTPUT_CSV}")
print(f"Remaining blank EndDateTimes: {df['EndDateTime'].isna().sum():,}")