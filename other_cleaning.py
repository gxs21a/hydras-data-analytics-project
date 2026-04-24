import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================
RAW_DATA_PATH  = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/rentals_all.csv'
MODELS_PATH    = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/raw/models.csv'
OUTPUT_PATH    = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/Imputed_Jan23March26.csv'
INTERMEDIATE_PATH = '/Users/GeorgiaSiegel/OneDrive - US Med-Equip, LLC/Desktop/hydras-data-analytics-project/data/clean/Justification.csv'

DATE_START     = pd.Timestamp('2023-01-01 00:00:01')   # floor: clamp early dates to here
DATE_END       = pd.Timestamp('2026-03-31 00:00:01')   # ceiling: clamp late dates to here
ACTIVE_END     = pd.Timestamp('2025-12-31 00:00:00')   # EndDateTime for confirmed active rentals

N_SIMULATIONS  = 2000
MEDIAN_SIM_IDX = 999   # row index of the median simulation (0-indexed, for 2000 sims)

# Subtypes to bootstrap
BOOTSTRAP_SUBTYPES = [
    "Heated Humidifier",
    "Mattress",
    "Pump Module",
    "Suction Unit",
    "Ventilator",
]

# Statuses that indicate a rental ended and should have a known EndDateTime.
# These are used as the "known duration" pool for bootstrapping.
KNOWN_STATUSES = ['Completed', 'Active']

# Statuses where EndDateTime is missing and needs to be imputed via bootstrapping.
IMPUTE_STATUSES = ['Delivered', 'Out For Delivery', 'Partial', 'Turn Down']


# =====================================================
# STEP 1 — LOAD & PARSE
# (from V2_1_Filtered_DataFrame.ipynb)
# =====================================================
print("Loading data...")
df = pd.read_csv(RAW_DATA_PATH, low_memory=False)

print(f"  Raw rows: {df.shape[0]:,}")

df['Delivery_CallDateTime'] = pd.to_datetime(df['Delivery_CallDateTime'], errors='coerce')
df['StartDateTime']         = pd.to_datetime(df['StartDateTime'],         errors='coerce')
df['EndDateTime']           = pd.to_datetime(df['EndDateTime'],           errors='coerce')


# =====================================================
# STEP 2 — DROP CLOSED BRANCHES
# Remove any branch whose name contains "closed" or "zClosed" (case-insensitive).
# =====================================================
print("\nDropping closed branches...")
before = df.shape[0]

mask_closed = df['Delivery_BranchName'].str.contains(
    r'\bclosed\b|zClosed', case=False, na=False
)
df = df[~mask_closed].copy()
print(f"  Removed {before - df.shape[0]:,} rows | Remaining: {df.shape[0]:,}")


# =====================================================
# STEP 3 — DATE RANGE FILTER
# Keep rentals whose EndDateTime >= Jan 2023 (or is null — still ongoing)
# and whose Delivery_CallDateTime <= Mar 2026.
# =====================================================
print("\nFiltering by date range...")
before = df.shape[0]

# Keep: EndDateTime is null OR ends after the start of our window
df = df[df['EndDateTime'].isna() | (df['EndDateTime'] >= DATE_START)].reset_index(drop=True)

# Keep: Delivery_CallDateTime is null OR starts before the end of our window
df = df[df['Delivery_CallDateTime'].isna() | (df['Delivery_CallDateTime'] <= DATE_END)].reset_index(drop=True)

print(f"  Removed {before - df.shape[0]:,} rows | Remaining: {df.shape[0]:,}")


# =====================================================
# STEP 4 — CLAMP EARLY DATES TO DATE_START
# If Delivery_CallDateTime or StartDateTime is before Jan 2023, set it to Jan 2023.
# This handles legacy data where dates were incorrectly imputed (e.g. May 2022).
# =====================================================
print("\nClamping early dates to DATE_START...")

early_delivery = (df['Delivery_CallDateTime'] < DATE_START)
early_start    = (df['StartDateTime']         < DATE_START)
print(f"  Delivery_CallDateTime rows clamped: {early_delivery.sum():,}")
print(f"  StartDateTime rows clamped:         {early_start.sum():,}")

df.loc[early_delivery, 'Delivery_CallDateTime'] = DATE_START
df.loc[early_start,    'StartDateTime']         = DATE_START


# =====================================================
# STEP 5 — CLAMP LATE END DATES TO DATE_END
# Cap any EndDateTime that exceeds the data export date.
# =====================================================
print("\nClamping late EndDateTime to DATE_END...")

late_end = df['EndDateTime'].notna() & (df['EndDateTime'] >= DATE_END)
print(f"  EndDateTime rows clamped: {late_end.sum():,}")
df.loc[late_end, 'EndDateTime'] = DATE_END


# =====================================================
# STEP 6 — DROP JUNK ROWS BY STATUS
# Per notebook 2: drop rows where EndDateTime is null and the status
# indicates the rental never actually started or was clearly invalid.
# Turn Down is intentionally kept for bootstrapping.
# =====================================================
print("\nDropping invalid rows by DeliveryStatusDesc...")
before = df.shape[0]

def drop_condition(df, status, extra_conditions=None):
    """
    Base drop condition shared across most statuses:
      - EndDateTime is null
      - DeliveryStatusDesc matches status
      - IsVoid is null or 0
      - Pickup_CallLogID is null
      - IsActiveRental != 1
      - IsSwapped != 1
    Extra conditions (list of boolean Series) are AND-ed in.
    """
    cond = (
        df['EndDateTime'].isna()
        & (df['DeliveryStatusDesc'] == status)
        & (df['IsVoid'].isna() | (df['IsVoid'] == 0))
        & df['Pickup_CallLogID'].isna()
        & (df['IsActiveRental'] != 1)
        & (df['IsSwapped'] != 1)
    )
    if extra_conditions:
        for c in extra_conditions:
            cond = cond & c
    return cond


# Canceled — but only if DeliveryActionDesc is NOT Turn Down
cond_canceled = drop_condition(
    df, 'Canceled',
    extra_conditions=[df['DeliveryActionDesc'] != 'Turn Down']
)

# Delivered — requires StartDateTime also null
cond_delivered = drop_condition(
    df, 'Delivered',
    extra_conditions=[df['StartDateTime'].isna()]
)

# Completed — requires StartDateTime also null
cond_completed = drop_condition(
    df, 'Completed',
    extra_conditions=[df['StartDateTime'].isna()]
)

# Pending Branch Confirmation — requires StartDateTime also null
cond_pending_branch = drop_condition(
    df, 'Pending Branch Confirmation',
    extra_conditions=[df['StartDateTime'].isna()]
)

# Out For Delivery — two passes:
#   pass 1: StartDateTime null + standard guards
#   pass 2: any remaining Out For Delivery with null EndDateTime
cond_ofd_1 = drop_condition(
    df, 'Out For Delivery',
    extra_conditions=[df['StartDateTime'].isna()]
)
cond_ofd_2 = (
    df['EndDateTime'].isna()
    & (df['DeliveryStatusDesc'] == 'Out For Delivery')
)

# Partial — requires StartDateTime null and BilledToDate null
cond_partial = drop_condition(
    df, 'Partial',
    extra_conditions=[df['StartDateTime'].isna(), df['BilledToDate'].isna()]
)

# Active — requires StartDateTime null (no IsVoid check for Active)
cond_active = (
    df['StartDateTime'].isna()
    & df['EndDateTime'].isna()
    & (df['DeliveryStatusDesc'] == 'Active')
    & df['Pickup_CallLogID'].isna()
    & (df['IsActiveRental'] != 1)
    & (df['IsSwapped'] != 1)
)

# NaN status — requires StartDateTime null
cond_nan_status = drop_condition(
    df, None,
    extra_conditions=[df['StartDateTime'].isna(), df['DeliveryStatusDesc'].isna()]
)
# Override: drop_condition checks == status which won't work for NaN
cond_nan_status = (
    df['StartDateTime'].isna()
    & df['EndDateTime'].isna()
    & df['DeliveryStatusDesc'].isna()
    & (df['IsVoid'].isna() | (df['IsVoid'] == 0))
    & df['Pickup_CallLogID'].isna()
    & (df['IsActiveRental'] != 1)
    & (df['IsSwapped'] != 1)
)

# Pending Approval of Partner Account — requires StartDateTime null
cond_pending_approval = drop_condition(
    df, 'Pending Approval of Partner Account',
    extra_conditions=[df['StartDateTime'].isna()]
)

# Combine all drop conditions and apply in one pass
drop_mask = (
    cond_canceled
    | cond_delivered
    | cond_completed
    | cond_pending_branch
    | cond_ofd_1
    | cond_ofd_2
    | cond_partial
    | cond_active
    | cond_nan_status
    | cond_pending_approval
)

df = df[~drop_mask].reset_index(drop=True)
print(f"  Removed {before - df.shape[0]:,} junk rows | Remaining: {df.shape[0]:,}")


# =====================================================
# STEP 7 — IMPUTE ACTIVE RENTALS
# For rows where IsActiveRental == 1, StartDateTime is known, but
# EndDateTime is null: set EndDateTime to ACTIVE_END.
# These are confirmed still-running rentals at the time of export.
# =====================================================
print("\nImputing EndDateTime for confirmed active rentals...")

mask_active = (
    df['EndDateTime'].isna()
    & df['StartDateTime'].notna()
    & (df['IsActiveRental'] == 1)
)
print(f"  Active rentals imputed: {mask_active.sum():,}")
df.loc[mask_active, 'EndDateTime'] = ACTIVE_END


# =====================================================
# STEP 8 — DROP DUPLICATES
# =====================================================
print("\nDropping duplicate rows...")
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
print(f"  Removed {before - df.shape[0]:,} duplicates | Remaining: {df.shape[0]:,}")


# =====================================================
# STEP 9 — ADD SUBTYPE & RENTAL LENGTH
# =====================================================
print("\nMerging subtype from models.csv...")
model_df  = pd.read_csv(MODELS_PATH, encoding='latin1')
price_map = dict(zip(model_df['ModelID'], model_df['ModelSubTypeName']))
df['Subtype'] = df['ModelID'].map(price_map)

df['rental_length'] = (
    (df['EndDateTime'] - df['Delivery_CallDateTime'])
    .dt.total_seconds() / (3600 * 24)
)

# Intermediate save — this is the "Justification.csv" equivalent
df.to_csv(INTERMEDIATE_PATH, index=False)
print(f"  Intermediate file saved: {INTERMEDIATE_PATH}  ({df.shape[0]:,} rows)")


# =====================================================
# STEP 10 — BOOTSTRAP IMPUTATION
# (from V3_3_BootstrapCode.ipynb)
#
# For each subtype in BOOTSTRAP_SUBTYPES:
#   1. Build a pool of known rental durations from Completed/Active rentals.
#   2. Identify CallLogIDs that still have a null EndDateTime.
#   3. Run N_SIMULATIONS bootstrap samples over those IDs.
#   4. Pick the median simulation row and use its sampled durations to
#      impute EndDateTime for the null rows.
# =====================================================
print("\nRunning bootstrap imputation...")

# Pre-compute rental_length from known rows (needed as the sampling pool)
df['rental_length'] = (
    (df['EndDateTime'] - df['Delivery_CallDateTime'])
    .dt.total_seconds() / (3600 * 24)
)

# Pool: rows with known end dates from 'Completed' or 'Active'
known = df[df['DeliveryStatusDesc'].isin(KNOWN_STATUSES)].copy()

# Rows that still need imputation after all the dropping above
call_ids_missing = df[
    df['DeliveryStatusDesc'].isin(IMPUTE_STATUSES) &
    df['EndDateTime'].isna()
]['CallLogID'].dropna().unique()

print(f"  CallLogIDs needing imputation: {len(call_ids_missing):,}")

for subtype in BOOTSTRAP_SUBTYPES:

    # Known durations for this subtype (the sampling pool)
    known_sub = known[known['Subtype'] == subtype]['rental_length'].dropna()

    if len(known_sub) == 0:
        print(f"  [{subtype}] No known durations — skipping bootstrap.")
        continue

    # IDs of this subtype that need imputation
    boot_ids = (
        df[
            (df['Subtype'] == subtype) &
            (df['CallLogID'].isin(call_ids_missing))
        ]['CallLogID']
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(boot_ids) == 0:
        print(f"  [{subtype}] No missing CallLogIDs — nothing to impute.")
        continue

    print(f"  [{subtype}] Bootstrapping {len(boot_ids):,} IDs from {len(known_sub):,} known durations...")

    # Run simulations: shape (N_SIMULATIONS, n_ids)
    sim_array = np.random.choice(
        known_sub.values,
        size=(N_SIMULATIONS, len(boot_ids)),
        replace=True
    )

    sim_df = pd.DataFrame(
        sim_array,
        columns=[f"boot_{cid}" for cid in boot_ids]
    )
    sim_df['sum']     = sim_df.sum(axis=1)
    sim_df['average'] = sim_df['sum'] / len(boot_ids)
    sim_df = sim_df.sort_values(by='sum').reset_index(drop=True)

    # Use median simulation row
    chosen_row  = sim_df.iloc[MEDIAN_SIM_IDX]
    boot_values = chosen_row.filter(like="boot_")
    boot_ids_str = boot_values.index.str.replace("boot_", "", regex=False).astype(str)
    boot_series  = pd.Series(boot_values.values, index=boot_ids_str)

    # Subset of df that needs imputation for this subtype
    boot_sub = df[
        (df['Subtype'] == subtype) &
        (df['DeliveryStatusDesc'].isin(IMPUTE_STATUSES))
    ].copy()
    boot_sub['CallLogID'] = boot_sub['CallLogID'].astype(str)
    boot_sub['imputed_duration'] = boot_sub['CallLogID'].map(boot_series)

    # Fallback for any IDs not matched
    if boot_sub['imputed_duration'].isna().any():
        fallback = boot_series.median()
        boot_sub['imputed_duration'] = boot_sub['imputed_duration'].fillna(fallback)

    # Fill any missing Delivery_CallDateTime before computing EndDateTime
    missing_start = boot_sub['Delivery_CallDateTime'].isna()
    if missing_start.any():
        median_start = df['Delivery_CallDateTime'].median()
        boot_sub.loc[missing_start, 'Delivery_CallDateTime'] = median_start

    # Compute imputed EndDateTime
    boot_sub['imputed_EndDateTime'] = (
        boot_sub['Delivery_CallDateTime'] +
        pd.to_timedelta(boot_sub['imputed_duration'], unit='D')
    ).dt.floor('s')

    # Apply only to rows still missing EndDateTime
    mask_missing = df.loc[boot_sub.index, 'EndDateTime'].isna()
    df.loc[boot_sub.index[mask_missing], 'EndDateTime']    = boot_sub.loc[mask_missing, 'imputed_EndDateTime'].values
    df.loc[boot_sub.index[mask_missing], 'rental_length']  = boot_sub.loc[mask_missing, 'imputed_duration'].values

    n_imputed = mask_missing.sum()
    print(f"  [{subtype}] Imputed {n_imputed:,} rows.")


# Final safety net: any remaining nulls within bootstrap scope get the global median duration
final_mask = (
    df['Subtype'].isin(BOOTSTRAP_SUBTYPES) &
    df['DeliveryStatusDesc'].isin(IMPUTE_STATUSES) &
    df['EndDateTime'].isna()
)
if final_mask.sum() > 0:
    fallback_duration = df['rental_length'].median()
    fallback_start    = df['Delivery_CallDateTime'].median()
    df.loc[final_mask, 'Delivery_CallDateTime'] = df.loc[final_mask, 'Delivery_CallDateTime'].fillna(fallback_start)
    df.loc[final_mask, 'EndDateTime'] = (
        df.loc[final_mask, 'Delivery_CallDateTime'] +
        pd.to_timedelta(fallback_duration, unit='D')
    )
    df.loc[final_mask, 'rental_length'] = fallback_duration
    print(f"  Safety net applied to {final_mask.sum():,} remaining nulls.")


# =====================================================
# STEP 11 — FINAL CHECKS & SAVE
# =====================================================
print("\n" + "="*55)
print("  FINAL SUMMARY")
print("="*55)
print(f"  Total rows:            {df.shape[0]:,}")
print(f"  Null EndDateTime:      {df['EndDateTime'].isna().sum():,}")
print(f"  Null Delivery_CallDT:  {df['Delivery_CallDateTime'].isna().sum():,}")

# Remaining null EndDateTime by status (should be near zero for bootstrap subtypes)
remaining_nulls = df.loc[df['EndDateTime'].isna(), 'DeliveryStatusDesc'].value_counts(dropna=False)
if len(remaining_nulls) > 0:
    print("\n  Remaining null EndDateTime by status:")
    print(remaining_nulls.to_string())

# Null check per bootstrap subtype
print("\n  Null EndDateTime per bootstrap subtype (within imputation scope):")
for subtype in BOOTSTRAP_SUBTYPES:
    n_null = (
        df[
            (df['Subtype'] == subtype) &
            df['DeliveryStatusDesc'].isin(IMPUTE_STATUSES)
        ]['EndDateTime'].isna().sum()
    )
    print(f"    {subtype:<25} {n_null:,} remaining nulls")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")