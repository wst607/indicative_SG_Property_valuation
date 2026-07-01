#!/usr/bin/env python3
"""
NMRK Model Training Script — 3 Time-Period Models + GLS Pipeline Features
===========================================================================
Trains 3 separate RandomForest models:
  - 6 months of data  → rf_model_6m.pkl
  - 1 year of data    → rf_model_1y.pkl
  - 3 years of data   → rf_model_3y.pkl  (recommended)

Run once:
    python nmrk_train_model.py

Requirements:
    pip install pandas numpy scikit-learn openpyxl python-dateutil
"""

import pickle, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── EDIT THIS PATH ─────────────────────────────────────────────────────────────
DATASET_PATH = Path(r"C:\Users\SW154611\NMRK Tenant (8209549)\Newmark SG Research - Documents\Research\For Valuation\Non-Landed Caveats.xlsx")

# Output folder — same folder as this script
OUTPUT_DIR = Path(__file__).parent
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 60)
print("  NMRK Indicative Valuation — Multi-Period Model Training")
print("=" * 60)
print()

# ── Load full dataset ──────────────────────────────────────────────────────────
print(f"Loading dataset …")
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found at:\n  {DATASET_PATH}\n"
        "Edit the DATASET_PATH variable at the top of this script."
    )

df_raw = pd.read_excel(DATASET_PATH, sheet_name="Non landed caveats excluding EC")
print(f"  Loaded {len(df_raw):,} rows  |  Date range: "
      f"{pd.to_datetime(df_raw['Sale Date']).min().strftime('%b %Y')} – "
      f"{pd.to_datetime(df_raw['Sale Date']).max().strftime('%b %Y')}")

# Keep a clean copy of raw transactions for comparables lookup (saved per period)
COMP_COLS = [
    "Project Name", "Address", "Unit", "Level",
    "Sale Date", "Transacted Price ($)", "Area (SQFT)", "Unit Price ($ PSF)",
    "Type of Sale", "Property Type", "Tenure",
    "Planning Area", "Market segment", "Postal Code",
]
df_comps_raw = df_raw[[c for c in COMP_COLS if c in df_raw.columns]].copy()
df_comps_raw["Sale Date"] = pd.to_datetime(df_comps_raw["Sale Date"])
df_comps_raw["Postal Code"] = df_comps_raw["Postal Code"].astype(str).str.zfill(6)
df_comps_raw["Planning Area"] = df_comps_raw["Planning Area"].str.strip().str.title()
df_comps_raw["Market segment"] = df_comps_raw["Market segment"].str.strip().str.upper()

# ── Load GLS pipeline ─────────────────────────────────────────────────────────
GLS_CSV = OUTPUT_DIR / "gls_pipeline.csv"
gls_df = None
if GLS_CSV.exists():
    gls_df = pd.read_csv(GLS_CSV)
    gls_df["Date of Award"] = pd.to_datetime(gls_df["Date of Award"])
    gls_df["Award Year"] = gls_df["Date of Award"].dt.year
    gls_df["Award Half"] = gls_df["Date of Award"].dt.month.apply(lambda m: 1 if m <= 6 else 2)
    gls_df["award_period"] = gls_df["Award Year"] * 2 + gls_df["Award Half"]
    gls_df["Market Segment"] = gls_df["Market Segment"].str.strip().str.upper()
    gls_df["Planning Area"] = gls_df["Planning Area"].str.strip().str.title()
    print(f"  Loaded GLS pipeline: {len(gls_df)} awarded sites")
else:
    print(f"  gls_pipeline.csv not found — GLS features will be set to 0.")
    print(f"  Copy gls_pipeline.csv to: {OUTPUT_DIR}")

def get_gls_features(sale_year, sale_half, market_segment, planning_area):
    """
    For each transaction, looks back up to 18 months of awarded GLS sites
    (the pipeline that will launch within the next 12-18 months).
    Returns 4 features capturing supply pressure.
    """
    if gls_df is None:
        return {
            "GLS Sites Segment": 0,
            "GLS Units Segment": 0,
            "GLS Units Area": 0,
            "GLS Avg psf ppr Segment": 0,
        }
    sale_period = int(sale_year) * 2 + int(sale_half)
    # Sites awarded in the 18-month window before this transaction
    # (award_period in [sale_period - 3, sale_period])
    window = gls_df[
        (gls_df["award_period"] >= sale_period - 3) &
        (gls_df["award_period"] <= sale_period)
    ]
    seg_sites = window[window["Market Segment"] == str(market_segment).strip().upper()]
    gls_num_sites = len(seg_sites)
    median_du = gls_df["Est DUs"].median() if gls_df["Est DUs"].notna().any() else 400
    gls_units_seg = float(seg_sites["Est DUs"].fillna(median_du).sum())
    gls_avg_psf   = float(seg_sites["psf_ppr"].mean()) if gls_num_sites > 0 else 0.0

    area_sites = window[
        window["Planning Area"].str.lower() == str(planning_area).strip().lower()
    ]
    gls_units_area = float(area_sites["Est DUs"].fillna(0).sum())

    return {
        "GLS Sites Segment": gls_num_sites,
        "GLS Units Segment": gls_units_seg,
        "GLS Units Area": gls_units_area,
        "GLS Avg psf ppr Segment": round(gls_avg_psf, 2),
    }

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target = "Unit Price ($ PSF)"

    # Remove only extreme outliers (p0.5/p99.5 — wide band preserves CCR high-end)
    Q1, Q3 = df[target].quantile(0.005), df[target].quantile(0.995)
    df = df[(df[target] >= Q1) & (df[target] <= Q3)]
    df = df[(df["Area (SQFT)"] >= 200) & (df["Area (SQFT)"] <= 10000)]

    # Date features
    df["Sale Date"]    = pd.to_datetime(df["Sale Date"])
    df["Sale Year"]    = df["Sale Date"].dt.year
    df["Sale Month"]   = df["Sale Date"].dt.month
    df["Sale Quarter"] = df["Sale Date"].dt.quarter
    df["Sale Half"]    = df["Sale Date"].dt.month.apply(lambda m: 1 if m <= 6 else 2)

    # Floor level
    df["Level"]       = df["Level"].fillna("00")
    df["Floor Level"] = pd.to_numeric(df["Level"].str.strip(), errors="coerce").fillna(0)

    # Tenure
    def simplify_tenure(t):
        t = str(t).lower()
        if "freehold" in t: return "Freehold"
        if "999" in t:      return "999-yr"
        if "99" in t:       return "99-yr"
        return "Other"

    # Completion
    def completion_bucket(c):
        c = str(c).strip().lower()
        if "uncompleted" in c: return "Uncompleted"
        try:
            yr = int(c[:4])
            if yr < 2000: return "Pre-2000"
            if yr < 2010: return "2000-2009"
            if yr < 2020: return "2010-2019"
            return "2020+"
        except:
            return "Unknown"

    df["Tenure Group"]      = df["Tenure"].apply(simplify_tenure)
    df["Completion Bucket"] = df["Completion Date"].apply(completion_bucket)

    # Clean planning area / market segment for GLS merge
    df["Planning Area clean"] = df["Planning Area"].str.strip().str.title()
    df["Market segment clean"] = df["Market segment"].str.strip().str.upper()

    # Drop unused columns
    drop_cols = [
        "Source.Name", "Address", "Unit", "Postal Sector",
        "Area (SQM)", "Unit Price ($ PSM)", "Nett Price($)", "Number of Units",
        "Level", "Tenure", "Completion Date", "Postal District",
        "Transacted Price ($)",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

df_full = engineer_features(df_raw)
latest_date = df_full["Sale Date"].max()

# ── Merge postal enrichment features if available ─────────────────────────────
POSTAL_CSV = OUTPUT_DIR / "postal_lookup.csv"
if POSTAL_CSV.exists():
    postal_df = pd.read_csv(POSTAL_CSV, dtype={"Postal Code": str})
    postal_df["Postal Code"] = postal_df["Postal Code"].str.zfill(6)
    extra_loc_cols = [c for c in EXTRA_LOC_FEATURES if c in postal_df.columns] \
        if "EXTRA_LOC_FEATURES" in dir() else []

    # Detect which enrichment cols are present
    EXTRA_LOC_FEATURES = [
        "MRT Distance (km)",
        "Top School Distance (km)",
        "Top School 1km",
        "Top School 2km",
    ]
    extra_loc_cols = [c for c in EXTRA_LOC_FEATURES if c in postal_df.columns]

    if extra_loc_cols:
        merge_cols = ["Postal Code"] + extra_loc_cols
        df_full["Postal Code"] = df_raw["Postal Code"].astype(str).str.zfill(6)
        df_full = df_full.merge(
            postal_df[merge_cols], on="Postal Code", how="left"
        )
        df_full.drop(columns=["Postal Code"], inplace=True)
        print(f"  Merged postal features: {extra_loc_cols}")
    else:
        print(f"  postal_lookup.csv found but no MRT/school columns yet.")
        print(f"  Run nmrk_enrich_postal.py to add those features.")
        EXTRA_LOC_FEATURES = []
        extra_loc_cols = []
else:
    EXTRA_LOC_FEATURES = []
    extra_loc_cols = []

# ── Add GLS pipeline features to every transaction ────────────────────────────
GLS_FEATURES = [
    "GLS Sites Segment",
    "GLS Units Segment",
    "GLS Units Area",
    "GLS Avg psf ppr Segment",
]

if gls_df is not None:
    print("  Computing GLS pipeline features for all transactions …", end=" ", flush=True)
    gls_rows = df_full.apply(
        lambda r: get_gls_features(
            r["Sale Year"], r["Sale Half"],
            r["Market segment clean"], r["Planning Area clean"]
        ), axis=1
    )
    gls_feature_df = pd.DataFrame(list(gls_rows), index=df_full.index)
    df_full = pd.concat([df_full, gls_feature_df], axis=1)
    print("done.")
else:
    for col in GLS_FEATURES:
        df_full[col] = 0

# Drop helper columns used only for GLS lookup
df_full.drop(columns=["Sale Half", "Planning Area clean", "Market segment clean"],
             errors="ignore", inplace=True)

print(f"  After cleaning: {len(df_full):,} rows  |  Latest sale: {latest_date.strftime('%b %Y')}")
print()

# ── Training periods ───────────────────────────────────────────────────────────
try:
    from dateutil.relativedelta import relativedelta
    cutoffs = {
        "6m": latest_date - relativedelta(months=6),
        "1y": latest_date - relativedelta(years=1),
        "3y": latest_date - relativedelta(years=3),
    }
except ImportError:
    cutoffs = {
        "6m": latest_date - pd.DateOffset(months=6),
        "1y": latest_date - pd.DateOffset(years=1),
        "3y": latest_date - pd.DateOffset(years=3),
    }

labels = {
    "6m": "Last 6 Months",
    "1y": "Last 1 Year",
    "3y": "Last 3 Years",
}

CAT_COLS = [
    "Project Name", "Type of Sale", "Type of Area", "Property Type",
    "Purchaser Address Indicator", "Planning Region", "Planning Area",
    "Market segment", "Tenure Group", "Completion Bucket",
]
TARGET = "Unit Price ($ PSF)"

# All extra numeric features (loc + GLS)
EXTRA_FEATURES = EXTRA_LOC_FEATURES + GLS_FEATURES

results_summary = []

for period, cutoff in cutoffs.items():
    label = labels[period]
    print(f"{'─' * 60}")
    print(f"  Training: {label}  (from {cutoff.strftime('%b %Y')})")
    print(f"{'─' * 60}")

    # Filter to period
    df = df_full[df_full["Sale Date"] >= cutoff].copy()
    df = df.drop(columns=["Sale Date"])
    print(f"  Rows: {len(df):,}")

    # Label encode
    le_dict = {}
    df_enc = df.copy()
    for col in CAT_COLS:
        df_enc[col] = df_enc[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    # Fill missing enrichment features with median
    for col in EXTRA_FEATURES:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].fillna(df_enc[col].median())

    feature_cols = [c for c in df_enc.columns if c != TARGET]
    X, y = df_enc[feature_cols], df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}  (including {len([c for c in GLS_FEATURES if c in feature_cols])} GLS features)")

    # Train
    print(f"  Training model …", end=" ", flush=True)
    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("done.")

    # Evaluate
    y_pred = rf.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    print(f"  R²={r2:.4f}  MAE=S${mae:.0f} psf  MAPE={mape:.2f}%")

    # Top GLS feature importances
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols)
    gls_imp = feat_imp[[c for c in GLS_FEATURES if c in feat_imp.index]]
    if len(gls_imp) > 0:
        print(f"  GLS feature importances:")
        for fname, fimp in gls_imp.sort_values(ascending=False).items():
            print(f"    {fname}: {fimp:.4f}")

    # Save comparable transactions for this period
    # Keep all transactions from this period window for the app to query
    comps_period = df_comps_raw[df_comps_raw["Sale Date"] >= cutoff].copy()
    comps_period["Sale Date"] = comps_period["Sale Date"].dt.strftime("%d %b %Y")
    comps_period["Unit Price ($ PSF)"] = comps_period["Unit Price ($ PSF)"].round(0).astype(int)
    comps_period["Transacted Price ($)"] = comps_period["Transacted Price ($)"].round(0).astype(int)
    comps_path = OUTPUT_DIR / f"comparables_{period}.csv"
    comps_period.to_csv(comps_path, index=False)
    print(f"  Saved comparables: {comps_path.name}  ({len(comps_period):,} transactions)")

    # Save residuals
    residuals_df = pd.DataFrame({
        "Actual PSF ($)"   : y_test.values,
        "Predicted PSF ($)": y_pred,
        "Error ($)"        : y_pred - y_test.values,
        "Error (%)"        : (y_pred - y_test.values) / y_test.values * 100,
    })
    res_path = OUTPUT_DIR / f"predictions_sample_{period}.csv"
    residuals_df.to_csv(res_path, index=False)

    # Save model
    model_path = OUTPUT_DIR / f"rf_model_{period}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {model_path.name}  ({size_mb:.1f} MB)")

    # Save encoders
    le_path = OUTPUT_DIR / f"le_dict_{period}.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le_dict, f)

    # Save feature cols
    feat_path = OUTPUT_DIR / f"feature_cols_{period}.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)

    results_summary.append({
        "Period"    : label,
        "Rows"      : f"{len(df):,}",
        "R²"        : f"{r2:.4f}",
        "MAE (psf)" : f"S${mae:.0f}",
        "MAPE"      : f"{mape:.2f}%",
        "File size" : f"{size_mb:.1f} MB",
    })
    print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  TRAINING COMPLETE — Summary")
print("=" * 60)
print()
print(f"  {'Period':<18} {'Rows':<10} {'R²':<8} {'MAE':<12} {'MAPE':<8} {'Size'}")
print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*12} {'─'*8} {'─'*8}")
for r in results_summary:
    print(f"  {r['Period']:<18} {r['Rows']:<10} {r['R²']:<8} {r['MAE (psf)']:<12} {r['MAPE']:<8} {r['File size']}")

print()
print("  GLS features added:")
for f in GLS_FEATURES:
    print(f"    • {f}")
print()
print("  Recommendation: use the 3-year model (best accuracy + recency)")
print()
print("  Next step:")
print("    streamlit run nmrk_valuation_app.py")
print("=" * 60)
