#!/usr/bin/env python3
"""
NMRK Model Training Script — 4 Time-Period Models
===================================================
Trains 4 separate RandomForest models:
  - 6 months  → rf_model_6m.pkl
  - 1 year    → rf_model_1y.pkl
  - 18 months → rf_model_18m.pkl
  - 3 years   → rf_model_3y.pkl

Split: 40% train / 30% validation / 30% test
  - Train:      model learns from this
  - Validation: used to report in-training performance (early overfitting check)
  - Test:       final holdout — never seen during training

GLS pipeline data is loaded for display in the app only.
It is NOT used as a model training feature (removed to avoid supply-signal distortion).

Removed features vs previous version:
  - Purchaser Address Indicator (buyer residency — not a property characteristic)
  - GLS Sites Segment, GLS Units Segment, GLS Units Area, GLS Avg psf ppr Segment

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

print("=" * 65)
print("  NMRK Indicative Valuation — Multi-Period Model Training")
print("  Split: 40% Train / 30% Validation / 30% Test")
print("=" * 65)
print()

# ── Load full dataset ──────────────────────────────────────────────────────────
print("Loading dataset …")
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
df_comps_raw["Sale Date"]     = pd.to_datetime(df_comps_raw["Sale Date"])
df_comps_raw["Postal Code"]   = df_comps_raw["Postal Code"].astype(str).str.zfill(6)
df_comps_raw["Planning Area"] = df_comps_raw["Planning Area"].str.strip().str.title()
df_comps_raw["Market segment"]= df_comps_raw["Market segment"].str.strip().str.upper()

# ── GLS pipeline — loaded for app display only, NOT used in model ──────────────
GLS_CSV = OUTPUT_DIR / "gls_pipeline.csv"
if GLS_CSV.exists():
    print(f"  GLS pipeline found — used for app display only (not a model feature).")
else:
    print(f"  gls_pipeline.csv not found — GLS display will be unavailable in app.")

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target = "Unit Price ($ PSF)"

    # Remove only extreme outliers (p0.5/p99.5 — preserves CCR high-end)
    Q1, Q3 = df[target].quantile(0.005), df[target].quantile(0.995)
    df = df[(df[target] >= Q1) & (df[target] <= Q3)]
    df = df[(df["Area (SQFT)"] >= 200) & (df["Area (SQFT)"] <= 10000)]

    # Date features
    df["Sale Date"]    = pd.to_datetime(df["Sale Date"])
    df["Sale Year"]    = df["Sale Date"].dt.year
    df["Sale Month"]   = df["Sale Date"].dt.month
    df["Sale Quarter"] = df["Sale Date"].dt.quarter

    # Floor level
    df["Level"]       = df["Level"].fillna("00")
    df["Floor Level"] = pd.to_numeric(df["Level"].str.strip(), errors="coerce").fillna(0)

    # Tenure grouping
    def simplify_tenure(t):
        t = str(t).lower()
        if "freehold" in t: return "Freehold"
        if "999" in t:      return "999-yr"
        if "99" in t:       return "99-yr"
        return "Other"

    # Completion bucket
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

    # Drop unused columns
    # NOTE: Purchaser Address Indicator removed — buyer residency is not a
    #       property characteristic and can introduce demographic bias.
    drop_cols = [
        "Source.Name", "Address", "Unit", "Postal Sector",
        "Area (SQM)", "Unit Price ($ PSM)", "Nett Price($)", "Number of Units",
        "Level", "Tenure", "Completion Date", "Postal District",
        "Transacted Price ($)",
        "Purchaser Address Indicator",   # ← removed
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

df_full = engineer_features(df_raw)
latest_date = df_full["Sale Date"].max()

# ── Merge postal enrichment features if available ─────────────────────────────
POSTAL_CSV = OUTPUT_DIR / "postal_lookup.csv"
EXTRA_LOC_FEATURES = [
    "MRT Distance (km)",
    "Top School Distance (km)",
    "Top School 1km",
    "Top School 2km",
]
extra_loc_cols = []

if POSTAL_CSV.exists():
    postal_df = pd.read_csv(POSTAL_CSV, dtype={"Postal Code": str})
    postal_df["Postal Code"] = postal_df["Postal Code"].str.zfill(6)
    extra_loc_cols = [c for c in EXTRA_LOC_FEATURES if c in postal_df.columns]

    if extra_loc_cols:
        merge_cols = ["Postal Code"] + extra_loc_cols
        df_full["Postal Code"] = df_raw["Postal Code"].astype(str).str.zfill(6)
        df_full = df_full.merge(postal_df[merge_cols], on="Postal Code", how="left")
        df_full.drop(columns=["Postal Code"], inplace=True)
        print(f"  Merged location features: {extra_loc_cols}")
    else:
        print("  postal_lookup.csv found but no MRT/school columns yet.")
        print("  Run nmrk_enrich_postal.py to add those features.")
        EXTRA_LOC_FEATURES = []
else:
    EXTRA_LOC_FEATURES = []

print(f"  After cleaning: {len(df_full):,} rows  |  Latest sale: {latest_date.strftime('%b %Y')}")
print()

# ── Training periods ───────────────────────────────────────────────────────────
try:
    from dateutil.relativedelta import relativedelta
    cutoffs = {
        "6m" : latest_date - relativedelta(months=6),
        "1y" : latest_date - relativedelta(years=1),
        "18m": latest_date - relativedelta(months=18),
        "3y" : latest_date - relativedelta(years=3),
    }
except ImportError:
    cutoffs = {
        "6m" : latest_date - pd.DateOffset(months=6),
        "1y" : latest_date - pd.DateOffset(years=1),
        "18m": latest_date - pd.DateOffset(months=18),
        "3y" : latest_date - pd.DateOffset(years=3),
    }

labels = {
    "6m" : "Last 6 Months",
    "1y" : "Last 1 Year",
    "18m": "Last 18 Months",
    "3y" : "Last 3 Years",
}

# CAT_COLS: Purchaser Address Indicator removed
CAT_COLS = [
    "Project Name", "Type of Sale", "Type of Area", "Property Type",
    "Planning Region", "Planning Area",
    "Market segment", "Tenure Group", "Completion Bucket",
]
TARGET = "Unit Price ($ PSF)"

# Extra numeric features (location only — GLS removed from model)
EXTRA_FEATURES = EXTRA_LOC_FEATURES  # no GLS

results_summary = []

for period, cutoff in cutoffs.items():
    label = labels[period]
    print(f"{'─' * 65}")
    print(f"  Training: {label}  (from {cutoff.strftime('%b %Y')})")
    print(f"{'─' * 65}")

    # Filter to period
    df = df_full[df_full["Sale Date"] >= cutoff].copy()
    df = df.drop(columns=["Sale Date"])
    print(f"  Rows in period: {len(df):,}")

    # Label encode categorical columns
    le_dict = {}
    df_enc  = df.copy()
    for col in CAT_COLS:
        if col not in df_enc.columns:
            continue
        df_enc[col] = df_enc[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    # Fill missing location features with median
    for col in EXTRA_FEATURES:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].fillna(df_enc[col].median())

    feature_cols = [c for c in df_enc.columns if c != TARGET]
    X, y = df_enc[feature_cols], df_enc[TARGET]

    # ── 40 / 30 / 30 split ────────────────────────────────────────────────────
    # Step 1: split off 40% train, 60% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.60, random_state=42
    )
    # Step 2: split temp 50/50 → 30% val, 30% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    print(f"  Split → Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}  (GLS removed, Purchaser Address removed)")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"  Training model …", end=" ", flush=True)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("done.")

    # ── Validation metrics (in-training check) ─────────────────────────────────
    y_val_pred = rf.predict(X_val)
    r2_val   = r2_score(y_val, y_val_pred)
    mae_val  = mean_absolute_error(y_val, y_val_pred)
    mape_val = float(np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100)
    print(f"  [Validation]  R²={r2_val:.4f}  MAE=S${mae_val:.0f} psf  MAPE={mape_val:.2f}%")

    # ── Test metrics (final holdout — not used during training) ───────────────
    y_test_pred = rf.predict(X_test)
    r2_test   = r2_score(y_test, y_test_pred)
    mae_test  = mean_absolute_error(y_test, y_test_pred)
    mape_test = float(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100)
    print(f"  [Test]        R²={r2_test:.4f}  MAE=S${mae_test:.0f} psf  MAPE={mape_test:.2f}%")

    # Flag if val vs test gap is large (sign of overfitting)
    r2_gap = abs(r2_val - r2_test)
    if r2_gap > 0.02:
        print(f"  ⚠  Val/Test R² gap = {r2_gap:.4f} — slight overfitting detected.")
    else:
        print(f"  ✓  Val/Test R² gap = {r2_gap:.4f} — model is stable.")

    # ── Top feature importances ────────────────────────────────────────────────
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols)
    print("  Top 8 features:")
    for fname, fimp in feat_imp.sort_values(ascending=False).head(8).items():
        print(f"    {fname:<35} {fimp:.4f}")

    # ── Save comparables ───────────────────────────────────────────────────────
    comps_period = df_comps_raw[df_comps_raw["Sale Date"] >= cutoff].copy()
    comps_period["Sale Date"]             = comps_period["Sale Date"].dt.strftime("%d %b %Y")
    comps_period["Unit Price ($ PSF)"]    = comps_period["Unit Price ($ PSF)"].round(0).astype(int)
    comps_period["Transacted Price ($)"]  = comps_period["Transacted Price ($)"].round(0).astype(int)
    comps_path = OUTPUT_DIR / f"comparables_{period}.csv"
    comps_period.to_csv(comps_path, index=False)
    print(f"  Saved comparables: {comps_path.name}  ({len(comps_period):,} transactions)")

    # ── Save residuals (test set only — true holdout) ─────────────────────────
    residuals_df = pd.DataFrame({
        "Actual PSF ($)"   : y_test.values,
        "Predicted PSF ($)": y_test_pred,
        "Error ($)"        : y_test_pred - y_test.values,
        "Error (%)"        : (y_test_pred - y_test.values) / y_test.values * 100,
    })
    res_path = OUTPUT_DIR / f"predictions_sample_{period}.csv"
    residuals_df.to_csv(res_path, index=False)

    # ── Save model + encoders + feature list ──────────────────────────────────
    model_path = OUTPUT_DIR / f"rf_model_{period}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {model_path.name}  ({size_mb:.1f} MB)")

    le_path = OUTPUT_DIR / f"le_dict_{period}.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le_dict, f)

    feat_path = OUTPUT_DIR / f"feature_cols_{period}.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)

    results_summary.append({
        "Period"      : label,
        "Rows"        : f"{len(df):,}",
        "Train"       : f"{len(X_train):,}",
        "Val R²"      : f"{r2_val:.4f}",
        "Test R²"     : f"{r2_test:.4f}",
        "Test MAE"    : f"S${mae_test:.0f}",
        "Test MAPE"   : f"{mape_test:.2f}%",
        "Val/Test Gap": f"{r2_gap:.4f}",
        "Size"        : f"{size_mb:.1f} MB",
    })
    print()

# ── Final Summary ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  TRAINING COMPLETE — Summary (40/30/30 Split)")
print("=" * 65)
print()
print(f"  {'Period':<18} {'Rows':<9} {'Train':<8} {'Val R²':<9} {'Test R²':<9} {'MAE':<12} {'MAPE':<8} {'Gap':<7} {'Size'}")
print(f"  {'─'*18} {'─'*9} {'─'*8} {'─'*9} {'─'*9} {'─'*12} {'─'*8} {'─'*7} {'─'*7}")
for r in results_summary:
    gap_flag = " ⚠" if float(r["Val/Test Gap"]) > 0.02 else " ✓"
    print(f"  {r['Period']:<18} {r['Rows']:<9} {r['Train']:<8} {r['Val R²']:<9} {r['Test R²']:<9} {r['Test MAE']:<12} {r['Test MAPE']:<8} {r['Val/Test Gap']:<7}{gap_flag}  {r['Size']}")

print()
print("  Features removed vs previous version:")
print("    • Purchaser Address Indicator  (buyer residency — not a property trait)")
print("    • GLS Sites/Units/Area/psf ppr (market-level signal — display only in app)")
print()
print("  Recommendation: use blended average of all 4 models in the app")
print()
print("  Next step:")
print("    streamlit run nmrk_valuation_app.py")
print("=" * 65)
