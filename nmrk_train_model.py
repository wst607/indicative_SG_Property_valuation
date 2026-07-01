#!/usr/bin/env python3
"""
NMRK Model Training Script — Segmented Resale / New Sale Models
================================================================
Trains 8 separate RandomForest models:
  2 sale types  × 4 time periods = 8 models

  Sale types : resale, newsale
  Periods    : 6m, 1y, 18m, 3y

  Output files per combination:
    rf_model_{saletype}_{period}.pkl
    le_dict_{saletype}_{period}.pkl
    feature_cols_{saletype}_{period}.json
    predictions_sample_{saletype}_{period}.csv
    comparables_{saletype}_{period}.csv

Split: 40% train / 30% validation / 30% test (true holdout)

Features removed vs original:
  - Purchaser Address Indicator  (buyer residency — not a property trait)
  - Type of Sale                 (redundant — each model is already sale-type-specific)
  - GLS features                 (market-level signal — display only in app)

Features added:
  + Property Age (Years)         (continuous age, r=-0.56 with PSF)

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

OUTPUT_DIR = Path(__file__).parent
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 70)
print("  SG Indicative Valuation — Segmented Model Training")
print("  Architecture: Resale + New Sale  ×  6m / 1y / 18m / 3y")
print("  Split: 40% Train / 30% Validation / 30% Test")
print("=" * 70)
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
print(f"  Sale types: {df_raw['Type of Sale'].value_counts().to_dict()}")

# ── Raw transactions for comparables lookup ────────────────────────────────────
COMP_COLS = [
    "Project Name", "Address", "Unit", "Level",
    "Sale Date", "Transacted Price ($)", "Area (SQFT)", "Unit Price ($ PSF)",
    "Type of Sale", "Property Type", "Tenure",
    "Planning Area", "Market segment", "Postal Code",
]
df_comps_raw = df_raw[[c for c in COMP_COLS if c in df_raw.columns]].copy()
df_comps_raw["Sale Date"]      = pd.to_datetime(df_comps_raw["Sale Date"])
df_comps_raw["Postal Code"]    = df_comps_raw["Postal Code"].astype(str).str.zfill(6)
df_comps_raw["Planning Area"]  = df_comps_raw["Planning Area"].str.strip().str.title()
df_comps_raw["Market segment"] = df_comps_raw["Market segment"].str.strip().str.upper()
df_comps_raw["Type of Sale"]   = df_comps_raw["Type of Sale"].str.strip()

# ── GLS pipeline note ─────────────────────────────────────────────────────────
GLS_CSV = OUTPUT_DIR / "gls_pipeline.csv"
if GLS_CSV.exists():
    print("  GLS pipeline found — used for app display only (not a model feature).")
else:
    print("  gls_pipeline.csv not found — GLS display unavailable in app.")

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target = "Unit Price ($ PSF)"

    # Outlier filter — p0.5/p99.5 (wide, preserves CCR high-end)
    Q1, Q3 = df[target].quantile(0.005), df[target].quantile(0.995)
    df = df[(df[target] >= Q1) & (df[target] <= Q3)]
    df = df[(df["Area (SQFT)"] >= 200) & (df["Area (SQFT)"] <= 10000)]

    # Date
    df["Sale Date"]    = pd.to_datetime(df["Sale Date"])
    df["Sale Year"]    = df["Sale Date"].dt.year
    df["Sale Month"]   = df["Sale Date"].dt.month
    df["Sale Quarter"] = df["Sale Date"].dt.quarter

    # Floor
    df["Level"]       = df["Level"].fillna("00")
    df["Floor Level"] = pd.to_numeric(df["Level"].str.strip(), errors="coerce").fillna(0)

    # Tenure
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

    # Exact completion year → property age
    def extract_completion_year(c):
        c = str(c).strip().lower()
        if "uncompleted" in c: return None
        try: return int(c[:4])
        except: return None

    df["Tenure Group"]         = df["Tenure"].apply(simplify_tenure)
    df["Completion Bucket"]    = df["Completion Date"].apply(completion_bucket)
    df["Completion Year"]      = df["Completion Date"].apply(extract_completion_year)
    df["Property Age (Years)"] = (
        df["Sale Year"] - df["Completion Year"]
    ).clip(lower=0, upper=60).fillna(0)

    # Drop unused / removed columns
    drop_cols = [
        "Source.Name", "Address", "Unit", "Postal Sector",
        "Area (SQM)", "Unit Price ($ PSM)", "Nett Price($)", "Number of Units",
        "Level", "Tenure", "Completion Date", "Postal District",
        "Transacted Price ($)",
        "Purchaser Address Indicator",  # removed — buyer residency not a property trait
        "Completion Year",              # helper — replaced by Property Age
        "Type of Sale",                 # removed — each model is sale-type-specific
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

df_full = engineer_features(df_raw)
# Keep Type of Sale for subsetting before it gets dropped
df_full["_sale_type"] = df_raw["Type of Sale"].str.strip().reindex(df_full.index)
latest_date = df_full["Sale Date"].max()

# ── Merge postal enrichment features ──────────────────────────────────────────
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
        EXTRA_LOC_FEATURES = []
else:
    EXTRA_LOC_FEATURES = []

print(f"  After cleaning: {len(df_full):,} rows  |  Latest sale: {latest_date.strftime('%b %Y')}")
print()

# ── Time period cutoffs ────────────────────────────────────────────────────────
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

period_labels = {
    "6m" : "Last 6 Months",
    "1y" : "Last 1 Year",
    "18m": "Last 18 Months",
    "3y" : "Last 3 Years",
}

# ── Sale type segments ─────────────────────────────────────────────────────────
# Sub Sale is grouped with New Sale (developer-linked, similar pricing)
SALE_SEGMENTS = {
    "resale" : ["Resale"],
    "newsale": ["New Sale", "Sub Sale"],
}
segment_labels = {
    "resale" : "Resale",
    "newsale": "New Sale / Sub Sale",
}

# ── Model feature columns ──────────────────────────────────────────────────────
# Project Name uses TARGET ENCODING (median PSF per project) instead of label
# encoding — this preserves the pricing signal for luxury outliers like
# Ardmore Park, Nassim Hill etc. that sit far above their segment average.
CAT_COLS = [
    # Project Name intentionally excluded from label-encoded cols — target encoded below
    "Type of Area", "Property Type",
    "Planning Region", "Planning Area",
    "Market segment", "Tenure Group", "Completion Bucket",
    # Type of Sale intentionally excluded — redundant in segmented models
]
NUM_COLS = [
    "Floor Level", "Area (SQFT)",
    "Sale Year", "Sale Month", "Sale Quarter",
    "Property Age (Years)",
]
TARGET = "Unit Price ($ PSF)"
EXTRA_FEATURES = EXTRA_LOC_FEATURES  # location features only, no GLS

# ── Training loop ──────────────────────────────────────────────────────────────
results_summary = []

for sale_key, sale_types in SALE_SEGMENTS.items():
    sale_label = segment_labels[sale_key]
    print(f"\n{'=' * 70}")
    print(f"  SEGMENT: {sale_label}")
    print(f"{'=' * 70}")

    # Subset by sale type
    df_seg = df_full[df_full["_sale_type"].isin(sale_types)].copy()
    print(f"  Total rows in segment: {len(df_seg):,}")

    for period, cutoff in cutoffs.items():
        plabel = period_labels[period]
        print(f"\n  {'─' * 60}")
        print(f"  [{sale_label}]  Period: {plabel}  (from {cutoff.strftime('%b %Y')})")
        print(f"  {'─' * 60}")

        # Filter to period
        df = df_seg[df_seg["Sale Date"] >= cutoff].copy()
        df = df.drop(columns=["Sale Date", "_sale_type"], errors="ignore")
        print(f"  Rows: {len(df):,}")

        if len(df) < 500:
            print(f"  ⚠  Too few rows ({len(df)}) — skipping this combination.")
            continue

        # ── 40 / 30 / 30 split (done before encoding to prevent leakage) ───────
        # We split first so target encoding is fitted on train set only
        from sklearn.model_selection import train_test_split as _tts
        df_train_raw, df_temp_raw = _tts(df, test_size=0.60, random_state=42)
        df_val_raw,   df_test_raw = _tts(df_temp_raw, test_size=0.50, random_state=42)

        # ── Target encode Project Name on FULL dataset ──────────────────────
        # Built from all historical data so rare luxury projects (e.g. Ardmore
        # Park with only 2-3 transactions per year) are never missing from the
        # map. This is safe: TE uses project-level median across all time, not
        # future prices for the specific rows being predicted.
        global_median = float(df[TARGET].median())
        project_te_map = (
            df.groupby("Project Name")[TARGET]
            .median()
            .to_dict()
        )
        # Save the map so the app can look up project PSF at prediction time
        te_path = OUTPUT_DIR / f"project_te_{sale_key}_{period}.json"
        with open(te_path, "w") as f:
            json.dump({"map": project_te_map, "global_median": global_median}, f)
        print(f"  Target encoding map: {len(project_te_map):,} projects  |  global median S${global_median:.0f} psf")

        def apply_te(df_slice):
            df_out = df_slice.copy()
            df_out["Project Name TE"] = (
                df_out["Project Name"]
                .fillna("Unknown")
                .astype(str)
                .map(project_te_map)
                .fillna(global_median)
            )
            df_out = df_out.drop(columns=["Project Name"], errors="ignore")
            return df_out

        df_train_raw = apply_te(df_train_raw)
        df_val_raw   = apply_te(df_val_raw)
        df_test_raw  = apply_te(df_test_raw)

        # ── Label encode remaining CAT_COLS ───────────────────────────────────
        le_dict = {}
        df_enc_train = df_train_raw.copy()
        df_enc_val   = df_val_raw.copy()
        df_enc_test  = df_test_raw.copy()

        for col in CAT_COLS:
            if col not in df_enc_train.columns:
                continue
            df_enc_train[col] = df_enc_train[col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            df_enc_train[col] = le.fit_transform(df_enc_train[col])
            le_dict[col] = le
            # Apply same encoder to val/test — unseen labels → "Unknown" → 0
            for df_slice in [df_enc_val, df_enc_test]:
                df_slice[col] = df_slice[col].fillna("Unknown").astype(str)
                known = set(le.classes_)
                df_slice[col] = df_slice[col].apply(
                    lambda x: x if x in known else ("Unknown" if "Unknown" in known else le.classes_[0])
                )
                df_slice[col] = le.transform(df_slice[col])

        # Fill missing location features
        for col in EXTRA_FEATURES:
            for df_slice in [df_enc_train, df_enc_val, df_enc_test]:
                if col in df_slice.columns:
                    df_slice[col] = df_slice[col].fillna(df_enc_train[col].median() if col in df_enc_train.columns else 0)

        feature_cols = [c for c in df_enc_train.columns if c != TARGET]

        X_train = df_enc_train[feature_cols]
        y_train = df_enc_train[TARGET]
        X_val   = df_enc_val[feature_cols]
        y_val   = df_enc_val[TARGET]
        X_test  = df_enc_test[feature_cols]
        y_test  = df_enc_test[TARGET]
        print(f"  Split → Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

        # ── Train ──────────────────────────────────────────────────────────────
        print(f"  Training …", end=" ", flush=True)
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        print("done.")

        # ── Metrics ───────────────────────────────────────────────────────────
        y_val_pred  = rf.predict(X_val)
        y_test_pred = rf.predict(X_test)

        r2_val  = r2_score(y_val,  y_val_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae     = mean_absolute_error(y_test, y_test_pred)
        mape    = float(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100)
        gap     = abs(r2_val - r2_test)

        print(f"  [Val]   R²={r2_val:.4f}  MAE=S${mean_absolute_error(y_val, y_val_pred):.0f} psf")
        print(f"  [Test]  R²={r2_test:.4f}  MAE=S${mae:.0f} psf  MAPE={mape:.2f}%")
        flag = "⚠  overfitting" if gap > 0.02 else "✓  stable"
        print(f"  Val/Test gap: {gap:.4f}  {flag}")

        # Top features
        feat_imp = pd.Series(rf.feature_importances_, index=feature_cols)
        print("  Top 6 features:")
        for fname, fimp in feat_imp.sort_values(ascending=False).head(6).items():
            print(f"    {fname:<35} {fimp:.4f}")

        # ── Save comparables ───────────────────────────────────────────────────
        comps_types = sale_types
        comps_period = df_comps_raw[
            (df_comps_raw["Sale Date"] >= cutoff) &
            (df_comps_raw["Type of Sale"].isin(comps_types))
        ].copy()
        comps_period["Sale Date"]            = comps_period["Sale Date"].dt.strftime("%d %b %Y")
        comps_period["Unit Price ($ PSF)"]   = comps_period["Unit Price ($ PSF)"].round(0).astype(int)
        comps_period["Transacted Price ($)"] = comps_period["Transacted Price ($)"].round(0).astype(int)
        comps_path = OUTPUT_DIR / f"comparables_{sale_key}_{period}.csv"
        comps_period.to_csv(comps_path, index=False)
        print(f"  Saved comparables: {comps_path.name}  ({len(comps_period):,} rows)")

        # ── Save residuals (test set) ──────────────────────────────────────────
        residuals_df = pd.DataFrame({
            "Actual PSF ($)"   : y_test.values,
            "Predicted PSF ($)": y_test_pred,
            "Error ($)"        : y_test_pred - y_test.values,
            "Error (%)"        : (y_test_pred - y_test.values) / y_test.values * 100,
        })
        res_path = OUTPUT_DIR / f"predictions_sample_{sale_key}_{period}.csv"
        residuals_df.to_csv(res_path, index=False)

        # ── Save model + encoders + feature list ───────────────────────────────
        model_path = OUTPUT_DIR / f"rf_model_{sale_key}_{period}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"  Saved: {model_path.name}  ({size_mb:.1f} MB)")

        le_path = OUTPUT_DIR / f"le_dict_{sale_key}_{period}.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(le_dict, f)

        feat_path = OUTPUT_DIR / f"feature_cols_{sale_key}_{period}.json"
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f)

        results_summary.append({
            "Segment" : sale_label,
            "Period"  : plabel,
            "Rows"    : f"{len(df):,}",
            "Val R²"  : f"{r2_val:.4f}",
            "Test R²" : f"{r2_test:.4f}",
            "MAE"     : f"S${mae:.0f}",
            "MAPE"    : f"{mape:.2f}%",
            "Gap"     : f"{gap:.4f}",
            "Size"    : f"{size_mb:.1f} MB",
            "stable"  : gap <= 0.02,
        })

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  TRAINING COMPLETE — Summary")
print(f"{'=' * 70}")
print()
print(f"  {'Segment':<24} {'Period':<16} {'Rows':<8} {'Val R²':<9} {'Test R²':<9} {'MAE':<11} {'MAPE':<8} {'Gap':<7}")
print(f"  {'─'*24} {'─'*16} {'─'*8} {'─'*9} {'─'*9} {'─'*11} {'─'*8} {'─'*7}")
for r in results_summary:
    flag = " ✓" if r["stable"] else " ⚠"
    print(f"  {r['Segment']:<24} {r['Period']:<16} {r['Rows']:<8} {r['Val R²']:<9} "
          f"{r['Test R²']:<9} {r['MAE']:<11} {r['MAPE']:<8} {r['Gap']:<6}{flag}")

print()
print("  Output files (8 models × pkl + le_dict + feature_cols + predictions + comparables + project_te):")
print("    rf_model_resale_6m.pkl       rf_model_newsale_6m.pkl")
print("    rf_model_resale_1y.pkl       rf_model_newsale_1y.pkl")
print("    rf_model_resale_18m.pkl      rf_model_newsale_18m.pkl")
print("    rf_model_resale_3y.pkl       rf_model_newsale_3y.pkl")
print("    project_te_resale_6m.json    project_te_newsale_6m.json  (+ 1y, 18m, 3y)")
print()
print("  Encoding changes:")
print("    Project Name  — TARGET ENCODED (median PSF per project, train set only)")
print("                    Luxury outliers (Ardmore Park etc.) now correctly priced")
print("    All other cats — label encoded as before")
print()
print("  New features:")
print("    + Property Age (Years)  — continuous, r=-0.56 with PSF")
print("  Removed features:")
print("    - Type of Sale          — redundant in segmented models")
print("    - Purchaser Address     — buyer residency, not a property trait")
print("    - GLS features          — display only in app")
print()
print("  Next step:")
print("    streamlit run nmrk_valuation_app.py")
print("=" * 70)
