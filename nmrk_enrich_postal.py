#!/usr/bin/env python3
"""
NMRK Postal Enrichment Script
==============================
Adds lat/lon, nearest MRT distance, and school proximity to postal_lookup.csv.

Run ONCE before retraining:
    python nmrk_enrich_postal.py

Takes ~20-40 minutes (OneMap API lookup for ~6,330 postal codes).
Creates an enriched postal_lookup.csv replacing the original.

Requirements:
    pip install pandas numpy requests tqdm
"""

import time, warnings
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent

POSTAL_CSV  = BASE / "postal_lookup.csv"
MRT_CSV     = BASE / "mrt_stations.csv"
SCHOOL_CSV  = BASE / "primary_schools.csv"
OUTPUT_CSV  = BASE / "postal_lookup.csv"  # overwrites in place

import pandas as pd
import numpy as np
import requests

# ── Haversine distance (km) ───────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def nearest_distance(lat, lon, df_locs) -> float:
    """Return distance in km to nearest point in df_locs (must have Latitude, Longitude cols)."""
    dists = df_locs.apply(
        lambda r: haversine(lat, lon, r["Latitude"], r["Longitude"]), axis=1
    )
    return round(float(dists.min()), 4)

def nearest_top_school_distance(lat, lon, df_schools) -> float:
    top = df_schools[df_schools["Top School"] == 1]
    return nearest_distance(lat, lon, top)

def top_school_within(lat, lon, df_schools, radius_km) -> int:
    top = df_schools[df_schools["Top School"] == 1]
    dists = top.apply(
        lambda r: haversine(lat, lon, r["Latitude"], r["Longitude"]), axis=1
    )
    return int((dists <= radius_km).any())

# ── OneMap lat/lon lookup ─────────────────────────────────────────────────────

def get_latlon(postal_code: str, retries=3) -> tuple:
    """Look up lat/lon for a postal code via OneMap. Returns (lat, lon) or (None, None)."""
    url = (
        "https://www.onemap.gov.sg/api/common/elastic/search"
        f"?searchVal={postal_code}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=8)
            results = r.json().get("results", [])
            if results:
                lat = float(results[0].get("LATITUDE", 0) or 0)
                lon = float(results[0].get("LONGITUDE", 0) or 0)
                if lat != 0 and lon != 0:
                    return lat, lon
        except Exception:
            time.sleep(1)
    return None, None

# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  NMRK Postal Enrichment — Adding MRT & School Features")
print("=" * 60)
print()

# Load files
postal_df  = pd.read_csv(POSTAL_CSV, dtype={"Postal Code": str})
postal_df["Postal Code"] = postal_df["Postal Code"].str.zfill(6)
mrt_df     = pd.read_csv(MRT_CSV)
school_df  = pd.read_csv(SCHOOL_CSV)

print(f"  Postal codes to enrich : {len(postal_df):,}")
print(f"  MRT/LRT stations       : {len(mrt_df):,}")
print(f"  Primary schools        : {len(school_df):,}  ({school_df['Top School'].sum()} top schools)")
print()

# Check if already partially enriched (resume support)
already_done = "Latitude" in postal_df.columns and postal_df["Latitude"].notna().sum() > 0
if already_done:
    done_count = postal_df["Latitude"].notna().sum()
    print(f"  Resuming — {done_count:,} already enriched, {len(postal_df)-done_count:,} remaining")
else:
    postal_df["Latitude"]              = np.nan
    postal_df["Longitude"]             = np.nan
    postal_df["MRT Distance (km)"]     = np.nan
    postal_df["School Distance (km)"]  = np.nan
    postal_df["Top School Distance (km)"] = np.nan
    postal_df["Top School 1km"]        = np.nan
    postal_df["Top School 2km"]        = np.nan

print("  Looking up coordinates and computing distances …")
print("  (This takes 20-40 minutes — do not close the window)")
print()

total = len(postal_df)
errors = 0

try:
    from tqdm import tqdm
    iterator = tqdm(postal_df.iterrows(), total=total, desc="  Progress")
except ImportError:
    iterator = postal_df.iterrows()
    print(f"  (Install tqdm for a progress bar: pip install tqdm)")

for idx, row in iterator:
    # Skip if already done
    if pd.notna(row.get("Latitude")) and row.get("Latitude") != 0:
        continue

    postal = str(row["Postal Code"]).zfill(6)
    lat, lon = get_latlon(postal)

    if lat is None:
        errors += 1
        # Use Singapore centre as fallback
        lat, lon = 1.3521, 103.8198

    postal_df.at[idx, "Latitude"]  = lat
    postal_df.at[idx, "Longitude"] = lon

    # MRT distance
    postal_df.at[idx, "MRT Distance (km)"] = nearest_distance(lat, lon, mrt_df)

    # School distances
    postal_df.at[idx, "School Distance (km)"]     = nearest_distance(lat, lon, school_df)
    postal_df.at[idx, "Top School Distance (km)"] = nearest_top_school_distance(lat, lon, school_df)
    postal_df.at[idx, "Top School 1km"]           = top_school_within(lat, lon, school_df, 1.0)
    postal_df.at[idx, "Top School 2km"]           = top_school_within(lat, lon, school_df, 2.0)

    # Save every 100 rows (resume support)
    if idx % 100 == 0:
        postal_df.to_csv(OUTPUT_CSV, index=False)

# Final save
postal_df.to_csv(OUTPUT_CSV, index=False)

print()
print(f"  Done. {total - errors:,} enriched, {errors:,} fallback to SG centre.")
print()
print("  New columns added to postal_lookup.csv:")
print("    Latitude, Longitude")
print("    MRT Distance (km)         — distance to nearest MRT/LRT")
print("    School Distance (km)      — distance to nearest primary school")
print("    Top School Distance (km)  — distance to nearest top/popular school")
print("    Top School 1km            — 1 if any top school within 1km")
print("    Top School 2km            — 1 if any top school within 2km")
print()
print("  Next step: python nmrk_train_model.py")
print("=" * 60)
