"""
NMRK Indicative Valuation Tool — Streamlit App
===============================================
Run locally:
    streamlit run nmrk_valuation_app.py

Files needed in the same folder:
    rf_model_6m.pkl, rf_model_1y.pkl, rf_model_3y.pkl
    le_dict_6m.pkl, le_dict_1y.pkl, le_dict_3y.pkl
    feature_cols_6m.json, feature_cols_1y.json, feature_cols_3y.json
    predictions_sample_6m.csv, predictions_sample_1y.csv, predictions_sample_3y.csv
    postal_lookup.csv
    mrt_stations.csv       (optional — for MRT distance)
    primary_schools.csv    (optional — for school proximity)
    gls_pipeline.csv       (optional — for GLS supply context)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, re, requests, warnings
from pathlib import Path
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NMRK Indicative Valuation",
    page_icon="🏢",
    layout="wide",
)

BASE = Path(__file__).parent
POSTAL_CSV = BASE / "postal_lookup.csv"

MODEL_FILES = {
    "6 Months": {
        "model"      : BASE / "rf_model_6m.pkl",
        "encoders"   : BASE / "le_dict_6m.pkl",
        "features"   : BASE / "feature_cols_6m.json",
        "predictions": BASE / "predictions_sample_6m.csv",
        "comparables": BASE / "comparables_6m.csv",
    },
    "1 Year": {
        "model"      : BASE / "rf_model_1y.pkl",
        "encoders"   : BASE / "le_dict_1y.pkl",
        "features"   : BASE / "feature_cols_1y.json",
        "predictions": BASE / "predictions_sample_1y.csv",
        "comparables": BASE / "comparables_1y.csv",
    },
    "3 Years": {
        "model"      : BASE / "rf_model_3y.pkl",
        "encoders"   : BASE / "le_dict_3y.pkl",
        "features"   : BASE / "feature_cols_3y.json",
        "predictions": BASE / "predictions_sample_3y.csv",
        "comparables": BASE / "comparables_3y.csv",
    },
}

CAT_COLS = [
    "Project Name", "Type of Sale", "Type of Area", "Property Type",
    "Purchaser Address Indicator", "Planning Region", "Planning Area",
    "Market segment", "Tenure Group", "Completion Bucket",
]

EXTRA_LOC_FEATURES = [
    "MRT Distance (km)",
    "Top School Distance (km)",
    "Top School 1km",
    "Top School 2km",
]

GLS_FEATURES = [
    "GLS Sites Segment",
    "GLS Units Segment",
    "GLS Units Area",
    "GLS Avg psf ppr Segment",
]

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path):
    with open(path, "rb") as f: return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_encoders(path):
    with open(path, "rb") as f: return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_feature_cols(path):
    with open(path) as f: return json.load(f)

@st.cache_resource
def load_postal_lookup():
    df = pd.read_csv(POSTAL_CSV, dtype={"Postal Code": str})
    df["Postal Code"] = df["Postal Code"].str.zfill(6)
    return df.set_index("Postal Code").to_dict("index")

@st.cache_resource
def load_mrt():
    p = BASE / "mrt_stations.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_resource
def load_schools():
    p = BASE / "primary_schools.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_resource
def load_gls():
    p = BASE / "gls_pipeline.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["Date of Award"] = pd.to_datetime(df["Date of Award"])
        df["Award Year"] = df["Date of Award"].dt.year
        df["Award Half"] = df["Date of Award"].dt.month.apply(lambda m: 1 if m <= 6 else 2)
        df["award_period"] = df["Award Year"] * 2 + df["Award Half"]
        df["Market Segment"] = df["Market Segment"].str.strip().str.upper()
        df["Planning Area"] = df["Planning Area"].str.strip().str.title()
        return df
    return None

@st.cache_data
def load_residuals(path):
    p = Path(path)
    if p.exists():
        pdf = pd.read_csv(p)
        pred_col   = next((c for c in pdf.columns if "Predicted" in c), None)
        actual_col = next((c for c in pdf.columns if "Actual"    in c), None)
        if pred_col and actual_col:
            residuals = pdf[pred_col] - pdf[actual_col]
            return float(np.percentile(residuals, 2.5)), float(np.percentile(residuals, 97.5))
    return None, None

@st.cache_data
def load_comparables(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None

def get_comparables(period_key, project_name, planning_area, market_segment, area_sqft, n=5):
    """
    Find the most recent comparable transactions for the given property.
    Priority: same project → same planning area + similar size → same segment.
    """
    path = MODEL_FILES[period_key]["comparables"]
    df = load_comparables(str(path))
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df["Planning Area"] = df["Planning Area"].str.strip().str.title()
    df["Market segment"] = df["Market segment"].str.strip().str.upper()

    # Score each transaction by relevance
    area_lo, area_hi = area_sqft * 0.7, area_sqft * 1.3

    # Tier 1: same project, similar size
    t1 = df[
        (df["Project Name"].str.upper() == project_name.strip().upper()) &
        (df["Area (SQFT)"] >= area_lo) & (df["Area (SQFT)"] <= area_hi)
    ].copy()

    # Tier 2: same project, any size
    t2 = df[
        (df["Project Name"].str.upper() == project_name.strip().upper())
    ].copy()

    # Tier 3: same planning area, similar size, same segment
    t3 = df[
        (df["Planning Area"].str.lower() == planning_area.strip().lower()) &
        (df["Market segment"] == market_segment.strip().upper()) &
        (df["Area (SQFT)"] >= area_lo) & (df["Area (SQFT)"] <= area_hi)
    ].copy()

    # Tier 4: same segment, similar size
    t4 = df[
        (df["Market segment"] == market_segment.strip().upper()) &
        (df["Area (SQFT)"] >= area_lo) & (df["Area (SQFT)"] <= area_hi)
    ].copy()

    # Build result — fill from tiers until we have n rows
    seen = set()
    rows = []
    for tier in [t1, t2, t3, t4]:
        tier_sorted = tier.sort_values("Sale Date", ascending=False)
        for _, row in tier_sorted.iterrows():
            key = (row["Project Name"], row.get("Unit", ""), row["Sale Date"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
            if len(rows) >= n:
                break
        if len(rows) >= n:
            break

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)[["Project Name", "Unit", "Sale Date",
                                   "Area (SQFT)", "Unit Price ($ PSF)",
                                   "Transacted Price ($)", "Type of Sale",
                                   "Tenure", "Planning Area"]]
    result = result.rename(columns={
        "Project Name"        : "Project",
        "Unit"                : "Unit",
        "Sale Date"           : "Sale Date",
        "Area (SQFT)"         : "Area (sqft)",
        "Unit Price ($ PSF)"  : "PSF (S$)",
        "Transacted Price ($)": "Price (S$)",
        "Type of Sale"        : "Type",
        "Tenure"              : "Tenure",
        "Planning Area"       : "Location",
    })
    result["Price (S$)"] = result["Price (S$)"].apply(lambda x: f"S${int(x):,}")
    result["PSF (S$)"]   = result["PSF (S$)"].apply(lambda x: f"S${int(x):,}")
    return result.head(n)

# ── Helpers ────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def compute_location_features(lat, lon, mrt_df, school_df):
    features = {
        "MRT Distance (km)": 0.0,
        "Top School Distance (km)": 0.0,
        "Top School 1km": 0,
        "Top School 2km": 0,
        "Nearest MRT": "N/A",
        "Nearest Top School": "N/A",
    }
    if mrt_df is not None and lat and lon:
        mrt_df = mrt_df.copy()
        mrt_df["dist"] = mrt_df.apply(
            lambda r: haversine(lat, lon, r["Latitude"], r["Longitude"]), axis=1
        )
        nearest = mrt_df.loc[mrt_df["dist"].idxmin()]
        features["MRT Distance (km)"] = round(nearest["dist"], 3)
        features["Nearest MRT"] = nearest["Station Name"]

    if school_df is not None and lat and lon:
        top = school_df[school_df["Top School"] == 1].copy()
        if len(top) > 0:
            top["dist"] = top.apply(
                lambda r: haversine(lat, lon, r["Latitude"], r["Longitude"]), axis=1
            )
            nearest_school = top.loc[top["dist"].idxmin()]
            d = round(nearest_school["dist"], 3)
            features["Top School Distance (km)"] = d
            features["Nearest Top School"] = nearest_school["School Name"]
            features["Top School 1km"] = 1 if d <= 1.0 else 0
            features["Top School 2km"] = 1 if d <= 2.0 else 0
    return features

def get_gls_features(sale_year, sale_half, market_segment, planning_area, gls_df):
    """Return GLS pipeline supply features for the given transaction period."""
    if gls_df is None:
        return {f: 0 for f in GLS_FEATURES}
    sale_period = int(sale_year) * 2 + int(sale_half)
    window = gls_df[
        (gls_df["award_period"] >= sale_period - 3) &
        (gls_df["award_period"] <= sale_period)
    ]
    seg_sites = window[window["Market Segment"] == str(market_segment).strip().upper()]
    gls_num_sites = len(seg_sites)
    median_du = float(gls_df["Est DUs"].median()) if gls_df["Est DUs"].notna().any() else 400
    gls_units_seg = float(seg_sites["Est DUs"].fillna(median_du).sum())
    gls_avg_psf   = float(seg_sites["psf_ppr"].mean()) if gls_num_sites > 0 else 0.0
    area_sites = window[
        window["Planning Area"].str.lower() == str(planning_area).strip().lower()
    ]
    gls_units_area = float(area_sites["Est DUs"].fillna(0).sum())
    return {
        "GLS Sites Segment"       : gls_num_sites,
        "GLS Units Segment"       : gls_units_seg,
        "GLS Units Area"          : gls_units_area,
        "GLS Avg psf ppr Segment" : round(gls_avg_psf, 2),
    }

def get_gls_outlook(gls_df, market_segment, planning_area, sale_year, sale_half):
    """
    Returns a list of upcoming GLS launches (sites awarded in last 18 months
    that are expected to launch within 12–18 months from their award date).
    """
    if gls_df is None:
        return pd.DataFrame()
    sale_period = int(sale_year) * 2 + int(sale_half)
    window = gls_df[
        (gls_df["award_period"] >= sale_period - 3) &
        (gls_df["award_period"] <= sale_period)
    ]
    seg = window[window["Market Segment"] == str(market_segment).strip().upper()].copy()
    seg = seg.sort_values("Date of Award", ascending=False)
    return seg[[
        "Date of Award", "Location", "Planning Area", "Market Segment",
        "Est DUs", "psf_ppr", "Launch Window Lo", "Launch Window Hi", "Num Bids"
    ]].head(10)

def extract_floor(unit_number: str) -> int:
    s = unit_number.strip().lstrip("#").upper()
    if s.startswith("B"): return 0
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else 0

def onemap_search(query: str) -> list:
    url = (
        "https://www.onemap.gov.sg/api/common/elastic/search"
        f"?searchVal={query.replace(' ', '+')}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return r.json().get("results", [])
    except:
        return []

def encode_value(le, value: str) -> int:
    val = str(value)
    classes = list(le.classes_)
    if val in classes:
        return int(le.transform([val])[0])
    lower_map = {c.lower(): c for c in classes}
    if val.lower() in lower_map:
        return int(le.transform([lower_map[val.lower()]])[0])
    if "Unknown" in classes:
        return int(le.transform(["Unknown"])[0])
    return 0

def predict(period_key, row, feature_cols_override=None):
    """Run prediction for a given period. Returns (psf, lo, hi) or None."""
    files = MODEL_FILES[period_key]
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        return None, None, None
    model        = load_model(str(files["model"]))
    le_dict      = load_encoders(str(files["encoders"]))
    feature_cols = load_feature_cols(str(files["features"]))
    res_lo, res_hi = load_residuals(str(files["predictions"]))

    encoded = {}
    for col in feature_cols:
        if col in CAT_COLS:
            encoded[col] = encode_value(le_dict[col], row.get(col, "Unknown"))
        else:
            encoded[col] = row.get(col, 0)

    X = pd.DataFrame([encoded])[feature_cols]
    psf = float(model.predict(X)[0])

    if res_lo is not None:
        lo, hi = psf + res_lo, psf + res_hi
    else:
        lo, hi = psf * 0.95, psf * 1.05

    return psf, lo, hi

# ── Check files ────────────────────────────────────────────────────────────────

if not POSTAL_CSV.exists():
    st.error("postal_lookup.csv not found.")
    st.stop()

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🏢 NMRK Indicative Valuation")
st.caption("Singapore Residential  |  Powered by URA Caveats Data")
st.divider()

# ── Address input ──────────────────────────────────────────────────────────────
st.subheader("Property")

col1, col2 = st.columns([3, 1])
with col1:
    address_input = st.text_input("Address or project name", placeholder="e.g. BISHAN 8 or 61 Bishan Street 21")
with col2:
    unit_input = st.text_input("Unit number", placeholder="#12-34")

prop_attrs   = None
loc_features = {}
addr_lat, addr_lon = None, None
mrt_df    = load_mrt()
school_df = load_schools()
gls_data  = load_gls()

if address_input:
    with st.spinner("Looking up address…"):
        results = onemap_search(address_input)

    if results:
        options = {f"{r.get('ADDRESS','N/A')}  [Postal: {r.get('POSTAL','?')}]": r for r in results[:10]}
        selected_label = st.selectbox("Select the correct address:", list(options.keys()))
        chosen = options[selected_label]
        postal = chosen.get("POSTAL", "").strip()
        addr_lat = float(chosen.get("LATITUDE") or 0)
        addr_lon = float(chosen.get("LONGITUDE") or 0)
        if postal and postal != "NIL":
            postal_dict = load_postal_lookup()
            prop_attrs  = postal_dict.get(postal.zfill(6))

        if prop_attrs:
            st.success(f"**{prop_attrs['Project Name']}** — {prop_attrs['Planning Area']}, {prop_attrs['Planning Region']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Segment", prop_attrs["Market segment"])
            c2.metric("Tenure",         prop_attrs["Tenure Group"])
            c3.metric("Completion",     prop_attrs.get("Completion Bucket", "Unknown"))

            # MRT + school distances
            if addr_lat and mrt_df is not None:
                loc_features = compute_location_features(addr_lat, addr_lon, mrt_df, school_df)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Nearest MRT", loc_features.get("Nearest MRT", "N/A"),
                          f"{loc_features.get('MRT Distance (km)', 0):.2f} km")
                m2.metric("Nearest Top School", loc_features.get("Nearest Top School", "N/A"),
                          f"{loc_features.get('Top School Distance (km)', 0):.2f} km")
                m3.metric("Top School within 1km", "Yes" if loc_features.get("Top School 1km") else "No")
                m4.metric("Top School within 2km", "Yes" if loc_features.get("Top School 2km") else "No")
        else:
            st.warning("Postal code not in lookup. Fill in details manually below.")
    else:
        st.warning("Address not found via OneMap. Fill in details manually below.")

# ── Manual override ────────────────────────────────────────────────────────────
with st.expander("Override or fill in property details manually", expanded=(prop_attrs is None)):
    regions = ["Central Region","East Region","North Region","North-East Region","West Region"]
    man_project    = st.text_input("Project Name",    value=prop_attrs["Project Name"]     if prop_attrs else "")
    man_region     = st.selectbox("Planning Region",  regions,
                                  index=regions.index(prop_attrs["Planning Region"]) if prop_attrs and prop_attrs["Planning Region"] in regions else 0)
    man_area       = st.text_input("Planning Area",   value=prop_attrs["Planning Area"]    if prop_attrs else "")
    man_segment    = st.selectbox("Market Segment",   ["CCR","RCR","OCR"],
                                  index=["CCR","RCR","OCR"].index(prop_attrs["Market segment"]) if prop_attrs else 1)
    man_proptype   = st.selectbox("Property Type",    ["Apartment","Condominium","Executive Condominium"],
                                  index=["Apartment","Condominium","Executive Condominium"].index(prop_attrs["Property Type"]) if prop_attrs else 1)
    man_tenure     = st.selectbox("Tenure",           ["Freehold","999-yr","99-yr","Other"],
                                  index=["Freehold","999-yr","99-yr","Other"].index(prop_attrs["Tenure Group"]) if prop_attrs else 2)
    man_completion = st.selectbox("Completion Period",["Pre-2000","2000-2009","2010-2019","2020+","Uncompleted"],
                                  index=["Pre-2000","2000-2009","2010-2019","2020+","Uncompleted"].index(prop_attrs.get("Completion Bucket","2010-2019")) if prop_attrs and prop_attrs.get("Completion Bucket","Unknown") not in ("Unknown","") else 2)

    final_attrs = {
        "Project Name"     : man_project,
        "Planning Region"  : man_region,
        "Planning Area"    : man_area,
        "Market segment"   : man_segment,
        "Property Type"    : man_proptype,
        "Tenure Group"     : man_tenure,
        "Completion Bucket": man_completion,
    }

st.divider()

# ── Sale details ───────────────────────────────────────────────────────────────
st.subheader("Sale Details")

now = datetime.now()
col1, col2, col3 = st.columns(3)
with col1:
    area_sqft    = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1000, step=10)
with col2:
    sale_year    = st.selectbox("Sale Year",  list(range(2000, now.year+2)), index=list(range(2000, now.year+2)).index(now.year))
with col3:
    sale_month   = st.selectbox("Sale Month", list(range(1,13)), index=now.month-1)

type_of_sale = st.radio("Type of Sale", ["Resale","New Sale","Sub Sale"], horizontal=True)

st.divider()

# ── GLS Pipeline Preview ───────────────────────────────────────────────────────
if gls_data is not None and final_attrs.get("Market segment"):
    sale_half = 1 if sale_month <= 6 else 2
    gls_preview = get_gls_outlook(
        gls_data,
        final_attrs["Market segment"],
        final_attrs["Planning Area"],
        sale_year, sale_half
    )
    if len(gls_preview) > 0:
        with st.expander(f"🏗️ GLS Pipeline — {final_attrs['Market segment']} Segment ({len(gls_preview)} recent sites)", expanded=False):
            st.caption(
                "Government land sale sites awarded in the last 18 months. "
                "These typically launch as new projects 12–18 months after award, "
                "adding supply pressure to the segment."
            )
            display_df = gls_preview.copy()
            display_df["Date of Award"] = display_df["Date of Award"].dt.strftime("%b %Y")
            display_df["Est DUs"] = display_df["Est DUs"].fillna(0).astype(int)
            display_df["psf_ppr"] = display_df["psf_ppr"].apply(lambda x: f"S${x:,.0f}" if pd.notna(x) else "N/A")
            display_df.columns = ["Awarded", "Location", "Planning Area", "Segment",
                                   "Est Units", "Land psf ppr", "Launch From", "Launch To", "No. of Bids"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Run all 3 models ───────────────────────────────────────────────────────────
run_btn = st.button("Get Indicative Valuation", type="primary", use_container_width=True)

if run_btn:
    if not unit_input:
        st.error("Please enter a unit number (e.g. #12-34).")
        st.stop()
    if not final_attrs["Project Name"]:
        st.error("Please enter or look up a property address first.")
        st.stop()

    floor_level  = extract_floor(unit_input)
    sale_quarter = (sale_month - 1) // 3 + 1
    sale_half    = 1 if sale_month <= 6 else 2

    # GLS features for this transaction
    gls_feat = get_gls_features(
        sale_year, sale_half,
        final_attrs["Market segment"],
        final_attrs["Planning Area"],
        gls_data
    )

    row = {
        "Project Name"               : final_attrs["Project Name"],
        "Area (SQFT)"                : float(area_sqft),
        "Type of Sale"               : type_of_sale,
        "Type of Area"               : "Strata",
        "Property Type"              : final_attrs["Property Type"],
        "Purchaser Address Indicator": "Private",
        "Planning Region"            : final_attrs["Planning Region"],
        "Planning Area"              : final_attrs["Planning Area"],
        "Market segment"             : final_attrs["Market segment"],
        "Tenure Group"               : final_attrs["Tenure Group"],
        "Completion Bucket"          : final_attrs["Completion Bucket"],
        "Sale Year"                  : sale_year,
        "Sale Month"                 : sale_month,
        "Sale Quarter"               : sale_quarter,
        "Floor Level"                : floor_level,
        # Location features (0 if enrichment not yet run)
        "MRT Distance (km)"          : loc_features.get("MRT Distance (km)", 0),
        "Top School Distance (km)"   : loc_features.get("Top School Distance (km)", 0),
        "Top School 1km"             : loc_features.get("Top School 1km", 0),
        "Top School 2km"             : loc_features.get("Top School 2km", 0),
        # GLS pipeline features
        "GLS Sites Segment"          : gls_feat["GLS Sites Segment"],
        "GLS Units Segment"          : gls_feat["GLS Units Segment"],
        "GLS Units Area"             : gls_feat["GLS Units Area"],
        "GLS Avg psf ppr Segment"    : gls_feat["GLS Avg psf ppr Segment"],
    }

    # Run all 3 models
    with st.spinner("Running all 3 models…"):
        results_all = {}
        for period in ["6 Months", "1 Year", "3 Years"]:
            psf, lo, hi = predict(period, row)
            results_all[period] = {"psf": psf, "lo": lo, "hi": hi}

    # ── Results side by side ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Indicative Valuation — All Models")
    st.caption(f"{final_attrs['Project Name']}  |  Unit {unit_input} (Floor {floor_level})  |  {area_sqft:,} sqft  |  {type_of_sale}  |  {sale_year} Q{sale_quarter}")

    col6m, col1y, col3y = st.columns(3)
    cols   = {"6 Months": col6m, "1 Year": col1y, "3 Years": col3y}
    colors = {"6 Months": "🟡", "1 Year": "🟠", "3 Years": "🟢"}
    notes  = {"6 Months": "Most current pricing", "1 Year": "Balanced view", "3 Years": "Recommended — most stable"}

    for period, col in cols.items():
        r = results_all[period]
        with col:
            st.markdown(f"### {colors[period]} Last {period}")
            st.caption(notes[period])
            if r["psf"] is None:
                st.warning("Model not found")
            else:
                total    = r["psf"] * area_sqft
                lo_total = r["lo"] * area_sqft
                hi_total = r["hi"] * area_sqft
                st.metric("Unit Price (PSF)", f"S${r['psf']:,.0f}")
                st.metric("Total Value",      f"S${total:,.0f}")
                st.markdown(
                    f"<div style='font-size:0.8em; color:grey;'>"
                    f"95% CI PSF: S${r['lo']:,.0f} – S${r['hi']:,.0f}<br>"
                    f"95% CI Total: S${lo_total:,.0f} – S${hi_total:,.0f}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # ── GLS Supply Context ─────────────────────────────────────────────────────
    if gls_data is not None:
        st.markdown("**GLS Supply Context (used in model)**")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("GLS Sites in Pipeline", f"{int(gls_feat['GLS Sites Segment'])}",
                  help="Residential GLS sites awarded in last 18 months in this segment")
        g2.metric("Est. Units in Pipeline", f"{int(gls_feat['GLS Units Segment']):,}",
                  help="Total estimated dwelling units from those sites")
        g3.metric("Area Pipeline Units", f"{int(gls_feat['GLS Units Area']):,}",
                  help="Estimated units in this planning area specifically")
        g4.metric("Avg Land Cost (psf ppr)", f"S${gls_feat['GLS Avg psf ppr Segment']:,.0f}",
                  help="Average land psf per plot ratio — signal of developer cost expectations")

        # Interpretation
        units = int(gls_feat["GLS Units Segment"])
        if units > 3000:
            st.info(
                f"⚠️  **High pipeline supply** — {units:,} units coming to market in the {final_attrs['Market segment']} segment. "
                "This level of new supply can moderate price growth in the near term."
            )
        elif units > 1500:
            st.info(
                f"📊  **Moderate pipeline supply** — {units:,} units in the {final_attrs['Market segment']} segment. "
                "Manageable supply; less likely to exert strong downward pressure."
            )
        elif units > 0:
            st.success(
                f"✅  **Low pipeline supply** — {units:,} units in the {final_attrs['Market segment']} segment. "
                "Tight supply supports pricing."
            )
        else:
            st.success("✅  No recent GLS pipeline for this segment — supply is tight.")

    st.divider()

    # Summary table
    st.markdown("**Property Summary**")
    summary = {
        "Project"          : final_attrs["Project Name"],
        "Unit"             : f"{unit_input} (Floor {floor_level})",
        "Area"             : f"{area_sqft:,} sqft",
        "Market Segment"   : final_attrs["Market segment"],
        "Location"         : f"{final_attrs['Planning Area']}, {final_attrs['Planning Region']}",
        "Tenure"           : final_attrs["Tenure Group"],
        "Completion Period": final_attrs["Completion Bucket"],
        "Sale Type"        : f"{type_of_sale} ({sale_year} Q{sale_quarter})",
    }
    if loc_features.get("Nearest MRT"):
        summary["Nearest MRT"] = f"{loc_features['Nearest MRT']} ({loc_features['MRT Distance (km)']:.2f} km)"
    if loc_features.get("Nearest Top School"):
        summary["Nearest Top School"] = f"{loc_features['Nearest Top School']} ({loc_features['Top School Distance (km)']:.2f} km)"
    st.table(pd.DataFrame(summary.items(), columns=["", "Value"]).set_index(""))

    # PSF comparison bar chart
    st.markdown("**PSF Comparison Across Models**")
    chart_data = pd.DataFrame({
        "Model"  : ["Last 6 Months", "Last 1 Year", "Last 3 Years"],
        "PSF (S$)": [results_all["6 Months"]["psf"], results_all["1 Year"]["psf"], results_all["3 Years"]["psf"]],
    }).set_index("Model")
    st.bar_chart(chart_data)

    st.divider()

    # ── Comparable Transactions ────────────────────────────────────────────────
    st.subheader("Comparable Transactions")
    st.caption(
        "Most recent transactions used from each training period. "
        "Prioritised by: same project → same area + similar size → same segment."
    )

    tab6m, tab1y, tab3y = st.tabs(["🟡 Last 6 Months", "🟠 Last 1 Year", "🟢 Last 3 Years"])
    for tab, period_key in zip([tab6m, tab1y, tab3y], ["6 Months", "1 Year", "3 Years"]):
        with tab:
            comps = get_comparables(
                period_key,
                final_attrs["Project Name"],
                final_attrs["Planning Area"],
                final_attrs["Market segment"],
                area_sqft,
                n=5,
            )
            if comps.empty:
                st.info("No comparable transactions found for this period.")
            else:
                st.dataframe(comps, use_container_width=True, hide_index=True)

    st.caption("⚠️ Indicative estimate only. Not a formal valuation. Model: RandomForest | Trained on URA caveats data.")
