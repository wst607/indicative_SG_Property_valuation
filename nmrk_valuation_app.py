"""
NMRK Indicative Valuation Tool — Streamlit App
===============================================
Run locally:
    streamlit run nmrk_valuation_app.py

Files needed in the same folder:
    rf_model.pkl          — trained model (run nmrk_train_model.py once)
    le_dict.pkl           — label encoders (auto-saved by training script)
    feature_cols.json     — feature order (auto-saved by training script)
    postal_lookup.csv     — postal code lookup table
    predictions_sample.csv — residuals for CI (auto-saved by training script)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, re, os, requests, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NMRK Indicative Valuation",
    page_icon="🏢",
    layout="centered",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

POSTAL_CSV      = BASE / "postal_lookup.csv"

MODEL_FILES = {
    "Last 3 Years (recommended)": {
        "model"      : BASE / "rf_model_3y.pkl",
        "encoders"   : BASE / "le_dict_3y.pkl",
        "features"   : BASE / "feature_cols_3y.json",
        "predictions": BASE / "predictions_sample_3y.csv",
    },
    "Last 1 Year": {
        "model"      : BASE / "rf_model_1y.pkl",
        "encoders"   : BASE / "le_dict_1y.pkl",
        "features"   : BASE / "feature_cols_1y.json",
        "predictions": BASE / "predictions_sample_1y.csv",
    },
    "Last 6 Months": {
        "model"      : BASE / "rf_model_6m.pkl",
        "encoders"   : BASE / "le_dict_6m.pkl",
        "features"   : BASE / "feature_cols_6m.json",
        "predictions": BASE / "predictions_sample_6m.csv",
    },
}

# ── Load model + encoders (cached so they only load once) ─────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading encoders…")
def load_encoders(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_cols(path: str):
    with open(path) as f:
        return json.load(f)

@st.cache_resource
def load_postal_lookup():
    df = pd.read_csv(POSTAL_CSV, dtype={"Postal Code": str})
    df["Postal Code"] = df["Postal Code"].str.zfill(6)
    return df.set_index("Postal Code").to_dict("index")

@st.cache_data
def load_residuals(path: str):
    p = Path(path)
    if p.exists():
        pdf = pd.read_csv(p)
        pred_col   = next((c for c in pdf.columns if "Predicted" in c), None)
        actual_col = next((c for c in pdf.columns if "Actual"    in c), None)
        if pred_col and actual_col:
            residuals = pdf[pred_col] - pdf[actual_col]
            return float(np.percentile(residuals, 2.5)), float(np.percentile(residuals, 97.5))
    return None, None

# ── Helpers ───────────────────────────────────────────────────────────────────
CAT_COLS = [
    "Project Name", "Type of Sale", "Type of Area", "Property Type",
    "Purchaser Address Indicator", "Planning Region", "Planning Area",
    "Market segment", "Tenure Group", "Completion Bucket",
]

def extract_floor(unit_number: str) -> int:
    s = unit_number.strip().lstrip("#").upper()
    if s.startswith("B"):
        return 0
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

# ── Check postal lookup exists ───────────────────────────────────────────────
if not POSTAL_CSV.exists():
    st.error("postal_lookup.csv not found. Place it in the same folder as this script.")
    st.stop()

# ── Model selector (sidebar) ──────────────────────────────────────────────────
st.sidebar.title("Model Settings")
selected_period = st.sidebar.radio(
    "Training Period",
    list(MODEL_FILES.keys()),
    help="More recent data = more current prices. 3 years gives the best balance."
)

files = MODEL_FILES[selected_period]
missing = [k for k, v in files.items() if not v.exists()]
if missing:
    st.error(f"Model files for '{selected_period}' not found. Run `python nmrk_train_model.py` first.")
    st.stop()

# ── Load selected model ───────────────────────────────────────────────────────
model        = load_model(str(files["model"]))
le_dict      = load_encoders(str(files["encoders"]))
feature_cols = load_feature_cols(str(files["features"]))
postal_dict  = load_postal_lookup()
res_lo, res_hi = load_residuals(str(files["predictions"]))

st.sidebar.caption(f"Using: {files['model'].name}")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏢 NMRK Indicative Valuation")
st.caption("Singapore Residential  |  Powered by URA Caveats Data")
st.divider()

# ── Address input ─────────────────────────────────────────────────────────────
st.subheader("Property")

col1, col2 = st.columns([3, 1])
with col1:
    address_input = st.text_input("Address or project name", placeholder="e.g. BISHAN 8 or 61 Bishan Street 21")
with col2:
    unit_input = st.text_input("Unit number", placeholder="#12-34")

# OneMap lookup
prop_attrs = None
addr_display = ""

if address_input:
    with st.spinner("Looking up address…"):
        results = onemap_search(address_input)

    if results:
        # Build dropdown of results
        options = {f"{r.get('ADDRESS', 'N/A')} [Postal: {r.get('POSTAL','?')}]": r for r in results[:10]}
        selected_label = st.selectbox("Select the correct address:", list(options.keys()))
        chosen = options[selected_label]

        postal = chosen.get("POSTAL", "").strip()
        if postal and postal != "NIL":
            prop_attrs = postal_dict.get(postal.zfill(6))
            addr_display = chosen.get("ADDRESS", "")

        if prop_attrs:
            st.success(f"**{prop_attrs['Project Name']}** — {prop_attrs['Planning Area']}, {prop_attrs['Planning Region']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Segment", prop_attrs["Market segment"])
            c2.metric("Tenure",         prop_attrs["Tenure Group"])
            c3.metric("Completion",     prop_attrs.get("Completion Bucket", "Unknown"))
        else:
            st.warning("Postal code not in lookup table. Fill in details manually below.")
    else:
        st.warning("Address not found via OneMap. Fill in details manually below.")

# Manual override / fallback
with st.expander("Override property details (or fill manually if not found above)", expanded=(prop_attrs is None)):
    regions = ["Central Region", "East Region", "North Region", "North-East Region", "West Region"]

    man_project  = st.text_input("Project Name",     value=prop_attrs["Project Name"]    if prop_attrs else "")
    man_region   = st.selectbox("Planning Region",   regions,
                                index=regions.index(prop_attrs["Planning Region"]) if prop_attrs and prop_attrs["Planning Region"] in regions else 0)
    man_area     = st.text_input("Planning Area",    value=prop_attrs["Planning Area"]   if prop_attrs else "")
    man_segment  = st.selectbox("Market Segment",    ["CCR", "RCR", "OCR"],
                                index=["CCR","RCR","OCR"].index(prop_attrs["Market segment"]) if prop_attrs else 1)
    man_proptype = st.selectbox("Property Type",     ["Apartment", "Condominium", "Executive Condominium"],
                                index=["Apartment","Condominium","Executive Condominium"].index(prop_attrs["Property Type"]) if prop_attrs else 1)
    man_tenure   = st.selectbox("Tenure",            ["Freehold", "999-yr", "99-yr", "Other"],
                                index=["Freehold","999-yr","99-yr","Other"].index(prop_attrs["Tenure Group"]) if prop_attrs else 2)
    man_completion = st.selectbox("Completion Period", ["Pre-2000","2000-2009","2010-2019","2020+","Uncompleted"],
                                  index=["Pre-2000","2000-2009","2010-2019","2020+","Uncompleted"].index(prop_attrs.get("Completion Bucket","2010-2019")) if prop_attrs and prop_attrs.get("Completion Bucket","Unknown") != "Unknown" else 2)

    # Always use manual values (they're pre-filled from lookup if available)
    final_attrs = {
        "Project Name"    : man_project,
        "Planning Region" : man_region,
        "Planning Area"   : man_area,
        "Market segment"  : man_segment,
        "Property Type"   : man_proptype,
        "Tenure Group"    : man_tenure,
        "Completion Bucket": man_completion,
    }

st.divider()

# ── Sale details ──────────────────────────────────────────────────────────────
st.subheader("Sale Details")

now = datetime.now()
col1, col2, col3 = st.columns(3)
with col1:
    area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1000, step=10)
with col2:
    sale_year = st.selectbox("Sale Year", list(range(2000, now.year + 2)), index=list(range(2000, now.year + 2)).index(now.year))
with col3:
    sale_month = st.selectbox("Sale Month", list(range(1, 13)), index=now.month - 1)

type_of_sale = st.radio("Type of Sale", ["Resale", "New Sale", "Sub Sale"], horizontal=True)

st.divider()

# ── Run valuation ─────────────────────────────────────────────────────────────
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
    }

    # Encode
    encoded = {}
    for col in feature_cols:
        if col in CAT_COLS:
            encoded[col] = encode_value(le_dict[col], row[col])
        else:
            encoded[col] = row[col]

    X = pd.DataFrame([encoded])[feature_cols]

    with st.spinner("Running model…"):
        psf_pred    = float(model.predict(X)[0])
        total_price = psf_pred * area_sqft

    # CI
    if res_lo is not None:
        psf_lo, psf_hi = psf_pred + res_lo, psf_pred + res_hi
    else:
        psf_lo, psf_hi = psf_pred * 0.95, psf_pred * 1.05

    total_lo = psf_lo * area_sqft
    total_hi = psf_hi * area_sqft

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Indicative Valuation Result")

    r1, r2 = st.columns(2)
    r1.metric("Unit Price (PSF)", f"S${psf_pred:,.0f}")
    r2.metric("Total Value",      f"S${total_price:,.0f}")

    st.caption(f"95% Confidence Interval — PSF: S${psf_lo:,.0f} – S${psf_hi:,.0f}  |  Total: S${total_lo:,.0f} – S${total_hi:,.0f}")

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
    st.table(pd.DataFrame(summary.items(), columns=["", "Value"]).set_index(""))

    st.caption("⚠️ Indicative estimate only. Not a formal valuation. Model: RandomForest | R²≈0.95 | MAE≈S$80 psf")
