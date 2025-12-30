import re
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Filter Produk", layout="wide")
st.title("📦 Filter Produk Shopee")

# =========================================================
# HEADER TETAP
# =========================================================
COLUMNS = ["No","Link Produk","Nama Produk","Harga","Stock","Terjual Bulanan","Terjual Semua","Komisi %","Komisi Rp","Ratting"]

# =========================================================
# SESSION STATE CONTROL
# =========================================================
if "ready" not in st.session_state:
    st.session_state.ready = False

# =========================================================
# HELPERS
# =========================================================
def clean_num(x):
    if pd.isna(x): return 0
    s = re.sub(r"[^\d]", "", str(x))
    return int(s) if s else 0

def load_text(text):
    sep = "\t" if "\t" in text else ","
    df = pd.read_csv(StringIO(text), sep=sep, header=None, names=COLUMNS)
    return df

def export_csv(df): return df.to_csv(index=False).encode()
def export_txt(df): return df.to_csv(index=False, sep="\t").encode()

# =========================================================
# INPUT SECTION
# =========================================================
with st.sidebar:
    st.header("1️⃣ Input Data")
    mode = st.radio("Sumber Data", ["Paste", "Upload (.txt / .csv)"])

    raw = None

    if mode == "Paste":
        raw = st.text_area("Paste data TANPA header", height=200)
    else:
        file = st.file_uploader("Upload file", type=["txt","csv"])
        if file: raw = file.read().decode("utf-8")

    if st.button("▶️ MULAI PROSES"):
        st.session_state.raw = raw
        st.session_state.ready = True

# =========================================================
# STOP JIKA BELUM DIKLIK
# =========================================================
if not st.session_state.ready:
    st.warning("Silakan input data lalu klik **▶️ MULAI PROSES**")
    st.stop()

# =========================================================
# LOAD & CLEAN
# =========================================================
df = load_text(st.session_state.raw)

for c in ["Harga","Stock","Terjual Bulanan","Terjual Semua","Komisi %","Komisi Rp","Ratting"]:
    df[c] = df[c].apply(clean_num)

# =========================================================
# PREVIEW
# =========================================================
st.subheader("Preview Data")
st.dataframe(df, use_container_width=True)

# =========================================================
# FILTER
# =========================================================
with st.sidebar:
    st.header("2️⃣ Filter")
    stock = st.number_input("Stock minimal", 0)
    sold = st.number_input("Terjual Bulanan minimal", 0)
    komisi_pct = st.slider("Komisi %", 0, int(df["Komisi %"].max()), (0, int(df["Komisi %"].max())))
    komisi_rp = st.slider("Komisi Rp", 0, int(df["Komisi Rp"].max()), (0, int(df["Komisi Rp"].max())))

    run = st.button("🚀 Jalankan Filter")

if "filtered" not in st.session_state:
    st.session_state.filtered = df

if run:
    f = df.copy()
    f = f[f["Stock"] >= stock]
    f = f[f["Terjual Bulanan"] >= sold]
    f = f[f["Komisi %"].between(*komisi_pct)]
    f = f[f["Komisi Rp"].between(*komisi_rp)]
    st.session_state.filtered = f

# =========================================================
# OUTPUT
# =========================================================
st.subheader("Hasil Filter")
st.dataframe(st.session_state.filtered, use_container_width=True)

# =========================================================
# EXPORT
# =========================================================
fmt = st.radio("Format Export", ["CSV","TXT"])

if fmt == "CSV":
    data = export_csv(st.session_state.filtered)
    name = "hasil.csv"
else:
    data = export_txt(st.session_state.filtered)
    name = "hasil.txt"

st.download_button("⬇️ Download", data, name)
