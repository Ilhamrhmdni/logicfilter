# app.py
import re
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Filter Produk (No Header)", layout="wide")
st.title("📦 Filter Produk (Input Tanpa Header) — Trigger Filter + Export TXT/CSV")

# =========================================================
# HEADER TETAP (USER TIDAK PERLU PASTE HEADER)
# =========================================================
FIXED_COLS = [
    "No",
    "Link Produk",
    "Nama Produk",
    "Harga",
    "Stock",
    "Terjual Bulanan",
    "Terjual Semua",
    "Komisi %",
    "Komisi Rp",
    "Ratting",
]

# =========================================================
# Helpers
# =========================================================
def clean_number_id(x):
    """
    Bersihin angka model Indonesia:
    '43.800' -> 43800
    '59,999' -> 59999
    'Rp 10.860' -> 10860
    '-' / '' -> NaN
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "-", "null"]:
        return np.nan

    s = s.replace("Rp", "").replace("rp", "").strip()
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\.,]", "", s)

    # dataset kamu umumnya titik/koma pemisah ribuan
    s = s.replace(".", "")
    s = s.replace(",", "")

    return pd.to_numeric(s, errors="coerce")


def detect_sep_from_text(text: str) -> str:
    # kalau ada tab → TSV; kalau tidak → CSV (koma)
    return "\t" if "\t" in text else ","


def read_no_header_text(text: str) -> pd.DataFrame:
    """
    Baca data TANPA HEADER dari teks (paste / file).
    Otomatis deteksi pemisah (TAB atau koma).
    """
    text = text.strip()
    sep = detect_sep_from_text(text)

    df = pd.read_csv(
        StringIO(text),
        sep=sep,
        header=None,
        names=FIXED_COLS,
        engine="python",
        dtype=str,
    )
    return df


@st.cache_data
def load_uploaded_file(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    return read_no_header_text(text)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # rapikan teks
    df["Link Produk"] = df["Link Produk"].astype(str).str.strip()
    df["Nama Produk"] = df["Nama Produk"].astype(str).str.strip()

    # convert angka
    numeric_cols = ["No", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Ratting"]
    for c in numeric_cols:
        df[c] = df[c].apply(clean_number_id)

    return df


def fmt_id(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.0f}".replace(",", ".")
    except Exception:
        return str(x)


# ===== EXPORTERS (INI YANG BIKIN TXT BENERAN TXT) =====
def export_csv_bytes(df: pd.DataFrame) -> bytes:
    # CSV pakai koma
    return df.to_csv(index=False, sep=",").encode("utf-8")

def export_txt_bytes(df: pd.DataFrame) -> bytes:
    # TXT pakai TAB (TSV) supaya aman kalau nama produk ada koma
    # tetap ada header di output, biar bisa dipakai ulang
    return df.to_csv(index=False, sep="\t").encode("utf-8")


# =========================================================
# INPUT
# =========================================================
with st.sidebar:
    st.header("Input (Tanpa Header)")
    st.caption("Header sudah ditanam di script. Jadi kamu paste/upload data BARIS SAJA.")
    source_mode = st.radio("Sumber input", ["Paste", "Upload file (.txt / .csv)"], index=0)

df_raw = None

if source_mode == "Paste":
    pasted = st.text_area(
        "Paste data (tanpa header). Bisa TAB (TSV) atau koma (CSV).",
        height=240,
        placeholder="Contoh baris:\n1\thttps://...\tNama Produk\t43.800\t0\t269\t50000\t0\t0\t4",
    )
    if pasted.strip():
        df_raw = read_no_header_text(pasted)
else:
    up = st.file_uploader("Upload file .txt atau .csv (tanpa header)", type=["txt", "csv"])
    if up is not None:
        df_raw = load_uploaded_file(up)

if df_raw is None:
    st.info("Masukkan data via sidebar (Paste atau Upload).")
    st.stop()

df = coerce_types(df_raw)

# =========================================================
# PREVIEW
# =========================================================
st.subheader("Preview Data (Header otomatis dari script)")
st.dataframe(df, use_container_width=True, height=320)

# =========================================================
# FILTER (TRIGGER BUTTON) — HANYA: Stock, Terjual Bulanan, Komisi %, Komisi Rp
# =========================================================
st.subheader("Filter (Hanya 4 kolom)")

with st.sidebar:
    st.header("Filter")
    with st.form("filter_form"):
        stock_min = st.number_input("Stock minimal", min_value=0, value=0, step=1)
        tb_min = st.number_input("Terjual Bulanan minimal", min_value=0, value=0, step=1)

        # Komisi %
        if df["Komisi %"].notna().any():
            kmin = float(df["Komisi %"].min())
            kmax = float(df["Komisi %"].max())
        else:
            kmin, kmax = 0.0, 0.0
        komisi_pct_rng = st.slider("Rentang Komisi %", kmin, kmax, (kmin, kmax))

        # Komisi Rp
        if df["Komisi Rp"].notna().any():
            rmin = float(df["Komisi Rp"].min())
            rmax = float(df["Komisi Rp"].max())
        else:
            rmin, rmax = 0.0, 0.0
        komisi_rp_rng = st.slider("Rentang Komisi Rp", rmin, rmax, (rmin, rmax))

        run_filter = st.form_submit_button("🚀 Jalankan Filter")

# default: belum difilter
df_f = df.copy()

if run_filter:
    df_f = df_f[df_f["Stock"].fillna(0) >= stock_min]
    df_f = df_f[df_f["Terjual Bulanan"].fillna(0) >= tb_min]
    df_f = df_f[(df_f["Komisi %"].fillna(0) >= komisi_pct_rng[0]) & (df_f["Komisi %"].fillna(0) <= komisi_pct_rng[1])]
    df_f = df_f[(df_f["Komisi Rp"].fillna(0) >= komisi_rp_rng[0]) & (df_f["Komisi Rp"].fillna(0) <= komisi_rp_rng[1])]
else:
    st.info("Atur filter di sidebar lalu klik **🚀 Jalankan Filter** untuk memproses.")

# =========================================================
# OUTPUT + METRICS
# =========================================================
st.subheader("Hasil")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total baris (awal)", len(df))
m2.metric("Total baris (hasil)", len(df_f))
m3.metric("Total Terjual Bulanan (hasil)", fmt_id(df_f["Terjual Bulanan"].fillna(0).sum()))
m4.metric("Total Komisi Rp (hasil)", fmt_id(df_f["Komisi Rp"].fillna(0).sum()))

st.dataframe(df_f, use_container_width=True, height=420)

# =========================================================
# EXPORT (TXT / CSV)
# =========================================================
st.subheader("Export Hasil")

export_fmt = st.radio("Pilih format export", ["CSV", "TXT"], horizontal=True, index=0)

if export_fmt == "CSV":
    data_bytes = export_csv_bytes(df_f)
    file_name = "hasil_filter_produk.csv"
    mime = "text/csv"
else:
    data_bytes = export_txt_bytes(df_f)
    file_name = "hasil_filter_produk.txt"
    mime = "text/plain"

st.download_button(
    f"⬇️ Download hasil ({export_fmt})",
    data=data_bytes,
    file_name=file_name,
    mime=mime,
)

st.caption("TXT diexport sebagai TSV (dipisah TAB) supaya rapi dan aman kalau Nama Produk mengandung koma.")
