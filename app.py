import re
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Filter Produk Shopee", layout="wide")
st.title("📦 Filter & Olah Data Produk (Shopee)")

# ---------------------------
# Helpers
# ---------------------------
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
    # ambil hanya digit, titik, koma
    s = re.sub(r"[^0-9\.,]", "", s)

    # kasus umum dataset kamu: titik = pemisah ribuan
    # jadi hapus semua titik
    s = s.replace(".", "")

    # kalau koma dipakai pemisah ribuan juga, hapus
    # (kalau kamu punya desimal beneran, bilang ya)
    s = s.replace(",", "")

    return pd.to_numeric(s, errors="coerce")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Samakan nama kolom rating (kadang "Ratting")
    if "Ratting" in df.columns and "Rating" not in df.columns:
        df = df.rename(columns={"Ratting": "Rating"})

    # Pastikan kolom penting ada
    expected = ["No", "Link Produk", "Nama Produk", "Harga", "Stock",
                "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Rating"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning(f"Kolom kurang: {missing}. App tetap jalan untuk kolom yang ada.")
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = [c for c in ["No", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Rating"] if c in df.columns]
    for c in num_cols:
        df[c] = df[c].apply(clean_number_id)

    # rapikan teks
    for c in ["Link Produk", "Nama Produk"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # hitung komisi Rp kalau kosong tapi ada harga & komisi%
    if all(c in df.columns for c in ["Komisi Rp", "Harga", "Komisi %"]):
        df["Komisi Rp_calc"] = (df["Harga"] * df["Komisi %"] / 100.0).round(0)
        # kalau Komisi Rp kosong/0, pakai hasil kalkulasi
        df["Komisi Rp_final"] = np.where(
            (df["Komisi Rp"].isna()) | (df["Komisi Rp"] == 0),
            df["Komisi Rp_calc"],
            df["Komisi Rp"]
        )
    return df

def load_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Format file tidak didukung.")

def parse_pasted_tsv(text: str) -> pd.DataFrame:
    # Data kamu mirip hasil copy dari Excel → tab-separated
    from io import StringIO
    return pd.read_csv(StringIO(text), sep="\t")

# ---------------------------
# Input section
# ---------------------------
with st.sidebar:
    st.header("Input Data")
    mode = st.radio("Sumber data", ["Upload CSV/Excel", "Paste (TSV dari Excel / teks)"], index=1)

df_raw = None

if mode == "Upload CSV/Excel":
    up = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
    if up:
        df_raw = load_uploaded(up)
else:
    sample_hint = "Paste data kamu (tab-separated). Contoh: copy dari Excel lalu paste ke sini."
    pasted = st.text_area(sample_hint, height=220)
    if pasted.strip():
        df_raw = parse_pasted_tsv(pasted)

if df_raw is None:
    st.info("Masukkan data dulu via sidebar.")
    st.stop()

df = normalize_columns(df_raw)
df = coerce_types(df)

# ---------------------------
# Preview
# ---------------------------
st.subheader("Preview (setelah dibersihkan)")
st.dataframe(df, use_container_width=True, height=320)

# ---------------------------
# Filters
# ---------------------------
st.subheader("Filter")
with st.sidebar:
    st.header("Filter")

    keyword = st.text_input("Cari nama produk", placeholder="mis: turtleneck / inara / knit ...")

    # Harga
    if "Harga" in df.columns and df["Harga"].notna().any():
        hmin = float(df["Harga"].min())
        hmax = float(df["Harga"].max())
        harga_rng = st.slider("Rentang Harga", hmin, hmax, (hmin, hmax))

    # Stock minimal
    stock_min = st.number_input("Stock minimal", min_value=0, value=0, step=1)

    # Rating
    if "Rating" in df.columns and df["Rating"].notna().any():
        rmin = float(df["Rating"].min())
        rmax = float(df["Rating"].max())
        rating_rng = st.slider("Rentang Rating", rmin, rmax, (rmin, rmax))

    # Terjual Bulanan
    tb_min = st.number_input("Terjual Bulanan minimal", min_value=0, value=0, step=1)

    # Terjual Semua
    ts_min = st.number_input("Terjual Semua minimal", min_value=0, value=0, step=1)

    # Komisi %
    if "Komisi %" in df.columns and df["Komisi %"].notna().any():
        kmin = float(df["Komisi %"].min())
        kmax = float(df["Komisi %"].max())
        komisi_rng = st.slider("Rentang Komisi %", kmin, kmax, (kmin, kmax))

# Apply filters
df_f = df.copy()

if keyword and "Nama Produk" in df_f.columns:
    df_f = df_f[df_f["Nama Produk"].astype(str).str.lower().str.contains(keyword.lower(), na=False)]

if "Harga" in df_f.columns and df_f["Harga"].notna().any():
    df_f = df_f[(df_f["Harga"] >= harga_rng[0]) & (df_f["Harga"] <= harga_rng[1])]

if "Stock" in df_f.columns:
    df_f = df_f[df_f["Stock"].fillna(0) >= stock_min]

if "Rating" in df_f.columns and df_f["Rating"].notna().any():
    df_f = df_f[(df_f["Rating"] >= rating_rng[0]) & (df_f["Rating"] <= rating_rng[1])]

if "Terjual Bulanan" in df_f.columns:
    df_f = df_f[df_f["Terjual Bulanan"].fillna(0) >= tb_min]

if "Terjual Semua" in df_f.columns:
    df_f = df_f[df_f["Terjual Semua"].fillna(0) >= ts_min]

if "Komisi %" in df_f.columns and df_f["Komisi %"].notna().any():
    df_f = df_f[(df_f["Komisi %"] >= komisi_rng[0]) & (df_f["Komisi %"] <= komisi_rng[1])]

# ---------------------------
# Sorting & output columns
# ---------------------------
st.subheader("Hasil Filter")
c1, c2, c3, c4 = st.columns([2, 2, 2, 6])

with c1:
    sort_col = st.selectbox("Sort kolom", ["(tanpa sort)"] + df_f.columns.tolist())
with c2:
    sort_dir = st.radio("Urutan", ["Asc", "Desc"], horizontal=True)
with c3:
    topn = st.number_input("Tampilkan Top N", min_value=1, value=min(200, max(1, len(df_f))), step=10)

with c4:
    show_cols = st.multiselect(
        "Kolom ditampilkan",
        df_f.columns.tolist(),
        default=[c for c in ["No", "Nama Produk", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp_final", "Rating", "Link Produk"] if c in df_f.columns]
    )

df_out = df_f.copy()
if sort_col != "(tanpa sort)":
    df_out = df_out.sort_values(by=sort_col, ascending=(sort_dir == "Asc"))

if show_cols:
    df_out = df_out[show_cols]

df_out = df_out.head(int(topn))

# ---------------------------
# KPIs
# ---------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah produk", len(df_out))
if "Harga" in df_out.columns:
    k2.metric("Rata-rata harga", f"{df_out['Harga'].mean():,.0f}".replace(",", "."))
if "Terjual Bulanan" in df_out.columns:
    k3.metric("Total terjual bulanan", f"{df_out['Terjual Bulanan'].sum():,.0f}".replace(",", "."))
if "Komisi Rp_final" in df_out.columns:
    k4.metric("Estimasi total komisi (Rp)", f"{df_out['Komisi Rp_final'].sum():,.0f}".replace(",", "."))

st.dataframe(df_out, use_container_width=True, height=420)

# ---------------------------
# Download
# ---------------------------
st.download_button(
    "⬇️ Download hasil (CSV)",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name="hasil_filter_produk.csv",
    mime="text/csv",
)
