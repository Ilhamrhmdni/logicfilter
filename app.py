# app.py
import re
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Filter Produk (Locked Pipeline)", layout="wide")
st.title("📦 Filter & Olah Data Produk (Shopee)")

# =========================================================
# Struktur data tetap (ditanam di script)
# =========================================================
COLUMNS = [
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
# Session State (kunci alur)
# =========================================================
if "stage" not in st.session_state:
    # "input" -> belum proses, "ready" -> data sudah diproses, siap filter/export
    st.session_state.stage = "input"

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "df" not in st.session_state:
    st.session_state.df = None

if "df_filtered" not in st.session_state:
    st.session_state.df_filtered = None

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

    # umum: titik/koma sebagai pemisah ribuan
    s = s.replace(".", "").replace(",", "")
    return pd.to_numeric(s, errors="coerce")


def detect_sep(text: str) -> str:
    # TSV (tab) kalau ada tab, selain itu anggap CSV koma
    return "\t" if "\t" in text else ","


def read_no_header_text(text: str) -> pd.DataFrame:
    """
    Baca data TANPA header dari teks (paste/upload).
    Otomatis deteksi pemisah TAB atau koma.
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=COLUMNS)

    sep = detect_sep(text)
    df = pd.read_csv(
        StringIO(text),
        sep=sep,
        header=None,
        names=COLUMNS,     # header ditanam di script
        engine="python",
        dtype=str,
    )
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # rapikan teks
    df["Link Produk"] = df["Link Produk"].astype(str).str.strip()
    df["Nama Produk"] = df["Nama Produk"].astype(str).str.strip()

    # ubah kolom numerik
    for c in ["No", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Ratting"]:
        df[c] = df[c].apply(clean_number_id)

    return df


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    # CSV koma, include header
    return df.to_csv(index=False, sep=",").encode("utf-8")


def export_txt_bytes(df: pd.DataFrame) -> bytes:
    # TXT sebagai TSV (tab), include header
    return df.to_csv(index=False, sep="\t").encode("utf-8")


def fmt_id(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.0f}".replace(",", ".")
    except Exception:
        return str(x)

# =========================================================
# Sidebar: Input + Tombol kontrol
# (Tidak ada parsing/cleaning/preview sebelum MULAI PROSES)
# =========================================================
with st.sidebar:
    st.header("Input Data")

    source_mode = st.radio(
        "Sumber data",
        ["Paste (TSV dari Excel / CSV)", "Upload (.txt / .csv)"],
        index=0
    )

    if source_mode.startswith("Paste"):
        st.session_state.raw_text = st.text_area(
            "Paste data TANPA header",
            height=240,
            placeholder="Contoh baris (tanpa header):\n1\thttps://...\tNama Produk\t43.800\t0\t269\t50000\t0\t0\t4"
        )
    else:
        up = st.file_uploader("Upload file TANPA header", type=["txt", "csv"])
        if up is not None:
            raw = up.getvalue()
            try:
                st.session_state.raw_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                st.session_state.raw_text = raw.decode("latin-1", errors="ignore")

    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶️ MULAI PROSES", use_container_width=True)
    with c2:
        reset_btn = st.button("🔄 RESET", use_container_width=True)

# RESET: balikin ke kondisi awal
if reset_btn:
    st.session_state.stage = "input"
    st.session_state.raw_text = ""
    st.session_state.df = None
    st.session_state.df_filtered = None
    st.rerun()

# =========================================================
# STAGE: INPUT (LOCKED) — TIDAK ADA PROSES DI SINI
# =========================================================
if st.session_state.stage == "input":
    st.info("Masukkan data via sidebar lalu klik **▶️ MULAI PROSES**. Tidak ada parsing/cleaning/preview sebelum tombol itu ditekan.")

    if start_btn:
        if not st.session_state.raw_text.strip():
            st.error("Data masih kosong. Paste atau upload dulu.")
            st.stop()

        # === Baru mulai parsing + cleaning setelah tombol ditekan ===
        df0 = read_no_header_text(st.session_state.raw_text)
        if df0.empty:
            st.error("Data terbaca kosong. Pastikan format TSV (tab) atau CSV (koma) tanpa header.")
            st.stop()

        df0 = coerce_types(df0)

        st.session_state.df = df0
        st.session_state.df_filtered = df0.copy()
        st.session_state.stage = "ready"
        st.rerun()

    st.stop()  # wajib stop agar tidak ada proses lanjut

# =========================================================
# STAGE: READY — data sudah diproses, baru boleh preview/filter/export
# =========================================================
df = st.session_state.df
if df is None or df.empty:
    st.error("Data tidak tersedia. Klik RESET lalu input ulang.")
    st.stop()

st.subheader("Preview Data (setelah dibersihkan)")
st.dataframe(df, use_container_width=True, height=320)

# =========================================================
# Filter (manual trigger) — hanya 4 kolom
# =========================================================
st.subheader("Filter (Manual Trigger)")

with st.sidebar:
    st.header("Filter")
    with st.form("filter_form"):
        stock_min = st.number_input("Stock minimal", min_value=0, value=0, step=1)
        tb_min = st.number_input("Terjual Bulanan minimal", min_value=0, value=0, step=1)

        kmax = float(df["Komisi %"].max()) if df["Komisi %"].notna().any() else 0.0
        komisi_pct_rng = st.slider("Rentang Komisi %", 0.0, kmax, (0.0, kmax))

        rmax = float(df["Komisi Rp"].max()) if df["Komisi Rp"].notna().any() else 0.0
        komisi_rp_rng = st.slider("Rentang Komisi Rp", 0.0, rmax, (0.0, rmax))

        run_filter_btn = st.form_submit_button("🚀 JALANKAN FILTER")

# Filter hanya jalan kalau tombol ditekan
if run_filter_btn:
    f = df.copy()
    f = f[f["Stock"].fillna(0) >= stock_min]
    f = f[f["Terjual Bulanan"].fillna(0) >= tb_min]
    f = f[(f["Komisi %"].fillna(0) >= komisi_pct_rng[0]) & (f["Komisi %"].fillna(0) <= komisi_pct_rng[1])]
    f = f[(f["Komisi Rp"].fillna(0) >= komisi_rp_rng[0]) & (f["Komisi Rp"].fillna(0) <= komisi_rp_rng[1])]
    st.session_state.df_filtered = f

df_out = st.session_state.df_filtered if st.session_state.df_filtered is not None else df

# =========================================================
# Output + metrics
# =========================================================
st.subheader("Hasil Filter")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total baris (awal)", len(df))
m2.metric("Total baris (hasil)", len(df_out))
m3.metric("Total Terjual Bulanan", fmt_id(df_out["Terjual Bulanan"].fillna(0).sum()))
m4.metric("Total Komisi Rp", fmt_id(df_out["Komisi Rp"].fillna(0).sum()))

st.dataframe(df_out, use_container_width=True, height=420)

# =========================================================
# Export (CSV / TXT)
# =========================================================
st.subheader("Export Hasil")

export_fmt = st.radio("Pilih format export", ["CSV", "TXT"], horizontal=True, index=0)

if export_fmt == "CSV":
    bytes_out = export_csv_bytes(df_out)
    fname = "hasil_filter_produk.csv"
    mime = "text/csv"
else:
    bytes_out = export_txt_bytes(df_out)
    fname = "hasil_filter_produk.txt"
    mime = "text/plain"

st.download_button(
    f"⬇️ Download hasil ({export_fmt})",
    data=bytes_out,
    file_name=fname,
    mime=mime,
)

st.caption("TXT diexport sebagai TSV (dipisah TAB) + header, supaya mudah dipakai ulang.")
