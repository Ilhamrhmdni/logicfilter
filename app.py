# app.py
import re
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Filter Produk (Locked Pipeline)", layout="wide")
st.title("📦 Filter & Olah Data Produk (Shopee)")

# =========================================================
# HEADER TETAP (ditanam di script)
# =========================================================
COLUMNS_10 = [
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

# Format input TANPA kolom No (9 kolom)
COLUMNS_9_NO_NO = [
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
# SESSION STATE (kunci alur)
# =========================================================
if "stage" not in st.session_state:
    st.session_state.stage = "input"  # input -> ready

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "df" not in st.session_state:
    st.session_state.df = None

if "df_filtered" not in st.session_state:
    st.session_state.df_filtered = None

if "df_rejected" not in st.session_state:
    st.session_state.df_rejected = None

# =========================================================
# HELPERS
# =========================================================
def clean_number_id(x):
    """
    Bersihin angka model Indonesia:
    '43.800' -> 43800
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

    # umumnya titik/koma sebagai pemisah ribuan
    s = s.replace(".", "").replace(",", "")
    return pd.to_numeric(s, errors="coerce")


def detect_sep(text: str) -> str:
    return "\t" if "\t" in text else ","


def read_no_header_text(text: str) -> pd.DataFrame:
    """
    Baca data TANPA header dari paste/upload.
    Mendukung 2 format:
      - 10 kolom (ada No di depan)
      - 9 kolom (tanpa No) -> No dibuat otomatis 1..n
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=COLUMNS_10)

    sep = detect_sep(text)

    df_raw = pd.read_csv(
        StringIO(text),
        sep=sep,
        header=None,
        engine="python",
        dtype=str,
    )

    ncols = df_raw.shape[1]
    if ncols == 10:
        df_raw.columns = COLUMNS_10
        return df_raw

    if ncols == 9:
        df_raw.columns = COLUMNS_9_NO_NO
        df_raw.insert(0, "No", range(1, len(df_raw) + 1))
        return df_raw

    raise ValueError(
        f"Jumlah kolom tidak sesuai. Ditemukan {ncols} kolom. "
        "Harus 10 kolom (dengan No) atau 9 kolom (tanpa No)."
    )


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Link Produk"] = df["Link Produk"].astype(str).str.strip()
    df["Nama Produk"] = df["Nama Produk"].astype(str).str.strip()

    for c in ["No", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Ratting"]:
        df[c] = df[c].apply(clean_number_id)

    # rapikan jadi integer nullable biar export gak banyak .0
    int_cols = ["No", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Ratting"]
    for c in int_cols:
        df[c] = df[c].round(0).astype("Int64")

    return df


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    # CSV normal: include header + semua kolom
    return df.to_csv(index=False, sep=",").encode("utf-8")


def export_txt_bytes(df: pd.DataFrame) -> bytes:
    """
    TXT hasil lolos sesuai format kamu:
    - TANPA header
    - TANPA kolom "No"
    - dipisah TAB (TSV)
    """
    cols_txt = [
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
    df_txt = df[cols_txt].copy()
    return df_txt.to_csv(index=False, header=False, sep="\t").encode("utf-8")


def export_reject_csv_bytes(df_rej: pd.DataFrame) -> bytes:
    return df_rej.to_csv(index=False, sep=",").encode("utf-8")


def export_reject_txt_bytes(df_rej: pd.DataFrame) -> bytes:
    """
    TXT log reject:
    - TANPA header
    - TANPA kolom No
    - + kolom terakhir: Alasan
    - dipisah TAB
    """
    cols_txt = [
        "Link Produk",
        "Nama Produk",
        "Harga",
        "Stock",
        "Terjual Bulanan",
        "Terjual Semua",
        "Komisi %",
        "Komisi Rp",
        "Ratting",
        "Alasan",
    ]
    df_txt = df_rej[cols_txt].copy()
    return df_txt.to_csv(index=False, header=False, sep="\t").encode("utf-8")


def fmt_id(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


def sanitize_basename(name: str, fallback: str = "hasil_filter_produk") -> str:
    name = (name or "").strip()
    if not name:
        name = fallback

    # buang ekstensi kalau user mengetik .csv/.txt
    name = re.sub(r"\.(csv|txt)\s*$", "", name, flags=re.IGNORECASE)

    # karakter ilegal windows -> _
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)

    # rapikan
    name = name.strip().strip(".")
    return name or fallback


def parse_terms(raw: str):
    """Split keyword by comma/semicolon/newline, strip empty."""
    parts = re.split(r"[,\n;]+", (raw or ""))
    return [p.strip() for p in parts if p.strip()]


def add_reason(reason_map: dict, idxs, reason: str):
    for i in idxs:
        reason_map.setdefault(i, []).append(reason)

# =========================================================
# SIDEBAR INPUT + CONTROL
# Tidak ada parsing/cleaning/preview sebelum ▶️ MULAI PROSES
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
            placeholder=(
                "Bisa 10 kolom (dengan No) atau 9 kolom (tanpa No).\n\n"
                "Contoh 9 kolom:\n"
                "https://...\tNama Produk\t43800\t0\t269\t50000\t0\t0\t4"
            ),
        )

        # ====== INFO UKURAN PASTE + WARNING ======
        raw = st.session_state.raw_text or ""
        char_count = len(raw)
        size_bytes = len(raw.encode("utf-8"))
        size_mb = size_bytes / (1024 * 1024)

        st.caption(
            f"📏 Panjang paste: **{char_count:,}** karakter • **{size_mb:.2f} MB** (UTF-8)".replace(",", ".")
        )

        THRESH_MB = 1.5  # ambang warning (1–2 MB)
        if size_mb >= THRESH_MB:
            st.warning("⚠️ Paste kamu sudah besar. Disarankan **upload file (.txt/.csv)** agar lebih stabil dan tidak lag/timeout.")

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

# RESET
if reset_btn:
    st.session_state.stage = "input"
    st.session_state.raw_text = ""
    st.session_state.df = None
    st.session_state.df_filtered = None
    st.session_state.df_rejected = None
    st.rerun()

# =========================================================
# STAGE INPUT (LOCKED)
# =========================================================
if st.session_state.stage == "input":
    st.info("Masukkan data via sidebar lalu klik **▶️ MULAI PROSES**. Tidak ada parsing/cleaning/preview sebelum tombol itu ditekan.")

    if start_btn:
        if not st.session_state.raw_text.strip():
            st.error("Data masih kosong. Paste atau upload dulu.")
            st.stop()

        try:
            df0 = read_no_header_text(st.session_state.raw_text)
        except Exception as e:
            st.error(f"Gagal membaca data: {e}")
            st.stop()

        if df0.empty:
            st.error("Data terbaca kosong. Pastikan TSV (tab) atau CSV (koma) tanpa header.")
            st.stop()

        # parsing + cleaning hanya setelah tombol ditekan
        df0 = coerce_types(df0)

        st.session_state.df = df0
        st.session_state.df_filtered = df0.copy()
        st.session_state.df_rejected = pd.DataFrame(columns=list(df0.columns) + ["Alasan"])
        st.session_state.stage = "ready"
        st.rerun()

    st.stop()

# =========================================================
# READY
# =========================================================
df = st.session_state.df
if df is None or df.empty:
    st.error("Data tidak tersedia. Klik RESET lalu input ulang.")
    st.stop()

st.subheader("Preview Data (setelah dibersihkan)")
st.dataframe(df, use_container_width=True, height=320)

# =========================================================
# FILTER MIN + WAJIB MENGANDUNG KATA PADA NAMA PRODUK (include) + LOG REJECT
# =========================================================
st.subheader("Hasil Filter")

with st.sidebar:
    st.header("Filter (MIN - input angka)")
    with st.form("filter_form"):
        min_stock = st.number_input("Stock minimal", min_value=0, value=0, step=1)
        min_tb = st.number_input("Terjual Bulanan minimal", min_value=0, value=0, step=1)
        min_kpct = st.number_input("Komisi % minimal", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        min_krp = st.number_input("Komisi Rp minimal", min_value=0.0, value=0.0, step=100.0, format="%.0f")

        # ✅ PERUBAHAN: wajib mengandung kata -> jika tidak mengandung, dibuang
        include_words_raw = st.text_input(
            'Nama Produk WAJIB mengandung kata (pisahkan koma/enter)',
            value="",
            placeholder="contoh: anak, bayi, kids",
        )

        run_filter_btn = st.form_submit_button("🚀 JALANKAN FILTER")

if run_filter_btn:
    reason_map = {}
    f = df.copy()

    # --- Stock
    stock_fail = f[f["Stock"].fillna(0).astype(float) < float(min_stock)].index
    add_reason(reason_map, stock_fail, f"Stock < {int(min_stock)}")
    f = f[f["Stock"].fillna(0).astype(float) >= float(min_stock)]

    # --- Terjual Bulanan
    tb_fail = f[f["Terjual Bulanan"].fillna(0).astype(float) < float(min_tb)].index
    add_reason(reason_map, tb_fail, f"Terjual Bulanan < {int(min_tb)}")
    f = f[f["Terjual Bulanan"].fillna(0).astype(float) >= float(min_tb)]

    # --- Komisi %
    kp_fail = f[f["Komisi %"].fillna(0).astype(float) < float(min_kpct)].index
    add_reason(reason_map, kp_fail, f"Komisi % < {min_kpct:g}")
    f = f[f["Komisi %"].fillna(0).astype(float) >= float(min_kpct)]

    # --- Komisi Rp
    kr_fail = f[f["Komisi Rp"].fillna(0).astype(float) < float(min_krp)].index
    add_reason(reason_map, kr_fail, f"Komisi Rp < {min_krp:g}")
    f = f[f["Komisi Rp"].fillna(0).astype(float) >= float(min_krp)]

    # --- Nama Produk wajib mengandung (OR: salah satu term cukup)
    terms = parse_terms(include_words_raw)
    if terms:
        pattern = "|".join(re.escape(t) for t in terms)
        name_series = f["Nama Produk"].fillna("").astype(str)

        # yang gagal = yang tidak mengandung pattern
        name_fail = f[~name_series.str.contains(pattern, case=False, regex=True)].index
        add_reason(reason_map, name_fail, f"Nama tidak mengandung: {', '.join(terms)}")

        # yang lolos = yang mengandung
        f = f[name_series.str.contains(pattern, case=False, regex=True)]

    # simpan lolos
    st.session_state.df_filtered = f

    # rejected = semua index original yang tidak lolos
    passed_idx = set(f.index.tolist())
    all_idx = df.index.tolist()
    rejected_idx = [i for i in all_idx if i not in passed_idx]

    df_rej = df.loc[rejected_idx].copy()
    alasan_list = []
    for i in rejected_idx:
        reasons = reason_map.get(i)
        if not reasons:
            reasons = ["Tidak lolos filter"]
        alasan_list.append(" | ".join(reasons))

    df_rej["Alasan"] = alasan_list
    st.session_state.df_rejected = df_rej

df_out = st.session_state.df_filtered if st.session_state.df_filtered is not None else df
df_rej_out = st.session_state.df_rejected

# =========================================================
# METRICS + TABLE
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Produk", len(df))
m2.metric("Total Produk Lolos", len(df_out))
m3.metric("Total Terjual Bulanan (lolos)", fmt_id(df_out["Harga"].fillna(0).sum()))
m4.metric("Total Komisi Rp (lolos)", fmt_id(df_out["Komisi Rp"].fillna(0).sum()))

st.dataframe(df_out, use_container_width=True, height=420)

# =========================================================
# LOG DATA TIDAK LOLOS (ditampilkan + bisa export)
# =========================================================
with st.expander("🧾 Log data tidak lolos filter"):
    if df_rej_out is None or df_rej_out.empty:
        st.info("Belum ada data log. Klik **🚀 JALANKAN FILTER** dulu.")
    else:
        st.caption(f"Jumlah data tidak lolos: {len(df_rej_out)}")
        st.dataframe(df_rej_out, use_container_width=True, height=320)

# =========================================================
# EXPORT HASIL LOLOS (CSV / TXT) + Nama file custom
# =========================================================
st.subheader("Export Hasil (Lolos)")

colA, colB = st.columns([2, 3])
with colA:
    export_fmt = st.radio("Format export", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_pass")
with colB:
    base_name = st.text_input("Nama file", value="Masukan_nama_file", key="name_pass")

base_name = sanitize_basename(base_name)

if export_fmt == "CSV":
    bytes_out = export_csv_bytes(df_out)
    fname = f"{base_name}.csv"
    mime = "text/csv"
else:
    bytes_out = export_txt_bytes(df_out)
    fname = f"{base_name}.txt"
    mime = "text/plain"

st.download_button(
    f"⬇️ Download ({export_fmt})",
    data=bytes_out,
    file_name=fname,
    mime=mime,
)
st.caption(note)

# =========================================================
# EXPORT LOG TIDAK LOLOS
# =========================================================
st.subheader("Export Log (Tidak Lolos)")

if df_rej_out is None or df_rej_out.empty:
    st.info("Log belum tersedia. Klik **🚀 JALANKAN FILTER** dulu untuk membuat log data tidak lolos.")
else:
    colC, colD = st.columns([2, 3])
    with colC:
        log_fmt = st.radio("Format log", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_log")
    with colD:
        log_name = st.text_input(
            "Nama file log",
            value=f"{base_name}",
            key="name_log"
        )

    log_name = sanitize_basename(log_name, fallback=f"{base_name}_log_tidak_lolos")

    if log_fmt == "CSV":
        log_bytes = export_reject_csv_bytes(df_rej_out)
        log_fname = f"{log_name}.csv"
        log_mime = "text/csv"
    else:
        log_bytes = export_reject_txt_bytes(df_rej_out)
        log_fname = f"{log_name}.txt"
        log_mime = "text/plain"

    st.download_button(
        f"⬇️ Download LOG ({log_fmt})",
        data=log_bytes,
        file_name=log_fname,
        mime=log_mime,
    )
    st.caption(log_note)
