# app.py
import re
import csv
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

NUM_COLS = ["No", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua", "Komisi %", "Komisi Rp", "Ratting"]

LOG_EXTRA_COLS = ["Sumber", "Baris Ke", "Kolom Error", "Alasan", "Baris Asli"]

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

if "df_rejected_parse" not in st.session_state:
    st.session_state.df_rejected_parse = pd.DataFrame(columns=COLUMNS_10 + LOG_EXTRA_COLS)

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
    # izinkan angka + titik + koma
    s = re.sub(r"[^0-9\.,]", "", s)

    # umumnya titik/koma sebagai pemisah ribuan
    s = s.replace(".", "").replace(",", "")
    return pd.to_numeric(s, errors="coerce")


def detect_sep(text: str) -> str:
    return "\t" if "\t" in text else ","


def parse_terms(raw: str):
    parts = re.split(r"[,\n;]+", (raw or ""))
    return [p.strip() for p in parts if p.strip()]


def fmt_id(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


def sanitize_basename(name: str, fallback: str) -> str:
    name = (name or "").strip()
    if not name:
        name = fallback
    name = re.sub(r"\.(csv|txt)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = name.strip().strip(".")
    return name or fallback


def export_csv_bytes(df: pd.DataFrame) -> bytes:
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
    # CSV log berisi header
    return df_rej.to_csv(index=False, sep=",").encode("utf-8")


def export_reject_txt_bytes(df_rej: pd.DataFrame) -> bytes:
    """
    TXT log reject:
    - TANPA header
    - dipisah TAB
    - berisi data (tanpa No) + Kolom Error + Alasan + Sumber + Baris Ke + Baris Asli
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
        "Kolom Error",
        "Alasan",
        "Sumber",
        "Baris Ke",
        "Baris Asli",
    ]
    df_txt = df_rej.reindex(columns=cols_txt).copy()
    return df_txt.to_csv(index=False, header=False, sep="\t").encode("utf-8")


def parse_input_to_df_and_bad(text: str):
    """
    Parse per-baris supaya:
    - baris rusak (jumlah kolom != 9/10) DISKIP
    - baris rusak masuk log (df_bad)
    """
    text = (text or "")
    sep = detect_sep(text)

    good_rows = []
    bad_rows = []

    auto_no = 1
    for lineno, line in enumerate(text.splitlines(), start=1):
        raw_line = line
        if not raw_line.strip():
            continue

        try:
            row = next(csv.reader([raw_line], delimiter=sep))
        except Exception as e:
            bad_rows.append({
                **{c: "" for c in COLUMNS_10},
                "Sumber": "PARSE",
                "Baris Ke": lineno,
                "Kolom Error": "Parsing",
                "Alasan": f"Gagal parsing: {e}",
                "Baris Asli": raw_line,
            })
            continue

        n = len(row)

        if n == 10:
            rec = dict(zip(COLUMNS_10, row))
            rec["__lineno__"] = lineno
            good_rows.append(rec)
        elif n == 9:
            rec = {"No": str(auto_no)}
            rec.update(dict(zip(COLUMNS_9_NO_NO, row)))
            rec["__lineno__"] = lineno
            good_rows.append(rec)
            auto_no += 1
        else:
            # baris tidak sesuai → skip & log
            bad_rows.append({
                **{c: "" for c in COLUMNS_10},
                "Sumber": "PARSE",
                "Baris Ke": lineno,
                "Kolom Error": "Jumlah Kolom",
                "Alasan": f"Jumlah kolom {n} tidak sesuai (harus 9 atau 10).",
                "Baris Asli": raw_line,
            })

    df_good = pd.DataFrame(good_rows)
    df_bad = pd.DataFrame(bad_rows, columns=COLUMNS_10 + LOG_EXTRA_COLS)

    return df_good, df_bad


def validate_and_clean_numbers(df_raw_good: pd.DataFrame):
    """
    - Deteksi baris yang punya nilai angka invalid → skip & log kolom error
    - Baris valid → dibersihkan jadi numeric Int64
    """
    if df_raw_good is None or df_raw_good.empty:
        return pd.DataFrame(columns=COLUMNS_10), pd.DataFrame(columns=COLUMNS_10 + LOG_EXTRA_COLS)

    df_raw_good = df_raw_good.copy()

    # pastikan semua kolom ada
    for c in COLUMNS_10:
        if c not in df_raw_good.columns:
            df_raw_good[c] = ""

    # ambil lineno sebelum dibuang
    if "__lineno__" not in df_raw_good.columns:
        df_raw_good["__lineno__"] = pd.NA

    bad_idx = []
    bad_logs = []

    for idx, row in df_raw_good.iterrows():
        bad_cols = []
        for c in NUM_COLS:
            orig = "" if pd.isna(row.get(c)) else str(row.get(c)).strip()

            # kalau kosong / '-' dianggap boleh (jadi NaN)
            if orig == "" or orig.lower() in ["nan", "none", "-", "null"]:
                continue

            conv = clean_number_id(orig)
            if pd.isna(conv):
                bad_cols.append(c)

        if bad_cols:
            bad_idx.append(idx)
            bad_logs.append({
                **{c: ("" if pd.isna(row.get(c)) else row.get(c)) for c in COLUMNS_10},
                "Sumber": "CLEAN",
                "Baris Ke": int(row["__lineno__"]) if pd.notna(row["__lineno__"]) else "",
                "Kolom Error": ", ".join(bad_cols),
                "Alasan": "Nilai angka tidak valid pada kolom tersebut.",
                "Baris Asli": "",  # bisa dikosongkan karena sudah ada kolom per-kolom
            })

    # drop baris invalid dulu
    df_ok = df_raw_good.drop(index=bad_idx).copy()

    # cleaning numeric untuk yang OK
    df_ok["Link Produk"] = df_ok["Link Produk"].astype(str).str.strip()
    df_ok["Nama Produk"] = df_ok["Nama Produk"].astype(str).str.strip()

    for c in NUM_COLS:
        df_ok[c] = df_ok[c].apply(clean_number_id)

    for c in NUM_COLS:
        df_ok[c] = df_ok[c].round(0).astype("Int64")

    # buang helper
    if "__lineno__" in df_ok.columns:
        df_ok = df_ok.drop(columns=["__lineno__"], errors="ignore")

    df_bad_num = pd.DataFrame(bad_logs, columns=COLUMNS_10 + LOG_EXTRA_COLS)

    return df_ok, df_bad_num


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

        raw = st.session_state.raw_text or ""
        char_count = len(raw)
        size_bytes = len(raw.encode("utf-8"))
        size_mb = size_bytes / (1024 * 1024)

        st.caption(
            f"📏 Panjang paste: **{char_count:,}** karakter • **{size_mb:.2f} MB** (UTF-8)".replace(",", ".")
        )

        THRESH_MB = 1.5
        if size_mb >= THRESH_MB:
            st.warning("⚠️ Paste kamu sudah besar. Disarankan **upload file (.txt/.csv)** agar lebih stabil.")

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
    st.session_state.df_rejected_parse = pd.DataFrame(columns=COLUMNS_10 + LOG_EXTRA_COLS)
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

        # ✅ parse per baris: skip baris rusak → masuk log
        df_good_raw, df_bad_parse = parse_input_to_df_and_bad(st.session_state.raw_text)

        # ✅ validasi angka: baris dengan angka invalid → skip → masuk log
        df_good, df_bad_num = validate_and_clean_numbers(df_good_raw)

        # gabung log parse + clean
        df_rej_parse = pd.concat([df_bad_parse, df_bad_num], ignore_index=True)
        st.session_state.df_rejected_parse = df_rej_parse

        if df_good.empty:
            st.error("Tidak ada baris valid yang bisa diproses. Cek log (baris rusak/angka invalid).")
            st.session_state.df = pd.DataFrame(columns=COLUMNS_10)
            st.session_state.df_filtered = pd.DataFrame(columns=COLUMNS_10)
            st.session_state.stage = "ready"
            st.rerun()

        st.session_state.df = df_good
        st.session_state.df_filtered = df_good.copy()
        st.session_state.stage = "ready"
        st.rerun()

    st.stop()

# =========================================================
# READY
# =========================================================
df = st.session_state.df
if df is None:
    st.error("Data tidak tersedia. Klik RESET lalu input ulang.")
    st.stop()

# Info jika ada baris rusak yang diskip
rej_parse = st.session_state.df_rejected_parse
if rej_parse is not None and not rej_parse.empty:
    st.warning(f"⚠️ Ada {len(rej_parse)} baris yang di-skip karena rusak / angka invalid. Lihat di Log Tidak Lolos.")

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

        include_words_raw = st.text_input(
            "Nama Produk WAJIB mengandung kata (pisahkan koma/enter)",
            value="",
            placeholder="contoh: anak, bayi, kids",
        )

        run_filter_btn = st.form_submit_button("🚀 JALANKAN FILTER")

def add_reason(reason_map: dict, idxs, reason: str):
    for i in idxs:
        reason_map.setdefault(i, []).append(reason)

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

    # --- Nama Produk wajib mengandung (OR)
    terms = parse_terms(include_words_raw)
    if terms:
        pattern = "|".join(re.escape(t) for t in terms)
        name_series = f["Nama Produk"].fillna("").astype(str)

        name_fail = f[~name_series.str.contains(pattern, case=False, regex=True)].index
        add_reason(reason_map, name_fail, f"Nama tidak mengandung: {', '.join(terms)}")

        f = f[name_series.str.contains(pattern, case=False, regex=True)]

    st.session_state.df_filtered = f

    # build log filter rejects
    passed_idx = set(f.index.tolist())
    all_idx = df.index.tolist()
    rejected_idx = [i for i in all_idx if i not in passed_idx]

    df_rej_filter = df.loc[rejected_idx].copy()
    alasan_list = []
    for i in rejected_idx:
        reasons = reason_map.get(i) or ["Tidak lolos filter"]
        alasan_list.append(" | ".join(reasons))

    df_rej_filter["Sumber"] = "FILTER"
    df_rej_filter["Baris Ke"] = ""
    df_rej_filter["Kolom Error"] = ""
    df_rej_filter["Alasan"] = alasan_list
    df_rej_filter["Baris Asli"] = ""

    # gabung dengan log parse/clean
    df_all_log = pd.concat([st.session_state.df_rejected_parse, df_rej_filter], ignore_index=True)
    st.session_state.df_rejected = df_all_log

# output
df_out = st.session_state.df_filtered if st.session_state.df_filtered is not None else df
df_rej_out = st.session_state.df_rejected
if df_rej_out is None:
    df_rej_out = st.session_state.df_rejected_parse

# =========================================================
# METRICS + TABLE
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total baris (awal, valid)", len(df))
m2.metric("Total baris (lolos)", len(df_out))
m3.metric("Total Terjual Bulanan (lolos)", fmt_id(df_out["Terjual Bulanan"].fillna(0).sum()))
m4.metric("Total Komisi Rp (lolos)", fmt_id(df_out["Komisi Rp"].fillna(0).sum()))

st.dataframe(df_out, use_container_width=True, height=420)

# =========================================================
# LOG DATA TIDAK LOLOS
# =========================================================
with st.expander("🧾 Log data tidak lolos (rusak/invalid/filter)"):
    if df_rej_out is None or df_rej_out.empty:
        st.info("Belum ada log.")
    else:
        st.caption(f"Jumlah data/log tidak lolos: {len(df_rej_out)}")
        st.dataframe(df_rej_out, use_container_width=True, height=320)

# =========================================================
# EXPORT HASIL LOLOS
# =========================================================
st.subheader("Export Hasil (Lolos)")

colA, colB = st.columns([2, 3])
with colA:
    export_fmt = st.radio("Format export", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_pass")
with colB:
    base_name = st.text_input("Nama file (tanpa ekstensi)", value="hasil_filter_produk", key="name_pass")

base_name = sanitize_basename(base_name, "hasil_filter_produk")

if export_fmt == "CSV":
    bytes_out = export_csv_bytes(df_out)
    fname = f"{base_name}.csv"
    mime = "text/csv"
    note = "CSV berisi semua kolom + header."
else:
    bytes_out = export_txt_bytes(df_out)
    fname = f"{base_name}.txt"
    mime = "text/plain"
    note = "TXT (TSV TAB) tanpa header & tanpa kolom No (sesuai format kamu)."

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
    st.info("Log belum tersedia.")
else:
    colC, colD = st.columns([2, 3])
    with colC:
        log_fmt = st.radio("Format log", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_log")
    with colD:
        log_name = st.text_input(
            "Nama file log (tanpa ekstensi)",
            value=f"{base_name}_log_tidak_lolos",
            key="name_log"
        )

    log_name = sanitize_basename(log_name, f"{base_name}_log_tidak_lolos")

    if log_fmt == "CSV":
        log_bytes = export_reject_csv_bytes(df_rej_out)
        log_fname = f"{log_name}.csv"
        log_mime = "text/csv"
        log_note = "CSV log berisi data + Sumber + Kolom Error + Alasan + Baris Asli (dengan header)."
    else:
        log_bytes = export_reject_txt_bytes(df_rej_out)
        log_fname = f"{log_name}.txt"
        log_mime = "text/plain"
        log_note = "TXT log = TSV TAB tanpa header, berisi data + Kolom Error + Alasan + info sumber."

    st.download_button(
        f"⬇️ Download LOG ({log_fmt})",
        data=log_bytes,
        file_name=log_fname,
        mime=log_mime,
    )
    st.caption(log_note)
