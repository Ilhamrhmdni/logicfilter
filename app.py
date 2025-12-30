# app.py
import os
import re
import csv
import uuid
import shutil
import tempfile
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Filter Produk (Big Data Mode)", layout="wide")
st.title("📦 Filter & Olah Data Produk (Big Data Mode)")

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

# Log columns (disimpan bareng data)
REJ_LOG_COLS_CSV = COLUMNS_10 + ["Sumber", "Baris Ke", "Kolom Error", "Alasan"]
# untuk TXT log: tanpa header, tanpa No, TAB, plus meta
REJ_LOG_COLS_TXT = [
    "Link Produk", "Nama Produk", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua",
    "Komisi %", "Komisi Rp", "Ratting",
    "Sumber", "Baris Ke", "Kolom Error", "Alasan"
]

# =========================================================
# SESSION STATE (kunci alur)
# =========================================================
if "stage" not in st.session_state:
    st.session_state.stage = "input"  # input -> ready -> filtered

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix=f"shopee_{st.session_state.session_id}_")

if "paths" not in st.session_state:
    wd = st.session_state.workdir
    st.session_state.paths = {
        "raw_input": os.path.join(wd, "raw_input.txt"),
        "cleaned_tsv": os.path.join(wd, "cleaned.tsv"),
        "passed_csv": os.path.join(wd, "passed.csv"),
        "passed_txt": os.path.join(wd, "passed.txt"),
        "reject_csv": os.path.join(wd, "reject_log.csv"),
        "reject_txt": os.path.join(wd, "reject_log.txt"),
    }

if "prep_stats" not in st.session_state:
    st.session_state.prep_stats = {}

if "filter_stats" not in st.session_state:
    st.session_state.filter_stats = {}

if "preview_df" not in st.session_state:
    st.session_state.preview_df = None

if "passed_preview_df" not in st.session_state:
    st.session_state.passed_preview_df = None


# =========================================================
# HELPERS
# =========================================================
def detect_sep_from_sample(sample: str) -> str:
    return "\t" if "\t" in (sample or "") else ","


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
    s = s.replace(".", "").replace(",", "")
    return pd.to_numeric(s, errors="coerce")


def sanitize_basename(name: str, fallback: str) -> str:
    name = (name or "").strip()
    if not name:
        name = fallback
    name = re.sub(r"\.(csv|txt)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = name.strip().strip(".")
    return name or fallback


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


def write_reject_header(csv_path: str):
    # CSV log: pakai header, hanya tulis kalau file belum ada / kosong
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    if need_header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(REJ_LOG_COLS_CSV)


def append_reject_csv(csv_path: str, rows: list):
    # rows: list[dict] keys minimal REJ_LOG_COLS_CSV
    if not rows:
        return
    write_reject_header(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r.get(c, "") for c in REJ_LOG_COLS_CSV])


def append_reject_txt(txt_path: str, rows: list):
    # TXT log: tanpa header, TSV
    if not rows:
        return
    with open(txt_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for r in rows:
            # tanpa No + meta
            w.writerow([r.get(c, "") for c in REJ_LOG_COLS_TXT])


def init_cleaned_file(cleaned_path: str):
    # cleaned.tsv internal: pakai header supaya gampang dibaca chunk
    with open(cleaned_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(COLUMNS_10)


def append_cleaned_rows(cleaned_path: str, df_valid: pd.DataFrame):
    if df_valid is None or df_valid.empty:
        return
    with open(cleaned_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for _, r in df_valid.iterrows():
            out = []
            for c in COLUMNS_10:
                v = r.get(c, "")
                if pd.isna(v):
                    out.append("")
                else:
                    out.append(str(int(v)) if c in NUM_COLS else str(v))
            w.writerow(out)


def process_streaming_prepare(raw_path: str, cleaned_path: str, reject_csv: str, reject_txt: str,
                             sep: str, chunk_size: int = 100_000, preview_rows: int = 200):
    """
    Tahap ▶️ MULAI PROSES:
    - baca raw per-baris
    - skip baris rusak (kolom != 9/10) -> log (Sumber=PARSE)
    - clean angka, kalau ada kolom angka invalid -> skip -> log (Sumber=CLEAN, Kolom Error)
    - tulis baris valid ke cleaned.tsv (internal)
    - simpan sample preview (first N)
    """
    init_cleaned_file(cleaned_path)

    total_lines = 0
    good_rows_total = 0
    bad_parse = 0
    bad_clean = 0

    preview_records = []

    auto_no = 1
    buffer_rows = []
    buffer_meta = []  # lineno

    def flush_buffer():
        nonlocal good_rows_total, bad_clean, preview_records, buffer_rows, buffer_meta

        if not buffer_rows:
            return

        df = pd.DataFrame(buffer_rows, columns=COLUMNS_10)
        linenos = buffer_meta

        # strip text
        df["Link Produk"] = df["Link Produk"].astype(str).str.strip()
        df["Nama Produk"] = df["Nama Produk"].astype(str).str.strip()

        # cek invalid numerik per kolom (hanya untuk value yang tidak kosong)
        bad_cols_map = {}  # idx -> [col1, col2,...]

        for col in NUM_COLS:
            orig = df[col].fillna("").astype(str).str.strip()
            empty_mask = orig.eq("") | orig.str.lower().isin(["nan", "none", "-", "null"])
            cleaned = orig.map(clean_number_id)

            invalid_mask = (~empty_mask) & cleaned.isna()
            if invalid_mask.any():
                bad_idx = df.index[invalid_mask].tolist()
                for i in bad_idx:
                    bad_cols_map.setdefault(i, []).append(col)

            df[col] = cleaned

        if bad_cols_map:
            bad_idxs = sorted(bad_cols_map.keys())
            bad_clean += len(bad_idxs)

            bad_rows = []
            for i in bad_idxs:
                r = df.loc[i]
                bad_rows.append({
                    **{c: ("" if pd.isna(r.get(c)) else r.get(c)) for c in COLUMNS_10},
                    "Sumber": "CLEAN",
                    "Baris Ke": linenos[i],
                    "Kolom Error": ", ".join(bad_cols_map[i]),
                    "Alasan": "Nilai angka tidak valid pada kolom tersebut.",
                })
            append_reject_csv(reject_csv, bad_rows)
            append_reject_txt(reject_txt, bad_rows)

            df = df.drop(index=bad_idxs)

        if df.empty:
            buffer_rows = []
            buffer_meta = []
            return

        # convert numeric to Int64
        for col in NUM_COLS:
            df[col] = df[col].round(0).astype("Int64")

        # tulis ke cleaned file
        append_cleaned_rows(cleaned_path, df)

        # preview sample
        if len(preview_records) < preview_rows:
            needed = preview_rows - len(preview_records)
            sample = df.head(needed).to_dict(orient="records")
            preview_records.extend(sample)

        good_rows_total += len(df)

        buffer_rows = []
        buffer_meta = []

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            total_lines += 1
            raw_line = line.strip("\n")
            if not raw_line.strip():
                continue

            try:
                row = next(csv.reader([raw_line], delimiter=sep))
            except Exception as e:
                bad_parse += 1
                bad_rows = [{
                    **{c: "" for c in COLUMNS_10},
                    "Sumber": "PARSE",
                    "Baris Ke": lineno,
                    "Kolom Error": "Parsing",
                    "Alasan": f"Gagal parsing: {e}",
                }]
                append_reject_csv(reject_csv, bad_rows)
                append_reject_txt(reject_txt, bad_rows)
                continue

            if len(row) == 10:
                buffer_rows.append(row)
                buffer_meta.append(lineno)
            elif len(row) == 9:
                buffer_rows.append([str(auto_no)] + row)
                buffer_meta.append(lineno)
                auto_no += 1
            else:
                bad_parse += 1
                bad_rows = [{
                    **{c: "" for c in COLUMNS_10},
                    "Sumber": "PARSE",
                    "Baris Ke": lineno,
                    "Kolom Error": "Jumlah Kolom",
                    "Alasan": f"Jumlah kolom {len(row)} tidak sesuai (harus 9 atau 10).",
                }]
                append_reject_csv(reject_csv, bad_rows)
                append_reject_txt(reject_txt, bad_rows)
                continue

            if len(buffer_rows) >= chunk_size:
                flush_buffer()

    flush_buffer()

    preview_df = pd.DataFrame(preview_records, columns=COLUMNS_10) if preview_records else pd.DataFrame(columns=COLUMNS_10)

    stats = {
        "total_lines": total_lines,
        "valid_rows": good_rows_total,
        "bad_parse": bad_parse,
        "bad_clean": bad_clean,
    }
    return stats, preview_df


def run_streaming_filter(cleaned_path: str, passed_csv: str, passed_txt: str,
                         reject_csv: str, reject_txt: str,
                         min_stock: float, min_tb: float, min_kpct: float, min_krp: float,
                         include_terms: list, chunk_size: int = 200_000, preview_rows: int = 200):
    """
    Tahap 🚀 JALANKAN FILTER:
    - baca cleaned.tsv per chunk
    - apply filter numeric + nama wajib mengandung (OR)
    - tulis passed ke passed.csv (header) & passed.txt (tanpa header, tanpa No)
    - append rejected ke reject_log (Sumber=FILTER, Kolom Error=criteria yang gagal)
    - hitung totals (Terjual Bulanan & Komisi Rp) dari PASSED
    - ambil sample preview passed (first N)
    """
    # prepare output files
    with open(passed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS_10)

    with open(passed_txt, "w", newline="", encoding="utf-8") as f:
        pass  # no header

    # regex untuk nama
    pattern = None
    if include_terms:
        pattern = "|".join(re.escape(t) for t in include_terms)

    total_seen = 0
    total_passed = 0
    total_rejected_filter = 0
    sum_tb = 0
    sum_krp = 0

    passed_preview = []

    prog = st.progress(0)
    status = st.empty()

    # Perkiraan total valid untuk progress (kalau ada)
    approx_total = st.session_state.prep_stats.get("valid_rows", 0) or 0

    for chunk in pd.read_csv(cleaned_path, sep="\t", chunksize=chunk_size, dtype=str):
        total_seen += len(chunk)

        # numeric -> int (NA jadi 0 untuk compare)
        def to_int0(s):
            return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int64)

        stock = to_int0(chunk["Stock"])
        tb = to_int0(chunk["Terjual Bulanan"])
        kpct = to_int0(chunk["Komisi %"])
        krp = to_int0(chunk["Komisi Rp"])

        mask = (stock >= int(min_stock)) & (tb >= int(min_tb)) & (kpct >= int(min_kpct)) & (krp >= int(min_krp))

        if pattern:
            name_ok = chunk["Nama Produk"].fillna("").astype(str).str.contains(pattern, case=False, regex=True, na=False)
            mask = mask & name_ok

        passed = chunk[mask].copy()
        rejected = chunk[~mask].copy()

        # write passed CSV & TXT
        if not passed.empty:
            # update sums
            sum_tb += int(pd.to_numeric(passed["Terjual Bulanan"], errors="coerce").fillna(0).sum())
            sum_krp += int(pd.to_numeric(passed["Komisi Rp"], errors="coerce").fillna(0).sum())

            total_passed += len(passed)

            # preview passed
            if len(passed_preview) < preview_rows:
                need = preview_rows - len(passed_preview)
                passed_preview.extend(passed.head(need).to_dict(orient="records"))

            # write CSV
            with open(passed_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for _, r in passed.iterrows():
                    w.writerow([r.get(c, "") for c in COLUMNS_10])

            # write TXT (TSV, tanpa header, tanpa No)
            with open(passed_txt, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                for _, r in passed.iterrows():
                    w.writerow([
                        r.get("Link Produk", ""),
                        r.get("Nama Produk", ""),
                        r.get("Harga", ""),
                        r.get("Stock", ""),
                        r.get("Terjual Bulanan", ""),
                        r.get("Terjual Semua", ""),
                        r.get("Komisi %", ""),
                        r.get("Komisi Rp", ""),
                        r.get("Ratting", ""),
                    ])

        # log rejected (FILTER) + alasan + kolom gagal
        if not rejected.empty:
            total_rejected_filter += len(rejected)

            # recompute fail masks (buat alasan)
            stock_f = stock[~mask] < int(min_stock)
            tb_f = tb[~mask] < int(min_tb)
            kpct_f = kpct[~mask] < int(min_kpct)
            krp_f = krp[~mask] < int(min_krp)

            if pattern:
                name_ok_rej = rejected["Nama Produk"].fillna("").astype(str).str.contains(pattern, case=False, regex=True, na=False)
                name_f = ~name_ok_rej
            else:
                name_f = pd.Series(False, index=rejected.index)

            # build "Kolom Error" ringkas
            kol_err = pd.Series("", index=rejected.index, dtype=object)

            def add_col(cond, label):
                nonlocal kol_err
                add = np.where(cond.to_numpy(), label, "")
                kol_err = np.where(
                    kol_err == "",
                    add,
                    np.where(add == "", kol_err, kol_err + ", " + add)
                )

            add_col(stock_f, "Stock")
            add_col(tb_f, "Terjual Bulanan")
            add_col(kpct_f, "Komisi %")
            add_col(krp_f, "Komisi Rp")
            if pattern:
                add_col(name_f, "Nama Produk")

            # build alasan
            alasan = pd.Series("", index=rejected.index, dtype=object)

            def add_reason(cond, txt):
                nonlocal alasan
                add = np.where(cond.to_numpy(), txt, "")
                alasan = np.where(
                    alasan == "",
                    add,
                    np.where(add == "", alasan, alasan + " | " + add)
                )

            add_reason(stock_f, f"Stock < {int(min_stock)}")
            add_reason(tb_f, f"Terjual Bulanan < {int(min_tb)}")
            add_reason(kpct_f, f"Komisi % < {int(min_kpct)}")
            add_reason(krp_f, f"Komisi Rp < {int(min_krp)}")
            if pattern:
                add_reason(name_f, f"Nama tidak mengandung: {', '.join(include_terms)}")

            # append ke reject logs (CSV + TXT)
            rej_rows = []
            for i, r in rejected.iterrows():
                row_dict = {c: r.get(c, "") for c in COLUMNS_10}
                row_dict.update({
                    "Sumber": "FILTER",
                    "Baris Ke": "",
                    "Kolom Error": str(kol_err.loc[i]) if i in kol_err.index else "",
                    "Alasan": str(alasan.loc[i]) if i in alasan.index else "Tidak lolos filter",
                })
                rej_rows.append(row_dict)

            append_reject_csv(reject_csv, rej_rows)
            append_reject_txt(reject_txt, rej_rows)

        # progress
        if approx_total > 0:
            prog.progress(min(1.0, total_seen / approx_total))
            status.write(f"Memproses filter... {total_seen:,} / {approx_total:,} baris valid".replace(",", "."))
        else:
            status.write(f"Memproses filter... {total_seen:,} baris valid".replace(",", "."))

    prog.progress(1.0)
    status.write("✅ Selesai menjalankan filter.")

    stats = {
        "seen_valid_rows": total_seen,
        "passed_rows": total_passed,
        "rejected_filter_rows": total_rejected_filter,
        "sum_terjual_bulanan_passed": sum_tb,
        "sum_komisi_rp_passed": sum_krp,
    }
    passed_preview_df = pd.DataFrame(passed_preview, columns=COLUMNS_10) if passed_preview else pd.DataFrame(columns=COLUMNS_10)
    return stats, passed_preview_df


# =========================================================
# SIDEBAR INPUT (Paste/Upload) + ukuran paste + tombol
# =========================================================
with st.sidebar:
    st.header("Input Data")

    source_mode = st.radio(
        "Sumber data",
        ["Paste (TSV dari Excel / CSV)", "Upload (.txt / .csv)"],
        index=0
    )

    paste_too_big = False
    paste_size_mb = 0.0

    uploaded = None

    if source_mode.startswith("Paste"):
        st.session_state.raw_text = st.text_area(
            "Paste data TANPA header",
            height=220,
            placeholder=(
                "Bisa 10 kolom (dengan No) atau 9 kolom (tanpa No).\n"
                "Contoh 9 kolom:\n"
                "https://...\tNama Produk\t43800\t0\t269\t50000\t0\t0\t4"
            ),
        )

        raw = st.session_state.raw_text or ""
        char_count = len(raw)
        size_bytes = len(raw.encode("utf-8"))
        paste_size_mb = size_bytes / (1024 * 1024)

        st.caption(
            f"📏 Panjang paste: **{char_count:,}** karakter • **{paste_size_mb:.2f} MB** (UTF-8)".replace(",", ".")
        )

        WARN_MB = 1.5
        HARD_LIMIT_MB = 5.0  # big data: jangan paste > 5MB
        if paste_size_mb >= WARN_MB:
            st.warning("⚠️ Paste besar. Untuk data besar, pakai **upload file** biar stabil.")
        if paste_size_mb >= HARD_LIMIT_MB:
            paste_too_big = True
            st.error("❌ Paste terlalu besar. Silakan **upload file (.txt/.csv)**.")

    else:
        uploaded = st.file_uploader("Upload file TANPA header", type=["txt", "csv"])

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶️ MULAI PROSES", use_container_width=True, disabled=paste_too_big)
    with c2:
        reset_btn = st.button("🔄 RESET", use_container_width=True)

# RESET (hapus temp dir)
if reset_btn:
    try:
        shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    except Exception:
        pass
    for k in ["stage", "raw_text", "paths", "prep_stats", "filter_stats", "preview_df", "passed_preview_df", "workdir", "session_id"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# =========================================================
# STAGE: INPUT (LOCKED)
# =========================================================
if st.session_state.stage == "input":
    st.info("Masukkan data via sidebar lalu klik **▶️ MULAI PROSES**. Tidak ada parsing/preview sebelum tombol itu ditekan.")

    if start_btn:
        paths = st.session_state.paths

        # tulis raw_input dari paste/upload
        if source_mode.startswith("Paste"):
            if not (st.session_state.raw_text or "").strip():
                st.error("Data paste masih kosong.")
                st.stop()
            with open(paths["raw_input"], "w", encoding="utf-8", newline="") as f:
                f.write(st.session_state.raw_text)
            sample = (st.session_state.raw_text.splitlines()[0] if st.session_state.raw_text else "")
            sep = detect_sep_from_sample(sample)

        else:
            if uploaded is None:
                st.error("Belum upload file.")
                st.stop()
            data = uploaded.getvalue()
            try:
                txt = data.decode("utf-8")
            except UnicodeDecodeError:
                txt = data.decode("latin-1", errors="ignore")
            with open(paths["raw_input"], "w", encoding="utf-8", newline="") as f:
                f.write(txt)
            sample = txt.splitlines()[0] if txt else ""
            sep = detect_sep_from_sample(sample)

        # prepare reject log files (kosongkan dulu)
        for p in [paths["cleaned_tsv"], paths["passed_csv"], paths["passed_txt"], paths["reject_csv"], paths["reject_txt"]]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        # streaming prepare
        with st.spinner("Memproses input (streaming)..."):
            stats, preview_df = process_streaming_prepare(
                raw_path=paths["raw_input"],
                cleaned_path=paths["cleaned_tsv"],
                reject_csv=paths["reject_csv"],
                reject_txt=paths["reject_txt"],
                sep=sep,
                chunk_size=100_000,
                preview_rows=200,
            )

        st.session_state.prep_stats = stats
        st.session_state.preview_df = preview_df
        st.session_state.filter_stats = {}
        st.session_state.passed_preview_df = None
        st.session_state.stage = "ready"
        st.rerun()

    st.stop()

# =========================================================
# STAGE: READY (preview sample + filter form)
# =========================================================
paths = st.session_state.paths
prep = st.session_state.prep_stats or {}
preview_df = st.session_state.preview_df

st.subheader("✅ Preview (sample) setelah dibersihkan")
cA, cB, cC, cD = st.columns(4)
cA.metric("Total baris (raw)", f"{prep.get('total_lines', 0):,}".replace(",", "."))
cB.metric("Baris valid", f"{prep.get('valid_rows', 0):,}".replace(",", "."))
cC.metric("Skip rusak (PARSE)", f"{prep.get('bad_parse', 0):,}".replace(",", "."))
cD.metric("Skip angka invalid (CLEAN)", f"{prep.get('bad_clean', 0):,}".replace(",", "."))

if preview_df is not None:
    st.dataframe(preview_df, use_container_width=True, height=320)

st.divider()

# =========================================================
# FILTER (manual trigger) — tetap locked
# =========================================================
st.subheader("Filter (jalan hanya saat tombol 🚀 ditekan)")

with st.sidebar:
    st.header("Filter (MIN + Nama wajib mengandung)")
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

if run_filter_btn:
    terms = parse_terms(include_words_raw)

    with st.spinner("Menjalankan filter (streaming) + membuat file output..."):
        stats, passed_preview_df = run_streaming_filter(
            cleaned_path=paths["cleaned_tsv"],
            passed_csv=paths["passed_csv"],
            passed_txt=paths["passed_txt"],
            reject_csv=paths["reject_csv"],  # append ke log yang sama
            reject_txt=paths["reject_txt"],
            min_stock=min_stock,
            min_tb=min_tb,
            min_kpct=min_kpct,
            min_krp=min_krp,
            include_terms=terms,
            chunk_size=200_000,
            preview_rows=200,
        )

    st.session_state.filter_stats = stats
    st.session_state.passed_preview_df = passed_preview_df
    st.session_state.stage = "filtered"
    st.rerun()

# =========================================================
# STAGE: FILTERED (hasil + download)
# =========================================================
if st.session_state.stage == "filtered":
    fs = st.session_state.filter_stats or {}
    st.subheader("✅ Ringkasan hasil filter (dari data yang LOLOS)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baris valid diproses", f"{fs.get('seen_valid_rows', 0):,}".replace(",", "."))
    m2.metric("Baris lolos", f"{fs.get('passed_rows', 0):,}".replace(",", "."))
    m3.metric("Total Terjual Bulanan (lolos)", fmt_id(fs.get("sum_terjual_bulanan_passed", 0)))
    m4.metric("Total Komisi Rp (lolos)", fmt_id(fs.get("sum_komisi_rp_passed", 0)))

    st.subheader("Preview (sample) hasil LOLos")
    if st.session_state.passed_preview_df is not None:
        st.dataframe(st.session_state.passed_preview_df, use_container_width=True, height=320)

    with st.expander("🧾 Log tidak lolos (PARSE/CLEAN/FILTER)"):
        st.caption("Log ini berisi baris yang rusak/angka invalid + baris yang gagal kriteria filter, dengan Kolom Error & Alasan.")
        # demi performa, jangan load log penuh; tampilkan sample kecil
        if os.path.exists(paths["reject_csv"]) and os.path.getsize(paths["reject_csv"]) > 0:
            try:
                sample_log = pd.read_csv(paths["reject_csv"], nrows=200)
                st.dataframe(sample_log, use_container_width=True, height=320)
            except Exception:
                st.info("Log besar / tidak bisa dipreview. Silakan download log.")
        else:
            st.info("Tidak ada log.")

    st.divider()

    # =========================================================
    # EXPORT + nama file custom
    # =========================================================
    st.subheader("Export (file besar diproses streaming)")

    base_name = st.text_input("Nama file hasil (tanpa ekstensi)", value="hasil_filter_produk")
    base_name = sanitize_basename(base_name, "hasil_filter_produk")

    log_name = st.text_input("Nama file log (tanpa ekstensi)", value=f"{base_name}_log_tidak_lolos")
    log_name = sanitize_basename(log_name, f"{base_name}_log_tidak_lolos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⬇️ Download HASIL (LOLOS)")
        fmt = st.radio("Format hasil", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_pass")
        if fmt == "CSV":
            out_path = paths["passed_csv"]
            out_file = f"{base_name}.csv"
            mime = "text/csv"
        else:
            out_path = paths["passed_txt"]
            out_file = f"{base_name}.txt"
            mime = "text/plain"

        if os.path.exists(out_path):
            # Catatan: download file super besar via browser bisa berat.
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            if size_mb > 100:
                st.warning(f"⚠️ File hasil {size_mb:.1f} MB. Download via web bisa berat. Pertimbangkan filter lebih ketat / jalankan lokal.")
            with open(out_path, "rb") as f:
                st.download_button(
                    f"Download hasil ({fmt})",
                    data=f.read(),
                    file_name=out_file,
                    mime=mime,
                )
        else:
            st.info("Belum ada file hasil. Jalankan filter dulu.")

    with col2:
        st.markdown("### ⬇️ Download LOG (TIDAK LOLOS)")
        fmt2 = st.radio("Format log", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_log")
        if fmt2 == "CSV":
            log_path = paths["reject_csv"]
            log_file = f"{log_name}.csv"
            log_mime = "text/csv"
        else:
            log_path = paths["reject_txt"]
            log_file = f"{log_name}.txt"
            log_mime = "text/plain"

        if os.path.exists(log_path):
            size_mb = os.path.getsize(log_path) / (1024 * 1024)
            if size_mb > 100:
                st.warning(f"⚠️ File log {size_mb:.1f} MB. Download via web bisa berat.")
            with open(log_path, "rb") as f:
                st.download_button(
                    f"Download log ({fmt2})",
                    data=f.read(),
                    file_name=log_file,
                    mime=log_mime,
                )
        else:
            st.info("Belum ada log.")

else:
    st.caption("Klik 🚀 JALANKAN FILTER untuk membuat output & log.")
