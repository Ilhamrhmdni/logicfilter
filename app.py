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

st.set_page_config(page_title="Shopee Product Filter", layout="wide")

# =========================
# HEADER TETAP
# =========================
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

REJ_LOG_COLS_CSV = COLUMNS_10 + ["Sumber", "Baris Ke", "Kolom Error", "Alasan"]
REJ_LOG_COLS_TXT = [
    "Link Produk", "Nama Produk", "Harga", "Stock", "Terjual Bulanan", "Terjual Semua",
    "Komisi %", "Komisi Rp", "Ratting",
    "Sumber", "Baris Ke", "Kolom Error", "Alasan"
]

# =========================
# SESSION STATE
# =========================
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


# =========================
# HELPERS
# =========================
def detect_sep_from_sample(sample: str) -> str:
    return "\t" if "\t" in (sample or "") else ","


def clean_number_id(x):
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
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    if need_header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(REJ_LOG_COLS_CSV)


def append_reject_csv(csv_path: str, rows: list):
    if not rows:
        return
    write_reject_header(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r.get(c, "") for c in REJ_LOG_COLS_CSV])


def append_reject_txt(txt_path: str, rows: list):
    if not rows:
        return
    with open(txt_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for r in rows:
            w.writerow([r.get(c, "") for c in REJ_LOG_COLS_TXT])


def init_cleaned_file(cleaned_path: str):
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
    init_cleaned_file(cleaned_path)

    total_lines = 0
    good_rows_total = 0
    bad_parse = 0
    bad_clean = 0

    preview_records = []
    auto_no = 1
    buffer_rows = []
    buffer_meta = []

    def flush_buffer():
        nonlocal good_rows_total, bad_clean, preview_records, buffer_rows, buffer_meta

        if not buffer_rows:
            return

        df = pd.DataFrame(buffer_rows, columns=COLUMNS_10)
        linenos = buffer_meta

        df["Link Produk"] = df["Link Produk"].astype(str).str.strip()
        df["Nama Produk"] = df["Nama Produk"].astype(str).str.strip()

        bad_cols_map = {}

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
            buffer_rows.clear()
            buffer_meta.clear()
            return

        for col in NUM_COLS:
            df[col] = df[col].round(0).astype("Int64")

        append_cleaned_rows(cleaned_path, df)

        if len(preview_records) < preview_rows:
            needed = preview_rows - len(preview_records)
            preview_records.extend(df.head(needed).to_dict(orient="records"))

        good_rows_total += len(df)

        buffer_rows.clear()
        buffer_meta.clear()

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
    with open(passed_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(COLUMNS_10)
    with open(passed_txt, "w", newline="", encoding="utf-8") as f:
        pass

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
    approx_total = st.session_state.prep_stats.get("valid_rows", 0) or 0

    for chunk in pd.read_csv(cleaned_path, sep="\t", chunksize=chunk_size, dtype=str):
        total_seen += len(chunk)

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

        if not passed.empty:
            sum_tb += int(pd.to_numeric(passed["Terjual Bulanan"], errors="coerce").fillna(0).sum())
            sum_krp += int(pd.to_numeric(passed["Komisi Rp"], errors="coerce").fillna(0).sum())
            total_passed += len(passed)

            if len(passed_preview) < preview_rows:
                need = preview_rows - len(passed_preview)
                passed_preview.extend(passed.head(need).to_dict(orient="records"))

            with open(passed_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for _, r in passed.iterrows():
                    w.writerow([r.get(c, "") for c in COLUMNS_10])

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

        if not rejected.empty:
            total_rejected_filter += len(rejected)

            rej_rows = []
            for _, r in rejected.iterrows():
                # kolom gagal dibuat ringkas (tanpa teks panjang)
                fails = []
                if int(r.get("Stock") or 0) < int(min_stock): fails.append("Stock")
                if int(r.get("Terjual Bulanan") or 0) < int(min_tb): fails.append("Terjual Bulanan")
                if int(r.get("Komisi %") or 0) < int(min_kpct): fails.append("Komisi %")
                if int(r.get("Komisi Rp") or 0) < int(min_krp): fails.append("Komisi Rp")
                if pattern and not re.search(pattern, str(r.get("Nama Produk") or ""), flags=re.IGNORECASE): fails.append("Nama Produk")

                alasan = []
                if "Stock" in fails: alasan.append(f"Stock < {int(min_stock)}")
                if "Terjual Bulanan" in fails: alasan.append(f"Terjual Bulanan < {int(min_tb)}")
                if "Komisi %" in fails: alasan.append(f"Komisi % < {int(min_kpct)}")
                if "Komisi Rp" in fails: alasan.append(f"Komisi Rp < {int(min_krp)}")
                if "Nama Produk" in fails and include_terms:
                    alasan.append(f"Nama tidak mengandung: {', '.join(include_terms)}")

                row_dict = {c: r.get(c, "") for c in COLUMNS_10}
                row_dict.update({
                    "Sumber": "FILTER",
                    "Baris Ke": "",
                    "Kolom Error": ", ".join(fails),
                    "Alasan": " | ".join(alasan) if alasan else "Tidak lolos filter",
                })
                rej_rows.append(row_dict)

            append_reject_csv(reject_csv, rej_rows)
            append_reject_txt(reject_txt, rej_rows)

        if approx_total > 0:
            prog.progress(min(1.0, total_seen / approx_total))
            status.write(f"Processing... {total_seen:,} / {approx_total:,}".replace(",", "."))
        else:
            status.write(f"Processing... {total_seen:,}".replace(",", "."))

    prog.progress(1.0)
    status.empty()

    stats = {
        "seen_valid_rows": total_seen,
        "passed_rows": total_passed,
        "rejected_filter_rows": total_rejected_filter,
        "sum_terjual_bulanan_passed": sum_tb,
        "sum_komisi_rp_passed": sum_krp,
    }
    passed_preview_df = pd.DataFrame(passed_preview, columns=COLUMNS_10) if passed_preview else pd.DataFrame(columns=COLUMNS_10)
    return stats, passed_preview_df


# =========================
# HEADER (PROFESIONAL)
# =========================
st.markdown("## Shopee Product Filter")
st.caption("Input tanpa header • Proses streaming • Export CSV/TXT + Log")

# =========================
# SIDEBAR INPUT
# =========================
with st.sidebar:
    st.header("Input Data")

    source_mode = st.radio(
        "Sumber data",
        ["Paste (TSV/CSV)", "Upload (.txt / .csv)"],
        index=0
    )

    paste_too_big = False
    uploaded = None

    HARD_LIMIT_MB = 5.0  # biar paste gak bikin lemot/timeout

    if source_mode.startswith("Paste"):
        st.session_state.raw_text = st.text_area(
            "Paste data (tanpa header)",
            height=220,
            placeholder="Tempel TSV dari Excel atau CSV tanpa header.",
        )

        raw = st.session_state.raw_text or ""
        size_mb = len(raw.encode("utf-8")) / (1024 * 1024)

        # (hapus teks panjang/ukuran) -> tetap enforce limit
        if size_mb >= HARD_LIMIT_MB:
            paste_too_big = True
            st.error("Paste terlalu besar. Gunakan upload file.")

    else:
        uploaded = st.file_uploader("Upload file (tanpa header)", type=["txt", "csv"])

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶️ MULAI PROSES", use_container_width=True, disabled=paste_too_big)
    with c2:
        reset_btn = st.button("🔄 RESET", use_container_width=True)

    st.divider()
    st.subheader("Filter")
    with st.form("filter_form"):
        min_stock = st.number_input("Stock minimal", min_value=0, value=0, step=1)
        min_tb = st.number_input("Terjual Bulanan minimal", min_value=0, value=0, step=1)
        min_kpct = st.number_input("Komisi % minimal", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        min_krp = st.number_input("Komisi Rp minimal", min_value=0.0, value=0.0, step=100.0, format="%.0f")
        include_words_raw = st.text_input("Nama Produk wajib mengandung", value="", placeholder="contoh: anak, bayi, kids")
        run_filter_btn = st.form_submit_button("🚀 JALANKAN FILTER")

# RESET
if reset_btn:
    try:
        shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    except Exception:
        pass
    for k in ["stage", "raw_text", "paths", "prep_stats", "filter_stats", "preview_df", "passed_preview_df", "workdir", "session_id"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# =========================
# STAGE INPUT
# =========================
if st.session_state.stage == "input":
    if start_btn:
        paths = st.session_state.paths

        if source_mode.startswith("Paste"):
            if not (st.session_state.raw_text or "").strip():
                st.error("Data kosong.")
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

        # bersihkan file output lama
        for p in [paths["cleaned_tsv"], paths["passed_csv"], paths["passed_txt"], paths["reject_csv"], paths["reject_txt"]]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        with st.spinner("Processing..."):
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

# =========================
# READY VIEW
# =========================
paths = st.session_state.paths
prep = st.session_state.prep_stats or {}
preview_df = st.session_state.preview_df

# Ringkasan bersih (tanpa heading panjang)
cA, cB, cC, cD = st.columns(4)
cA.metric("Total baris (raw)", f"{prep.get('total_lines', 0):,}".replace(",", "."))
cB.metric("Baris valid", f"{prep.get('valid_rows', 0):,}".replace(",", "."))
cC.metric("Skip (PARSE)", f"{prep.get('bad_parse', 0):,}".replace(",", "."))
cD.metric("Skip (CLEAN)", f"{prep.get('bad_clean', 0):,}".replace(",", "."))

if preview_df is not None:
    st.dataframe(preview_df, use_container_width=True, height=360)

# =========================
# FILTER TRIGGER
# =========================
if run_filter_btn:
    terms = parse_terms(include_words_raw)

    with st.spinner("Filtering..."):
        stats, passed_preview_df = run_streaming_filter(
            cleaned_path=paths["cleaned_tsv"],
            passed_csv=paths["passed_csv"],
            passed_txt=paths["passed_txt"],
            reject_csv=paths["reject_csv"],
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

# =========================
# FILTERED VIEW
# =========================
if st.session_state.stage == "filtered":
    fs = st.session_state.filter_stats or {}

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid diproses", f"{fs.get('seen_valid_rows', 0):,}".replace(",", "."))
    c2.metric("Lolos", f"{fs.get('passed_rows', 0):,}".replace(",", "."))
    c3.metric("Total Terjual Bulanan", fmt_id(fs.get("sum_terjual_bulanan_passed", 0)))
    c4.metric("Total Komisi Rp", fmt_id(fs.get("sum_komisi_rp_passed", 0)))

    st.subheader("Preview Hasil (sample)")
    if st.session_state.passed_preview_df is not None:
        st.dataframe(st.session_state.passed_preview_df, use_container_width=True, height=360)

    with st.expander("Log Tidak Lolos (sample)"):
        if os.path.exists(paths["reject_csv"]) and os.path.getsize(paths["reject_csv"]) > 0:
            try:
                sample_log = pd.read_csv(paths["reject_csv"], nrows=200)
                st.dataframe(sample_log, use_container_width=True, height=320)
            except Exception:
                st.info("Log terlalu besar untuk preview. Silakan download log.")
        else:
            st.info("Tidak ada log.")

    st.divider()
    st.subheader("Export")

    base_name = sanitize_basename(st.text_input("Nama file hasil", value="hasil_filter_produk"), "hasil_filter_produk")
    log_name = sanitize_basename(st.text_input("Nama file log", value=f"{base_name}_log_tidak_lolos"),
                                 f"{base_name}_log_tidak_lolos")

    col1, col2 = st.columns(2)

    with col1:
        fmt = st.radio("Format hasil", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_pass")
        out_path = paths["passed_csv"] if fmt == "CSV" else paths["passed_txt"]
        out_file = f"{base_name}.csv" if fmt == "CSV" else f"{base_name}.txt"
        mime = "text/csv" if fmt == "CSV" else "text/plain"

        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button("Download hasil", data=f.read(), file_name=out_file, mime=mime)
        else:
            st.info("Belum ada hasil. Jalankan filter dulu.")

    with col2:
        fmt2 = st.radio("Format log", ["CSV", "TXT"], horizontal=True, index=0, key="fmt_log")
        log_path = paths["reject_csv"] if fmt2 == "CSV" else paths["reject_txt"]
        log_file = f"{log_name}.csv" if fmt2 == "CSV" else f"{log_name}.txt"
        log_mime = "text/csv" if fmt2 == "CSV" else "text/plain"

        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                st.download_button("Download log", data=f.read(), file_name=log_file, mime=log_mime)
        else:
            st.info("Belum ada log.")
