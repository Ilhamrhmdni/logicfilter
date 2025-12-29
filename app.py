import re
import io
from datetime import datetime
from typing import Tuple

import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Data Filter Bot - Streamlit", layout="wide")

NUM_COLS_PARSE_DOT = {"harga", "stok", "terjual bln", "terjual semua", "komisi rp", "rating"}
NUM_COLS_PARSE_PERCENT = {"komisi"}

SHOPID_REGEX = re.compile(r"shopee\.co\.id\/product\/(\d+)\/\d+", re.IGNORECASE)

SAMPLE_TEXT = """no\tlink\tnama\tharga\tstok\tterjual bln\tterjual semua\tkomisi\tkomisi rp\trating
1\thttps://shopee.co.id/product/12345678/99887766\tBeras 5kg Premium\t75.000\t50\t120\t890\t6%\t4.500\t4,9
2\thttps://shopee.co.id/product/12345678/88776655\tMinyak Goreng 2L\t38.500\t80\t210\t1320\t5%\t2.200\t4,8
3\thttps://shopee.co.id/product/77777777/11112222\tRak Piring Stainless\t250.000\t10\t2\t35\t2%\t1.500\t4,6
4\thttps://shopee.co.id/product/99999999/44445555\tGula Pasir 1kg\t16.500\t200\t320\t2100\t7%\t3.000\t4,9
"""

# =========================
# UTIL
# =========================
def normalize_header(h: str) -> str:
    return (h or "").strip().lower()

def parse_number_dot_thousand(val: str) -> float:
    # replace '.' thousand sep -> ''
    # replace ',' decimal -> '.'
    if val is None:
        return 0.0
    s = str(val).strip()
    if s == "":
        return 0.0
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def parse_percent(val: str) -> float:
    if val is None:
        return 0.0
    s = str(val).strip().replace("%", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def extract_shopid(link: str) -> int:
    if not link:
        return 0
    m = SHOPID_REGEX.search(str(link))
    return int(m.group(1)) if m else 0

def to_id_int(x) -> int:
    try:
        return int(x)
    except:
        return 0

def format_id(n) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except:
        return str(n)

def parse_tsv_text(raw_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return: (df_ok, df_err)
    df_err contains: line_no, reason, raw
    """
    if not raw_text:
        return pd.DataFrame(), pd.DataFrame()

    lines = [ln for ln in raw_text.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        return pd.DataFrame(), pd.DataFrame()

    headers = [normalize_header(h) for h in lines[0].split("\t")]
    header_len = len(headers)

    ok_rows = []
    err_rows = []

    for i, ln in enumerate(lines[1:], start=2):  # line number in file (1-based)
        values = [v.strip() for v in ln.split("\t")]

        if len(values) < header_len:
            err_rows.append({
                "line_no": i,
                "reason": f"kolom kurang ({len(values)}/{header_len})",
                "raw": ln
            })
            continue

        row = {}
        for idx, h in enumerate(headers):
            v = values[idx] if idx < len(values) else ""
            if h in NUM_COLS_PARSE_DOT:
                row[h] = parse_number_dot_thousand(v)
            elif h in NUM_COLS_PARSE_PERCENT:
                row[h] = parse_percent(v)
            else:
                row[h] = v

        row["shopid"] = extract_shopid(row.get("link", ""))
        row["no_src"] = i - 1  # order from file (data row index)
        ok_rows.append(row)

    df_ok = pd.DataFrame(ok_rows)
    df_err = pd.DataFrame(err_rows)
    return df_ok, df_err

def make_csv_bytes(df: pd.DataFrame) -> bytes:
    # Output columns like JS
    cols = [
        ("ShopID", "shopid"),
        ("Link", "link"),
        ("Nama", "nama"),
        ("Harga", "harga"),
        ("Stok", "stok"),
        ("Terjual_Bln", "terjual bln"),
        ("Terjual_Semua", "terjual semua"),
        ("Komisi%", "komisi"),
        ("Komisi_Rp", "komisi rp"),
        ("Rating", "rating"),
    ]

    out = pd.DataFrame()
    out["No"] = range(1, len(df) + 1)
    for out_col, src_col in cols:
        out[out_col] = df[src_col] if src_col in df.columns else ""

    bio = io.StringIO()
    out.to_csv(bio, index=False, sep=";")  # semicolon ; like JS
    return bio.getvalue().encode("utf-8")

def build_summary_text(
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame,
    terjual_min: float,
    komisi_min: float,
    komisirp_min: float,
    out_csv_name: str
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_all = len(df_all)
    total_filtered = len(df_filtered)
    pct = (total_filtered / total_all * 100) if total_all else 0.0

    total_sold = float(df_filtered.get("terjual bln", pd.Series(dtype=float)).sum()) if total_filtered else 0.0
    total_comm = float(df_filtered.get("komisi rp", pd.Series(dtype=float)).sum()) if total_filtered else 0.0

    lines = []
    lines.append("=" * 70)
    lines.append("SUMMARY HASIL FILTER DATA SHOPEE (STREAMLIT)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Tanggal     : {now}")
    lines.append(f"File Output : {out_csv_name}")
    lines.append("")
    lines.append("STATISTIK:")
    lines.append("-" * 40)
    lines.append(f"Total Data Awal     : {total_all}")
    lines.append(f"Data Lolos Filter   : {total_filtered}")
    lines.append(f"Persentase Lolos    : {pct:.1f}%")
    lines.append(f"Jumlah Toko (ShopID): {df_filtered['shopid'].nunique() if total_filtered else 0}")
    lines.append("")
    lines.append(f"Total Terjual per Bulan : {total_sold:,.0f}".replace(",", "."))
    lines.append(f"Total Komisi (Rp)       : Rp {total_comm:,.0f}".replace(",", "."))
    lines.append("")
    lines.append("FILTER YANG DITERAPKAN:")
    lines.append("-" * 40)
    lines.append(f"Minimum Terjual per Bulan : {terjual_min}")
    lines.append(f"Minimum Komisi            : {komisi_min}%")
    lines.append(f"Minimum Komisi (Rp)       : Rp {komisirp_min:,.0f}".replace(",", "."))
    lines.append("")
    lines.append("DETAIL PER SHOPID:")
    lines.append("-" * 40)

    if total_filtered:
        for shopid, g in df_filtered.sort_values("shopid").groupby("shopid"):
            shop_sold = float(g.get("terjual bln", pd.Series(dtype=float)).sum())
            shop_comm = float(g.get("komisi rp", pd.Series(dtype=float)).sum())
            lines.append("")
            lines.append(f"ShopID: {shopid}")
            lines.append(f"  Jumlah Produk : {len(g)}")
            lines.append(f"  Total Terjual : {shop_sold:,.0f}".replace(",", "."))
            lines.append(f"  Total Komisi  : Rp {shop_comm:,.0f}".replace(",", "."))
    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by Data Filter Bot - Streamlit")
    return "\n".join(lines)

def ensure_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = ["terjual bln", "komisi", "komisi rp", "rating", "harga", "stok", "terjual semua"]
    for c in need:
        if c not in df.columns:
            df[c] = 0.0
    return df


# =========================
# SESSION DEFAULTS
# =========================
st.session_state.setdefault("raw_text", "")
st.session_state.setdefault("terjual_min", 0.0)
st.session_state.setdefault("komisi_min", 0.0)
st.session_state.setdefault("komisirp_min", 0.0)

st.session_state.setdefault("last_run_ok", False)
st.session_state.setdefault("df_all", None)
st.session_state.setdefault("df_err", None)
st.session_state.setdefault("df_filtered", None)
st.session_state.setdefault("csv_bytes", None)
st.session_state.setdefault("summary_text", None)
st.session_state.setdefault("csv_name", None)
st.session_state.setdefault("summary_name", None)


# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("📊 Data Filter Bot")
menu = st.sidebar.radio(
    "Menu",
    ["📝 Input Data", "⚙️ Setting Filter", "🚀 Jalankan & Export"]
)

st.title("📌 Data Filter Bot — Streamlit")

with st.expander("ℹ️ Format input (TSV/tab) yang didukung", expanded=False):
    st.write(
        "Header minimal:\n\n"
        "`no[tab]link[tab]nama[tab]harga[tab]stok[tab]terjual bln[tab]terjual semua[tab]komisi[tab]komisi rp[tab]rating`\n\n"
        "Angka diparse: titik ribuan dihapus, koma jadi desimal."
    )

# =========================
# MENU: INPUT DATA
# =========================
if menu == "📝 Input Data":
    st.subheader("1) Input Data")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Load sample data"):
            st.session_state["raw_text"] = SAMPLE_TEXT
            st.success("Sample data loaded.")

    with c2:
        uploaded = st.file_uploader("Upload file .txt / .tsv", type=["txt", "tsv"])
        if uploaded is not None:
            st.session_state["raw_text"] = uploaded.read().decode("utf-8", errors="replace")
            st.success("File uploaded & loaded.")

    st.session_state["raw_text"] = st.text_area(
        "Paste data TSV di sini:",
        value=st.session_state["raw_text"],
        height=280,
        placeholder="Paste data TSV / tab-separated di sini..."
    )

    if st.session_state["raw_text"].strip():
        df_preview, df_err_preview = parse_tsv_text(st.session_state["raw_text"])
        st.caption(f"Preview parse: {len(df_preview)} baris OK, {len(df_err_preview)} baris error")
        if not df_err_preview.empty:
            st.warning("Ada baris error (kolom kurang). Lihat tabel error:")
            st.dataframe(df_err_preview, use_container_width=True)
        if not df_preview.empty:
            st.dataframe(df_preview.head(15), use_container_width=True)

# =========================
# MENU: SETTING FILTER
# =========================
elif menu == "⚙️ Setting Filter":
    st.subheader("2) Setting Filter (Minimum)")

    st.session_state["terjual_min"] = st.number_input(
        "Minimum Terjual per Bulan",
        min_value=0.0,
        value=float(st.session_state["terjual_min"]),
        step=1.0
    )
    st.session_state["komisi_min"] = st.number_input(
        "Minimum Komisi (%)",
        min_value=0.0,
        value=float(st.session_state["komisi_min"]),
        step=0.5
    )
    st.session_state["komisirp_min"] = st.number_input(
        "Minimum Komisi (Rp)",
        min_value=0.0,
        value=float(st.session_state["komisirp_min"]),
        step=100.0
    )

    st.info("Setting tersimpan otomatis (session). Pindah menu tidak hilang.")

# =========================
# MENU: RUN & EXPORT
# =========================
else:
    st.subheader("3) Jalankan Filter & Export")

    run_btn = st.button("🚀 PROSES DATA", type="primary")

    if run_btn:
        raw_text = st.session_state["raw_text"]
        if not raw_text.strip():
            st.error("❌ Data kosong. Masuk menu **Input Data** dulu.")
        else:
            df_all, df_err = parse_tsv_text(raw_text)
            if df_all.empty:
                st.error("❌ Data gagal diparse. Pastikan ada header + minimal 1 baris data.")
            else:
                df_all = ensure_numeric_cols(df_all)

                # apply filter
                terjual_min = float(st.session_state["terjual_min"])
                komisi_min = float(st.session_state["komisi_min"])
                komisirp_min = float(st.session_state["komisirp_min"])

                df_filtered = df_all[
                    (df_all["terjual bln"] >= terjual_min) &
                    (df_all["komisi"] >= komisi_min) &
                    (df_all["komisi rp"] >= komisirp_min)
                ].copy()

                df_filtered["shopid"] = df_filtered["shopid"].apply(to_id_int)
                df_filtered = df_filtered.sort_values(["shopid", "no_src"], ascending=[True, True]).reset_index(drop=True)

                # export build
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_name = f"hasil_filter_{ts}.csv"
                summary_name = f"summary_{ts}.txt"
                csv_bytes = make_csv_bytes(df_filtered)
                summary_text = build_summary_text(df_all, df_filtered, terjual_min, komisi_min, komisirp_min, csv_name)

                # save to session
                st.session_state["last_run_ok"] = True
                st.session_state["df_all"] = df_all
                st.session_state["df_err"] = df_err
                st.session_state["df_filtered"] = df_filtered
                st.session_state["csv_bytes"] = csv_bytes
                st.session_state["summary_text"] = summary_text
                st.session_state["csv_name"] = csv_name
                st.session_state["summary_name"] = summary_name

                st.success("✅ Proses selesai. Scroll ke bawah untuk hasil & export.")

    # =========================
    # SHOW RESULTS (if exist)
    # =========================
    if st.session_state["last_run_ok"] and st.session_state["df_all"] is not None:
        df_all = st.session_state["df_all"]
        df_err = st.session_state["df_err"]
        df_filtered = st.session_state["df_filtered"]

        st.divider()
        st.subheader("📊 Hasil Proses")

        # stats awal
        total_products = len(df_all)
        shop_ids = sorted([sid for sid in df_all["shopid"].dropna().unique().tolist() if to_id_int(sid) > 0])
        avg_sold = float(df_all["terjual bln"].mean()) if total_products else 0.0
        avg_comm = float(df_all["komisi"].mean()) if total_products else 0.0

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("📦 Total Produk", format_id(total_products))
        s2.metric("🏪 Total Toko (ShopID)", format_id(len(shop_ids)))
        s3.metric("📈 Rata-rata Terjual/Bulan", f"{avg_sold:,.0f}".replace(",", "."))
        s4.metric("💰 Rata-rata Komisi", f"{avg_comm:.1f}%")

        if df_err is not None and not df_err.empty:
            st.warning(f"⚠️ Ada {len(df_err)} baris error (di-skip).")
            st.dataframe(df_err, use_container_width=True)

        total_filtered = len(df_filtered)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Data Lolos", format_id(total_filtered))
        c2.metric("❌ Tidak Lolos", format_id(total_products - total_filtered))
        pct = (total_filtered / total_products * 100) if total_products else 0.0
        c3.metric("📊 Persentase Lolos", f"{pct:.1f}%")
        c4.metric("🏪 ShopID Lolos", format_id(df_filtered["shopid"].nunique() if total_filtered else 0))

        if total_filtered == 0:
            st.error("Tidak ada data yang memenuhi filter. Ubah setting filter di menu Setting Filter.")
        else:
            total_sold = float(df_filtered["terjual bln"].sum())
            total_comm = float(df_filtered["komisi rp"].sum())
            avg_rating = float(df_filtered["rating"].mean()) if total_filtered else 0.0

            a1, a2, a3 = st.columns(3)
            a1.metric("📊 Total Terjual/Bulan", f"{total_sold:,.0f}".replace(",", "."))
            a2.metric("💰 Total Komisi (Rp)", f"Rp {total_comm:,.0f}".replace(",", "."))
            a3.metric("⭐ Rata-rata Rating", f"{avg_rating:.2f}")

            st.subheader("👁️ Preview Data (terurut ShopID)")
            preview_cols = ["shopid", "nama", "harga", "stok", "terjual bln", "komisi", "komisi rp", "rating", "link"]
            preview_cols = [c for c in preview_cols if c in df_filtered.columns]
            st.dataframe(df_filtered[preview_cols].head(50), use_container_width=True)

            st.subheader("🏪 Ringkasan per ShopID")
            g = df_filtered.groupby("shopid", as_index=False).agg(
                jumlah_produk=("shopid", "count"),
                total_terjual_bln=("terjual bln", "sum"),
                total_komisi_rp=("komisi rp", "sum")
            ).sort_values("shopid", ascending=True)
            st.dataframe(g, use_container_width=True)

            st.subheader("💾 Export")
            dl1, dl2 = st.columns(2)

            with dl1:
                st.download_button(
                    "⬇️ Download CSV (semicolon ;)",
                    data=st.session_state["csv_bytes"],
                    file_name=st.session_state["csv_name"],
                    mime="text/csv"
                )
            with dl2:
                st.download_button(
                    "⬇️ Download Summary (.txt)",
                    data=st.session_state["summary_text"].encode("utf-8"),
                    file_name=st.session_state["summary_name"],
                    mime="text/plain"
                )

            with st.expander("📋 Lihat isi summary"):
                st.code(st.session_state["summary_text"], language="text")
    else:
        st.info("Belum ada hasil. Klik **PROSES DATA** dulu.")
