# app.py
# Streamlit lengkap:
# - Input: Upload atau Copy–Paste
# - Tombol: Load sample data
# - Validasi format (header TAB, kolom wajib, minimal baris)
# - Highlight baris error saat parsing
# - Filter numeric + filter sembako (rule) + labeling (memory) + ML + auto-suggest keyword CORE_FOOD

import os
import io
import re
import time
import joblib
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================
# SETTINGS / PATHS
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(APP_DIR, "labels.csv")
MODEL_PATH = os.path.join(APP_DIR, "sembako_model.pkl")
CUSTOM_KW_PATH = os.path.join(APP_DIR, "custom_keywords.csv")

st.set_page_config(page_title="Filter Produk Sembako (Belajar)", layout="wide")


# =========================
# SAMPLE DATA
# =========================
SAMPLE_TEXT = (
    "no\tlink\tnama\tharga\tstok\tterjual bln\tterjual semua\tkomisi\tkomisi rp\trating\n"
    "1\thttps://shopee.co.id/product/111/999\tBeras Ramos 5kg\t75.000\t100\t50\t300\t10%\t5.000\t4,8\n"
    "2\thttps://shopee.co.id/product/222/888\tRak Besi 4 Susun\t250.000\t20\t10\t50\t5%\t2.000\t4,2\n"
    "3\thttps://shopee.co.id/product/333/777\tMinyak Goreng Bimoli 2L\t35.000\t200\t120\t900\t12%\t8.000\t4,9\n"
    "4\thttps://shopee.co.id/product/444/666\tSabun Cuci Piring 800ml\t18.000\t150\t80\t600\t8%\t3.000\t4,7\n"
    "5\thttps://shopee.co.id/product/555/555\tOli Motor 1L\t50.000\t90\t60\t400\t7%\t2.500\t4,6\n"
    "6\thttps://shopee.co.id/product/666/444\tTeh Celup 50 sachet\t12.500\t300\t200\t1500\t10%\t4.000\t4,9\n"
    "7\thttps://shopee.co.id/product/777/333\tGula Pasir 1kg\t16.000\t250\t160\t1200\t9%\t3.500\t4,8\n"
    # baris error contoh: kurang kolom / komisi salah
    "8\thttps://shopee.co.id/product/888/222\tBaju Kaos Oblong\t35.000\t50\t40\t300\tabc\t2.000\t4,1\n"
    "9\thttps://shopee.co.id/product/999/111\tSusu UHT 1L\t18.000\t200\t110\t800\t10%\tX\t4,8\n"
)


# =========================
# NORMALIZATION
# =========================
def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_phrase(haystack: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", haystack) is not None


# =========================
# RULE-BASED SEMBAKO (RICH)
# =========================
CORE_FOOD = [
    "beras","gula","garam","tepung","minyak","telur","mie","mi","bihun","soun",
    "oat","oatmeal","sereal",
    "bawang","cabai","cabe","ketumbar","merica","lada","kunyit","jahe","lengkuas","kemiri",
    "bumbu","penyedap","kaldu","kecap","saus","sambal","santan","cuka",
    "sarden","kornet","abon","nugget","sosis","bakso","ikan","ayam","daging",
    "teh","kopi","coklat","sirup","susu","air mineral",
    "biskuit","snack","keripik","wafer","permen","cokelat","selai","madu",
    "mentega","margarin","keju","yogurt"
]

FOOD_SIGNAL = [
    "instan","premium","murni","asli","organik","halal",
    "bubuk","serbuk","butiran","kental","uht",
    "masak","goreng","panggang","resep"
]

HOUSEHOLD = [
    "sabun","sabun mandi","sabun cuci","deterjen","detergen","pelembut","pewangi","pengharum",
    "pembersih","pembersih lantai","karbol","pemutih",
    "tisu","tissue","cuci piring","sampo","shampoo","pasta gigi","sikat gigi"
]

BLACKLIST_HARD = [
    "rak","etalase","lemari","meja","kursi","gondola","display","hanger","manekin",
    "baju","celana","jaket","hijab","kaos","sepatu","sandal","topi",
    "sparepart","suku cadang","motor","mobil","oli","ban","aki","busi","rantai","kampas",
    "charger","kabel","headset","speaker","lampu","setrika","rice cooker","blender",
    "bor","obeng","tang","perkakas",
    "cat","semen","paku","baut","keramik","pasir","baja"
]

BLACKLIST_EXCEPTIONS = ["rak telur", "rak piring"]

PACK_UNITS_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?)\s*(kg|g|gr|gram|ml|l|liter|pcs|pc|sachet|sct|pack|pak|dus|box|botol|btl|refill)\b",
    re.IGNORECASE
)


# =========================
# CUSTOM KEYWORDS (learned)
# =========================
def load_custom_keywords() -> pd.DataFrame:
    if os.path.exists(CUSTOM_KW_PATH):
        df = pd.read_csv(CUSTOM_KW_PATH)
        for col in ["keyword", "type", "added_at"]:
            if col not in df.columns:
                df[col] = ""
        df["keyword"] = df["keyword"].fillna("").astype(str)
        df["type"] = df["type"].fillna("").astype(str)
        df["added_at"] = df["added_at"].fillna("").astype(str)
        return df[["keyword", "type", "added_at"]]
    return pd.DataFrame(columns=["keyword", "type", "added_at"])

def add_custom_keyword(keyword: str, kw_type: str = "CORE_FOOD"):
    keyword = normalize(keyword)
    if not keyword:
        return
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = load_custom_keywords()
    df = df[df["keyword"] != keyword]
    df = pd.concat([df, pd.DataFrame([{"keyword": keyword, "type": kw_type, "added_at": now}])], ignore_index=True)
    df.to_csv(CUSTOM_KW_PATH, index=False)

def get_core_food_list() -> List[str]:
    base = list(CORE_FOOD)
    dfc = load_custom_keywords()
    extra = dfc[dfc["type"] == "CORE_FOOD"]["keyword"].dropna().astype(str).tolist()
    return sorted(set(base + extra))


@dataclass
class RuleResult:
    label: str
    score: int
    hits: Dict[str, List[str]]
    hard_black: bool

def rule_classify(name: str, mode: str = "ketat") -> RuleResult:
    text = normalize(name)
    score = 0
    hits = {"core": [], "signal": [], "house": [], "pack": [], "black": []}

    # hard blacklist
    for bad in BLACKLIST_HARD:
        if contains_phrase(text, bad):
            if any(exc in text for exc in BLACKLIST_EXCEPTIONS):
                hits["black"].append(bad)
                return RuleResult(label="REVIEW", score=1, hits=hits, hard_black=False)
            hits["black"].append(bad)
            return RuleResult(label="NON", score=-10, hits=hits, hard_black=True)

    # core food
    for kw in get_core_food_list():
        if contains_phrase(text, kw):
            score += 3
            hits["core"].append(kw)

    # food signals (cap 2)
    sig = 0
    for kw in FOOD_SIGNAL:
        if contains_phrase(text, kw):
            sig += 1
            hits["signal"].append(kw)
    score += min(sig, 2) * 2

    # household (kelontong only, cap 2)
    if mode == "kelontong":
        house = 0
        for kw in HOUSEHOLD:
            if contains_phrase(text, kw):
                house += 1
                hits["house"].append(kw)
        score += min(house, 2) * 1

    # packaging boosters
    if PACK_UNITS_PATTERN.search(text):
        score += 1
        hits["pack"].append("pack_unit")
    if any(contains_phrase(text, x) for x in ["isi", "refill", "kemasan", "pouch"]):
        score += 1
        hits["pack"].append("pack_word")

    if score >= 5:
        return RuleResult("SEMBAKO", score, hits, False)
    if score >= 2:
        return RuleResult("REVIEW", score, hits, False)
    return RuleResult("NON", score, hits, False)


# =========================
# LABEL MEMORY
# =========================
def load_labels_df() -> pd.DataFrame:
    if os.path.exists(LABELS_PATH):
        df = pd.read_csv(LABELS_PATH)
        for col in ["key", "text", "label", "updated_at"]:
            if col not in df.columns:
                df[col] = ""
        for c in ["key", "text", "label", "updated_at"]:
            df[c] = df[c].fillna("").astype(str)
        return df[["key", "text", "label", "updated_at"]]
    return pd.DataFrame(columns=["key", "text", "label", "updated_at"])

def upsert_label(text: str, label: str):
    key = normalize(text)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = load_labels_df()
    df = df[df["key"] != key]
    df = pd.concat([df, pd.DataFrame([{"key": key, "text": text, "label": label, "updated_at": now}])], ignore_index=True)
    df.to_csv(LABELS_PATH, index=False)

def labels_map() -> Dict[str, str]:
    df = load_labels_df()
    return dict(zip(df["key"], df["label"]))


# =========================
# ML MODEL
# =========================
def train_model_from_labels(min_samples: int = 80) -> Tuple[Optional[Pipeline], str]:
    df = load_labels_df()
    df = df[df["label"].isin(["SEMBAKO", "NON"])]
    if len(df) < min_samples:
        return None, f"Label masih {len(df)} baris. Minimal {min_samples} biar training enak."
    if df["label"].nunique() < 2:
        return None, "Butuh label SEMBAKO dan NON (dua kelas) biar bisa training."

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    pipe.fit(df["text"].astype(str), df["label"].astype(str))
    joblib.dump(pipe, MODEL_PATH)
    return pipe, f"Model berhasil ditrain dari {len(df)} label dan disimpan ke {os.path.basename(MODEL_PATH)}."

def load_model() -> Optional[Pipeline]:
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def predict_with_model(model: Pipeline, text: str) -> Tuple[str, float]:
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    i = int(proba.argmax())
    return classes[i], float(proba[i])


# =========================
# HYBRID CLASSIFIER
# =========================
@dataclass
class FinalResult:
    label: str
    score_rule: int
    source: str
    confidence: float
    detail: str

def classify_hybrid(name: str, mode: str, ml_conf_threshold: float) -> FinalResult:
    key = normalize(name)

    mem = labels_map()
    if key in mem:
        return FinalResult(mem[key], 999, "memory", 1.0, "override label")

    rr = rule_classify(name, mode=mode)
    if rr.hard_black:
        return FinalResult("NON", rr.score, "rule_blacklist", 1.0, "hard blacklist")

    if rr.label == "SEMBAKO" and rr.score >= 6:
        return FinalResult("SEMBAKO", rr.score, "rule", 1.0, "rule confident")

    model = load_model()
    if model is not None:
        pred, conf = predict_with_model(model, name)
        if conf >= ml_conf_threshold:
            return FinalResult(pred, rr.score, "ml", conf, "ml confident")

    if rr.label == "REVIEW":
        return FinalResult("REVIEW", rr.score, "rule_review", 0.0, "needs review")
    return FinalResult(rr.label, rr.score, "rule", 0.0, "rule fallback")


# =========================
# AUTO-SUGGEST KEYWORDS (from labels)
# =========================
ID_STOPWORDS = set("""
yang dan di ke dari untuk dengan atau pada ini itu serta juga agar karena jadi adalah akan
promo murah diskon gratis original ori baru best seller kualitas ukuran varian warna
pack pcs sachet isi refill kemasan botol dus box kg gr gram ml liter l
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")

def extract_ngrams(text: str, n: int = 2) -> List[str]:
    t = normalize(text)
    tokens = [x for x in TOKEN_RE.findall(t) if x and x not in ID_STOPWORDS]
    tokens = [x for x in tokens if not x.isdigit() and len(x) >= 3]
    grams: List[str] = []
    grams += tokens
    if n >= 2:
        grams += [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    return grams

def auto_suggest_keywords(top_n: int = 40, min_yes: int = 3, min_ratio: float = 2.0) -> pd.DataFrame:
    df = load_labels_df()
    df = df[df["label"].isin(["SEMBAKO", "NON"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["keyword", "yes_cnt", "no_cnt", "score", "ratio"])

    yes_texts = df[df["label"] == "SEMBAKO"]["text"].astype(str).tolist()
    no_texts = df[df["label"] == "NON"]["text"].astype(str).tolist()

    from collections import Counter
    yes_counter = Counter()
    no_counter = Counter()

    for s in yes_texts:
        yes_counter.update(set(extract_ngrams(s, n=2)))
    for s in no_texts:
        no_counter.update(set(extract_ngrams(s, n=2)))

    existing = set(get_core_food_list())

    rows = []
    for kw, y in yes_counter.items():
        if kw in existing:
            continue
        n = no_counter.get(kw, 0)
        if y < min_yes:
            continue
        ratio = (y + 1) / (n + 1)
        if ratio < min_ratio:
            continue
        score = (y - n) + (ratio - 1)
        rows.append((kw, y, n, score, ratio))

    out = pd.DataFrame(rows, columns=["keyword", "yes_cnt", "no_cnt", "score", "ratio"])
    out = out.sort_values(["score", "yes_cnt", "ratio"], ascending=False).head(top_n)
    return out


# =========================
# PARSING + VALIDATION + ERROR HIGHLIGHT
# =========================
NUMERIC_DOT_FIELDS = {"harga", "stok", "terjual bln", "terjual semua", "komisi rp", "rating"}

REQUIRED_COLS = ["link", "nama"]  # numeric cols can be missing; will be filled with 0

def parse_id_number(val: str) -> float:
    s = (val or "").strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def parse_percent(val: str) -> float:
    s = (val or "").strip().replace("%", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

SHOPID_RE = re.compile(r"shopee\.co\.id\/product\/(\d+)\/\d+", re.IGNORECASE)

def extract_shopid(link: str) -> int:
    if not link:
        return 0
    m = SHOPID_RE.search(link)
    return int(m.group(1)) if m else 0

def validate_raw_text(raw_text: str) -> Tuple[bool, str]:
    txt = (raw_text or "").strip()
    if not txt:
        return False, "Teks kosong."
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False, "Minimal harus ada 2 baris: header + 1 baris data."
    if "\t" not in lines[0]:
        return False, "Header harus TAB-separated (pakai tombol TAB, bukan spasi / koma)."
    headers = [h.strip().lower() for h in lines[0].split("\t")]
    missing = [c for c in REQUIRED_COLS if c not in headers]
    if missing:
        return False, f"Kolom wajib tidak ada di header: {missing}. Minimal butuh: link, nama."
    return True, "OK"

def parse_txt_to_df_with_errors(raw_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      df_ok: baris berhasil diparse
      df_err: baris error (di-highlight)
    """
    lines = [ln for ln in (raw_text or "").splitlines() if ln.strip()]
    headers = [h.strip().lower() for h in lines[0].split("\t")]

    ok_rows = []
    err_rows = []

    for line_no, line in enumerate(lines[1:], start=2):  # 1-indexed display; header = line 1
        vals = [v.strip() for v in line.split("\t")]

        if len(vals) != len(headers):
            err_rows.append({
                "line_no": line_no,
                "error": f"Jumlah kolom tidak sama dengan header (got {len(vals)}, expected {len(headers)})",
                "raw_line": line[:2000]
            })
            continue

        row = {}
        row_errors = []

        for idx, header in enumerate(headers):
            v = vals[idx]
            try:
                if header in NUMERIC_DOT_FIELDS:
                    row[header] = parse_id_number(v)
                elif header == "komisi":
                    # komisi percent; if not parseable -> mark error
                    pv = parse_percent(v)
                    if v.strip() and pv == 0.0 and not re.search(r"0+(\.0+)?%?$", v.strip().replace(",", ".")):
                        # heuristic: string non-empty, parse=0, and not explicitly "0" -> likely error
                        row_errors.append(f"komisi invalid: '{v}'")
                    row[header] = pv
                else:
                    row[header] = v
            except Exception as e:
                row_errors.append(f"{header} parse error: {e}")

        # derived
        row["shopid"] = extract_shopid(row.get("link", ""))
        row["no"] = line_no - 1  # data row index (mirip JS)

        if not row.get("nama"):
            row_errors.append("nama kosong")
        if not row.get("link"):
            row_errors.append("link kosong")

        # numeric "komisi rp" sanity (optional)
        if "komisi rp" in row and isinstance(row["komisi rp"], (int, float)) and row["komisi rp"] == 0.0:
            # allow 0, but if raw looks like letters, it would've become 0; flag it
            raw_krp = vals[headers.index("komisi rp")] if "komisi rp" in headers else ""
            if raw_krp and re.search(r"[a-zA-Z]", raw_krp):
                row_errors.append(f"komisi rp invalid: '{raw_krp}'")

        if row_errors:
            err_rows.append({
                "line_no": line_no,
                "error": " | ".join(row_errors),
                "raw_line": line[:2000]
            })
            continue

        ok_rows.append(row)

    df_ok = pd.DataFrame(ok_rows)
    df_err = pd.DataFrame(err_rows, columns=["line_no", "error", "raw_line"])
    return df_ok, df_err


# =========================
# UI: INPUT (UPLOAD / PASTE) + LOAD SAMPLE
# =========================
st.title("🧺 Filter Produk Sembako — Copy–Paste + Validasi + Error Highlight + Belajar")

st.subheader("📥 Input Data")
col_in1, col_in2 = st.columns([3, 2])

with col_in1:
    input_mode = st.radio("Metode input:", ["Copy–Paste Text", "Upload File"], horizontal=True)

with col_in2:
    if st.button("🧪 Load sample data"):
        st.session_state["raw_text"] = SAMPLE_TEXT

raw_text = st.session_state.get("raw_text", "")

if input_mode == "Copy–Paste Text":
    raw_text = st.text_area(
        "Paste data di sini (TAB-separated, baris pertama header).",
        height=260,
        value=raw_text,
        placeholder="no\tlink\tnama\tharga\tstok\tterjual bln\tterjual semua\tkomisi\tkomisi rp\trating\n1\thttps://...\tBeras 5kg\t75000\t100\t50\t300\t10%\t5000\t4.8"
    )
    st.session_state["raw_text"] = raw_text
else:
    uploaded = st.file_uploader("Upload file .txt (TAB-separated)", type=["txt"])
    if uploaded:
        raw_text = uploaded.read().decode("utf-8", errors="replace")
        st.session_state["raw_text"] = raw_text

ok, msg = validate_raw_text(raw_text)
if not ok:
    st.error(msg)
    st.stop()

# parse with errors
df, df_err = parse_txt_to_df_with_errors(raw_text)
if df.empty:
    st.error("Semua baris gagal diparse. Cek tabel error di bawah.")
    if not df_err.empty:
        st.subheader("🚨 Baris Error (highlight)")
        def _style_err(s):
            return ["background-color: #ffe5e5"] * len(s)
        st.dataframe(df_err.style.apply(_style_err, axis=1), use_container_width=True, height=350)
    st.stop()

# show errors if exist
if not df_err.empty:
    st.warning(f"Ada {len(df_err)} baris error dan di-skip.")
    st.subheader("🚨 Baris Error (highlight)")
    def _style_err(s):
        return ["background-color: #ffe5e5"] * len(s)
    st.dataframe(df_err.style.apply(_style_err, axis=1), use_container_width=True, height=350)

st.divider()

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("🎛️ Kontrol")
    mode = st.selectbox("Mode Sembako", ["ketat", "kelontong"], index=0)
    ml_conf = st.slider("Confidence ML minimal", 0.50, 0.95, 0.75, 0.01)
    include_review = st.checkbox("Izinkan REVIEW masuk hasil", value=True)

    st.markdown("---")
    st.subheader("📌 Filter Angka")
    terjual_min = st.number_input("Min Terjual/Bulan", min_value=0.0, value=0.0, step=1.0)
    komisi_min = st.number_input("Min Komisi (%)", min_value=0.0, value=0.0, step=0.5)
    komisi_rp_min = st.number_input("Min Komisi (Rp)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    st.subheader("🧠 Learning")
    if st.button("🧠 Train / Update Model dari labels.csv"):
        model, train_msg = train_model_from_labels(min_samples=80)
        st.success(train_msg)

    show_labels = st.checkbox("Tampilkan labels.csv", value=False)
    show_custom = st.checkbox("Tampilkan custom keywords", value=False)


# =========================
# PREPARE COLUMNS
# =========================
for col in ["nama", "link", "terjual bln", "komisi", "komisi rp", "rating", "harga", "stok", "terjual semua"]:
    if col not in df.columns:
        df[col] = "" if col in ["nama", "link"] else 0.0

# classify
results = df["nama"].fillna("").astype(str).apply(lambda x: classify_hybrid(x, mode, ml_conf))
df["kategori"] = results.apply(lambda r: r.label)
df["source"] = results.apply(lambda r: r.source)
df["score_rule"] = results.apply(lambda r: r.score_rule)
df["conf_ml"] = results.apply(lambda r: r.confidence)

# numeric filter
mask_numeric = (
    (df["terjual bln"].fillna(0) >= terjual_min) &
    (df["komisi"].fillna(0) >= komisi_min) &
    (df["komisi rp"].fillna(0) >= komisi_rp_min)
)

allowed = ["SEMBAKO"] + (["REVIEW"] if include_review else [])
mask_cat = df["kategori"].isin(allowed)

filtered = df[mask_numeric & mask_cat].copy()
filtered["shopid"] = filtered["shopid"].fillna(0).astype(int)
filtered = filtered.sort_values(by=["shopid"])


# =========================
# METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Produk (parsed OK)", f"{len(df):,}".replace(",", "."))
c2.metric("Lolos Filter", f"{len(filtered):,}".replace(",", "."))
c3.metric("ShopID Unik (lolos)", str(filtered["shopid"].nunique()))
pct = (len(filtered) / len(df) * 100) if len(df) else 0
c4.metric("% Lolos", f"{pct:.1f}%")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["✅ Hasil Lolos", "🟨 REVIEW + Auto-suggest", "⚙️ Diagnostics / Files"])

with tab1:
    st.subheader("✅ Hasil Lolos (SEMBAKO + optional REVIEW)")
    show_cols = [c for c in [
        "no","shopid","link","nama","harga","stok","terjual bln","terjual semua",
        "komisi","komisi rp","rating","kategori","source","score_rule","conf_ml"
    ] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, height=520)

    # Export CSV + Summary
    now = dt.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    rename_map = {
        "shopid": "ShopID",
        "link": "Link",
        "nama": "Nama",
        "harga": "Harga",
        "stok": "Stok",
        "terjual bln": "Terjual_Bln",
        "terjual semua": "Terjual_Semua",
        "komisi": "Komisi%",
        "komisi rp": "Komisi_Rp",
        "rating": "Rating",
        "kategori": "Kategori",
        "source": "Source",
        "score_rule": "Score_Rule",
        "conf_ml": "Conf_ML",
    }
    out = filtered.rename(columns=rename_map)
    preferred_order = [
        "ShopID","Link","Nama","Harga","Stok","Terjual_Bln","Terjual_Semua",
        "Komisi%","Komisi_Rp","Rating","Kategori","Source","Score_Rule","Conf_ML"
    ]
    out = out[[c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]]

    csv_buf = io.StringIO()
    out.to_csv(csv_buf, sep=";", index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    def make_summary(df_all: pd.DataFrame, df_f: pd.DataFrame) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("SUMMARY HASIL FILTER PRODUK (SEMBAKO LEARNING)")
        lines.append("=" * 70)
        lines.append(f"Tanggal: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Mode Sembako: {mode}")
        lines.append(f"Include REVIEW: {include_review}")
        lines.append(f"ML Confidence Min: {ml_conf}")
        lines.append("")
        lines.append("FILTER ANGKA:")
        lines.append(f"- Min terjual/bulan : {terjual_min}")
        lines.append(f"- Min komisi (%)    : {komisi_min}")
        lines.append(f"- Min komisi (Rp)   : {komisi_rp_min}")
        lines.append("")
        lines.append("RINGKASAN:")
        lines.append(f"- Total data parsed OK : {len(df_all)}")
        lines.append(f"- Lolos filter         : {len(df_f)} ({(len(df_f)/len(df_all)*100):.1f}%)")
        lines.append(f"- ShopID unik lolos     : {df_f['shopid'].nunique()}")
        lines.append(f"- Total terjual/bln     : {int(df_f['terjual bln'].fillna(0).sum())}")
        lines.append(f"- Total komisi (Rp)     : {int(df_f['komisi rp'].fillna(0).sum())}")
        lines.append("")
        lines.append("DETAIL PER SHOPID:")
        for sid, g in df_f.groupby("shopid"):
            lines.append(f"\nShopID {int(sid)}")
            lines.append(f"  Jumlah produk : {len(g)}")
            lines.append(f"  Total terjual : {int(g['terjual bln'].fillna(0).sum())}")
            lines.append(f"  Total komisi  : Rp {int(g['komisi rp'].fillna(0).sum()):,}".replace(",", "."))
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    summary_bytes = make_summary(df, filtered).encode("utf-8")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV (;)", data=csv_bytes, file_name=f"hasil_filter_{timestamp}.csv", mime="text/csv")
    d2.download_button("⬇️ Download Summary TXT", data=summary_bytes, file_name=f"summary_{timestamp}.txt", mime="text/plain")


with tab2:
    st.subheader("🟨 REVIEW — Labelin biar sistem belajar")
    st.caption("Klik ✅ SEMBAKO / ❌ NON. Disimpan ke labels.csv. Setelah cukup banyak, Train Model di sidebar.")

    review = df[df["kategori"] == "REVIEW"].copy()
    if review.empty:
        st.success("Tidak ada REVIEW saat ini.")
    else:
        max_show = 30
        review = review.head(max_show)

        for idx, row in review.iterrows():
            nama = str(row.get("nama", ""))
            left, mid, right = st.columns([7, 2, 2])

            with left:
                st.markdown(f"**{nama}**")
                st.write(
                    f"ShopID: {int(row.get('shopid', 0))} | "
                    f"Terjual/bln: {row.get('terjual bln', 0)} | "
                    f"Komisi%: {row.get('komisi', 0)} | "
                    f"Komisi Rp: {row.get('komisi rp', 0)}"
                )
                st.caption(f"rule_score={row.get('score_rule')} | source={row.get('source')} | ml_conf={row.get('conf_ml'):.2f}")

            with mid:
                if st.button("✅ SEMBAKO", key=f"btn_yes_{idx}"):
                    upsert_label(nama, "SEMBAKO")
                    st.success("Tersimpan: SEMBAKO")
                    time.sleep(0.12)
                    st.rerun()

            with right:
                if st.button("❌ NON", key=f"btn_no_{idx}"):
                    upsert_label(nama, "NON")
                    st.success("Tersimpan: NON")
                    time.sleep(0.12)
                    st.rerun()

    st.markdown("---")
    st.subheader("🤖 Auto-suggest Keyword CORE_FOOD (Belajar dari label kamu)")
    st.caption("Saran diambil dari kata/frasa yang jauh lebih sering muncul pada label SEMBAKO dibanding NON.")

    colx1, colx2, colx3 = st.columns([2, 2, 2])
    top_n = colx1.number_input("Top N saran", min_value=10, max_value=200, value=40, step=10)
    min_yes = colx2.number_input("Minimal muncul di SEMBAKO", min_value=1, max_value=20, value=3, step=1)
    min_ratio = colx3.number_input("Minimal rasio SEMBAKO vs NON", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    if st.button("🔍 Generate Suggestions"):
        st.session_state["suggestions_df"] = auto_suggest_keywords(
            top_n=int(top_n), min_yes=int(min_yes), min_ratio=float(min_ratio)
        )

    sug = st.session_state.get("suggestions_df")
    if isinstance(sug, pd.DataFrame) and not sug.empty:
        for i, r in sug.iterrows():
            k = r["keyword"]
            left, right = st.columns([8, 2])
            with left:
                st.write(f"**{k}** — yes:{int(r['yes_cnt'])} | no:{int(r['no_cnt'])} | ratio:{r['ratio']:.2f} | score:{r['score']:.2f}")
            with right:
                if st.button("➕ Add", key=f"add_kw_{k}_{i}"):
                    add_custom_keyword(k, "CORE_FOOD")
                    st.success(f"Added: {k}")
                    st.session_state["suggestions_df"] = auto_suggest_keywords(
                        top_n=int(top_n), min_yes=int(min_yes), min_ratio=float(min_ratio)
                    )
                    time.sleep(0.08)
                    st.rerun()
    else:
        st.info("Belum ada saran. Tambah label SEMBAKO/NON dulu, atau turunkan parameter.")

    st.markdown("---")
    st.subheader("📌 Custom CORE_FOOD Keywords (hasil belajar)")
    ck = load_custom_keywords()
    if ck.empty:
        st.write("Belum ada keyword custom.")
    else:
        st.dataframe(ck.sort_values("added_at", ascending=False), use_container_width=True, height=250)
        buf = io.StringIO()
        ck.to_csv(buf, index=False)
        st.download_button("⬇️ Download custom_keywords.csv", data=buf.getvalue().encode("utf-8"),
                           file_name="custom_keywords.csv", mime="text/csv")


with tab3:
    st.subheader("⚙️ Diagnostics / Files")

    st.write("Distribusi kategori:")
    st.dataframe(df["kategori"].value_counts().rename_axis("kategori").reset_index(name="count"))

    st.write("Distribusi source keputusan:")
    st.dataframe(df["source"].value_counts().rename_axis("source").reset_index(name="count"))

    st.write(f"CORE_FOOD total (base + custom): **{len(get_core_food_list())}**")

    if show_labels:
        st.markdown("---")
        st.subheader("📄 labels.csv (preview)")
        ldf = load_labels_df().sort_values("updated_at", ascending=False)
        st.dataframe(ldf, use_container_width=True, height=320)
        buf = io.StringIO()
        ldf.to_csv(buf, index=False)
        st.download_button("⬇️ Download labels.csv", data=buf.getvalue().encode("utf-8"),
                           file_name="labels.csv", mime="text/csv")

    if show_custom:
        st.markdown("---")
        st.subheader("📄 custom_keywords.csv (preview)")
        ck = load_custom_keywords().sort_values("added_at", ascending=False)
        st.dataframe(ck, use_container_width=True, height=320)
        buf = io.StringIO()
        ck.to_csv(buf, index=False)
        st.download_button("⬇️ Download custom_keywords.csv", data=buf.getvalue().encode("utf-8"),
                           file_name="custom_keywords.csv", mime="text/csv")

st.caption("Hybrid: memory label > hard blacklist > rule confident > ML confident > REVIEW. Input: paste/upload. Validasi & highlight error tersedia.")
