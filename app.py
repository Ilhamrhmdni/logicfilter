# app.py
# Streamlit lengkap: filter numeric + filter sembako (rule) + labeling (memory) + ML + auto-suggest keyword CORE_FOOD

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
LABELS_PATH = os.path.join(APP_DIR, "labels.csv")                 # memory labels
MODEL_PATH = os.path.join(APP_DIR, "sembako_model.pkl")           # trained ML model
CUSTOM_KW_PATH = os.path.join(APP_DIR, "custom_keywords.csv")     # learned keywords (rule enrichment)

st.set_page_config(page_title="Filter Produk Sembako (Belajar)", layout="wide")


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
    # bahan pokok
    "beras","gula","garam","tepung","minyak","telur","mie","mi","bihun","soun",
    "oat","oatmeal","sereal",
    # bumbu
    "bawang","cabai","cabe","ketumbar","merica","lada","kunyit","jahe","lengkuas","kemiri",
    "bumbu","penyedap","kaldu","kecap","saus","sambal","santan","cuka",
    # protein & olahan
    "sarden","kornet","abon","nugget","sosis","bakso","ikan","ayam","daging",
    # minuman
    "teh","kopi","coklat","sirup","susu","air mineral",
    # snack / pendamping
    "biskuit","snack","keripik","wafer","permen","cokelat","selai","madu",
    # dairy/lemak dapur
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

# non-sembako keras
BLACKLIST_HARD = [
    # furnitur/toko
    "rak","etalase","lemari","meja","kursi","gondola","display","hanger","manekin",
    # fashion
    "baju","celana","jaket","hijab","kaos","sepatu","sandal","topi",
    # otomotif
    "sparepart","suku cadang","motor","mobil","oli","ban","aki","busi","rantai","kampas",
    # elektronik/perkakas
    "charger","kabel","headset","speaker","lampu","setrika","rice cooker","blender",
    "bor","obeng","tang","perkakas",
    # bangunan
    "cat","semen","paku","baut","keramik","pasir","baja"
]

# exception contoh (kalau kamu mau REVIEW bukan drop)
BLACKLIST_EXCEPTIONS = ["rak telur", "rak piring"]

PACK_UNITS_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?)\s*(kg|g|gr|gram|ml|l|liter|pcs|pc|sachet|sct|pack|pak|dus|box|botol|btl|refill)\b",
    re.IGNORECASE
)


# =========================
# CUSTOM KEYWORDS (learned rule enrichment)
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
    merged = sorted(set(base + extra))
    return merged


@dataclass
class RuleResult:
    label: str          # SEMBAKO / NON / REVIEW
    score: int
    hits: Dict[str, List[str]]
    hard_black: bool

def rule_classify(name: str, mode: str = "ketat") -> RuleResult:
    """
    mode:
      - ketat: CORE_FOOD + FOOD_SIGNAL + packaging
      - kelontong: + HOUSEHOLD
    """
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

    # core food: +3 each
    for kw in get_core_food_list():
        if contains_phrase(text, kw):
            score += 3
            hits["core"].append(kw)

    # food signals: +2 each (cap)
    signal_count = 0
    for kw in FOOD_SIGNAL:
        if contains_phrase(text, kw):
            signal_count += 1
            hits["signal"].append(kw)
    score += min(signal_count, 2) * 2

    # household: +1 each (cap, only in kelontong)
    if mode == "kelontong":
        house_count = 0
        for kw in HOUSEHOLD:
            if contains_phrase(text, kw):
                house_count += 1
                hits["house"].append(kw)
        score += min(house_count, 2) * 1

    # packaging booster
    if PACK_UNITS_PATTERN.search(text):
        score += 1
        hits["pack"].append("pack_unit")

    if any(contains_phrase(text, x) for x in ["isi", "refill", "kemasan", "pouch"]):
        score += 1
        hits["pack"].append("pack_word")

    # decision
    if score >= 5:
        return RuleResult(label="SEMBAKO", score=score, hits=hits, hard_black=False)
    if score >= 2:
        return RuleResult(label="REVIEW", score=score, hits=hits, hard_black=False)
    return RuleResult(label="NON", score=score, hits=hits, hard_black=False)


# =========================
# LABEL MEMORY (labels.csv)
# =========================
def load_labels_df() -> pd.DataFrame:
    if os.path.exists(LABELS_PATH):
        df = pd.read_csv(LABELS_PATH)
        for col in ["key", "text", "label", "updated_at"]:
            if col not in df.columns:
                df[col] = ""
        df["key"] = df["key"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].fillna("").astype(str)
        df["updated_at"] = df["updated_at"].fillna("").astype(str)
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
# ML MODEL (TFIDF + Logistic Regression)
# =========================
def train_model_from_labels(min_samples: int = 80) -> Tuple[Optional[Pipeline], str]:
    df = load_labels_df()
    df = df[df["label"].isin(["SEMBAKO", "NON"])]
    if len(df) < min_samples:
        return None, f"Label masih {len(df)} baris. Minimal {min_samples} biar training enak."
    if df["label"].nunique() < 2:
        return None, "Butuh label SEMBAKO dan NON (dua kelas) biar bisa training."

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize,
            ngram_range=(1, 2),
            min_df=1
        )),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    pipe.fit(df["text"].astype(str), df["label"].astype(str))
    joblib.dump(pipe, MODEL_PATH)
    return pipe, f"Model berhasil ditrain dari {len(df)} label dan disimpan ke {os.path.basename(MODEL_PATH)}."

def load_model() -> Optional[Pipeline]:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def predict_with_model(model: Pipeline, text: str) -> Tuple[str, float]:
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = int(proba.argmax())
    return classes[idx], float(proba[idx])


# =========================
# HYBRID CLASSIFIER
# =========================
@dataclass
class FinalResult:
    label: str              # SEMBAKO / NON / REVIEW
    score_rule: int
    source: str             # memory / rule / ml / rule_blacklist / rule_review
    confidence: float       # 0..1 for ML
    detail: str

def classify_hybrid(name: str, mode: str, ml_conf_threshold: float) -> FinalResult:
    key = normalize(name)

    # 1) Memory label override
    mem = labels_map()
    if key in mem:
        return FinalResult(label=mem[key], score_rule=999, source="memory", confidence=1.0, detail="override label")

    # 2) Rule
    rr = rule_classify(name, mode=mode)
    if rr.hard_black:
        return FinalResult(label="NON", score_rule=rr.score, source="rule_blacklist", confidence=1.0, detail="hard blacklist")

    # If rule very confident SEMBAKO
    if rr.label == "SEMBAKO" and rr.score >= 6:
        return FinalResult(label="SEMBAKO", score_rule=rr.score, source="rule", confidence=1.0, detail="rule confident")

    # 3) ML (if available)
    model = load_model()
    if model is not None:
        pred, conf = predict_with_model(model, name)
        if conf >= ml_conf_threshold:
            return FinalResult(label=pred, score_rule=rr.score, source="ml", confidence=conf, detail="ml confident")

    # 4) fallback
    if rr.label == "REVIEW":
        return FinalResult(label="REVIEW", score_rule=rr.score, source="rule_review", confidence=0.0, detail="needs review")
    return FinalResult(label=rr.label, score_rule=rr.score, source="rule", confidence=0.0, detail="rule fallback")


# =========================
# AUTO-SUGGEST KEYWORDS (learn from labels)
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
        grams += [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    return grams

def auto_suggest_keywords(top_n: int = 40, min_yes: int = 3, min_ratio: float = 2.0) -> pd.DataFrame:
    """
    Kandidat keyword yang jauh lebih sering muncul di SEMBAKO dibanding NON.
    Sumber: labels.csv (hasil klik kamu).
    """
    df = load_labels_df()
    df = df[df["label"].isin(["SEMBAKO", "NON"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["keyword", "yes_cnt", "no_cnt", "score", "ratio"])

    yes_texts = df[df["label"] == "SEMBAKO"]["text"].astype(str).tolist()
    no_texts  = df[df["label"] == "NON"]["text"].astype(str).tolist()

    from collections import Counter
    yes_counter = Counter()
    no_counter = Counter()

    # pakai set() per item biar 1 produk tidak “nge-spam” kata yang sama berkali-kali
    for s in yes_texts:
        yes_counter.update(set(extract_ngrams(s, n=2)))
    for s in no_texts:
        no_counter.update(set(extract_ngrams(s, n=2)))

    existing = set(get_core_food_list())  # base + custom

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
# PARSING TXT TAB-SEPARATED
# =========================
NUMERIC_DOT_FIELDS = {"harga", "stok", "terjual bln", "terjual semua", "komisi rp", "rating"}

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

def parse_txt_to_df(text: str) -> pd.DataFrame:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    headers = [h.strip().lower() for h in lines[0].split("\t")]
    rows = []
    errors = 0

    for i, line in enumerate(lines[1:], start=1):
        try:
            vals = [v.strip() for v in line.split("\t")]
            if len(vals) < len(headers):
                continue

            row = {}
            for idx, header in enumerate(headers):
                if idx >= len(vals):
                    continue
                v = vals[idx]
                if header in NUMERIC_DOT_FIELDS:
                    row[header] = parse_id_number(v)
                elif header == "komisi":
                    row[header] = parse_percent(v)
                else:
                    row[header] = v

            row["shopid"] = extract_shopid(row.get("link", ""))
            row["no"] = i
            rows.append(row)
        except:
            errors += 1

    df = pd.DataFrame(rows)
    df.attrs["line_errors"] = errors
    df.attrs["headers"] = headers
    return df


# =========================
# UI
# =========================
st.title("🧺 Filter Produk Sembako — Auto + Belajar + Auto-Suggest Keyword")

with st.sidebar:
    st.header("🎛️ Kontrol")
    mode = st.selectbox(
        "Mode Sembako",
        ["ketat", "kelontong"],
        index=0,
        help="ketat: fokus pangan/dapur. kelontong: termasuk sabun/deterjen/tisu."
    )
    ml_conf = st.slider(
        "Confidence ML minimal",
        0.50, 0.95, 0.75, 0.01,
        help="Kalau model ML yakin ≥ ini, prediksi dipakai."
    )
    include_review = st.checkbox("Izinkan REVIEW masuk hasil", value=True)

    st.markdown("---")
    st.subheader("📌 Filter Angka")
    terjual_min = st.number_input("Min Terjual/Bulan", min_value=0.0, value=0.0, step=1.0)
    komisi_min = st.number_input("Min Komisi (%)", min_value=0.0, value=0.0, step=0.5)
    komisi_rp_min = st.number_input("Min Komisi (Rp)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    st.subheader("🧠 Learning")
    if st.button("🧠 Train / Update Model dari labels.csv"):
        model, msg = train_model_from_labels(min_samples=80)
        st.success(msg if model is not None else msg)

    colm1, colm2 = st.columns(2)
    with colm1:
        if st.button("🧾 Preview labels.csv"):
            st.session_state["show_labels"] = True
    with colm2:
        if st.button("🧾 Preview custom keywords"):
            st.session_state["show_custom"] = True

uploaded = st.file_uploader("Upload file .txt (TAB-separated)", type=["txt"])
if not uploaded:
    st.info("Upload dulu file .txt kamu.")
    st.stop()

raw_text = uploaded.read().decode("utf-8", errors="replace")
df = parse_txt_to_df(raw_text)
if df.empty:
    st.error("Format file tidak sesuai / kosong / hanya header.")
    st.stop()

if df.attrs.get("line_errors", 0):
    st.warning(f"Ada {df.attrs['line_errors']} baris yang error dan di-skip.")

# Ensure columns exist
for col in ["nama", "link", "terjual bln", "komisi", "komisi rp", "rating"]:
    if col not in df.columns:
        df[col] = "" if col in ["nama", "link"] else 0.0

# Hybrid classification columns
results = df["nama"].fillna("").astype(str).apply(lambda x: classify_hybrid(x, mode, ml_conf))
df["kategori"] = results.apply(lambda r: r.label)
df["source"] = results.apply(lambda r: r.source)
df["score_rule"] = results.apply(lambda r: r.score_rule)
df["conf_ml"] = results.apply(lambda r: r.confidence)

# Numeric filter
mask_numeric = (
    (df["terjual bln"].fillna(0) >= terjual_min) &
    (df["komisi"].fillna(0) >= komisi_min) &
    (df["komisi rp"].fillna(0) >= komisi_rp_min)
)

# Category filter
allowed = ["SEMBAKO"] + (["REVIEW"] if include_review else [])
mask_cat = df["kategori"].isin(allowed)

filtered = df[mask_numeric & mask_cat].copy()
filtered["shopid"] = filtered["shopid"].fillna(0).astype(int)
filtered = filtered.sort_values(by=["shopid"])

# Header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Produk", f"{len(df):,}".replace(",", "."))
c2.metric("Lolos Filter", f"{len(filtered):,}".replace(",", "."))
c3.metric("ShopID Unik (lolos)", str(filtered["shopid"].nunique()))
pct = (len(filtered)/len(df)*100) if len(df) else 0
c4.metric("% Lolos", f"{pct:.1f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["✅ Hasil Lolos", "🟨 REVIEW + Auto-Suggest", "⚙️ Diagnostics"])

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
        lines.append(f"- Total data awal   : {len(df_all)}")
        lines.append(f"- Lolos filter      : {len(df_f)} ({(len(df_f)/len(df_all)*100):.1f}%)")
        lines.append(f"- ShopID unik lolos : {df_f['shopid'].nunique()}")
        lines.append(f"- Total terjual/bln : {int(df_f['terjual bln'].fillna(0).sum())}")
        lines.append(f"- Total komisi (Rp) : {int(df_f['komisi rp'].fillna(0).sum())}")
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
    d1.download_button("⬇️ Download CSV (semicolon ;)", data=csv_bytes, file_name=f"hasil_filter_{timestamp}.csv", mime="text/csv")
    d2.download_button("⬇️ Download Summary TXT", data=summary_bytes, file_name=f"summary_{timestamp}.txt", mime="text/plain")

with tab2:
    st.subheader("🟨 REVIEW — Labelin biar sistem belajar")
    st.caption("Klik ✅ SEMBAKO / ❌ NON. Keputusan disimpan ke labels.csv. Setelah cukup banyak, Train Model di sidebar.")

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
                st.write(f"ShopID: {int(row.get('shopid', 0))} | Terjual/bln: {row.get('terjual bln', 0)} | Komisi%: {row.get('komisi', 0)} | Komisi Rp: {row.get('komisi rp', 0)}")
                st.caption(f"rule_score={row.get('score_rule')} | source={row.get('source')} | ml_conf={row.get('conf_ml'):.2f}")

            with mid:
                if st.button("✅ SEMBAKO", key=f"btn_yes_{idx}"):
                    upsert_label(nama, "SEMBAKO")
                    st.success("Tersimpan: SEMBAKO")
                    time.sleep(0.15)
                    st.rerun()

            with right:
                if st.button("❌ NON", key=f"btn_no_{idx}"):
                    upsert_label(nama, "NON")
                    st.success("Tersimpan: NON")
                    time.sleep(0.15)
                    st.rerun()

        st.info("Tips: minimal 80 label (SEMBAKO/NON) biar ML training stabil.")

    st.markdown("---")
    st.subheader("🤖 Auto-suggest Keyword CORE_FOOD (Belajar dari label kamu)")
    st.caption("Saran diambil dari kata/frasa yang jauh lebih sering muncul di label SEMBAKO dibanding NON.")

    colx1, colx2, colx3 = st.columns([2, 2, 2])
    top_n = colx1.number_input("Top N saran", min_value=10, max_value=200, value=40, step=10)
    min_yes = colx2.number_input("Minimal muncul di SEMBAKO", min_value=1, max_value=20, value=3, step=1)
    min_ratio = colx3.number_input("Minimal rasio SEMBAKO vs NON", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    if st.button("🔍 Generate Suggestions"):
        sug = auto_suggest_keywords(top_n=int(top_n), min_yes=int(min_yes), min_ratio=float(min_ratio))
        st.session_state["suggestions_df"] = sug

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
                    time.sleep(0.1)
                    st.rerun()
    else:
        st.info("Belum ada saran. Tambah label SEMBAKO/NON dulu, atau turunkan parameter.")

    st.markdown("---")
    st.subheader("📌 Custom CORE_FOOD Keywords (hasil belajar)")
    ck = load_custom_keywords()
    if ck.empty:
        st.write("Belum ada keyword custom.")
    else:
        st.dataframe(ck.sort_values("added_at", ascending=False), use_container_width=True, height=260)
        buf = io.StringIO()
        ck.to_csv(buf, index=False)
        st.download_button("⬇️ Download custom_keywords.csv", data=buf.getvalue().encode("utf-8"),
                           file_name="custom_keywords.csv", mime="text/csv")

    if st.session_state.get("show_labels"):
        st.markdown("---")
        st.subheader("📄 labels.csv (preview)")
        ldf = load_labels_df().sort_values("updated_at", ascending=False)
        st.dataframe(ldf, use_container_width=True, height=280)
        buf = io.StringIO()
        ldf.to_csv(buf, index=False)
        st.download_button("⬇️ Download labels.csv", data=buf.getvalue().encode("utf-8"),
                           file_name="labels.csv", mime="text/csv")

with tab3:
    st.subheader("⚙️ Diagnostics")
    st.write("Distribusi kategori:")
    st.dataframe(df["kategori"].value_counts().rename_axis("kategori").reset_index(name="count"))

    st.write("Distribusi source keputusan:")
    st.dataframe(df["source"].value_counts().rename_axis("source").reset_index(name="count"))

    st.write("CORE_FOOD total (base + custom):", len(get_core_food_list()))
    st.caption("Contoh keyword custom terbaru:")
    ck = load_custom_keywords()
    if not ck.empty:
        st.dataframe(ck.sort_values("added_at", ascending=False).head(20), use_container_width=True)

    st.write("Contoh NON teratas (cek false-negative):")
    non_sample = df[df["kategori"] == "NON"].head(20)
    st.dataframe(non_sample[[c for c in ["nama","shopid","source","score_rule","conf_ml"] if c in non_sample.columns]],
                 use_container_width=True)

    st.write("Contoh SEMBAKO teratas (cek false-positive):")
    yes_sample = df[df["kategori"] == "SEMBAKO"].head(20)
    st.dataframe(yes_sample[[c for c in ["nama","shopid","source","score_rule","conf_ml"] if c in yes_sample.columns]],
                 use_container_width=True)

st.caption("Hybrid: memory label > hard blacklist > rule confident > ML confident > REVIEW. Auto-suggest belajar dari labels.csv.")
