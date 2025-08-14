import re, joblib, numpy as np, pandas as pd, streamlit as st

# ====== kompat untuk artefak (unpickle) ======
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

class LexiconFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, pos=None, neg=None, neu=None):
        self.pos = set(pos or []); self.neg = set(neg or []); self.neu = set(neu or [])
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows = []
        for s in X:
            toks = set(str(s).split())
            rows.append([
                len(toks & self.pos),
                len(toks & self.neg),
                len(toks & self.neu),
                1 if "!" in str(s) else 0
            ])
        return sparse.csr_matrix(rows, dtype=float)

# =========================== KONFIG APP ===========================
st.set_page_config(page_title="Analisis Sentimen Ulasan", layout="wide")

# ---------- STOPWORDS / SLANG / PHRASE MAP ----------
STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the
their there they this to was will with
aku saya kamu kalian kita mereka dia ia ku mu nya
yang dan di ke dari untuk pada adalah itu ini
ada para saja atau serta sehingga karena namun tapi jadi maka
dengan tentang sebagai juga sudah belum masih hanya lebih kurang suatu tiap setiap
""".split())
NEGATORS = {"tidak","bukan","tak","no","not","n't","ga","gak","nggak","enggak","jangan"}
STOPWORDS = STOPWORDS.difference(NEGATORS)

SLANG = {
    "bgt":"banget","bngt":"banget","gk":"gak","ga":"gak","nggak":"gak","ngga":"gak",
    "rekomen":"rekomendasi","mantul":"mantap","anjir":"anjir","anjay":"anjay",
    "mager":"malas","ok":"oke","cs":"customer_service"
}
EMOJI_POS = {"🙂","😊","😁","😍","👍","🎉","🔥","😘","🤩","👌"}
EMOJI_NEG = {"😡","🤬","😞","😠","👎","😢","😭","💀","🤢","😫"}

PHRASE_MAP = {
    r"\bred flag\b": " red_flag ",
    r"\btidak (?:rekomendasi|direkomendasikan)\b": " tidak_rekomendasi ",
    r"\b(?:gak|ga|gk|enggak|nggak)\s+s(?:es)?uai\b": " gak_sesuai ",
    r"\bskip dulu\b": " skip_dulu ",
    r"\b(?:respon|response)\s+(?:buruk|lama|lambat)\b": " respon_buruk ",
    r"\bso[-\s]?so\b": " so_so ",
    r"\bb\s*aja\.?\b": " biasa ",
    r"\bnyesel\b": " menyesal ",
    r"\bzonk\b": " zonk ",
    r"\bampas\b": " ampas ",
    r"\b(?:mahal|kemahalan)\b": " mahal ",
    r"\blag(?:ging)?\b": " lag ",
    r"\b(gak|ga|gk|enggak|nggak)\s+sopan\b": " ga_sopan ",
    r"\btidak sopan\b": " tidak_sopan ",
    r"\bbau mulut\b": " bau_mulut ",
    r"\bbau badan\b": " bau_badan ",
}

PHRASE_MAP.update({
    r"\bworth it\b":                   " worth_it ",
    r"\buser friendly\b":              " user_friendly ",
    r"\b(keren)\s+(abis|parah)\b":     " keren_parah ",
    r"\bbagus banget\b":               " bagus_banget ",

    # respon cepat / lambat
    r"\bfast respon(se)?\b":           " respon_cepat ",
    r"\b(respon|response)\s+cepat\b":  " respon_cepat ",
    r"\bcepat tanggap\b":              " respon_cepat ",
    r"\bslow respon(se)?\b":           " slow_respon ",
    r"\b(respon|response)\s+(lama|lambat)\b": " respon_buruk ",
    r"\b(tidak|ga|gak|enggak|nggak)\s+responsif\b": " respon_buruk ",

    # netral / “meh”
    r"\bso[-\s]?so\b":                 " so_so ",
    r"\bnot bad\b":                    " not_bad ",
    r"\bok(e|elah)\s*saja\b|\bokelah\b": " oke_saja ",
    r"\bplus minus\b":                 " plus_minus ",
})

def clean_text(s: str) -> str:
    s = str(s).lower()
    for pat, repl in PHRASE_MAP.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    for e in EMOJI_POS: s = s.replace(e, " EMO_POS ")
    for e in EMOJI_NEG: s = s.replace(e, " EMO_NEG ")
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)
    toks = re.findall(r"[a-zA-Zá-ž_]+", s)
    toks = [SLANG.get(t, t) for t in toks]
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return " ".join(toks)

# ======================= LOAD MODEL/ARTEFAK =======================
@st.cache_resource(show_spinner=False)
def load_artifact(path="sentiment_pipeline_group.joblib"):
    art = joblib.load(path)
    pipe = art["pipeline"]
    meta = art.get("meta", {})
    classes = meta.get("class_order", list(pipe.classes_))
    thr_hr = float(meta.get("neg_threshold_high_recall", meta.get("neg_threshold", 0.40)))
    # Balanced: pakai dari meta jika ada; kalau tidak, buat konservatif: hr +0.08, minimal 0.35
    thr_bal = float(meta.get("neg_threshold_balanced", max(thr_hr + 0.08, 0.35)))
    return pipe, classes, meta, thr_hr, thr_bal

pipe, CLASSES, META, THR_HR, THR_BAL = load_artifact()
NEG_IDX = CLASSES.index("negative")

# ====================== LEXICON (untuk info & override) ======================
POS_LEXICON = {
    "bagus","keren","mantap","top","oke","wow","worth","recommended","rekomendasi",
    "puas","memuaskan","terbaik","cepat","ramah","bantu","solutif","jelas","mudah","stabil","rapi","aman"
}

POS_LEXICON.update({
    "mantapjiwa","mantul","gokil","keren_parah","bagus_banget",
    "top","topmarkotop","worth","worthit","worth_it","value",
    "recommended","rekomendasi","puas","memuaskan","terbaik",
    "cepat","respon_cepat","responsif","cepat_tanggap",
    "ramah","helpful","bantu","solutif","jelas","mudah",
    "stabil","rapi","aman","user_friendly","bermanfaat","murah",
})


NEG_LEXICON = {
    "jelek","buruk","parah","zonk","ampas","mahal","lambat","lemot","lelet","telat",
    "bohong","tipu","menyesal","kecewa","ga_sopan","tidak_sopan","respon_buruk",
    "error","bug","crash","ribet","susah","bawel","cerewet","nyebelin",
    "busuk","amis","bau","bau_mulut","bau_badan","jorok","jijik","menjijikkan","lag", 
}
NEU_LEXICON = {"biasa","so_so","lumayan","standar","cukup","seadanya","netral","overall"}
NEU_LEXICON.update({
    "biasa","so_so","lumayan","standar","cukup","seadanya","netral","overall",
    "okelah","oke_saja","plus_minus","not_bad","ya_gitu","gitu_aja","average","fair",
})

NEG_LEXICON.update({
    "jelek","buruk","payah","parah","kacau","hancur","sampah",
    "zonk","ampas","mahal","overprice","overpriced","kemahalan",
    "lambat","lemot","lelet","delay","telat","lag","ngelag","slow_respon",
    "error","bug","crash","hang","freeze","down",
    "ribet","susah","tidak_jelas","gak_jelas","gak_sesuai","tidak_sesuai",
    "tidak_rekomendasi","respon_buruk","tidak_responsif","slow",
    "bohong","tipu","penipuan","scam","palsu","fake",
    "kasar","jutek","ga_sopan","tidak_sopan","bawel","cerewet","nyebelin","rese",
    "norak","lebay","mengecewakan","kecewa","menyesal","nyesel",
    "bau","bau_mulut","bau_badan","busuk","jorok","kotor","cacat","rusak","retak",
})

def lexicon_hits(cleaned_text: str):
    toks = set(cleaned_text.split())
    return sorted(toks & POS_LEXICON), sorted(toks & NEG_LEXICON), sorted(toks & NEU_LEXICON)

NEG_OVERRIDE = NEG_LEXICON | {"zonk","ampas"}

NEG_OVERRIDE.update({"scam","penipuan","sampah","overpriced",
                     "tidak_rekomendasi","gak_sesuai","respon_buruk","tidak_responsif"})

# ====================== INFERENCE + PENJELASAN ======================
def predict_texts(texts, neg_threshold, use_override=True):
    """
    Return:
      preds, proba, infos
      infos[i] = {
        clean, conf, pos_hits, neg_hits, neu_hits,
        p_neg, thr_used, reason ('threshold' | 'override' | 'argmax')
      }
    """
    if isinstance(texts, str): texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    proba   = pipe.predict_proba(cleaned)

    preds, infos = [], []
    for raw, s, p in zip(texts, cleaned, proba):
        p_neg = float(p[NEG_IDX])
        cls_idx = int(np.argmax(p))
        label = CLASSES[cls_idx]
        reason = "argmax"  # default: pilih kelas proba maksimum

        # aturan threshold NEGATIVE
        if p_neg >= neg_threshold:
            label = "negative"; reason = "threshold"

        # override kuat untuk teks pendek / tidak yakin
        toks = s.split(); tokset = set(toks)
        if use_override and (tokset & NEG_OVERRIDE):
            if len(toks) <= 3 or (p_neg >= 0.15 and float(p.max()) < 0.95):
                label = "negative"; reason = "override"

        pos_h, neg_h, neu_h = lexicon_hits(s)
        preds.append(label)
        infos.append({
            "clean": s, "conf": float(p.max()),
            "pos_hits": pos_h, "neg_hits": neg_h, "neu_hits": neu_h,
            "p_neg": p_neg, "thr_used": float(neg_threshold), "reason": reason
        })
    return preds, proba, infos

# ============================ UI ============================
st.title("🧠 Analisis Sentimen Ulasan")
st.caption("Aplikasi ini membantu mengelompokkan ulasan menjadi positif, netral, atau negatif. Mode Seimbang menjaga presisi, sedangkan Tanggap Keluhan lebih sensitif menangkap komplain.")

with st.sidebar:
    st.subheader("⚙️ Pengaturan (sederhana)")

    # ---- default awal: Kustom + 0.50 ----
    if "profile" not in st.session_state:
        st.session_state.profile = "Kustom"
    if "thr_custom" not in st.session_state:
        st.session_state.thr_custom = 0.50

    profile = st.radio(
        "Profil Prediksi",
        ["Seimbang (presisi ↑)", "Tanggap Keluhan (recall ↑)", "Kustom"],
        index=["Seimbang (presisi ↑)", "Tanggap Keluhan (recall ↑)", "Kustom"].index(st.session_state.profile),
        help=(
            "**Seimbang**: ambang lebih tinggi → lebih jarang menandai negatif (lebih presisi).\n\n"
            "**Tanggap Keluhan**: ambang lebih rendah → lebih sensitif ke negatif (recall lebih tinggi)."
        )
    )
    st.session_state.profile = profile  # simpan pilihan terbaru

    if profile == "Kustom":
        thr = st.slider(
            "Ambang 'negative' (kustom)",
            0.0, 1.0, float(st.session_state.thr_custom), 0.01, key="thr_custom"
        )
    else:
        thr = THR_BAL if profile.startswith("Seimbang") else THR_HR

    st.caption(
        f"Ambang efektif: **{thr:.2f}**  •  Urutan kelas: {', '.join(CLASSES)}  "
        f"•  Seimbang={THR_BAL:.2f}  |  Tanggap={THR_HR:.2f}"
    )

    with st.expander("Pengaturan lanjutan (opsional)"):
        low_conf = st.slider("Batas 'low confidence'", 0.0, 1.0, 0.45, 0.01, key="low_conf")
        use_override = st.checkbox("Gunakan override kata negatif", value=True, key="use_override")
# default jika expander tak dibuka
if "low_conf" not in st.session_state: st.session_state.low_conf = 0.45
if "use_override" not in st.session_state: st.session_state.use_override = True
low_conf = st.session_state.low_conf
use_override = st.session_state.use_override


with st.expander("📚 Kosakata percakapan (referensi)"):
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Positif**"); c1.write(", ".join(sorted(POS_LEXICON)))
    c2.markdown("**Negatif**"); c2.write(", ".join(sorted(NEG_LEXICON)))
    c3.markdown("**Netral**");  c3.write(", ".join(sorted(NEU_LEXICON)))

tab1, tab2 = st.tabs(["🔎 Teks Tunggal", "📄 Batch CSV"])

with tab1:
    txt = st.text_area("Masukkan kalimat/ulasan", height=140,
                       placeholder="Contoh: Instrukturnya telat dan responnya lambat banget.")
    if st.button("Prediksi", type="primary") and txt.strip():
        labels, probas, infos = predict_texts(txt, neg_threshold=thr, use_override=use_override)
        label = labels[0]; p = probas[0]; info = infos[0]
        conf = info["conf"]
        flag = "⚠️ kepercayaan rendah" if conf < low_conf else "✅"
        why = f"(alasan: {info['reason']}; p_neg={info['p_neg']:.3f} ≥ ambang {info['thr_used']:.2f})" if info["reason"]!="argmax" else ""
        st.markdown(f"**Prediksi:** `{label}` | **Kepercayaan:** `{conf:.3f}` {flag}  {why}")
        st.caption(f"Tokens (clean): {info['clean']}")

        dfp = pd.DataFrame([p], columns=CLASSES).T.rename(columns={0:"probabilitas"}).sort_values("probabilitas", ascending=False)
        st.bar_chart(dfp)
        st.dataframe(dfp.style.format({"probabilitas": "{:.3f}"}))

        with st.expander("Detail kecocokan kosakata"):
            st.write(
                f"**Positif**: {', '.join(info['pos_hits']) or '-'}  \n"
                f"**Negatif**: {', '.join(info['neg_hits']) or '-'}  \n"
                f"**Netral**: {', '.join(info['neu_hits']) or '-'}"
            )

with tab2:
    st.markdown(
        "🔗 **Contoh data uji**: "
        "[Google Drive folder](https://drive.google.com/drive/folders/1vwuRYTuwpS5YhXCr4nwftWPuxePZPjeo?hl=ID)"
    )
    up = st.file_uploader("Unggah file (CSV / Excel)", type=["csv", "xlsx", "xls"])
    st.caption("Catatan: pastikan file memiliki **satu kolom teks** berisi ulasan/kalimat.")

    if up:
        # --- Baca file sesuai ekstensi ---
        try:
            name = up.name.lower()
            if name.endswith((".xlsx", ".xls")):
                # Jika error, install: pip install openpyxl
                df = pd.read_excel(up)
            else:
                df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        # --- Deteksi kandidat kolom teks ---
        preferred = ["review_text","text","review","comment","content","message"]
        cand = [c for c in preferred if c in df.columns]
        text_like = [c for c in df.columns if df[c].dtype == "object"]

        options = cand or text_like or list(df.columns)
        col = st.selectbox("Pilih kolom teks", options)
        if df[col].dtype != "object":
            st.warning("Kolom terpilih bukan tipe teks. Sistem akan mengubahnya menjadi string.")
        st.caption(
            "Kolom standar terdeteksi: "
            + (", ".join(cand) if cand else "— (tak ada nama standar, pilih manual)")
        )

        # --- Prediksi batch ---
        batch = df[col].astype(str).tolist()
        preds, probas, infos = predict_texts(batch, neg_threshold=thr, use_override=use_override)

        out = df.copy()
        out["prediksi"] = preds
        out["max_proba"] = np.max(probas, axis=1)
        out["flag_low_conf"] = (out["max_proba"] < low_conf)
        out["p_negative"] = [i["p_neg"] for i in infos]
        out["neg_threshold_used"] = [i["thr_used"] for i in infos]
        out["reason"] = [i["reason"] for i in infos]
        out["lex_pos_hits"] = [", ".join(i["pos_hits"]) for i in infos]
        out["lex_neg_hits"] = [", ".join(i["neg_hits"]) for i in infos]
        out["lex_neu_hits"] = [", ".join(i["neu_hits"]) for i in infos]

        st.success(f"Berhasil memprediksi {len(out)} baris.")
        st.dataframe(out.head(20))
        st.download_button(
            "⬇️ Unduh predictions.csv",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

