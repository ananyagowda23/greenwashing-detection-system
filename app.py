"""
app.py  —  Greenwashing Detection System
Run with:  streamlit run app.py
Make sure train_model.py has been run first!
"""

import streamlit as st
import pickle
import pandas as pd
import re
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

# ══════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════
st.set_page_config(
    page_title="Greenwashing Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════
# LOAD MODEL & DATA
# ══════════════════════════════════════
@st.cache_resource
def load_model():
    model      = pickle.load(open("model.pkl",      "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

try:
    model, vectorizer = load_model()
    data = load_data()

    _STOP = {'is','are','we','our','the','a','an','to','and','of','in','for',
             'on','with','by','this','that','all','from','be','been','was',
             'were','has','have','had','it','its','at','as','or','but','not','so'}
    def _preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s%]', ' ', text)
        return ' '.join([w for w in text.split() if w not in _STOP and len(w)>1])

    data['clean'] = data['Claim'].apply(_preprocess)

    from sklearn.model_selection import train_test_split as _split
    _, X_test_df, _, y_all = _split(
        data['clean'], data['Label'],
        test_size=0.25, random_state=18, stratify=data['Label']
    )
    X_all_vec  = vectorizer.transform(X_test_df)
    y_all_pred = model.predict(X_all_vec)
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ══════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════
# INGREDIENT KNOWLEDGE BASE
# ══════════════════════════════════════
HARMFUL_INGREDIENTS = {
    "methylparaben":           ("Paraben",             "endocrine disruptor, not biodegradable"),
    "ethylparaben":            ("Paraben",             "endocrine disruptor"),
    "propylparaben":           ("Paraben",             "endocrine disruptor, aquatic toxicity"),
    "butylparaben":            ("Paraben",             "strong endocrine disruptor"),
    "isobutylparaben":         ("Paraben",             "endocrine disruptor"),
    "isopropylparaben":        ("Paraben",             "endocrine disruptor"),
    "parabens":                ("Paraben family",      "endocrine disruptors, not biodegradable"),
    "sodium lauryl sulfate":   ("Sulfate surfactant",  "aquatic toxicity, skin irritant"),
    "sls":                     ("Sulfate surfactant",  "aquatic toxicity, skin irritant"),
    "sodium laureth sulfate":  ("Sulfate surfactant",  "may contain 1,4-dioxane carcinogen"),
    "sles":                    ("Sulfate surfactant",  "may contain 1,4-dioxane"),
    "ammonium lauryl sulfate": ("Sulfate surfactant",  "aquatic toxicity"),
    "fragrance":               ("Synthetic fragrance", "contains hidden phthalates, not biodegradable"),
    "parfum":                  ("Synthetic fragrance", "contains hidden phthalates"),
    "phthalates":              ("Phthalate",           "endocrine disruptor, persistent pollutant"),
    "diethyl phthalate":       ("Phthalate",           "endocrine disruptor"),
    "dibutyl phthalate":       ("Phthalate",           "reproductive toxin"),
    "dmdm hydantoin":          ("Formaldehyde releaser","releases carcinogen formaldehyde"),
    "imidazolidinyl urea":     ("Formaldehyde releaser","releases formaldehyde"),
    "diazolidinyl urea":       ("Formaldehyde releaser","releases formaldehyde"),
    "quaternium-15":           ("Formaldehyde releaser","releases formaldehyde, carcinogen risk"),
    "formaldehyde":            ("Carcinogen",          "banned in many countries"),
    "mineral oil":             ("Petroleum derivative","non-renewable, non-biodegradable"),
    "petroleum jelly":         ("Petroleum derivative","non-renewable, non-biodegradable"),
    "petrolatum":              ("Petroleum derivative","non-renewable, contaminant risk"),
    "paraffin":                ("Petroleum wax",       "non-renewable petroleum by-product"),
    "paraffinum liquidum":     ("Liquid paraffin",     "petroleum-derived, non-biodegradable"),
    "dimethicone":             ("Silicone polymer",    "not biodegradable, persistent pollutant"),
    "cyclomethicone":          ("Cyclic silicone",     "bioaccumulates, aquatic toxin"),
    "cyclopentasiloxane":      ("Cyclic silicone D5",  "EU restricted, bioaccumulative"),
    "cyclotetrasiloxane":      ("Cyclic silicone D4",  "endocrine disruptor, EU banned in wash-off"),
    "silicone":                ("Silicone polymer",    "persistent, not biodegradable"),
    "polyethylene":            ("Microplastic",        "persistent marine pollutant"),
    "polypropylene":           ("Microplastic",        "persistent marine pollutant"),
    "nylon":                   ("Synthetic polymer",   "microplastic shedder"),
    "microbeads":              ("Microplastic",        "banned in many countries, marine toxin"),
    "triclosan":               ("Antimicrobial",       "endocrine disruptor, aquatic toxin"),
    "triclocarban":            ("Antimicrobial",       "endocrine disruptor, persistent pollutant"),
    "oxybenzone":              ("Chemical UV filter",  "reef toxic, endocrine disruptor"),
    "octinoxate":              ("Chemical UV filter",  "reef toxic, hormone disruptor"),
    "homosalate":              ("UV filter",           "potential endocrine disruptor"),
    "diethanolamine":          ("Ethanolamine",        "may form carcinogenic nitrosamines"),
    "triethanolamine":         ("Ethanolamine",        "may form carcinogenic nitrosamines"),
    "cocamide dea":            ("DEA compound",        "potential carcinogen nitrosamine former"),
    "aluminum chlorohydrate":  ("Aluminum salt",       "potential neurotoxin, not biodegradable"),
    "aluminium chlorohydrate": ("Aluminum salt",       "potential neurotoxin"),
    "artificial dyes":         ("Synthetic colorant",  "petroleum-derived, aquatic toxin"),
    "synthetic dyes":          ("Synthetic colorant",  "petroleum-derived, aquatic toxin"),
    "coal tar":                ("Coal tar dye",        "carcinogen, persistent pollutant"),
    "polyethylene glycol":     ("PEG compound",        "petroleum-derived, contamination risk"),
    "sodium hypochlorite":     ("Chlorine bleach",     "aquatic toxin, toxic fumes"),
    "phosphates":              ("Phosphate",           "causes algal blooms, aquatic dead zones"),
    "bisphenol":               ("BPA/BPS",             "endocrine disruptor in plastics"),
    "bpa":                     ("Bisphenol A",         "endocrine disruptor, banned in some countries"),
    "talc":                    ("Talcum powder",       "potential asbestos contamination risk"),
}

SAFE_INGREDIENTS = {
    "aloe vera":               ("Plant extract",       "soothing, biodegradable"),
    "aloe barbadensis":        ("Aloe vera",           "plant-derived, biodegradable"),
    "coconut oil":             ("Plant oil",           "natural, biodegradable"),
    "argan oil":               ("Plant oil",           "sustainably sourced, biodegradable"),
    "jojoba oil":              ("Plant wax",           "sustainable desert crop, biodegradable"),
    "shea butter":             ("Plant butter",        "fair trade available, biodegradable"),
    "cocoa butter":            ("Plant butter",        "plant-derived, biodegradable"),
    "mango butter":            ("Plant butter",        "plant-derived, biodegradable"),
    "rosehip oil":             ("Plant oil",           "biodegradable, sustainably sourced"),
    "marula oil":              ("Plant oil",           "sustainably sourced"),
    "sunflower oil":           ("Plant oil",           "biodegradable, renewable"),
    "olive oil":               ("Plant oil",           "renewable, biodegradable"),
    "hemp seed oil":           ("Plant oil",           "sustainable low-water crop"),
    "baobab oil":              ("Plant oil",           "sustainably sourced, biodegradable"),
    "chamomile":               ("Plant extract",       "biodegradable, gentle"),
    "lavender":                ("Essential oil",       "natural, biodegradable"),
    "green tea extract":       ("Plant extract",       "antioxidant, biodegradable"),
    "turmeric":                ("Plant extract",       "natural colorant, biodegradable"),
    "calendula":               ("Plant extract",       "biodegradable, renewable"),
    "rose water":              ("Floral water",        "biodegradable, natural"),
    "rosemary extract":        ("Natural preservative","plant-derived, biodegradable"),
    "coco glucoside":          ("Plant surfactant",    "sugar + coconut, fully biodegradable"),
    "decyl glucoside":         ("Plant surfactant",    "sugar-derived, biodegradable"),
    "lauryl glucoside":        ("Plant surfactant",    "plant-derived, biodegradable"),
    "saponified coconut oil":  ("Natural soap",        "biodegradable, plant-derived"),
    "saponified olive oil":    ("Natural soap",        "biodegradable, plant-derived"),
    "vitamin e":               ("Natural preservative","plant-derived tocopherol"),
    "tocopherol":              ("Vitamin E",           "natural antioxidant preservative"),
    "vitamin c":               ("Ascorbic acid",       "natural antioxidant"),
    "benzyl alcohol":          ("Natural preservative","plant-origin, biodegradable"),
    "zinc oxide":              ("Mineral UV filter",   "reef-safe when non-nano"),
    "titanium dioxide":        ("Mineral UV filter",   "reef-safer than chemical filters"),
    "beeswax":                 ("Natural wax",         "renewable, biodegradable"),
    "carnauba wax":            ("Plant wax",           "sustainably sourced"),
    "candelilla wax":          ("Plant wax",           "vegan, biodegradable"),
    "oatmeal":                 ("Plant ingredient",    "biodegradable, gentle"),
    "colloidal oatmeal":       ("Plant ingredient",    "biodegradable, skin safe"),
    "hyaluronic acid":         ("Humectant",           "can be plant-fermented, biodegradable"),
    "bakuchiol":               ("Plant extract",       "natural retinol alternative, biodegradable"),
    "squalane":                ("Plant lipid",         "olive/sugarcane derived, biodegradable"),
    "niacinamide":             ("Vitamin B3",          "can be plant-derived, biodegradable"),
    "usda organic":            ("Certification",       "USDA certified organic"),
    "cosmos certified":        ("Certification",       "COSMOS organic/natural standard"),
    "ecocert":                 ("Certification",       "recognised eco-certification body"),
    "certified organic":       ("Certification",       "third-party organic verification"),
    "fair trade":              ("Certification",       "ethical and sustainable sourcing"),
    "rainforest alliance":     ("Certification",       "sustainable sourcing verified"),
}

# ══════════════════════════════════════
# CLAIMS KNOWLEDGE BASE
# ══════════════════════════════════════
VAGUE_WORDS = [
    "eco-friendly","eco friendly","green","sustainable","natural",
    "committed","responsible","environmentally","planet","future",
    "caring","cleaner","greener","conscious","dedicated","friendly",
    "kind","safe","proud","believe","support","gentle","pure","clean",
    "earth","organic","wholesome","love","care","protect","save","help",
    "awareness","mission","pledge","passionate","strive","aspire"
]

GENUINE_WORDS = [
    "verified","certified","audit","iso","percent","%","reduced",
    "achieved","measured","third-party","annually","report","baseline",
    "published","standard","tonnes","million","protocol","epa","ghg",
    "scope","recycled","renewable","biodegradable","fsc","usda","rspo",
    "leed","gots","sgs","bureau veritas","deloitte","intertek",
    "carbon trust","b corp","re100","sbti","gold standard",
    "rainforest alliance","fair trade","oeko-tex","independently"
]

VALID_CERTS = [
    "iso 14001","iso 14064","iso 50001","fsc","usda organic",
    "rainforest alliance","energy star","leed","gots","b corp",
    "rspo","epa","bureau veritas","sgs","carbon trust",
    "oeko-tex","fair trade","re100","sbti","science based targets",
    "gold standard","true certification","recyclass","deloitte",
    "1% for the planet","cosmos","intertek","ghg protocol"
]

CATEGORY_TIPS = {
    "🧴 Personal Care" : "Look for COSMOS, USDA Organic, or ECOCERT certifications.",
    "🧹 Household"     : "Genuine cleaners carry EPA Safer Choice or EU Ecolabel.",
    "🍵 Food & Bev"    : "Look for USDA Organic, Rainforest Alliance, or Fair Trade.",
    "👕 Clothing"      : "Genuine sustainable clothing uses GOTS, OEKO-TEX, or B Corp.",
    "⚡ Energy"        : "Look for RE100, Science Based Targets, or Gold Standard.",
    "📦 Packaging"     : "Genuine packaging uses FSC, RecyClass, or Cradle to Cradle.",
    "🏢 CSR / Other"   : "Always look for third-party verified claims with specific numbers."
}

# ══════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════
def has_numbers(text):
    return bool(re.search(r'\d+', text))

def check_certs(text):
    t = text.lower()
    return [c for c in VALID_CERTS if c in t]

def vagueness_score(text):
    t = text.lower()
    v = sum(1 for w in VAGUE_WORDS   if w in t)
    g = sum(1 for w in GENUINE_WORDS if w in t)
    n = 2 if has_numbers(t) else 0
    return max(0, min(100, v * 9 - g * 14 - n * 14 + 32))

def analyze_ingredients(text):
    t = text.lower()
    found_harmful, found_safe = {}, {}
    for ing, val in HARMFUL_INGREDIENTS.items():
        pat = r'\b' + re.escape(ing) + r'\b' if len(ing) <= 5 else re.escape(ing)
        if re.search(pat, t):
            found_harmful[ing] = val
    for ing, val in SAFE_INGREDIENTS.items():
        pat = r'\b' + re.escape(ing) + r'\b' if len(ing) <= 5 else re.escape(ing)
        if re.search(pat, t):
            found_safe[ing] = val
    return found_harmful, found_safe

def combined_eco_score(pred, prob, certs, vague, found_harmful, found_safe):
    risk = round((1 - prob) * 100) if pred == 0 else round(prob * 100)
    base = 100 - risk
    base += len(certs) * 8
    base -= vague * 0.2
    base -= len(found_harmful) * 10
    base += len(found_safe)   * 5
    return max(0, min(100, round(base)))

def get_final_verdict(pred, certs, nums, found_harmful, found_safe):
    h = len(found_harmful)
    s = len(found_safe)

    if h >= 3:
        return ("🔴 High Greenwashing Risk", "red",
                f"Found {h} harmful or non-eco ingredient(s). No eco-language justifies ingredients like parabens, SLS, or synthetic fragrance in a 'natural' product.")
    if h >= 1 and pred == 1:
        return ("🔴 Greenwashing Detected", "red",
                f"Both marketing language AND ingredients raise red flags — {h} harmful ingredient(s) alongside vague, unverified green claims.")
    if h >= 1 and pred == 0:
        return ("🟠 Claim Genuine — Ingredients Concerning", "orange",
                f"The sustainability claim uses credible language, but {h} concerning ingredient(s) were found in the formula.")
    if pred == 1 and s == 0:
        return ("🔴 Greenwashing Detected", "red",
                "Vague green marketing with no verifiable data, certifications, or genuinely natural ingredients.")
    if pred == 1 and s > 0:
        return ("🟡 Mixed Signals", "yellow",
                f"Some genuinely natural ingredients ({s} found) but marketing language is vague and unverified. Ask for third-party certification.")
    if pred == 0 and h == 0 and s > 0 and certs and nums:
        return ("✅ Genuinely Eco", "green",
                f"Credible claim backed by data, certifications, and {s} genuinely natural/organic ingredient(s). Low greenwashing risk.")
    if pred == 0 and h == 0 and nums:
        return ("✅ Likely Genuine", "green",
                "Claim contains verifiable data and no harmful ingredients detected. Adding a recognised certification would strengthen it further.")
    return ("🟡 Uncertain — Needs Verification", "yellow",
            "The claim language is cautiously positive but lacks specific data or certifications. No harmful ingredients detected.")

def highlight_text(text):
    result = text
    for w in GENUINE_WORDS:
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f'<span style="background:rgba(34,197,94,.25);color:#4ade80;border-radius:3px;padding:0 3px;font-weight:600">{m.group()}</span>',
            result
        )
    for w in VAGUE_WORDS:
        pattern = re.compile(r'\b' + re.escape(w) + r'\b', re.IGNORECASE)
        result = pattern.sub(
            lambda m: f'<span style="background:rgba(239,68,68,.2);color:#f87171;border-radius:3px;padding:0 3px;font-weight:600">{m.group()}</span>',
            result
        )
    return result

# ══════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #020617 0%, #022c22 100%); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031a0f, #022c22) !important;
    border-right: 1px solid #134a2a !important;
}

h1 { color: #22c55e !important; font-size: 36px !important; font-weight: 800 !important; }
h2, h3 { color: #4ade80 !important; }
p, label, .stMarkdown, .stCaption { color: #d1fae5 !important; }

.stTextArea textarea, .stTextInput input {
    background-color: #0f2d1a !important;
    color: #d1fae5 !important;
    border: 1px solid #166534 !important;
    border-radius: 10px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,.25) !important;
}

.stSelectbox [data-baseweb="select"] {
    background-color: #0f2d1a !important;
    border-color: #166534 !important;
    color: #d1fae5 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #16a34a, #22c55e) !important;
    color: #020617 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: all .3s !important;
    letter-spacing: .5px !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #22c55e, #4ade80) !important;
    box-shadow: 0 0 20px rgba(34,197,94,.45) !important;
    transform: translateY(-2px) !important;
}

[data-testid="metric-container"] {
    background: rgba(34,197,94,.07) !important;
    border: 1px solid rgba(34,197,94,.2) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #86efac !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #22c55e !important; font-size: 28px !important; font-weight: 800 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,0,0,.3) !important; border-radius: 10px !important; padding: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: #86efac !important; border-radius: 8px !important; }
.stTabs [aria-selected="true"] { background: rgba(34,197,94,.15) !important; color: #22c55e !important; }

.stSuccess { background: rgba(34,197,94,.1) !important; border: 1px solid rgba(34,197,94,.35) !important; border-radius:10px !important; }
.stError   { background: rgba(239,68,68,.1) !important;  border: 1px solid rgba(239,68,68,.35) !important;  border-radius:10px !important; }
.stWarning { background: rgba(234,179,8,.1) !important;  border: 1px solid rgba(234,179,8,.3) !important;   border-radius:10px !important; }
.stInfo    { background: rgba(59,130,246,.1) !important; border: 1px solid rgba(59,130,246,.3) !important;  border-radius:10px !important; }

.impact-card {
    background: linear-gradient(145deg, #022c22, #064e3b);
    border: 1px solid rgba(34,197,94,.2);
    padding: 22px; border-radius: 14px; color: white;
    min-height: 160px; transition: .3s; margin-bottom: 12px;
}
.impact-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(34,197,94,.3); }
.impact-title { font-size: 15px; font-weight: 700; color: #4ade80; margin: 8px 0 6px; }
.impact-desc  { font-size: 13px; opacity: .8; line-height: 1.5; }

.tag-bad  { display:inline-block; background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.3); border-radius:5px; padding:2px 9px; font-size:12px; margin:3px; font-family:monospace; }
.tag-good { display:inline-block; background:rgba(34,197,94,.12); color:#4ade80; border:1px solid rgba(34,197,94,.3); border-radius:5px; padding:2px 9px; font-size:12px; margin:3px; font-family:monospace; }
.tag-cert { display:inline-block; background:rgba(59,130,246,.12); color:#93c5fd; border:1px solid rgba(59,130,246,.3); border-radius:5px; padding:2px 9px; font-size:12px; margin:3px; font-family:monospace; }

.ing-row-bad  { background:rgba(239,68,68,.08); border-left:3px solid #ef4444; padding:8px 12px; border-radius:6px; margin:4px 0; }
.ing-row-good { background:rgba(34,197,94,.08); border-left:3px solid #22c55e; padding:8px 12px; border-radius:6px; margin:4px 0; }

hr { border-color: rgba(34,197,94,.2) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020617; }
::-webkit-scrollbar-thumb { background: #166534; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 Greenwashing Detection System")
    st.markdown("*Paste product text + ingredients together*")
    st.markdown("---")

    if MODEL_LOADED:
        st.markdown("### 📊 Model Performance")
        acc  = accuracy_score(y_all, y_all_pred)
        prec = precision_score(y_all, y_all_pred, zero_division=0)
        rec  = recall_score(y_all, y_all_pred, zero_division=0)
        f1   = f1_score(y_all, y_all_pred, zero_division=0)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy",  f"{acc*100:.1f}%")
        c2.metric("Precision", f"{prec*100:.1f}%")
        c1.metric("Recall",    f"{rec*100:.1f}%")
        c2.metric("F1 Score",  f"{f1*100:.1f}%")
        st.caption(f"Trained on {len(data)} samples")
    else:
        st.error("⚠️ Model not found!\nRun: `python train_model.py` first")

    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
1. Paste **all** product text in one box
   *(label claims + ingredient list)*
2. Select product category
3. ML model analyzes the full text
4. Ingredient scanner runs simultaneously
5. One unified verdict + Eco Score

**Tech Stack**
- Python · Scikit-learn
- TF-IDF · Gradient Boosting
- 80+ ingredient database
- Streamlit
""")

# ══════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:20px 0 10px'>
  <h1>🌿 Greenwashing Detection System</h1>
  <p style='color:#86efac; font-size:16px; font-style:italic;'>
    Paste the full product label — claims and ingredients together — for one unified verdict
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if not MODEL_LOADED:
    st.error("⚠️ **Model files not found!** Please run `python train_model.py` first, then restart the app.")
    st.stop()

# ══════════════════════════════════════
# TABS
# ══════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Analyze",
    "📊 Dashboard",
    "🧠 Model Performance",
    "🌍 Impact",
    "ℹ️ About"
])

# ─────────────────────────────────────
# TAB 1 — ANALYZE
# ─────────────────────────────────────
with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("#### Paste Full Product Label Text")
        st.caption("Include the sustainability claim AND the ingredient list — all in one box.")

        category = st.selectbox(
            "Product Category",
            list(CATEGORY_TIPS.keys()),
            help="Helps tailor the recommendation to your product type"
        )

        st.markdown("**Quick Samples:**")
        s1, s2, s3 = st.columns(3)
        if s1.button("🔴 Greenwash"):
            st.session_state["sample_claim"] = (
                "Our shampoo is 100% natural, eco-friendly and committed to protecting the planet. "
                "We care deeply about the environment and green living.\n\n"
                "Ingredients: Water, Sodium Lauryl Sulfate, Methylparaben, Propylparaben, "
                "Fragrance, Dimethicone, Mineral Oil, Synthetic Dyes, Cocamide DEA"
            )
        if s2.button("🟢 Genuine"):
            st.session_state["sample_claim"] = (
                "We reduced carbon emissions by 35 percent between 2021 and 2023 verified by ISO 14064. "
                "Packaging is 90 percent recycled certified by FSC.\n\n"
                "Ingredients: Aloe Vera, Certified Organic Coconut Oil, Jojoba Oil, Shea Butter, "
                "Lavender Essential Oil, Vitamin E (Tocopherol), Rosemary Extract, Coco Glucoside"
            )
        if s3.button("🟡 Mixed"):
            st.session_state["sample_claim"] = (
                "We are committed to sustainable fashion. Our product uses 65 percent less water "
                "verified by SGS 2022.\n\n"
                "Ingredients: Water, Aloe Vera, Sodium Lauryl Sulfate, Shea Butter, "
                "Fragrance, Jojoba Oil, Methylparaben, Chamomile Extract"
            )

        default_text = st.session_state.get("sample_claim", "")
        full_input = st.text_area(
            "Product Label Text + Ingredients",
            value=default_text,
            height=280,
            placeholder=(
                "Paste everything here, for example:\n\n"
                "Our eco-friendly shampoo cares for the planet...\n\n"
                "Ingredients: Water, Aloe Vera, Sodium Lauryl Sulfate, "
                "Coconut Oil, Methylparaben, Shea Butter, Fragrance..."
            )
        )

        analyze_btn = st.button("⚡ Analyze Product", use_container_width=True)

    # ── OUTPUT COLUMN ──
    with col_out:
        st.markdown("#### Analysis Result")

        if analyze_btn:
            if not full_input.strip():
                st.warning("⚠️ Please paste some product text first.")
            else:
                with st.spinner("🤖 Analyzing claim language and ingredients..."):

                    # ML on full combined text
                    vec  = vectorizer.transform([full_input])
                    pred = model.predict(vec)[0]
                    prob = max(model.predict_proba(vec)[0])

                    # Claim-level features
                    certs = check_certs(full_input)
                    vague = vagueness_score(full_input)
                    nums  = has_numbers(full_input)

                    # Ingredient scan on same text
                    found_harmful, found_safe = analyze_ingredients(full_input)

                    # Unified verdict and score
                    eco = combined_eco_score(pred, prob, certs, vague, found_harmful, found_safe)
                    verdict_label, verdict_color, verdict_explanation = get_final_verdict(
                        pred, certs, nums, found_harmful, found_safe
                    )

                    vague_found   = [w for w in VAGUE_WORDS   if w in full_input.lower()][:8]
                    genuine_found = [w for w in GENUINE_WORDS if w in full_input.lower()][:8]

                    result_label = "Greenwashing" if verdict_color in ("red", "orange") else "Genuine"
                    st.session_state.history.append({
                        "claim"  : full_input[:60] + "...",
                        "result" : result_label,
                        "eco"    : eco,
                        "conf"   : round(prob * 100, 1)
                    })

                # Verdict banner
                if verdict_color == "green":
                    st.success(f"**{verdict_label}**")
                elif verdict_color == "red":
                    st.error(f"**{verdict_label}**")
                else:
                    st.warning(f"**{verdict_label}**")

                st.markdown(f"*{verdict_explanation}*")

                # Score metrics — 4 cards
                m1, m2, m3, m4 = st.columns(4)
                eco_e = "🟢" if eco > 65 else "🟡" if eco > 35 else "🔴"
                m1.metric("Eco Score",       f"{eco_e} {eco}/100")
                m2.metric("ML Confidence",   f"{prob*100:.1f}%")
                m3.metric("⚠️ Harmful Ing.", len(found_harmful))
                m4.metric("✅ Safe Ing.",     len(found_safe))

                st.markdown("---")

                # Language highlight
                st.markdown("**🔍 Language Analysis:**")
                st.markdown(
                    f'<div style="background:rgba(0,0,0,.3);border:1px solid rgba(34,197,94,.2);'
                    f'border-radius:10px;padding:14px;font-size:14px;line-height:1.9;'
                    f'max-height:160px;overflow-y:auto;">'
                    f'{highlight_text(full_input)}</div>',
                    unsafe_allow_html=True
                )

                # Certifications
                st.markdown("**🏅 Certifications:**")
                if certs:
                    st.markdown(" ".join([f'<span class="tag-cert">✅ {c.upper()}</span>' for c in certs]), unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-bad">❌ No recognised certifications found</span>', unsafe_allow_html=True)

                # Numbers
                st.markdown("**🔢 Measurable Data:**")
                if nums:
                    st.markdown('<span class="tag-good">✅ Contains numerical data / percentages</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-bad">❌ No numbers or percentages found</span>', unsafe_allow_html=True)

                if vague_found:
                    st.markdown("**🚩 Vague Buzzwords:**")
                    st.markdown(" ".join([f'<span class="tag-bad">🚩 {w}</span>' for w in vague_found]), unsafe_allow_html=True)

                if genuine_found:
                    st.markdown("**✅ Genuine Indicators:**")
                    st.markdown(" ".join([f'<span class="tag-good">✅ {w}</span>' for w in genuine_found]), unsafe_allow_html=True)

                st.markdown("---")

                # Ingredient breakdown
                if found_harmful or found_safe:
                    st.markdown("**🧪 Ingredient Breakdown:**")

                    if found_harmful:
                        st.markdown(f"*🚫 {len(found_harmful)} Harmful Ingredient(s) Detected:*")
                        for ing, (cat, concern) in found_harmful.items():
                            st.markdown(
                                f'<div class="ing-row-bad">'
                                f'<span style="color:#f87171;font-weight:700;">⚠️ {ing.title()}</span> '
                                f'<span style="color:#fca5a5;font-size:12px;">({cat})</span> — '
                                f'<span style="color:#fca5a5;font-size:12px;">{concern}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    if found_safe:
                        st.markdown(f"*✅ {len(found_safe)} Natural/Organic Ingredient(s) Found:*")
                        for ing, (cat, benefit) in found_safe.items():
                            st.markdown(
                                f'<div class="ing-row-good">'
                                f'<span style="color:#4ade80;font-weight:700;">✅ {ing.title()}</span> '
                                f'<span style="color:#86efac;font-size:12px;">({cat})</span> — '
                                f'<span style="color:#86efac;font-size:12px;">{benefit}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown("**🧪 Ingredients:** *No ingredients from our database detected. For ingredient analysis, include the full INCI ingredient list in the input.*")

                # Final recommendation
                st.markdown("---")
                st.info(f"💡 **Recommendation:** {verdict_explanation}\n\n{CATEGORY_TIPS[category]}")

# ─────────────────────────────────────
# TAB 2 — DASHBOARD
# ─────────────────────────────────────
with tab2:
    st.markdown("### 📊 Analysis History Dashboard")
    history = st.session_state.history

    if not history:
        st.info("📭 No products analyzed yet. Go to the **Analyze** tab and run your first analysis!")
    else:
        df_h = pd.DataFrame(history)
        total = len(df_h)
        gw    = (df_h["result"] == "Greenwashing").sum()
        gen   = (df_h["result"] == "Genuine").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", total)
        c2.metric("Greenwashing",   gw)
        c3.metric("Genuine",        gen)
        c4.metric("Avg Eco Score",  f"{df_h['eco'].mean():.0f}/100")

        st.markdown("---")
        st.markdown("#### Distribution")
        st.bar_chart(df_h["result"].value_counts(), color="#22c55e")

        st.markdown("#### Analysis Log")
        st.dataframe(
            df_h.rename(columns={
                "claim":"Product Text (truncated)",
                "result":"Verdict",
                "eco":"Eco Score",
                "conf":"ML Confidence %"
            }),
            use_container_width=True, hide_index=True
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ─────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────
with tab3:
    st.markdown("### 🧠 Model Performance & Dataset Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Evaluation Metrics")
        acc  = accuracy_score(y_all, y_all_pred)
        prec = precision_score(y_all, y_all_pred, zero_division=0)
        rec  = recall_score(y_all, y_all_pred, zero_division=0)
        f1   = f1_score(y_all, y_all_pred, zero_division=0)

        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  f"{acc*100:.1f}%")
        m2.metric("Precision", f"{prec*100:.1f}%")
        m1.metric("Recall",    f"{rec*100:.1f}%")
        m2.metric("F1 Score",  f"{f1*100:.1f}%")

        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_all, y_all_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Genuine", "Actual: Greenwashing"],
            columns=["Pred: Genuine", "Pred: Greenwashing"]
        )
        st.dataframe(cm_df, use_container_width=True)

    with col2:
        st.markdown("#### Dataset Overview")
        d1, d2, d3 = st.columns(3)
        d1.metric("Total",        len(data))
        d2.metric("Greenwashing", (data["Label"] == 1).sum())
        d3.metric("Genuine",      (data["Label"] == 0).sum())

        st.markdown("#### ML + Ingredient Pipeline")
        st.code("""
Full Product Text (claim + ingredients)
    ↓
┌──────────────────┬─────────────────┐
│   ML Branch      │  Ingredient DB  │
│  TF-IDF + GB     │  Rule-based     │
│  → Probability   │  → Harmful/Safe │
└──────────┬───────┴────────┬────────┘
           └───────┬────────┘
             Combined Eco Score
             Unified Verdict
        """, language="text")

        st.markdown("#### Sample Training Data")
        st.dataframe(data.head(6), use_container_width=True, hide_index=True)

# ─────────────────────────────────────
# TAB 4 — IMPACT
# ─────────────────────────────────────
with tab4:
    st.markdown("### 🌍 Real World Environmental Impact")
    st.markdown("*Why detecting greenwashing matters for our planet and society.*")

    cards = [
        ("🌊", "8M Tonnes of Plastic in Oceans Yearly",
         "Much of it from products marketed as eco-friendly. Our system identifies the real culprits behind green labels."),
        ("🏭", "42% of Online Green Claims Are Exaggerated",
         "EU research confirms nearly half of all online sustainability claims are vague or unverifiable."),
        ("💸", "$1 Trillion in Misdirected ESG Investment",
         "Greenwashing diverts capital from genuinely sustainable companies. Risk scores help investors choose wisely."),
        ("👥", "Protects Low-Income Consumers Most",
         "Price-sensitive buyers rely entirely on label claims and cannot independently verify them."),
        ("🧴", "Hidden Chemicals Behind Natural Labels",
         "Parabens, SLS, and phthalates routinely appear in products labelled 'natural' — ingredient scanning exposes this."),
        ("⚖️", "Supports Regulatory Enforcement",
         "Regulators cannot audit every company manually. Our system pre-screens thousands of claims automatically."),
    ]

    col1, col2, col3 = st.columns(3)
    for i, (icon, title, desc) in enumerate(cards):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div class="impact-card">
                <div style="font-size:28px">{icon}</div>
                <div class="impact-title">{title}</div>
                <div class="impact-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────
# TAB 5 — ABOUT
# ─────────────────────────────────────
with tab5:
    st.markdown("### ℹ️ About Greenwashing Detection System")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Problem Statement**

Many companies use vague or exaggerated sustainability claims — known as **greenwashing** — making it difficult for consumers and investors to identify genuine environmental efforts. Products are also mislabelled as "natural" despite containing harmful synthetic chemicals.

**Solution**

Greenwashing Detection System uses NLP, supervised Machine Learning, and a curated ingredient database to detect greenwashing. Just paste the **full product label text** — marketing claims and ingredient list together — and get one unified verdict.

**What Makes It Different**

- Single input: paste claim + ingredients together
- ML model analyzes the full text for greenwashing language
- Ingredient scanner simultaneously flags 45+ harmful chemicals
- Recognizes 35+ genuinely natural/organic ingredients
- One unified Eco Score (0–100) merging both analyses
- Works in real-time on any product label
        """)

    with col2:
        st.markdown("""
**Tech Stack**
```
Language    : Python 3.x
ML Model    : Gradient Boosting (100 estimators)
Feature Ext : TF-IDF (ngram 1-2, 800 features)
NLP         : Regex + Custom Preprocessing
Ingredient  : Rule-based DB (80+ entries)
UI          : Streamlit
Dataset     : 160+ labeled samples
```




        """)

# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#166534;font-size:12px;font-family:monospace;padding:6px 0'>"
    "🌿 Greenwashing Detection System · NLP + Ingredient Analysis · Gradient Boosting + TF-IDF · GRIET Department of Data Science"
    "</div>",
    unsafe_allow_html=True
)