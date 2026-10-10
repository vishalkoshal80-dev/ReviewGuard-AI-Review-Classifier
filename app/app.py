import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Paths and custom preprocessing import
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "app" else Path.cwd()
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Needed when loading the saved sklearn pipeline, which references clean_text.
from preprocessing import clean_text  # noqa: F401

MODEL_PATH = ROOT / "models" / "review_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
CONFUSION_PATH = ROOT / "reports" / "confusion_matrix.png"

st.set_page_config(
    page_title="ReviewGuard | Review Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --rg-bg: #080d18;
        --rg-panel: rgba(17, 27, 46, 0.88);
        --rg-border: rgba(142, 168, 205, 0.20);
        --rg-text: #f1f6ff;
        --rg-muted: #9bacc7;
        --rg-cyan: #65e4ff;
        --rg-blue: #6c8cff;
        --rg-purple: #b18cff;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        color: var(--rg-text);
        background:
            radial-gradient(ellipse at 12% 0%, rgba(40, 114, 187, .19), transparent 38%),
            radial-gradient(ellipse at 92% 12%, rgba(128, 72, 191, .17), transparent 31%),
            linear-gradient(145deg, #080d18 0%, #0b1322 52%, #080d18 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1320px; padding: 2rem 2.1rem 3rem; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a2c 0%, #0b1220 100%);
        border-right: 1px solid var(--rg-border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.7rem; }

    .topline {
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; margin-bottom: 1.25rem;
    }
    .brand-mini { display:flex; align-items:center; gap:.7rem; color:#ecf5ff;
        font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; }
    .brand-icon { display:inline-flex; align-items:center; justify-content:center;
        height:42px; width:42px; border-radius:14px; font-size:1.45rem;
        background:linear-gradient(135deg,rgba(101,228,255,.18),rgba(108,140,255,.20));
        border:1px solid rgba(101,228,255,.30); box-shadow:0 0 28px rgba(75,164,255,.10); }
    .live-pill { display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .78rem;
        border:1px solid rgba(89,220,171,.28); border-radius:999px;
        background:rgba(44,178,131,.09); color:#92f0c8; font-size:.76rem; font-weight:700; }
    .live-dot { height:7px; width:7px; border-radius:100%; background:#59dca5; display:inline-block;
        box-shadow:0 0 10px rgba(89,220,165,.8); }

    .hero {
        position:relative; overflow:hidden; padding:2.25rem 2.35rem;
        border-radius:26px; border:1px solid rgba(133,167,220,.23);
        background:linear-gradient(112deg,rgba(20,43,74,.94) 0%,rgba(25,29,62,.93) 57%,rgba(45,28,70,.88) 100%);
        box-shadow:0 22px 60px rgba(0,0,0,.24); margin-bottom:1.4rem;
    }
    .hero:after { content:""; position:absolute; right:-95px; top:-150px; width:390px; height:390px;
        border-radius:50%; border:1px solid rgba(101,228,255,.15);
        box-shadow:0 0 0 32px rgba(101,228,255,.025),0 0 0 68px rgba(177,140,255,.025),0 0 85px rgba(101,228,255,.07); }
    .hero-kicker { position:relative; z-index:1; color:var(--rg-cyan); text-transform:uppercase;
        letter-spacing:2.2px; font-size:.72rem; font-weight:700; margin-bottom:.8rem; }
    .hero-title { position:relative; z-index:1; font-family:'Space Grotesk',sans-serif;
        font-size:clamp(2.25rem,4.2vw,3.45rem); font-weight:700; letter-spacing:-1.9px;
        line-height:1.08; margin:0; color:#f7fbff; }
    .hero-title .shine { background:linear-gradient(90deg,#66e5ff,#8ba4ff 58%,#c69bff);
        -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-copy { position:relative; z-index:1; max-width:690px; color:#b9c9e2;
        margin-top:1rem; font-size:1.02rem; line-height:1.75; }
    .hero-tags { position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.35rem; }
    .hero-tag { border:1px solid rgba(153,183,226,.22); background:rgba(6,13,28,.25);
        border-radius:999px; color:#d4e5fa; padding:.44rem .75rem; font-size:.76rem; font-weight:600; }

    .section-heading { font-family:'Space Grotesk',sans-serif; font-size:1.32rem; font-weight:700;
        letter-spacing:-.35px; color:#edf5ff; margin:.25rem 0 .25rem; }
    .section-copy { color:var(--rg-muted); font-size:.92rem; line-height:1.65; margin-bottom:1rem; }

    .metric-card { position:relative; overflow:hidden; height:100%; min-height:132px;
        border:1px solid var(--rg-border); border-radius:19px; padding:1.15rem 1.25rem;
        background:linear-gradient(145deg,rgba(20,33,55,.92),rgba(13,23,40,.92));
        box-shadow:0 10px 28px rgba(0,0,0,.12); }
    .metric-card:before { content:""; position:absolute; top:0; left:1.2rem; right:1.2rem; height:2px;
        background:linear-gradient(90deg,transparent,#65e4ff,#8f91ff,transparent); opacity:.9; }
    .metric-label { color:#9eb0cb; font-size:.79rem; font-weight:600; margin-bottom:.8rem; }
    .metric-value { font-family:'Space Grotesk',sans-serif; color:#f4f8ff; font-size:2rem;
        line-height:1.1; font-weight:700; letter-spacing:-.8px; }
    .metric-note { color:#8296b5; font-size:.73rem; margin-top:.55rem; }

    div[data-baseweb="tab-list"] { gap:.45rem; background:rgba(14,23,39,.65); padding:.42rem;
        border:1px solid var(--rg-border); border-radius:15px; }
    button[data-baseweb="tab"] { border-radius:10px; min-height:43px; padding:.55rem 1rem;
        color:#aebed5; font-weight:600; }
    button[data-baseweb="tab"][aria-selected="true"] { color:#f7fbff; background:linear-gradient(110deg,rgba(44,121,184,.32),rgba(113,91,184,.27)); }

    div[data-baseweb="textarea"] > div, div[data-baseweb="input"] > div {
        background:rgba(11,20,35,.94); border:1px solid #2c405e; border-radius:14px; }
    div[data-baseweb="textarea"] > div:focus-within { border-color:#65cffa; box-shadow:0 0 0 1px rgba(101,207,250,.20); }
    div[data-baseweb="textarea"] textarea { color:#eef5ff; line-height:1.7; }
    label, .stMarkdown p { color:#d5dfef; }
    .stButton button, .stDownloadButton button { border-radius:12px; min-height:44px; font-weight:700;
        transition:all .2s ease; }
    .stButton button[kind="primary"] { background:linear-gradient(100deg,#168bc8,#6269e8);
        border:1px solid rgba(141,205,255,.30); color:white; box-shadow:0 7px 22px rgba(48,111,215,.20); }
    .stButton button[kind="primary"]:hover { border-color:#9fe5ff; box-shadow:0 8px 30px rgba(70,157,255,.30); transform:translateY(-1px); }
    .stDownloadButton button { border:1px solid #3a5273; background:#15243a; color:#eaf3ff; }
    [data-testid="stMetric"] { background:rgba(16,27,46,.65); border:1px solid var(--rg-border);
        border-radius:15px; padding:1rem; }
    [data-testid="stMetricLabel"] { color:#9fb1ca; }
    [data-testid="stMetricValue"] { color:#f4f8ff; }

    .result-card { border-radius:20px; padding:1.4rem 1.5rem; margin:1rem 0;
        border:1px solid; box-shadow:0 12px 30px rgba(0,0,0,.14); }
    .result-cg { background:linear-gradient(115deg,rgba(75,48,20,.70),rgba(36,30,28,.80)); border-color:rgba(244,183,83,.42); }
    .result-or { background:linear-gradient(115deg,rgba(15,67,57,.70),rgba(17,40,43,.80)); border-color:rgba(88,224,179,.40); }
    .result-eyebrow { color:#b5c5da; text-transform:uppercase; letter-spacing:1.8px; font-weight:700; font-size:.70rem; }
    .result-label { color:#fbfdff; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.35rem; margin-top:.5rem; }
    .result-description { color:#c6d2e2; margin-top:.45rem; line-height:1.65; font-size:.9rem; }
    .disclaimer { border:1px solid rgba(135,159,191,.19); background:rgba(17,29,48,.63);
        border-radius:14px; padding:.95rem 1rem; color:#9eafc7; font-size:.82rem; line-height:1.65; }
    .mini-step { display:flex; align-items:flex-start; gap:.8rem; padding:.85rem 0; border-bottom:1px solid rgba(142,168,205,.12); }
    .step-number { display:flex; align-items:center; justify-content:center; min-width:31px; height:31px;
        border-radius:10px; color:#8de8ff; background:rgba(74,177,219,.12); border:1px solid rgba(101,228,255,.22); font-weight:700; }
    .footer { text-align:center; color:#6f83a1; font-size:.78rem; line-height:1.8;
        border-top:1px solid rgba(142,168,205,.14); padding-top:1.3rem; margin-top:2.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Resources
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` from the project root first.")
    st.stop()

model = load_model()
metrics = load_metrics()


def predict_reviews(texts):
    predictions = model.predict(texts)
    probability_rows = model.predict_proba(texts)
    classes = list(model.classes_)
    scores = [float(probability_rows[i][classes.index(predictions[i])]) * 100 for i in range(len(predictions))]
    return predictions, scores


def label_details(prediction):
    if int(prediction) == 1:
        return (
            "Likely computer-generated (CG)",
            "result-cg",
            "The text contains patterns the model associates with its computer-generated training examples.",
        )
    return (
        "Likely original (OR)",
        "result-or",
        "The text contains patterns the model associates with its original-review training examples.",
    )

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("<div class='brand-mini'><span class='brand-icon'>🛡️</span><span>ReviewGuard</span></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8ea3c2;margin:.35rem 0 1rem 3.4rem;font-size:.78rem;'>REVIEW INTELLIGENCE</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Model overview")
    st.markdown("**Algorithm**  \nLogistic Regression")
    st.markdown("**Text representation**  \nTF-IDF · Unigrams + bigrams")
    st.markdown("**Training labels**  \n`CG` · Computer-generated  \n`OR` · Original")
    st.markdown("---")
    st.markdown("### Test performance")
    if "accuracy" in metrics:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    if "f1_cg" in metrics:
        st.metric("F1-score · CG", f"{metrics['f1_cg'] * 100:.2f}%")
    st.caption("Results from the held-out test split.")
    st.markdown("---")
    st.markdown("<span class='live-pill'><span class='live-dot'></span> MODEL LOADED</span>", unsafe_allow_html=True)

# --------------------------------------------------
# Header / hero
# --------------------------------------------------
st.markdown(
    """
    <div class="topline">
        <div class="brand-mini"><span class="brand-icon">🛡️</span><span>ReviewGuard <span style="color:#8fa5c5;font-weight:500;">/ AI lab</span></span></div>
        <div class="live-pill"><span class="live-dot"></span> MODEL READY</div>
    </div>
    <div class="hero">
        <div class="hero-kicker">NLP · TEXT CLASSIFICATION · ML</div>
        <h1 class="hero-title">Understand the story<br>behind every <span class="shine">review.</span></h1>
        <div class="hero-copy">Analyze review text with a machine-learning model trained to distinguish computer-generated and original reviews according to its dataset labels.</div>
        <div class="hero-tags"><span class="hero-tag">✦ TF-IDF features</span><span class="hero-tag">◈ Logistic Regression</span><span class="hero-tag">▤ Batch CSV analysis</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric cards
metric_data = [
    ("TEST ACCURACY", f"{metrics.get('accuracy', 0) * 100:.2f}%", "Held-out split"),
    ("PRECISION · CG", f"{metrics.get('precision_cg', 0) * 100:.2f}%", "Positive class: CG"),
    ("RECALL · CG", f"{metrics.get('recall_cg', 0) * 100:.2f}%", "Positive class: CG"),
    ("F1-SCORE · CG", f"{metrics.get('f1_cg', 0) * 100:.2f}%", "Precision + recall"),
]
cols = st.columns(4)
for col, (label, value, note) in zip(cols, metric_data):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
single_tab, batch_tab, insights_tab = st.tabs(["✦  Review analyzer", "▤  Batch analysis", "◈  Model insights"])

with single_tab:
    st.markdown("<div class='section-heading'>Analyze a review</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Paste a review below to see the class predicted by your model.</div>", unsafe_allow_html=True)

    # Sample buttons can quickly populate the input box on rerun
    if "review_input" not in st.session_state:
        st.session_state.review_input = ""
    sample_col1, sample_col2 = st.columns([1, 1])
    with sample_col1:
        if st.button("Try sample review A", use_container_width=True):
            st.session_state.review_input = "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty"
    with sample_col2:
        if st.button("Try sample review B", use_container_width=True):
            st.session_state.review_input = "The headphones work well. Battery life is decent, and the sound quality is acceptable for the price."

    with st.form("review_form"):
        review_text = st.text_area(
            "REVIEW TEXT",
            key="review_input",
            height=165,
            max_chars=10000,
            placeholder="Example: The product is comfortable, works as described, and offers good value...",
        )
        submitted = st.form_submit_button("✦  Analyze review", type="primary", use_container_width=True)

    if submitted:
        if not review_text.strip():
            st.warning("Please enter a review before analyzing.")
        else:
            predictions, scores = predict_reviews([review_text])
            st.session_state.last_review_result = {
                "text": review_text,
                "prediction": int(predictions[0]),
                "score": float(scores[0]),
            }

    result = st.session_state.get("last_review_result")
    if result:
        label, card_class, description = label_details(result["prediction"])
        st.markdown(
            f"<div class='result-card {card_class}'><div class='result-eyebrow'>Prediction result</div><div class='result-label'>{label}</div><div class='result-description'>{description}</div></div>",
            unsafe_allow_html=True,
        )
        score_col, explainer_col = st.columns([1, 1.7])
        with score_col:
            st.markdown("**Predicted-class score**")
            st.markdown(f"<div style='font-family:Space Grotesk,sans-serif;font-size:2rem;font-weight:700;color:#f4f8ff;'>{result['score']:.2f}%</div>", unsafe_allow_html=True)
            st.progress(min(max(result["score"] / 100, 0), 1))
        with explainer_col:
            st.markdown("**How to read this result**")
            st.markdown("<div class='disclaimer'>This score reflects the model's selected class. It is not a verified probability that the review is fake, deceptive, or authentic.</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='disclaimer'><b>Important:</b> CG means computer-generated and OR means original according to the training dataset. These labels do not prove fraud or authenticity.</div>", unsafe_allow_html=True)

with batch_tab:
    st.markdown("<div class='section-heading'>Analyze reviews at scale</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Upload a CSV, choose its review-text column, and export predictions for every non-empty row.</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload review data (.csv)", type=["csv"], key="batch_upload")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if uploaded_df.empty:
                st.warning("This CSV has no rows.")
            else:
                common_names = ["Review_Text", "text_", "review_text", "review", "text"]
                selected_default = next((c for c in common_names if c in uploaded_df.columns), uploaded_df.columns[0])
                selected_index = list(uploaded_df.columns).index(selected_default)
                text_column = st.selectbox("Select the review text column", uploaded_df.columns.tolist(), index=selected_index)
                st.caption(f"{len(uploaded_df):,} rows · {len(uploaded_df.columns)} columns")
                if st.button("Analyze CSV", type="primary", use_container_width=True):
                    results = uploaded_df.copy()
                    valid_mask = uploaded_df[text_column].notna() & uploaded_df[text_column].astype(str).str.strip().ne("")
                    texts = uploaded_df.loc[valid_mask, text_column].astype(str).tolist()
                    results["prediction"] = "No text provided"
                    results["model_score_percent"] = pd.NA
                    if texts:
                        predictions, scores = predict_reviews(texts)
                        results.loc[valid_mask, "prediction"] = [label_details(int(p))[0] for p in predictions]
                        results.loc[valid_mask, "model_score_percent"] = [round(float(s), 2) for s in scores]
                    st.session_state.batch_results = results
            if "batch_results" in st.session_state:
                result_df = st.session_state.batch_results
                st.success(f"Analysis complete: {len(result_df):,} rows.")
                st.dataframe(result_df.head(100), use_container_width=True)
                st.download_button(
                    "⬇  Download prediction results",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="reviewguard_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

with insights_tab:
    st.markdown("<div class='section-heading'>Inside the model</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Review the evaluation metrics and see the core classification pipeline.</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("#### Evaluation summary")
        for label, key in [
            ("Accuracy", "accuracy"),
            ("Precision · CG", "precision_cg"),
            ("Recall · CG", "recall_cg"),
            ("F1-score · CG", "f1_cg"),
        ]:
            if key in metrics:
                st.metric(label, f"{metrics[key] * 100:.2f}%")
        st.caption("Metrics are calculated on a held-out test split from the same dataset source.")
    with right:
        st.markdown("#### Confusion matrix")
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate the confusion matrix image.")
    st.markdown("#### Classification pipeline")
    steps = [
        ("01", "Review text", "Raw review supplied by the user"),
        ("02", "Text preprocessing", "Normalize whitespace, case and URLs"),
        ("03", "TF-IDF vectorization", "Convert words and phrases into numeric features"),
        ("04", "Logistic Regression", "Predict CG or OR using learned text patterns"),
    ]
    for number, heading, detail in steps:
        st.markdown(f"<div class='mini-step'><div class='step-number'>{number}</div><div><b>{heading}</b><br><span style='color:#9bacc7;font-size:.88rem;'>{detail}</span></div></div>", unsafe_allow_html=True)
    st.markdown("<br><div class='disclaimer'><b>Limitations:</b> The model learns the CG/OR labels and patterns in its training data. It is not a universal fake-review detector. A CG review is not automatically fraudulent; an OR label does not prove truthfulness. Validate on independent datasets before drawing real-world conclusions.</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>🛡️ <b>ReviewGuard</b> · NLP & Machine Learning project<br>Built with Python · Scikit-learn · Streamlit</div>",
    unsafe_allow_html=True,
)
