# =========================================================
# streamlit_app.py  –  Fake-News Detector (3 Transformers)
# =========================================================
# • Expects the following files in the same folder:
#   distilbert_model.zip   bert_model.zip   albert_model.zip
#   train_dataset.csv
#   optuna_trials.csv  optuna_f1_plot.png (optional)
#   metrics_report.json  confusion_matrix.npy (optional)
#
# • Tested on Python 3.11 (CPU) with:
#   pip install streamlit torch==2.6.0 transformers pandas matplotlib seaborn scikit-learn plotly wordcloud
# ---------------------------------------------------------

# ── 0. Un-zip model archives on first run ─────────────────
# =========================================================
# streamlit_app.py  –  Fake-News Detector (DistilBERT | BERT | ALBERT)
# =========================================================
# • Put this file in your GitHub repo (no model files committed).
# • Add  gdown  to requirements.txt
# • Replace the three Google-Drive IDs below with your real IDs.
# ---------------------------------------------------------

# ──────────────────────────────────────────────────────────
# 0. Download & unzip models from Google Drive (one-time)
# ──────────────────────────────────────────────────────────
import types, torch

torch.classes.__path__ = []

import os, zipfile, subprocess, sys

GDRIVE_MODELS = {
    "distilbert_model": "1xwm50FzQJFrxZovV1AePsVOwtvugzFn7",
    "bert_model": "1hEUnQihsGjjtfkCqEfxjHAvujNLt-PTv",
    "albert_model": "1AezznK-va0QZ2hRrZydaRxNIWuvp6CY6",
}


def ensure_model(folder: str, file_id: str):
    """If <folder>/config.json is missing, download ZIP from Drive and extract."""
    if os.path.exists(os.path.join(folder, "config.json")):
        return
    zip_path = f"{folder}.zip"
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.isfile(zip_path):
        subprocess.run([sys.executable, "-m", "gdown", url, "-O", zip_path], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(folder)
    os.remove(zip_path)


for fld, fid in GDRIVE_MODELS.items():
    ensure_model(fld, fid)

# ──────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
import streamlit as st, torch, torch.nn.functional as F
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake-News Detector", layout="wide")


# ──────────────────────────────────────────────────────────
# 2. Helpers  (load model, predict)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model_tok(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, mdl


MODELS = {
    "DistilBERT": "distilbert_model",
    "BERT-base": "bert_model",
    "ALBERT-base": "albert_model",
}


def predict(model_dir: str, title: str, text: str):
    tok, mdl = load_model_tok(model_dir)
    inp = f"[TITLE] {title} [TEXT] {text}"
    enc = tok(inp, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = mdl(**enc).logits.squeeze()
    probs = F.softmax(logits, dim=0).cpu().numpy()
    return int(np.argmax(probs)), probs


# ──────────────────────────────────────────────────────────
# 3. Sidebar navigation
# ──────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview & Inference",
        "Dataset Visualization",
        "Hyperparameter Tuning",
        "Model Analysis",
    ],
)

# ──────────────────────────────────────────────────────────
# 4. Page 1 – Overview & Inference
# ──────────────────────────────────────────────────────────
if page == "Overview & Inference":
    st.title("📰 Fake-News Detector 🔮")
    mdl_name = st.selectbox("Choose a model", list(MODELS.keys()))
    col1, col2 = st.columns(2)
    with col1:
        title_in = st.text_input("News Title")
    with col2:
        text_in = st.text_area("News Content", height=180)

    if st.button("Predict") and title_in and text_in:
        label, probs = predict(MODELS[mdl_name], title_in, text_in)
        classes = ["Real", "Fake"]
        st.subheader(f"Prediction → **{classes[label]}**")
        st.write({classes[i]: f"{p*100:.2f} %" for i, p in enumerate(probs)})

    st.markdown(
        """
    ### Dataset
    We used the [Fake News Detection Dataset - English](https://huggingface.co/datasets/mohammadjavadpirhadi/fake-news-detection-dataset-english) from HuggingFace. 
    - **Train size:** 35,918 records  
    - **Test size:** 8,980 records  
    - **Labels:** 0 = Real, 1 = Fake

    ### Limitations of Prior Work
    Traditional fake news classifiers often rely on TF-IDF, bag-of-words, or logistic regression models that ignore contextual semantics.
    They also treat the title and body text separately, failing to capture the relationship between them. 
    Our approach combines title, subject, and content into one input and uses transformer models to understand the contextual semantics.

    ### Challenges in Dataset
    One of the main difficulties is the length of the articles. Over 15% of articles have more than 600 words, 
    and many transformer models can only process up to 512 tokens, leading to truncation and loss of information.

    ### Models Used
    We fine-tuned three transformer-based models:
    - DistilBERT
    - BERT-base
    - ALBERT-base

    All models were evaluated using accuracy, precision, recall, and F1-score.
    """
    )

# ──────────────────────────────────────────────────────────
# 5. Page 2 – Dataset Visualization
# ──────────────────────────────────────────────────────────
elif page == "Dataset Visualization":
    st.title("📊 Dataset Visualization")
    df = pd.read_csv("train_dataset.csv")
    df["words"] = df["full_text"].str.split().apply(len)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class distribution")
        st.bar_chart(df["label"].value_counts().sort_index())
    with col2:
        st.subheader("Word-count distribution")
        st.plotly_chart(px.histogram(df, x="words", nbins=50))


# ──────────────────────────────────────────────────────────
# 6. Page 3 – Hyperparameter Tuning
# ──────────────────────────────────────────────────────────
elif page == "Hyperparameter Tuning":
    st.title("⚙️ Hyperparameter Tuning (Optuna, 5 trials)")
    st.markdown("Best config → `lr 3e-5`, `batch 8`, `epochs 3`")
    if os.path.exists("optuna_f1_plot.png"):
        st.image("optuna_f1_plot.png")
    if os.path.exists("optuna_trials.csv"):
        st.dataframe(pd.read_csv("optuna_trials.csv"))
    else:
        st.info("Optuna result files not found.")

    """
    We optimized DistilBERT using Optuna for the following parameters:
    - Learning Rate
    - Batch Size
    - Number of Epochs

    Only 5 trials were used to minimize training time.
    """

# ──────────────────────────────────────────────────────────
# 7. Page 4 – Model Analysis
# ──────────────────────────────────────────────────────────
else:
    st.title("🧩 Model Analysis & Error Inspection")
    if os.path.exists("metrics_report.json"):
        st.subheader("Classification report")
        st.json(json.load(open("metrics_report.json")))
    if os.path.exists("confusion_matrix.npy"):
        cm = np.load("confusion_matrix.npy")
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("metrics_report.json / confusion_matrix.npy not found.")

        st.subheader("Error Analysis")
    st.markdown(
        """
    - **False Positives:** Some real news articles with clickbait titles were wrongly classified as fake.
    - **False Negatives:** Satirical news articles written with neutral tone were labeled as real.

    This suggests our model might benefit from:
    - Incorporating satire detection cues
    - More training examples with ambiguous language
    - Ensemble methods for better robustness
    """
    )
