
import streamlit as st
import pandas as pd
import joblib
import json
import re
import os
import gdown
import plotly.express as px

# ── Google Drive file IDs ──────────────────────────────────────────────────────
# After uploading files to Google Drive, get the share link for each file.
# The file ID is the long string in the link:
# https://drive.google.com/file/d/FILE_ID_HERE/view
# Replace the values below with your actual file IDs.

DRIVE_FILES = {
    "best_model.pkl":              "1mZLY71zY_UXl4vE5UleotYVN0lI1ht8-",
    "tfidf_vectorizer.pkl":        "1t18qwIrmNcsiWjW9G_g7rKv5gZ7tzkzQ",
    "label_encoder.pkl":           "1HMBXc7W6rBOlBH04OJCPqKLfRH9OMhlY",
    "model_metadata.json":         "1j40Q2yHMbKxYQJHfzz0_Y4lk9UoARa9Y",
    "reviews_for_dashboard.csv":   "1dYSCMKOEMNxLy1-G6uDdzVC-mr5zHdT8",
}

@st.cache_resource(show_spinner="Loading model from Google Drive...")
def download_and_load():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=True)

    model   = joblib.load("best_model.pkl")
    tfidf   = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return model, tfidf, encoder, meta

@st.cache_data(show_spinner="Loading review data...")
def load_data():
    if not os.path.exists("reviews_for_dashboard.csv"):
        url = f"https://drive.google.com/uc?id={DRIVE_FILES['reviews_for_dashboard.csv']}"
        gdown.download(url, "reviews_for_dashboard.csv", quiet=True)
    return pd.read_csv("reviews_for_dashboard.csv", parse_dates=["date"])

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethiopian Banking Sentiment Analyzer",
    page_icon="🇪🇹",
    layout="wide"
)

model, tfidf, encoder, meta = download_and_load()
df = load_data()

SENTIMENT_COLORS = {"positive": "#2ECC71", "neutral": "#F39C12", "negative": "#E74C3C"}
ORDER = ["positive", "neutral", "negative"]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", "", text)
    text = re.sub(r"[^\u1200-\u137Fa-zA-Z0-9\s\.\,\!\?\'\/-]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(text):
    cleaned  = clean_text(text)
    features = tfidf.transform([cleaned])
    pred_idx = model.predict(features)[0]
    return encoder.inverse_transform([pred_idx])[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🇪🇹 Ethiopian Banking\nSentiment Analyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔍 Predict Sentiment", "📊 Analytics Dashboard", "ℹ️ Model Info"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("Test Accuracy", f"{meta['test_accuracy']*100:.1f}%")
st.sidebar.metric("Test F1 (weighted)", f"{meta['test_f1']:.4f}")
st.sidebar.metric("CV F1 Mean", f"{meta['cv_mean_f1']:.4f} ± {meta['cv_std_f1']:.4f}")

# ── Page 1: Predict ────────────────────────────────────────────────────────────
if page == "🔍 Predict Sentiment":
    st.title("🔍 Predict Review Sentiment")
    st.markdown("""Enter a review in **English**, **Amharic (Ethiopic script)**,
    or **Romanized Amharic** (e.g. *betam tiru app new*) — the model handles all three.""")

    col1, col2 = st.columns([2, 1])
    with col1:
        review_input = st.text_area("Paste a review here:", height=150,
            placeholder="e.g. App yellem always crashing, betam annoying!")
        predict_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col2:
        st.markdown("**Example reviews to try:**")
        examples = [
            "betam tiru app new, always fast and reliable!",
            "app yellem always crashing, very frustrated",
            "Sometimes works sometimes not, average",
            "በጣም ጥሩ አፕሊኬሽን ነው፣ ፈጣን ነው",
            "Always shows error cannot transfer money",
        ]
        for ex in examples:
            if st.button(ex[:45] + "...", key=ex):
                review_input = ex
                predict_btn  = True

    if predict_btn and review_input.strip():
        sentiment = predict(review_input)
        color = SENTIMENT_COLORS[sentiment]
        icons = {"positive": "😊", "neutral": "😐", "negative": "😞"}
        st.markdown(
            f"""<div style="background:{color}22; border-left:5px solid {color};
            padding:20px; border-radius:8px; margin-top:10px;">
            <h2 style="color:{color};">{icons[sentiment]} {sentiment.upper()}</h2>
            <p style="color:#333;">The model classified this review as <b>{sentiment}</b>.</p>
            </div>""", unsafe_allow_html=True)

# ── Page 2: Dashboard ──────────────────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.title("📊 Sentiment Analytics Dashboard")
    st.markdown("Explore sentiment patterns across Ethiopian banking apps.")

    selected_apps = st.multiselect("Filter by app:",
        options=df["app_name"].unique().tolist(),
        default=df["app_name"].unique().tolist())
    dff = df[df["app_name"].isin(selected_apps)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(dff):,}")
    c2.metric("Positive", f"{(dff['sentiment']=='positive').sum():,}")
    c3.metric("Neutral",  f"{(dff['sentiment']=='neutral').sum():,}")
    c4.metric("Negative", f"{(dff['sentiment']=='negative').sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sc = dff["sentiment"].value_counts().reset_index()
        sc.columns = ["sentiment", "count"]
        st.plotly_chart(px.pie(sc, values="count", names="sentiment",
            title="Overall Sentiment", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS, hole=0.4), use_container_width=True)
    with col2:
        aps = dff.groupby(["app_name","sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(aps, x="app_name", y="count", color="sentiment",
            title="Sentiment by Bank", color_discrete_map=SENTIMENT_COLORS, barmode="stack")
        fig2.update_xaxes(tickangle=20)
        st.plotly_chart(fig2, use_container_width=True)

    avg = dff.groupby("app_name")["score"].mean().reset_index()
    avg.columns = ["app_name","avg_rating"]
    avg = avg.sort_values("avg_rating")
    st.plotly_chart(px.bar(avg, x="avg_rating", y="app_name", orientation="h",
        title="Average Star Rating by App", color="avg_rating",
        color_continuous_scale="RdYlGn", range_color=[1,5]), use_container_width=True)

    st.subheader("📋 Recent Reviews")
    st.dataframe(dff[["app_name","content","score","sentiment","date"]]
        .sort_values("date", ascending=False).head(20), use_container_width=True)

# ── Page 3: Model Info ─────────────────────────────────────────────────────────
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    st.markdown("""
    | Step | Method |
    |------|--------|
    | Data Collection | Google Play Scraper |
    | Labeling | Star rating → Negative / Neutral / Positive |
    | Text Cleaning | Regex — Amharic + Latin preserved |
    | Feature Extraction | TF-IDF character n-grams (2–4), vocab=50k |
    | Imbalance Handling | SMOTE (training set only) |
    | Models Compared | Logistic Regression, Naive Bayes, Linear SVM, Random Forest |
    | Evaluation | Accuracy, Weighted F1, Confusion Matrix, 5-Fold CV |
    | Deployment | Streamlit Community Cloud |
    """)
    st.markdown("### Handling Romanized Amharic")
    st.info(
        "Ethiopian users write Amharic phonetically in Latin script (e.g. 'betam tiru' = 'very good'). "
        "We use character-level n-gram TF-IDF which learns subword patterns correlated with sentiment "
        "directly from labeled data — no translation needed."
    )
    st.json(meta)
