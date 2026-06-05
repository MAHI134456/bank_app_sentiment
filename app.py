import streamlit as st
import pandas as pd
import joblib
import json
import re
import os
import gdown
import plotly.express as px

DRIVE_FILES = {
    "best_model.pkl":            "1VJu6yUdZbxIKF8TWHXxb-cB7COUlMSnI",
    "tfidf_vectorizer.pkl":      "1hSyq9eV3TcSkDtD0vgmTeIWdRCXXw90t",
    "label_encoder.pkl":         "1_s7Gi1Z9reKE_lE34IDvO_i2q0e-rgn6",
    "model_metadata.json":       "1M0s-Qek4jiQr-z4gIUqG_ryGIPM72-gh",
    "reviews_for_dashboard.csv": "1l5g45QBGsD7ExCRgVTDAmuVEoQZJkxLc",
}

@st.cache_resource(show_spinner='Loading model...')
def download_and_load():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=True)
    model = joblib.load("best_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return model, tfidf, encoder, meta

@st.cache_data(show_spinner='Loading review data...')
def load_data():
    if not os.path.exists("reviews_for_dashboard.csv"):
        url = f"https://drive.google.com/uc?id={DRIVE_FILES['reviews_for_dashboard.csv']}"
        gdown.download(url, "reviews_for_dashboard.csv", quiet=True)
    return pd.read_csv("reviews_for_dashboard.csv", parse_dates=["date"])

st.set_page_config(
    page_title="Ethiopian Banking Sentiment Analyzer",
    page_icon="🇪🇹",
    layout="wide",
)

model, tfidf, encoder, meta = download_and_load()
df = load_data()
COLORS = {"positive": "#2ECC71", "negative": "#E74C3C"}


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', '', text)
    text = re.sub(r"[^ሀ-፿a-zA-Z0-9\s\.,!?'/\-]", ' ', text)
    return re.sub(r'\s+', ' ', text.lower()).strip()


def predict(text):
    return encoder.inverse_transform(model.predict(tfidf.transform([clean_text(text)])))[0]

st.sidebar.title("🇪🇹 Ethiopian Banking\nSentiment Analyzer")
st.sidebar.markdown('---')
page = st.sidebar.radio("Navigate", ["Predict Sentiment", "Analytics Dashboard", "Model Info"])
st.sidebar.markdown('---')
st.sidebar.markdown('**Binary Model Performance**')
st.sidebar.metric('Test Accuracy', f"{meta['test_accuracy']*100:.1f}%")
st.sidebar.metric('Test F1 (negative)', f"{meta['test_f1']:.4f}")
st.sidebar.metric('ROC-AUC', f"{meta.get('test_auc', 'N/A')}")

if page == 'Predict Sentiment':
    st.title('Predict Review Sentiment')
    st.info(
        'This model classifies reviews as **Positive** or **Negative**. '
        'Three-star (mixed) reviews are excluded from the training data '
        'because their labels are too ambiguous to learn reliably.'
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        review_input = st.text_area(
            'Paste a review here:',
            height=150,
            placeholder='e.g. App yellem always crashing / betam tiru app!',
        )
        predict_btn = st.button('Analyze Sentiment', type='primary', use_container_width=True)
    with col2:
        st.markdown('**Examples to try:**')
        examples = [
            'betam tiru, always fast and reliable!',
            'yellem app always crashing very frustrated',
            'Great app, best banking app in Ethiopia',
            'Cannot transfer money, always shows error',
        ]
        for ex in examples:
            if st.button(ex[:45], key=ex):
                review_input = ex
                predict_btn = True
    if predict_btn and review_input.strip():
        sentiment = predict(review_input)
        color = COLORS[sentiment]
        icons = {"positive": "😊", "negative": "😞"}
        st.markdown(
            f'<div style="background:{color}22; border-left:5px solid {color}; '
            f'padding:20px; border-radius:8px; margin-top:10px;">'
            f'<h2 style="color:{color};">{icons[sentiment]} {sentiment.upper()}</h2>'
            f'<p>This review was classified as <b>{sentiment}</b>.</p></div>',
            unsafe_allow_html=True,
        )

elif page == 'Analytics Dashboard':
    st.title('Sentiment Analytics Dashboard')
    selected = st.multiselect(
        'Filter by app:',
        df['app_name'].unique().tolist(),
        default=df['app_name'].unique().tolist(),
    )
    dff = df[df['app_name'].isin(selected)]
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Reviews', f'{len(dff):,}')
    c2.metric('Positive', f"{(dff['sentiment'] == 'positive').sum():,}")
    c3.metric('Negative', f"{(dff['sentiment'] == 'negative').sum():,}")
    col1, col2 = st.columns(2)
    with col1:
        sc = dff['sentiment'].value_counts().reset_index()
        sc.columns = ['sentiment', 'count']
        st.plotly_chart(
            px.pie(
                sc,
                values='count',
                names='sentiment',
                title='Overall Sentiment',
                color='sentiment',
                color_discrete_map=COLORS,
                hole=0.4,
            ),
            use_container_width=True,
        )
    with col2:
        aps = dff.groupby(['app_name', 'sentiment']).size().reset_index(name='count')
        fig2 = px.bar(
            aps,
            x='app_name',
            y='count',
            color='sentiment',
            title='Sentiment by Bank',
            color_discrete_map=COLORS,
            barmode='group',
        )
        fig2.update_xaxes(tickangle=20)
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader('Recent Reviews')
    st.dataframe(
        dff[['app_name', 'content', 'score', 'sentiment', 'date']]
        .sort_values('date', ascending=False)
        .head(20),
        use_container_width=True,
    )

elif page == 'Model Info':
    st.title('Model Information')
    st.markdown('### Why Binary Classification?')
    st.warning(
        'This project originally attempted 3-class classification (Positive / Neutral / Negative). '
        'The neutral class consistently failed despite applying label refinement, SMOTE, and '
        'class_weight balancing. The root cause was structural: 3-star reviews have inherently '
        'contradictory labels (same star rating, opposite text sentiment), and at only 4.9% of '
        'the dataset the class had insufficient clean signal to learn from. '
        'Binary classification removes this ambiguity and covers 95% of the data with reliable labels.'
    )
    if 'design_decision' in meta:
        st.info(meta['design_decision'])
    st.json(meta)
