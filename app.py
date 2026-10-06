"""KDD Dashboard - Math Problem Topic Classifier
Run:  streamlit run app.py   (ML_PROJECT.csv must be in the same folder)
"""
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, adjusted_rand_score)

st.set_page_config(page_title="KDD Math Problem Classifier", page_icon="📊", layout="wide")
DATA_FILE = "ML_PROJECT.csv"


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"\$+", " ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop_duplicates(subset="problem").dropna(subset=["problem", "solution"])
    df = df[df["level"] != "Level ?"]
    df["problem_clean"] = df["problem"].apply(clean_text)
    return df


@st.cache_resource(show_spinner="Training 4 models (first run takes about a minute)...")
def train_all():
    df = load_data()
    data = df.dropna(subset=["type"]).copy()
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=20000)
    X = tfidf.fit_transform(data["problem_clean"])
    y = data["type"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
        "Linear SVM": LinearSVC(C=0.5),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[name] = p
        rows.append([name, accuracy_score(y_te, p),
                     precision_score(y_te, p, average="macro"),
                     recall_score(y_te, p, average="macro"),
                     f1_score(y_te, p, average="macro")])
    results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]) \
        .sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return dict(data=data, tfidf=tfidf, X=X, y=y, y_te=y_te, models=models,
                preds=preds, results=results)


@st.cache_resource(show_spinner="Clustering...")
def run_clusters():
    T = train_all()
    km = KMeans(n_clusters=7, n_init=10, random_state=42)
    clusters = km.fit_predict(T["X"])
    ari = adjusted_rand_score(T["y"], clusters)
    table = pd.crosstab(pd.Series(clusters, name="cluster"), T["y"].reset_index(drop=True))
    return ari, table


T = train_all()
data, results = T["data"], T["results"]

st.sidebar.title("📊 KDD Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Model Comparison", "Predict a Topic",
                                  "Top Words per Topic", "Clustering"])
st.sidebar.caption("Dataset: MATH problems with topic labels. Models predict the topic from the problem text.")

if page == "Overview":
    st.title("Math Problem Topic Classification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labelled problems", f"{len(data):,}")
    c2.metric("Topics", data["type"].nunique())
    c3.metric("Best model", results.loc[0, "Model"])
    c4.metric("Best accuracy", f"{results.loc[0, 'Accuracy']:.1%}")
    a, b = st.columns(2)
    a.subheader("Problems per topic")
    a.bar_chart(data["type"].value_counts())
    b.subheader("Problems per difficulty level")
    b.bar_chart(data["level"].value_counts().sort_index())
    st.subheader("Sample data")
    st.dataframe(data[["problem", "type", "level"]].sample(10, random_state=1), use_container_width=True)

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.dataframe(results.style.format({c: "{:.3f}" for c in results.columns[1:]}), use_container_width=True)
    st.bar_chart(results.set_index("Model")[["Accuracy", "F1"]])
    name = st.selectbox("Inspect a model", results["Model"])
    pred, y_te = T["preds"][name], T["y_te"]
    labels = sorted(y_te.unique())
    a, b = st.columns(2)
    a.subheader("Per-topic report")
    a.dataframe(pd.DataFrame(classification_report(y_te, pred, output_dict=True)).T.round(3))
    b.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_te, pred, labels=labels), annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    b.pyplot(fig)

elif page == "Predict a Topic":
    st.title("Predict the Topic of a Problem")
    name = st.selectbox("Model", list(T["models"]), index=0)
    text = st.text_area("Type or paste a math problem", height=150,
                        value="What is the probability of rolling a sum of 7 with two fair dice?")
    if st.button("Predict") and text.strip():
        vec = T["tfidf"].transform([clean_text(text)])
        model = T["models"][name]
        st.success(f"Predicted topic: **{model.predict(vec)[0]}**")
        if hasattr(model, "predict_proba"):
            st.bar_chart(pd.Series(model.predict_proba(vec)[0], index=model.classes_, name="probability"))

elif page == "Top Words per Topic":
    st.title("Words That Signal Each Topic")
    st.caption("Highest-weight words from the Logistic Regression model.")
    lr, terms = T["models"]["Logistic Regression"], T["tfidf"].get_feature_names_out()
    topic = st.selectbox("Topic", lr.classes_)
    i = list(lr.classes_).index(topic)
    idx = lr.coef_[i].argsort()[::-1][:15]
    st.bar_chart(pd.Series(lr.coef_[i][idx], index=terms[idx], name="weight"))

else:
    st.title("Clustering (KMeans, k = 7)")
    ari, table = run_clusters()
    st.metric("Adjusted Rand Index", f"{ari:.3f}")
    st.caption("1.0 = clusters match the real topics perfectly, 0 = random. "
               "A low score means the clusters do not line up with the topics.")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    st.pyplot(fig)
