"""
AI Lab Toolkit - one app that combines BFS/DFS, a rule-based expert system,
classic ML models, K-Means, and ANN/CNN training on MNIST.

Install:  pip install streamlit scikit-learn pandas numpy
          pip install tensorflow        (only needed for the Deep Learning tab)
Run:      streamlit run ai_lab_toolkit.py
"""
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AI Lab Toolkit", page_icon="🧠", layout="wide")


# ----------------------------------------------------------------- Graph search
def parse_graph(text):
    graph = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        node, neighbours = line.split(":", 1)
        node = node.strip()
        graph.setdefault(node, [])
        for n in (x.strip() for x in neighbours.split(",")):
            if n:
                graph[node].append(n)
                graph.setdefault(n, [])
    return graph


def bfs(graph, start):
    visited, queue, steps = [], deque([start]), []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            queue.extend(graph[node])
            steps.append({"Visit": node, "Frontier after": " ".join(queue) or "-"})
    return visited, steps


def dfs(graph, start):
    visited, stack, steps = [], [start], []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            stack.extend(reversed(graph[node]))  # keeps left-to-right order
            steps.append({"Visit": node, "Frontier after": " ".join(reversed(stack)) or "-"})
    return visited, steps


def graph_dot(graph, order):
    rank = {n: i + 1 for i, n in enumerate(order)}
    lines = ["digraph G { rankdir=TB; node [shape=circle, style=filled, fillcolor=white];"]
    for n in graph:
        if n in rank:
            lines.append(f'"{n}" [label="{n}\\n#{rank[n]}", fillcolor="#cfe8ff"];')
    for n, nbrs in graph.items():
        for m in nbrs:
            lines.append(f'"{n}" -> "{m}";')
    lines.append("}")
    return "\n".join(lines)


def tab_graph():
    st.subheader("Graph Explorer (BFS / DFS)")
    st.caption("One line per node: `A: B, C` means A connects to B and C.")
    c1, c2 = st.columns([1, 2])
    with c1:
        text = st.text_area("Graph", "A: B, C\nB: D, E\nC: F\nD:\nE:\nF:", height=200)
        graph = parse_graph(text)
        if not graph:
            st.warning("Enter at least one node.")
            return
        start = st.selectbox("Start node", list(graph))
        algo = st.radio("Algorithm", ["BFS", "DFS"], horizontal=True)
        goal = st.selectbox("Goal node (optional)", ["(none)"] + list(graph))
    order, steps = (bfs if algo == "BFS" else dfs)(graph, start)
    with c2:
        st.graphviz_chart(graph_dot(graph, order))
    st.success(f"{algo} order: " + " → ".join(order))
    if goal != "(none)":
        if goal in order:
            st.info(f"Goal '{goal}' reached after visiting {order.index(goal) + 1} node(s).")
        else:
            st.error(f"Goal '{goal}' is not reachable from '{start}'.")
    st.dataframe(pd.DataFrame(steps), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- Expert system
RULES = [
    ("Flu", lambda s: s["fever"] and s["cough"], "Fever AND cough"),
    ("Common Cold", lambda s: s["cold"] and s["cough"], "Cold AND cough"),
    ("Infection", lambda s: s["fever"], "Fever"),
]


def tab_expert():
    st.subheader("Rule-Based Symptom Checker")
    st.caption("Educational demo only - not medical advice.")
    c1, c2 = st.columns(2)
    with c1:
        s = {
            "fever": st.checkbox("Fever"),
            "cough": st.checkbox("Cough"),
            "cold": st.checkbox("Cold / runny nose"),
        }
    with c2:
        for label, cond, why in RULES:
            if cond(s):
                st.success(f"Result: **{label}**  \nRule fired: {why}")
                break
        else:
            st.info("Result: No clear diagnosis")
        with st.expander("Rule base"):
            for i, (label, _, why) in enumerate(RULES, 1):
                st.write(f"{i}. IF {why} THEN {label}")
            st.write(f"{len(RULES) + 1}. ELSE No clear diagnosis")


# ------------------------------------------------------------------- ML models
BUILTIN = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}


def get_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
        "SVM": make_pipeline(StandardScaler(), SVC()),
    }


def load_dataset():
    src = st.radio("Dataset", ["Built-in", "Upload CSV"], horizontal=True)
    if src == "Built-in":
        name = st.selectbox("Choose", list(BUILTIN))
        d = BUILTIN[name]()
        X = pd.DataFrame(d.data, columns=d.feature_names)
        y = pd.Series(d.target).map(dict(enumerate(d.target_names)))
        return X, y
    up = st.file_uploader("CSV file", type="csv")
    if up is None:
        return None, None
    df = pd.read_csv(up).dropna()
    target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    X = pd.get_dummies(df.drop(columns=[target]))
    return X, df[target].astype(str)


def tab_ml():
    st.subheader("Classifier Lab")
    X, y = load_dataset()
    if X is None:
        st.info("Upload a CSV to begin.")
        return
    c1, c2 = st.columns(2)
    chosen = c1.multiselect("Models", list(get_models()), default=list(get_models()))
    test_size = c2.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    if not chosen:
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42,
                                          stratify=y if y.value_counts().min() > 1 else None)
    rows, fitted = [], {}
    for name in chosen:
        m = get_models()[name].fit(Xtr, ytr)
        p = m.predict(Xte)
        fitted[name] = (m, p)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p, average="weighted", zero_division=0),
                     "Recall": recall_score(yte, p, average="weighted", zero_division=0),
                     "F1": f1_score(yte, p, average="weighted", zero_division=0)})
    res = pd.DataFrame(rows).set_index("Model")
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8ff"),
                 use_container_width=True)
    st.bar_chart(res)
    best = res["F1"].idxmax()
    st.success(f"Best model by F1: **{best}**")

    inspect = st.selectbox("Inspect model", chosen, index=chosen.index(best))
    model, pred = fitted[inspect]
    labels = sorted(y.unique())
    st.write("Confusion matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(yte, pred, labels=labels),
                              index=[f"true {l}" for l in labels],
                              columns=[f"pred {l}" for l in labels]))
    with st.expander("Try a prediction"):
        vals = {c: st.number_input(c, value=float(X[c].mean())) for c in X.columns[:12]}
        row = X.mean().to_dict()
        row.update(vals)
        st.write("Prediction:", model.predict(pd.DataFrame([row])[X.columns])[0])
        if len(X.columns) > 12:
            st.caption("Only the first 12 features are editable; others use their mean.")


# ------------------------------------------------------------------- Clustering
def tab_cluster():
    st.subheader("K-Means Clustering")
    src = st.radio("Data", ["Sample (Iris, no labels used)", "Upload CSV"], horizontal=True)
    if src.startswith("Sample"):
        df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
    else:
        up = st.file_uploader("CSV with numeric columns", type="csv", key="km")
        if up is None:
            return
        df = pd.read_csv(up).select_dtypes("number").dropna()
        if df.shape[1] < 1:
            st.error("No numeric columns found.")
            return
    k = st.slider("Clusters (k)", 2, 10, 3)
    Xs = StandardScaler().fit_transform(df)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    c1, c2 = st.columns(2)
    if Xs.shape[1] >= 2:
        pts = PCA(n_components=2).fit_transform(Xs)
    else:
        pts = np.c_[Xs[:, 0], np.zeros(len(Xs))]
    plot = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "cluster": km.labels_.astype(str)})
    c1.scatter_chart(plot, x="x", y="y", color="cluster")
    out = df.assign(cluster=km.labels_)
    c2.dataframe(out.groupby("cluster").mean().round(2).assign(size=out.groupby("cluster").size()))
    inertias = [KMeans(n_clusters=i, random_state=0, n_init=10).fit(Xs).inertia_ for i in range(1, 11)]
    st.write("Elbow curve (look for the bend)")
    st.line_chart(pd.DataFrame({"inertia": inertias}, index=range(1, 11)))
    st.download_button("Download clustered data", out.to_csv(index=False), "clusters.csv")


# ---------------------------------------------------------------- Deep learning
@st.cache_resource(show_spinner=False)
def load_mnist():
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    return (xtr / 255.0)[..., None], ytr, (xte / 255.0)[..., None], yte


def build(kind):
    from tensorflow.keras import layers, models
    if kind == "ANN (dense)":
        m = models.Sequential([layers.Input((28, 28, 1)), layers.Flatten(),
                               layers.Dense(128, activation="relu"),
                               layers.Dense(64, activation="relu"),
                               layers.Dense(10, activation="softmax")])
    else:
        m = models.Sequential([layers.Input((28, 28, 1)),
                               layers.Conv2D(32, 3, activation="relu"),
                               layers.MaxPooling2D(2), layers.Flatten(),
                               layers.Dense(64, activation="relu"),
                               layers.Dense(10, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def tab_dl():
    st.subheader("Digit Recognition: ANN vs CNN (MNIST)")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        st.error("TensorFlow is not installed. Run `pip install tensorflow` and restart.")
        return
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Model", ["ANN (dense)", "CNN"])
    epochs = c2.slider("Epochs", 1, 10, 3)
    n = c3.select_slider("Training samples", [5000, 10000, 30000, 60000], value=10000)
    if st.button("Train", type="primary"):
        with st.spinner("Loading data and training..."):
            xtr, ytr, xte, yte = load_mnist()
            model = build(kind)
            hist = model.fit(xtr[:n], ytr[:n], epochs=epochs, validation_split=0.1, verbose=0)
            loss, acc = model.evaluate(xte, yte, verbose=0)
        st.session_state["dl"] = (model, hist.history, acc, kind)
    if "dl" in st.session_state:
        model, h, acc, kname = st.session_state["dl"]
        st.metric(f"{kname} test accuracy", f"{acc:.2%}")
        st.line_chart(pd.DataFrame({"train acc": h["accuracy"], "val acc": h["val_accuracy"]}))
        _, _, xte, yte = load_mnist()
        if st.button("Predict a random test digit"):
            i = np.random.randint(len(xte))
            probs = model.predict(xte[i:i + 1], verbose=0)[0]
            a, b = st.columns(2)
            a.image(xte[i, :, :, 0], width=200, clamp=True)
            b.write(f"Predicted **{probs.argmax()}** (actual {yte[i]})")
            b.bar_chart(pd.Series(probs, index=range(10)))


# -------------------------------------------------------------------------- Main
st.title("🧠 AI Lab Toolkit")
tabs = st.tabs(["Graph Search", "Expert System", "Classifier Lab", "Clustering", "Deep Learning"])
for tab, fn in zip(tabs, [tab_graph, tab_expert, tab_ml, tab_cluster, tab_dl]):
    with tab:
        fn()