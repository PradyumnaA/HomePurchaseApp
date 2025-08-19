# app.py — minimal single-record predictor (prints 1 or 0)
import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
st.set_page_config(page_title="Bank Predictor", layout="centered")

ART_DIR = "/run/media/pradyumnakubear/easystore/KaggleBankingProjectPython"  # folder containing preprocessor.pkl, model.keras, meta.json

@st.cache_resource
def load_artifacts():
    pre = joblib.load(os.path.join(ART_DIR, "preprocessor.pkl"))
    model = keras.models.load_model(os.path.join(ART_DIR, "model.keras"))
    with open(os.path.join(ART_DIR, "meta.json")) as f:
        meta = json.load(f)
    return pre, model, meta

def to_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)

st.title("🔮 Enter values → get 1 (yes) or 0 (no)")

# Load once
try:
    preprocessor, nn, meta = load_artifacts()
except Exception as e:
    st.error(f"Could not load artifacts from '{ART_DIR}': {e}")
    st.stop()

cat_cols = meta.get("categorical_cols", [])
num_cols = meta.get("numerical_cols", [])
threshold = float(meta.get("threshold", 0.5))

# Try to discover categorical options from the saved OneHotEncoder
def get_cat_options():
    try:
        cat_tx = preprocessor.named_transformers_["cat"]
        if hasattr(cat_tx, "categories_"):
            return {c: list(opts) for c, opts in zip(cat_cols, cat_tx.categories_)}
    except Exception:
        pass
    return {c: [] for c in cat_cols}

cat_options = get_cat_options()

st.caption(f"Using decision threshold = **{threshold:.2f}**")

with st.form("single_input"):
    st.subheader("Provide feature values")

    # Categorical inputs
    for c in cat_cols:
        opts = cat_options.get(c, [])
        if opts:
            st.session_state.setdefault(f"cat_{c}", opts[0])
            val = st.selectbox(c, options=opts, key=f"cat_{c}")
        else:
            val = st.text_input(c, key=f"cat_{c}")
        st.session_state[c] = val

    # Numeric inputs
    for c in num_cols:
        st.session_state.setdefault(c, 0.0)
        val = st.number_input(c, value=float(st.session_state[c]), format="%.6f")
        st.session_state[c] = val

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Build single-row DataFrame in training order
        row = {c: st.session_state[c] for c in (cat_cols + num_cols)}
        df = pd.DataFrame([row])[cat_cols + num_cols]

        X = preprocessor.transform(df)
        X = to_dense_float32(X)
        p = float(nn.predict(X, verbose=0).ravel()[0])
        yhat = int(p >= threshold)

        # Output exactly as you asked
        st.markdown("### Result")
        st.write(f"**Probability**: {p:.6f}")
        st.write(f"**Prediction (1=yes, 0=no)**: **{yhat}**")
        st.success("yes (1)" if yhat == 1 else "no (0)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
