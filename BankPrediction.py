# ==============================
# Binary Classification (Keras)
# - Robust train/val/test split
# - OneHot + Scale
# - Class weights
# - EarlyStopping + LR schedule
# - Threshold tuning (no ROC curve plot)
# - Training curves (loss/auc/accuracy)
# - Save artifacts for download (.pkl/.keras/.json + .zip)
# ==============================

import os, json, zipfile, joblib
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, f1_score
)
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import *
from tensorflow.keras import *

# --------------------
# Reproducibility
# --------------------
tf.random.set_seed(42)
np.random.seed(42)

# --------------------
# Load & prepare data
# --------------------
dataset = pd.read_csv('train.csv').dropna()

X = dataset.drop(columns=['id', 'y'])
y = dataset['y']

# Map y to 0/1 if it's text like 'yes'/'no'
if y.dtype == 'object':
    y = y.astype(str).str.lower().map({'yes': 1, 'no': 0})
    if y.isna().any():
        bad_vals = dataset.loc[y.isna(), 'y'].unique()
        raise ValueError(f"Unexpected y values: {bad_vals}. Map them to 0/1.")
y = y.astype(int)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
    ],
    remainder='drop'
)

# --------------------
# Split: train / test, then train / val
# --------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.10, random_state=70, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=70, stratify=y_train_full
)

# --------------------
# Fit preprocessor on TRAIN only; transform all
# --------------------
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

# OHE can be sparse; Keras wants dense
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_val_proc   = X_val_proc.toarray()
    X_test_proc  = X_test_proc.toarray()

# Save memory and speed up training
X_train_proc = X_train_proc.astype(np.float32)
X_val_proc   = X_val_proc.astype(np.float32)
X_test_proc  = X_test_proc.astype(np.float32)

input_dim = X_train_proc.shape[1]

# --------------------
# Build NN
# --------------------
def build_model(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.30),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.30),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )
    return model

nn = build_model(input_dim)

# --------------------
# Optional: handle class imbalance
# --------------------
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: float(cw[0]), 1: float(cw[1])}

# --------------------
# Callbacks
# --------------------
es = keras.callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=3, restore_best_weights=True, verbose=1
)
rlr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-5, verbose=1
)

# --------------------
# Train
# --------------------
history = nn.fit(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    epochs=100,
    batch_size=512,
    callbacks=[es, rlr],
    verbose=1,
    class_weight=class_weights  # remove if not desired
)

# --------------------
# Tune threshold on validation (0.1..0.9)
# --------------------
val_prob = nn.predict(X_val_proc, verbose=0).ravel()
candidates = np.linspace(0.1, 0.9, 33)

accs, f1s = [], []
for t in candidates:
    pred = (val_prob >= t).astype(int)
    accs.append(accuracy_score(y_val, pred))
    f1s.append(f1_score(y_val, pred))

best_acc_t = float(candidates[int(np.argmax(accs))])
best_f1_t  = float(candidates[int(np.argmax(f1s))])

print(f"\nBest accuracy threshold on VAL: {best_acc_t:.3f} | acc={max(accs):.4f}")
print(f"Best F1 threshold on VAL:       {best_f1_t:.3f} | f1={max(f1s):.4f}")

# Choose which to use for TEST (pick accuracy-oriented by default)
chosen_t = best_acc_t

# --------------------
# Predict & Evaluate on TEST
# --------------------
test_prob = nn.predict(X_test_proc, verbose=0).ravel()
y_pred_class = (test_prob >= chosen_t).astype(int)

print("\nConfusion Matrix (TEST):\n", confusion_matrix(y_test, y_pred_class))
print("\nClassification Report (TEST):\n", classification_report(y_test, y_pred_class, digits=4))
print("ROC AUC (TEST):", roc_auc_score(y_test, test_prob))

# --------------------
# Plots: Training curves (no ROC curve plot)
# --------------------
def plot_history(metric):
    plt.figure(figsize=(7,4))
    plt.plot(history.history.get(metric, []), label=metric)
    val_key = 'val_' + metric
    if val_key in history.history:
        plt.plot(history.history[val_key], label=val_key)
    plt.title(f'Training history: {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

plot_history('loss')
plot_history('auc')
plot_history('accuracy')

# --------------------
# SAVE & DOWNLOAD ARTIFACTS (for Kaggle)
# --------------------
OUT_DIR = "/run/media/pradyumnakubear/easystore/KaggleBankingProjectPython"
PREPROC_PATH = os.path.join(OUT_DIR, "preprocessor.pkl")
MODEL_PATH   = os.path.join(OUT_DIR, "model.keras")
META_PATH    = os.path.join(OUT_DIR, "meta.json")
BUNDLE_PATH  = os.path.join(OUT_DIR, "model_bundle.zip")

# 1) Save preprocessing pipeline
joblib.dump(preprocessor, PREPROC_PATH, compress=3)

# 2) Save model (Keras-native format)
nn.save(MODEL_PATH)

# 3) Save metadata
meta = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "input_dim": int(input_dim),
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "class_weights": class_weights,
    "threshold": float(chosen_t),
    "val_auc_best": float(np.max(history.history.get("val_auc", [np.nan]))),
    "notes": "Binary classifier with OneHot+Scale, class weights, ES+ReduceLROnPlateau"
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

# 4) Zip everything for one-click download
with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(PREPROC_PATH, arcname=os.path.basename(PREPROC_PATH))
    zf.write(MODEL_PATH,   arcname=os.path.basename(MODEL_PATH))
    zf.write(META_PATH,    arcname=os.path.basename(META_PATH))

print(f"\nSaved artifacts to {OUT_DIR}:")
print(" -", PREPROC_PATH)
print(" -", MODEL_PATH)
print(" -", META_PATH)
print(" -", BUNDLE_PATH)

# (Optional) Show clickable links inside Kaggle notebook UI
try:
    from IPython.display import display, FileLink
    display(FileLink(PREPROC_PATH))
    display(FileLink(MODEL_PATH))
    display(FileLink(META_PATH))
    display(FileLink(BUNDLE_PATH))
except Exception as _e:
    pass