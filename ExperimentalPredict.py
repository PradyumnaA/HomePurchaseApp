# app.py — Voice form for all columns except 'y' → random fallbacks → predict → speak yes/no
# Run: streamlit run app.py

import io
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from gtts import gTTS
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ------------------ Paths ------------------
ART_DIR  = "/run/media/pradyumnakubear/easystore/KaggleBankingProjectPython"
PRE_PATH = os.path.join(ART_DIR, "preprocessor.pkl")
MOD_PATH = os.path.join(ART_DIR, "model.keras")
META_PATH= os.path.join(ART_DIR, "meta.json")

# ------------------ UI ------------------
st.set_page_config(page_title="🗣️ Voice Form — Bank Predictor", page_icon="🗣️", layout="centered")

# ---- Global CSS: force light/white background + clean cards/buttons/typography ----
st.markdown("""
<style>
/* Force light theme feel */
:root, .stApp, .stApp [data-testid="stAppViewContainer"], .main, body {
  background: #ffffff !important;
  color: #111827 !important; /* slate-900 */
}

/* Centered, narrower content with breathing room */
.block-container {
  max-width: 880px !important;
  padding-top: 1.25rem !important;
}

/* Headings */
h1, h2, h3 {
  letter-spacing: .2px;
}

/* Streamlit widget polish */
.stButton > button {
  border-radius: 12px;
  padding: .7rem 1rem;
  font-weight: 600;
  border: 1px solid #e5e7eb;
  background: #111827;
  color: white;
}
.stButton > button:hover { filter: brightness(1.05); }

/* Cards */
.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 2px 10px rgb(0 0 0 / 4%);
}

/* Status + log box in component */
.voice-status {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  margin: .5rem 0 0.6rem;
  font-size: 1.05rem;
}
.voice-log {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  padding: .75rem;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: .5rem;
  max-height: 260px;
  overflow: auto;
  white-space: pre-wrap;
}
.voice-summary {
  display:none;
  margin-top: 12px;
  padding: .75rem;
  background: #fafafa;
  border: 1px dashed #d1d5db;
  border-radius: .5rem;
}
.small-cap { color:#6b7280; } /* slate-500 */
.key { color:#374151; font-weight:600; }
.val { color:#111827; }
</style>
""", unsafe_allow_html=True)

st.title("🗣️ Voice Form — Bank Dataset → Yes/No")

with st.expander("Health check", expanded=False):
    checks = {
        "preprocessor.pkl": os.path.isfile(PRE_PATH),
        "model.keras": os.path.isfile(MOD_PATH),
        "meta.json": os.path.isfile(META_PATH),
    }
    for k, ok in checks.items():
        st.write(f"{'✅' if ok else '❌'} {k}")
    st.caption(f"ART_DIR = {ART_DIR}")

if not (os.path.isfile(PRE_PATH) and os.path.isfile(MOD_PATH) and os.path.isfile(META_PATH)):
    st.error("Missing model artifacts. Ensure preprocessor.pkl, model.keras, meta.json are in ART_DIR.")
    st.stop()

# ------------------ Load artifacts ------------------
@st.cache_resource
def load_artifacts():
    pre = joblib.load(PRE_PATH)
    model = keras.models.load_model(MOD_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pre, model, meta

preprocessor, nn, meta = load_artifacts()
cat_cols  = meta.get("categorical_cols", [])
num_cols  = meta.get("numerical_cols", [])
threshold = float(meta.get("threshold", 0.5))

# We purposefully do NOT include 'y'
ordered_cols = [*cat_cols, *num_cols]

def get_cat_options():
    try:
        tx = preprocessor.named_transformers_["cat"]
        if hasattr(tx, "categories_"):
            return {c: list(opts) for c, opts in zip(cat_cols, tx.categories_)}
    except Exception:
        pass
    return {c: [] for c in cat_cols}

cat_options = get_cat_options()

# Friendly prompts + numeric ranges for common bank columns
PROMPTS = {
    "job":        "What is the job role? (e.g., technician, admin, management)",
    "age":        "What is the age?",
    "marital":    "What is the marital status? (single, married, divorced)",
    "education":  "What is the education level? (primary, secondary, tertiary, unknown)",
    "default":    "Any credit default? (yes or no)",
    "balance":    "What is the average account balance?",
    "housing":    "Any housing loan? (yes or no)",
    "loan":       "Any personal loan? (yes or no)",
    "contact":    "Preferred contact method? (cellular or telephone)",
    "day":        "What is the last contact day of month? (1-31)",
    "month":      "What is the last contact month? (jan, feb, ...)",
    # duration is special-cased in JS (seconds first, then minutes fallback)
    "duration":   "What was the last contact duration in seconds?",
    "campaign":   "How many contacts during this campaign?",
    "pdays":      "How many previous days since the last contact in a previous campaign? Say 999 if never.",
    "previous":   "How many contacts before this campaign?",
    "poutcome":   "Outcome of previous campaign? (success, failure, other, unknown)",
}
GENERIC_CAT = lambda f: f"Please provide a value for {f}."
GENERIC_NUM = lambda f: f"Please provide a numeric value for {f}."

# Heuristic numeric ranges (used for random fallback)
NUM_RANGES = {
    "age":       (10, 120),
    "balance":   (-100000, 100000),
    "day":       (1, 31),
    "duration":  (0, 10000),
    "campaign":  (0, 100),
    "pdays":     (0, 999),
    "previous":  (0, 100),
}
DEFAULT_NUM_RANGE = (0, 1000)

# Pretty labels for the summary panel (so it shows “previous days” instead of “pdays”)
DISPLAY_LABEL = {
    "pdays": "previous days",
    "duration": "duration (seconds)",
}

def to_dense_float32(X):
    if hasattr(X, "toarray"): X = X.toarray()
    return X.astype(np.float32)

st.caption(f"Decision threshold: **{threshold:.2f}**")

# Build a schema for the browser (JS) so it knows types/options/prompts/ranges
schema = []
for c in ordered_cols:
    if c in cat_cols:
        opts = cat_options.get(c, [])
        schema.append({
            "key": c,
            "type": "cat",
            "prompt": PROMPTS.get(c, GENERIC_CAT(c)),
            "options": opts
        })
    else:
        lo, hi = NUM_RANGES.get(c, DEFAULT_NUM_RANGE)
        schema.append({
            "key": c,
            "type": "num",
            "prompt": PROMPTS.get(c, GENERIC_NUM(c)),
            "min": lo,
            "max": hi
        })

# ---- Primary CTA inside a subtle card ----
st.markdown('<div class="card">', unsafe_allow_html=True)
start = st.button("▶️ Start voice form")
st.caption("Tip: Chrome works best. Allow microphone + sound. Say “don’t know” or “skip” to auto-pick a random value.")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Voice interaction (browser) with random fallback ------------------
html_template = """
<div class="card">
  <div class="voice-status" id="status">Ready.</div>
  <div class="voice-log" id="log"></div>
  <div class="voice-summary" id="summary"></div>
  <div class="small-cap">Voice is synthesized via your browser (SpeechSynthesis). Recognition uses the Web Speech API.</div>
</div>

<script>
(async () => {
  const status = document.getElementById("status");
  const log = document.getElementById("log");
  const summary = document.getElementById("summary");
  const schema = %%SCHEMA%%;

  function say(t) {
    return new Promise((resolve) => {
      try {
        const u = new SpeechSynthesisUtterance(t);
        u.rate = 1.0; u.pitch = 1.0;
        u.onend = resolve; window.speechSynthesis.speak(u);
      } catch(e) { resolve(); }
    });
  }

  function listenOnce(timeoutMs=15000) {
    return new Promise((resolve, reject) => {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) return reject(new Error("SpeechRecognition API not supported."));
      const r = new SR();
      r.lang = "en-US";
      r.interimResults = false;
      r.maxAlternatives = 1;
      const to = setTimeout(() => { try{r.abort();}catch(_){ } reject(new Error("timeout")); }, timeoutMs);
      r.onresult = (e) => { clearTimeout(to); resolve((e.results[0][0].transcript || "").trim()); };
      r.onerror = (e) => { clearTimeout(to); reject(new Error(e.error || "recog error")); };
      try { r.start(); } catch (e) { clearTimeout(to); reject(e); }
    });
  }

  function addLog(line) {
    log.textContent += line + "\\n";
    log.scrollTop = log.scrollHeight;
  }

  // helpers
  function parseNumber(text) {
    const m = (text||"").replace(/[,]/g,"").match(/-?\\d+(?:\\.\\d+)?/);
    return m ? Number(m[0]) : null;
  }
  function clampInt(v, lo, hi) {
    if (v == null || isNaN(v)) return null;
    v = Math.round(v);
    if (v < lo || v > hi) return null;
    return v;
  }
  function pickFromList(text, list) {
    if (!list || list.length === 0) return null;
    const t = (text||"").toLowerCase();
    for (const v of list) if (t.includes(String(v).toLowerCase())) return v;
    const s = t.replace(/[^a-z0-9]/g,"");
    for (const v of list) {
      const vs = String(v).toLowerCase().replace(/[^a-z0-9]/g,"");
      if (s.includes(vs)) return v;
    }
    return null;
  }
  function userDoesntKnow(s) {
    const t = (s||"").toLowerCase();
    return t.includes("dont know") || t.includes("don't know") || t.includes("no idea") || t.includes("skip") || t.includes("not sure") || t.includes("idk");
  }

  // Random fallbacks (equal probability for categories)
  function randInt(lo, hi) {
    return Math.floor(lo + Math.random() * (hi - lo + 1));
  }
  function fallbackFor(field) {
    if (field.type === "cat") {
      const opts = field.options || [];
      if (opts.length) {
        const i = Math.floor(Math.random() * opts.length);
        return opts[i];
      }
      return "unknown";
    } else {
      let lo = Number.isFinite(field.min) ? field.min : 0;
      let hi = Number.isFinite(field.max) ? field.max : 1000;
      if (lo > hi) { const t = lo; lo = hi; hi = t; }
      return randInt(lo, hi);
    }
  }

  function dispKey(k) {
    if (k === "pdays") return "previous days";
    if (k === "duration") return "duration (seconds)";
    return k;
  }

  function addSummary(answers) {
    const lines = Object.entries(answers).map(([k,v]) =>
      "<span class='key'>" + dispKey(k) + "</span>: <span class='val'>" + v + "</span>"
    );
    summary.style.display = "block";
    summary.innerHTML = "<b>Collected:</b><br>" + lines.join("<br>");
  }

  // Special handler for duration:
  async function askDuration(field) {
    // try seconds
    for (let tries = 0; tries < 2; tries++) {
      const prompt = field.prompt || "Provide duration in seconds.";
      status.textContent = prompt;
      await say(prompt);
      try {
        const heard = await listenOnce();
        addLog("Heard (duration): " + heard);

        if (userDoesntKnow(heard)) break; // go to minutes path

        const n = parseNumber(heard);
        const v = clampInt(n, Number.isFinite(field.min)?field.min:0, Number.isFinite(field.max)?field.max:1000000);
        if (v != null) {
          status.textContent = "Duration noted.";
          await say("Duration noted.");
          return v;
        }
        await say("Sorry, I didn't get that.");
      } catch (e) {
        addLog("Recognition error: " + e.message);
        await say("Please repeat.");
      }
    }

    // ask minutes
    for (let tries = 0; tries < 2; tries++) {
      const promptMin = "How many minutes was the conversation?";
      status.textContent = promptMin;
      await say(promptMin);
      try {
        const heard = await listenOnce();
        addLog("Heard (minutes): " + heard);

        if (userDoesntKnow(heard)) break;

        const nMin = parseNumber(heard);
        const minutes = clampInt(nMin, 0, 1000000);
        if (minutes != null) {
          const seconds = minutes * 60;
          status.textContent = "Duration noted.";
          await say("Duration noted.");
          return seconds;
        }
        await say("Sorry, I didn't get that.");
      } catch (e) {
        addLog("Recognition error: " + e.message);
        await say("Please repeat.");
      }
    }

    // fallback
    const fbVal = fallbackFor(field);
    status.textContent = "I'll assume a duration.";
    await say("I'll assume a duration.");
    return fbVal;
  }

  async function askAndGet(field) {
    if (field.key === "duration") {
      return await askDuration(field);
    }

    const prompt = field.prompt || ("Provide value for " + field.key);
    for (let tries=0; tries<3; tries++) {
      status.textContent = prompt;
      await say(prompt);
      try {
        const heard = await listenOnce();
        addLog("Heard: " + heard);

        if (userDoesntKnow(heard)) {
          const fbVal = fallbackFor(field);
          const conf = `No problem. I'll choose ${fbVal} for ${field.key === "pdays" ? "previous days" : field.key}.`;
          status.textContent = conf; await say(conf);
          return fbVal;
        }

        if (field.type === "cat") {
          const val = pickFromList(heard, field.options||[]);
          if (val != null) {
            const conf = `${field.key === "pdays" ? "previous days" : field.key} ${val}.`;
            status.textContent = conf; await say(conf);
            return val;
          }
        } else {
          const n = parseNumber(heard);
          const v = clampInt(n, Number.isFinite(field.min)?field.min:0, Number.isFinite(field.max)?field.max:1000000);
          if (v != null) {
            const label = field.key === "pdays" ? "previous days" : field.key;
            const conf = `${label} ${v}.`;
            status.textContent = conf; await say(conf);
            return v;
          }
        }
        await say("Sorry, I didn't get that.");
      } catch (e) {
        addLog("Recognition error: " + e.message);
        await say("Please repeat.");
      }
    }
    const fbVal = fallbackFor(field);
    const conf = `I'll assume ${fbVal} for ${field.key === "pdays" ? "previous days" : field.key}.`;
    status.textContent = conf; await say(conf);
    return fbVal;
  }

  async function flow() {
    status.textContent = "Starting voice form…";
    await say("Let's fill your details by voice.");
    const answers = {};

    for (const f of schema) {
      const v = await askAndGet(f);
      answers[f.key] = v;
    }

    addSummary(answers);
    status.textContent = "Review your details. Sending to predictor…";
    await say("Sending to predictor.");

    window.parent.postMessage({
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: answers
    }, "*");
  }

  %%RUN%%
})();
</script>
"""

html = (
    html_template
    .replace("%%SCHEMA%%", json.dumps(schema))
    .replace("%%RUN%%", "flow();" if start else "")
)

comp_val = st.components.v1.html(html, height=560)

# ------------------ Predict once answers arrive ------------------
result_box = st.empty()

if comp_val:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    answers = comp_val if isinstance(comp_val, dict) else {}
    st.subheader("Collected (from voice)")
    pretty_answers = {DISPLAY_LABEL.get(k, k): v for k, v in answers.items()}
    st.write(pretty_answers)
    st.markdown('</div>', unsafe_allow_html=True)

    # Build single-row DataFrame in expected column order (actual keys unchanged)
    row = {c: answers.get(c, None) for c in ordered_cols}
    df = pd.DataFrame([row])[ordered_cols]

    try:
        X = preprocessor.transform(df)
        if hasattr(X, "toarray"): X = X.toarray()
        X = X.astype(np.float32)
        p = float(nn.predict(X, verbose=0).ravel()[0])
        yhat = int(p >= threshold)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")
        st.write(f"**Probability**: {p:.6f}")
        st.write(f"**Prediction (1=yes, 0=no)**: **{yhat}**")
        st.markdown('</div>', unsafe_allow_html=True)

        # Speak yes/no with gTTS
        spoken = "yes" if yhat == 1 else "no"
        mp3 = io.BytesIO()
        gTTS(text=f"The prediction is {spoken}. for purchasing home", lang="en").write_to_fp(mp3)
        mp3.seek(0)
        st.audio(mp3.read(), format="audio/mp3")

        result_box.success(f"{'yes (1)' if yhat==1 else 'no (0)'} — prediction complete. for home purchase")
    except Exception as e:
        result_box.error(f"Prediction failed: {e}")
