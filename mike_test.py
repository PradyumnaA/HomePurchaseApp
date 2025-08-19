# app.py — Voice Q&A "Hey Property" → predict → speak yes/no
# Works with: vosk-model-small-en-us-0.15
# Run: streamlit run app.py

import os, json, re, difflib, queue, tempfile, joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Optional imports (we degrade gracefully if missing)
WRTC_OK = True
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
    import av
except Exception as e:
    WRTC_OK = False
    WRTC_ERR = str(e)

VOSK_OK = True
try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception as e:
    VOSK_OK = False
    VOSK_ERR = str(e)

TTS_OK = True
try:
    import pyttsx3
except Exception as e:
    TTS_OK = False
    TTS_ERR = str(e)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
st.set_page_config(page_title="Hey Property — Voice Predictor", layout="centered")

# ---------- Paths ----------
ART_DIR  = "/run/media/pradyumnakubear/easystore/KaggleBankingProjectPython"
VOSK_DIR = os.path.join(ART_DIR, "vosk-model-small-en-us-0.15")

# ---------- Small utils ----------
def file_exists(p): return os.path.isfile(p)
def dir_exists(p): return os.path.isdir(p)

def speak(text: str):
    if not TTS_OK: return
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfp:
            path = tfp.name
        eng = pyttsx3.init()
        eng.setProperty("rate", 170)
        eng.save_to_file(text, path)
        eng.runAndWait()
        with open(path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
        os.unlink(path)
    except Exception as e:
        st.caption(f"TTS error: {e}")

def to_dense_float32(X):
    if hasattr(X, "toarray"): X = X.toarray()
    return X.astype(np.float32)

def fuzzy_pick(text: str, options: list[str]) -> str:
    if not text: return options[0] if options else ""
    if not options: return text
    for o in options:
        if o.lower() == text.lower(): return o
    match = difflib.get_close_matches(text, options, n=1, cutoff=0.0)
    return match[0] if match else options[0]

_num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def parse_num(text: str, default: float = 0.0) -> float:
    m = _num_pat.search(text or "")
    if not m: return default
    try: return float(m.group(0))
    except: return default

# ---------- Health check ----------
pre_path   = os.path.join(ART_DIR, "preprocessor.pkl")
model_path = os.path.join(ART_DIR, "model.keras")
meta_path  = os.path.join(ART_DIR, "meta.json")

st.title("🗣️ Hey Property — Voice Predictor")
with st.expander("🔎 Health check"):
    checks = {
        "preprocessor.pkl": file_exists(pre_path),
        "model.keras": file_exists(model_path),
        "meta.json": file_exists(meta_path),
        "vosk dir exists": dir_exists(VOSK_DIR),
        "vosk import": VOSK_OK,
        "streamlit-webrtc import": WRTC_OK,
        "pyttsx3 (TTS)": TTS_OK,
    }
    for k, ok in checks.items():
        st.write(f"{'✅' if ok else '❌'} {k}")
    if not VOSK_OK: st.code(VOSK_ERR)
    if not WRTC_OK: st.code(WRTC_ERR)
    if not TTS_OK: st.code(TTS_ERR)
    st.caption(f"ART_DIR = {ART_DIR}")
    st.caption(f"VOSK_DIR = {VOSK_DIR}")
if not (file_exists(pre_path) and file_exists(model_path) and file_exists(meta_path)):
    st.error("Missing model artifacts. Ensure preprocessor.pkl, model.keras, meta.json are in ART_DIR.")
    st.stop()

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    pre = joblib.load(pre_path)
    model = keras.models.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return pre, model, meta

preprocessor, nn, meta = load_artifacts()
cat_cols  = meta.get("categorical_cols", [])
num_cols  = meta.get("numerical_cols", [])
threshold = float(meta.get("threshold", 0.5))

def get_cat_options():
    try:
        tx = preprocessor.named_transformers_["cat"]
        if hasattr(tx, "categories_"):
            return {c: list(opts) for c, opts in zip(cat_cols, tx.categories_)}
    except Exception:
        pass
    return {c: [] for c in cat_cols}

cat_options = get_cat_options()

# Friendly prompts for common bank dataset fields
PROMPTS = {
    "job":        "Where is he working? (job role like technician, admin, etc.)",
    "age":        "What is his age?",
    "marital":    "What is his marital status? (single, married, divorced)",
    "education":  "What is his education level? (primary, secondary, tertiary)",
    "default":    "Does he have any credit default? (yes or no)",
    "balance":    "What is his average account balance?",
    "housing":    "Does he have a housing loan? (yes or no)",
    "loan":       "Does he have any personal loan? (yes or no)",
    "contact":    "Preferred contact method? (cellular or telephone)",
    "day":        "What is the last contact day of the month? (number)",
    "month":      "What is the last contact month? (jan, feb, ...)",
    "duration":   "What was the last contact duration in seconds?",
    "campaign":   "How many contacts were performed during this campaign?",
    "pdays":      "How many days passed after the client was last contacted? (999 means never)",
    "previous":   "How many contacts were performed before this campaign?",
    "poutcome":   "What was the outcome of the previous marketing campaign? (success, failure, unknown)",
}
GENERIC_CAT = lambda f: f"Please provide a value for {f}."
GENERIC_NUM = lambda f: f"Please provide a numeric value for {f}."
ALL_FIELDS = [(c, "cat") for c in cat_cols] + [(c, "num") for c in num_cols]
def prompt_for(field, ftype): return PROMPTS.get(field, GENERIC_CAT(field) if ftype == "cat" else GENERIC_NUM(field))

# ---------- State ----------
ss = st.session_state
ss.setdefault("answers", {})
ss.setdefault("q_idx", 0)
ss.setdefault("voice_awake", False)
ss.setdefault("transcripts", [])

# ---------- Safe Mode toggle ----------
SAFE_MODE = st.toggle("🛟 Safe Mode (per-question mic; no hotword)", value=not WRTC_OK)

@st.cache_resource
def load_vosk():
    if not VOSK_OK:
        raise RuntimeError("Vosk not installed (pip install vosk).")
    if not dir_exists(VOSK_DIR):
        raise RuntimeError(f"Vosk directory not found: {VOSK_DIR}")
    return VoskModel(VOSK_DIR)

# ---------- SAFE MODE (works everywhere) ----------
def safe_mode_screen():
    if not ALL_FIELDS:
        st.warning("No fields found in meta.json."); return
    fld, ftype = ALL_FIELDS[ss.q_idx]
    prompt = prompt_for(fld, ftype)
    st.subheader(f"Q{ss.q_idx+1}/{len(ALL_FIELDS)} — {prompt}")

    audio = st.audio_input("🎙️ Speak, then stop")
    if audio and VOSK_OK and dir_exists(VOSK_DIR):
        try:
            vm = load_vosk()
            rec = KaldiRecognizer(vm, 16000)
            rec.SetWords(False)
            b = audio.getvalue()
            for i in range(0, len(b), 8192):
                rec.AcceptWaveform(b[i:i+8192])
            import json as _json
            out = (_json.loads(rec.FinalResult()).get("text") or "").strip()
            if out:
                ss.transcripts.append(out)
                if ftype == "cat":
                    ss.answers[fld] = fuzzy_pick(out, cat_options.get(fld, []))
                else:
                    ss.answers[fld] = parse_num(out, default=float(ss.answers.get(fld, 0.0) or 0.0))
                st.success(f"Heard: {out} → {ss.answers[fld]}")
        except Exception as e:
            st.warning(f"STT error: {e}")

    # manual correction
    if ftype == "cat":
        opts = cat_options.get(fld, [])
        if opts:
            cur = ss.answers.get(fld, opts[0] if opts else "")
            if cur not in opts: cur = fuzzy_pick(str(cur), opts)
            ss.answers[fld] = st.selectbox("Or pick an option", options=opts, index=opts.index(cur))
        else:
            ss.answers[fld] = st.text_input("Or type", value=str(ss.answers.get(fld, "")))
    else:
        try:  val = float(ss.answers.get(fld, 0.0))
        except: val = 0.0
        ss.answers[fld] = st.number_input("Or type number", value=val, format="%.6f")

    c1, c2 = st.columns(2)
    if c1.button("Save & Next ➡️", disabled=(ss.q_idx >= len(ALL_FIELDS)-1)):
        ss.q_idx = min(len(ALL_FIELDS)-1, ss.q_idx+1); st.rerun()
    if c2.button("⬅️ Prev", disabled=(ss.q_idx == 0)):
        ss.q_idx = max(0, ss.q_idx-1); st.rerun()

# ---------- HOTWORD MODE (uses queued batches to avoid dropped frames) ----------
def hotword_mode_screen():
    if not WRTC_OK:
        st.error("streamlit-webrtc not available. Use Safe Mode or install/repair it."); return

    class VoskAudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.model = load_vosk()
            self.rec = KaldiRecognizer(self.model, 16000)
            self.rec.SetWords(False)
            self.result_queue: "queue.Queue[str]" = queue.Queue()

        # Batch processing prevents dropped frames
        def recv_queued(self, frames: list["av.AudioFrame"]) -> list["av.AudioFrame"]:
            try:
                if not frames:
                    return frames
                pcm_chunks = []
                sr = None
                for f in frames:
                    # Get true 16-bit PCM
                    pcm_i16 = f.to_ndarray(format="s16")
                    if pcm_i16.ndim == 2:
                        pcm_i16 = pcm_i16[0]
                    pcm_chunks.append(pcm_i16)
                    sr = f.sample_rate
                pcm_i16 = np.concatenate(pcm_chunks)

                # Resample to 16 kHz if needed
                if sr and sr != 16000:
                    duration = pcm_i16.shape[0] / sr
                    target_len = int(16000 * duration)
                    x_old = np.linspace(0, 1, pcm_i16.shape[0], endpoint=False)
                    x_new = np.linspace(0, 1, target_len, endpoint=False)
                    pcm_i16 = np.interp(x_new, x_old, pcm_i16.astype(np.float32)).astype(np.int16)

                if self.rec.AcceptWaveform(pcm_i16.tobytes()):
                    res = self.rec.Result()
                else:
                    res = self.rec.PartialResult()
                import json as _json
                j = _json.loads(res)
                text = (j.get("text") or j.get("partial") or "").strip()
                if text:
                    self.result_queue.put_nowait(text)
            except Exception:
                pass
            return frames

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="hey-property-voice",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=2048,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "audio": {
                "echoCancellation": False,
                "noiseSuppression": False,
                "autoGainControl": False,
            },
            "video": False,
        },
        async_processing=True,   # required for recv_queued
        audio_processor_factory=VoskAudioProcessor,
    )

    # status
    c1, c2, c3 = st.columns(3)
    c1.metric("Listening", "Yes" if (webrtc_ctx and webrtc_ctx.state.playing) else "No")
    c2.metric("Awake", "Yes" if ss.voice_awake else "No")
    c3.metric("Answered", f"{len(ss.answers)}/{len(ALL_FIELDS)}")
    st.info('Say **"hey property"** to start. Keep the tab focused and mic allowed.')

    def handle_phrase(phrase: str):
        low = phrase.lower().strip()
        ss.transcripts.append(low)
        # Wake
        if not ss.voice_awake and ("hey property" in low or "hay property" in low):
            ss.voice_awake = True
            ss.q_idx = 0
            st.success("👋 Hello user — starting the interview.")
            speak("Hello user. Let's begin. Please provide the first value.")
            return
        if not ss.voice_awake or not ALL_FIELDS:
            return
        # Answer current question and advance
        fld, ftype = ALL_FIELDS[ss.q_idx]
        if ftype == "cat":
            ss.answers[fld] = fuzzy_pick(low, cat_options.get(fld, []))
        else:
            ss.answers[fld] = parse_num(low, default=float(ss.answers.get(fld, 0.0) or 0.0))
        if ss.q_idx < len(ALL_FIELDS) - 1:
            ss.q_idx += 1
        else:
            speak("All questions answered. Click Submit to get the result.")

    # drain queue (cap per rerun)
    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        q = webrtc_ctx.audio_processor.result_queue
        updates, recent = 0, []
        while updates < 32:
            try:
                phrase = q.get_nowait()
                updates += 1
                recent.append(phrase)
                handle_phrase(phrase)
            except queue.Empty:
                break
        if recent:
            st.caption("🎧 Heard: " + " | ".join(recent[-5:]))
            st.rerun()

    # current question
    if ss.voice_awake and ALL_FIELDS:
        fld, ftype = ALL_FIELDS[ss.q_idx]
        st.markdown(f"### ❓ {prompt_for(fld, ftype)}")
        st.caption(f"Captured for **{fld}**: {ss.answers.get(fld, '…')}")

# ---------- Render ----------
st.caption(f"Decision threshold: **{threshold:.2f}** | Questions: {len(ALL_FIELDS)}")

# Show transcript (both modes)
if st.toggle("Show transcript"):
    st.write("\n".join(ss.transcripts[-50:]) or "—")

if SAFE_MODE:
    st.warning("Safe Mode is ON — per-question mic. Turn OFF to try the wake word.")
    safe_mode_screen()
else:
    hotword_mode_screen()

st.divider()

# Review / correct answers
with st.expander("✏️ Review / correct answers"):
    for c, t in ALL_FIELDS:
        if t == "cat":
            opts = cat_options.get(c, [])
            if opts:
                cur = ss.answers.get(c, opts[0] if opts else "")
                if cur not in opts: cur = fuzzy_pick(str(cur), opts)
                ss.answers[c] = st.selectbox(c, options=opts, index=opts.index(cur))
            else:
                ss.answers[c] = st.text_input(c, value=str(ss.answers.get(c, "")))
        else:
            try:    val = float(ss.answers.get(c, 0.0))
            except: val = 0.0
            ss.answers[c] = st.number_input(c, value=val, format="%.6f")

st.divider()

# ---------- Submit & Predict ----------
ready = all(k in ss.answers for k,_ in ALL_FIELDS)
if st.button("✅ Submit", type="primary", disabled=not ready):
    try:
        row = {}
        for c, t in ALL_FIELDS:
            if t == "cat":
                row[c] = fuzzy_pick(str(ss.answers.get(c, "")), cat_options.get(c, []))
            else:
                row[c] = float(ss.answers.get(c, 0.0))
        df = pd.DataFrame([row])[[c for c,_ in ALL_FIELDS]]
        X = preprocessor.transform(df)
        X = to_dense_float32(X)
        p = float(nn.predict(X, verbose=0).ravel()[0])
        yhat = int(p >= threshold)

        st.subheader("Result")
        st.write(f"**Probability**: {p:.6f}")
        st.write(f"**Prediction (1=yes, 0=no)**: **{yhat}**")
        msg = "yes" if yhat == 1 else "no"
        st.success(f"{'yes (1)' if yhat==1 else 'no (0)'} — user { 'can' if yhat==1 else 'cannot' }.")
        speak(f"The prediction is {msg}.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Reset tools
cA, cB = st.columns(2)
if cA.button("🔁 Reset session"):
    ss.clear(); st.rerun()
if cB.button("🧹 Clear transcripts"):
    ss.transcripts = []; st.rerun()
