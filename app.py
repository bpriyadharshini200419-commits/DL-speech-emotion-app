"""Bilingual (Tamil / English) Speech Emotion Recognition - Streamlit app."""
import os
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from faster_whisper import WhisperModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SR = 16000
MAX_SECONDS = 20
# Wav2Vec2-XLSR (multilingual pre-training) fine-tuned for 8 emotions.
# Replace with your own fine-tuned folder/hub id after running train.py.
EMOTION_MODEL = os.environ.get(
    "EMOTION_MODEL", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
EMOJI = {"angry": "😠", "calm": "😌", "fearful": "😨",
         "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲"}

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    fe = AutoFeatureExtractor.from_pretrained(EMOTION_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(EMOTION_MODEL).eval()
    return fe, model


@st.cache_resource(show_spinner="Loading Whisper...")
def load_whisper(size: str):
    return WhisperModel(size, device="cpu", compute_type="int8")


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)          # remove silence
    y = y[: SR * MAX_SECONDS]
    if len(y) < SR // 2:
        return None
    return y / (np.max(np.abs(y)) + 1e-9)               # normalise


def predict_emotion(y):
    fe, model = load_emotion_model()
    inputs = fe(y, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
    labels = [model.config.id2label[i].lower() for i in range(len(probs))]
    series = pd.Series(probs, index=labels)
    series = series.drop(labels=["disgust"], errors="ignore")   # exclude disgust
    series = series / series.sum()                              # renormalise to 100%
    return series.sort_values(ascending=False)


def transcribe(y, size, language=None):
    segments, info = load_whisper(size).transcribe(
        y.astype(np.float32), beam_size=5, language=language, task="transcribe")
    return " ".join(s.text.strip() for s in segments), info.language, info.language_probability


# ---------------- UI ----------------
st.title("🎙️ Bilingual Speech Emotion Recognition")
st.caption("Tamil + English speech → emotion (Wav2Vec2) + transcript (Whisper)")

with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    lang_choice = st.selectbox("Speech language", ["Auto detect", "Tamil", "English"])
    LANG_CODE = {"Auto detect": None, "Tamil": "ta", "English": "en"}[lang_choice]
    st.markdown("**Tips**: speak 3–15 s, quiet room, one speaker.")
    st.markdown("Emotions: " + ", ".join(EMOJI))

tab_rec, tab_up = st.tabs(["🎤 Record", "📁 Upload"])
audio_file = None
with tab_rec:
    rec = st.audio_input("Record your voice")
    if rec:
        audio_file = rec
with tab_up:
    up = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if up:
        audio_file = up

if audio_file is not None:
    st.audio(audio_file)
    suffix = os.path.splitext(getattr(audio_file, "name", "rec.wav"))[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getvalue())
        path = tmp.name

    if st.button("Analyse", type="primary"):
        y = load_audio(path)
        if y is None:
            st.error("Audio too short or silent. Please try again.")
        else:
            with st.spinner("Analysing..."):
                probs = predict_emotion(y)
                text, lang, lang_p = transcribe(y, whisper_size, LANG_CODE)

            top = probs.index[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted emotion", f"{EMOJI.get(top, '')} {top.title()}")
            c2.metric("Confidence", f"{probs.iloc[0]*100:.1f}%")
            c3.metric("Detected language", f"{ {'ta': 'Tamil', 'en': 'English'}.get(lang, lang)} ({lang_p*100:.0f}%)")

            left, right = st.columns(2)
            with left:
                st.subheader("Emotion probabilities")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.barh(probs.index[::-1], probs.values[::-1], color="#4C78A8")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                st.pyplot(fig)
            with right:
                st.subheader("Waveform")
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.plot(np.arange(len(y)) / SR, y, lw=0.5)
                ax2.set_xlabel("Time (s)")
                st.pyplot(fig2)

            st.subheader("📝 Transcript (what you said)")
            if text:
                st.success(text)
                st.text_area("Copy text", text, height=120)
            else:
                st.warning("No speech detected")
else:
    st.info("Record or upload a Tamil/English voice clip to begin.")
