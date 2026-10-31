# Bilingual Speech Emotion Recognition (Tamil + English)

Deep-learning web app: Wav2Vec2 predicts emotion from speech; Whisper transcribes and detects Tamil/English.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
(ffmpeg is needed for mp3/m4a: `sudo apt install ffmpeg`)

## Train your own model (improves Tamil accuracy)
```bash
python train.py --data data --out ser_model
EMOTION_MODEL=ser_model streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `README.md` to a GitHub repo.
2. Create `packages.txt` containing the line `ffmpeg`.
3. Go to share.streamlit.io → New app → pick repo, branch, `app.py` → Deploy.
4. Your public URL appears after build (first load downloads models).
