"""Fine-tune Wav2Vec2 for speech emotion recognition.

Folder layout (one sub-folder per emotion; mix English + Tamil clips):
    data/angry/*.wav  data/happy/*.wav  data/sad/*.wav  data/neutral/*.wav ...
Datasets: RAVDESS, TESS, CREMA-D (English); Tamil: record your own / use a Tamil emotion corpus.

Run:  python train.py --data data --out ser_model
Then: EMOTION_MODEL=ser_model streamlit run app.py
"""
import argparse

import numpy as np
import torch
from datasets import Audio, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          Trainer, TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--out", default="ser_model")
p.add_argument("--base", default="facebook/wav2vec2-base")
p.add_argument("--epochs", type=int, default=10)
a = p.parse_args()

ds = load_dataset("audiofolder", data_dir=a.data)["train"].cast_column("audio", Audio(sampling_rate=16000))
ds = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
labels = ds["train"].features["label"].names
fe = AutoFeatureExtractor.from_pretrained(a.base)


def prep(b):
    arr = [x["array"][: 16000 * 6] for x in b["audio"]]
    return fe(arr, sampling_rate=16000, max_length=16000 * 6, truncation=True, padding="max_length")


ds = ds.map(prep, batched=True, remove_columns=["audio"])
model = AutoModelForAudioClassification.from_pretrained(
    a.base, num_labels=len(labels),
    id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
model.freeze_feature_encoder()


def metrics(ev):
    pred = np.argmax(ev.predictions, axis=1)
    return {"accuracy": accuracy_score(ev.label_ids, pred),
            "macro_f1": f1_score(ev.label_ids, pred, average="macro")}


args = TrainingArguments(a.out, eval_strategy="epoch", save_strategy="epoch",
                         learning_rate=3e-5, per_device_train_batch_size=8,
                         gradient_accumulation_steps=2, num_train_epochs=a.epochs,
                         warmup_ratio=0.1, load_best_model_at_end=True,
                         metric_for_best_model="macro_f1", fp16=torch.cuda.is_available(),
                         logging_steps=20, report_to="none")
tr = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
             compute_metrics=metrics)
tr.train()
out = tr.predict(ds["test"])
print(classification_report(out.label_ids, np.argmax(out.predictions, 1), target_names=labels))
tr.save_model(a.out)
fe.save_pretrained(a.out)
