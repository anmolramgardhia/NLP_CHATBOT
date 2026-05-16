"""
intent_classifier.py — Intent classification module
NLP Chatbot Project — Restaurant Booking Domain

Two-stage approach:
  Stage 1 (baseline) : TF-IDF + Logistic Regression  — fast, interpretable
  Stage 2 (production): Fine-tuned DistilBERT         — high accuracy
"""

import json
import joblib
import numpy as np
from pathlib import Path
from preprocess import clean_text

DATA_PATH  = Path(__file__).parent.parent / "data" / "intents.json"
MODEL_DIR  = Path(__file__).parent.parent / "models"
TFIDF_PATH = MODEL_DIR / "tfidf_pipeline.joblib"
BERT_DIR   = MODEL_DIR / "bert_intent"

CONFIDENCE_THRESHOLD = 0.55   # below this → out_of_scope


# ── Data loading ──────────────────────────────────────────────────────────────

def load_intent_data(path: str = None) -> tuple:
    """
    Load intent examples from intents.json.

    Returns:
        Tuple of (texts: list[str], labels: list[str])
    """
    file_path = Path(path) if path else DATA_PATH
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for example in intent["examples"]:
            texts.append(clean_text(example))
            labels.append(intent["tag"])

    print(f"Loaded {len(texts)} examples across {len(set(labels))} intents")
    return texts, labels


# ── Stage 1: TF-IDF Baseline ─────────────────────────────────────────────────

class TFIDFIntentClassifier:
    """
    Baseline intent classifier using TF-IDF + Logistic Regression.
    Target accuracy: >90%. Trains in seconds, no GPU needed.
    """

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
            ("clf",   LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")),
        ])
        self.classes_ = None

    def fit(self, texts: list, labels: list):
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        print(f"TF-IDF model trained on {len(texts)} examples")
        return self

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        probs   = self.pipeline.predict_proba([cleaned])[0]
        idx     = int(np.argmax(probs))
        intent  = self.classes_[idx]
        confidence = float(probs[idx])
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "out_of_scope"
        return {
            "intent":     intent,
            "confidence": round(confidence, 4),
            "all_scores": {c: round(float(p), 4) for c, p in zip(self.classes_, probs)},
            "model":      "tfidf",
        }

    def save(self, path: str = None):
        save_path = Path(path) if path else TFIDF_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        print(f"TF-IDF model saved to {save_path}")

    @classmethod
    def load(cls, path: str = None):
        load_path = Path(path) if path else TFIDF_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"No TF-IDF model at {load_path}. Run train_intent.py first.")
        return joblib.load(load_path)


# ── Stage 2: DistilBERT Fine-tuned ───────────────────────────────────────────

class BERTIntentClassifier:
    """
    Production intent classifier using fine-tuned DistilBERT.
    Target accuracy: >95%. Requires GPU for training.
    """

    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    EPOCHS     = 5
    LR         = 2e-5

    def __init__(self):
        self.pipeline  = None
        self.label2id  = {}
        self.id2label  = {}

    def train(self, texts: list, labels: list, epochs: int = None):
        """Fine-tune DistilBERT on intent examples."""
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer
        )
        import torch
        from torch.utils.data import Dataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        label_ids = le.fit_transform(labels)
        self.label2id = {l: i for i, l in enumerate(le.classes_)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        X_tr, X_val, y_tr, y_val = train_test_split(
            texts, label_ids, test_size=0.15, stratify=label_ids, random_state=42
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        class IntentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.enc    = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
                self.labels = labels
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        train_ds = IntentDataset(X_tr,  y_tr,  tokenizer, self.MAX_LENGTH)
        val_ds   = IntentDataset(X_val, y_val, tokenizer, self.MAX_LENGTH)

        args = TrainingArguments(
            output_dir=str(BERT_DIR),
            num_train_epochs=epochs or self.EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            learning_rate=self.LR,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()
        self.save(model=model, tokenizer=tokenizer)

    def save(self, model=None, tokenizer=None, path: str = None):
        save_dir = Path(path) if path else BERT_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        if model and tokenizer:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        with open(save_dir / "label_map.json", "w") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f)

    def load(self, path: str = None):
        from transformers import pipeline as hf_pipeline
        load_dir = Path(path) if path else BERT_DIR
        if not load_dir.exists():
            raise FileNotFoundError(f"No BERT model at {load_dir}. Run train_intent.py --bert first.")
        with open(load_dir / "label_map.json") as f:
            maps = json.load(f)
        self.label2id = maps["label2id"]
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        self.pipeline = hf_pipeline(
            "text-classification", model=str(load_dir),
            tokenizer=str(load_dir), return_all_scores=True, device=-1,
        )
        return self

    def predict(self, text: str) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        cleaned = clean_text(text)
        scores  = self.pipeline(cleaned)[0]
        scores_dict = {s["label"]: round(s["score"], 4) for s in scores}
        best    = max(scores, key=lambda x: x["score"])
        intent  = best["label"]
        confidence = round(best["score"], 4)
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "out_of_scope"
        return {
            "intent":     intent,
            "confidence": confidence,
            "all_scores": scores_dict,
            "model":      "bert",
        }


if __name__ == "__main__":
    texts, labels = load_intent_data()
    clf = TFIDFIntentClassifier()
    clf.fit(texts, labels)
    clf.save()
    test_inputs = [
        "Book a table for 2 at 7pm",
        "What time do you close?",
        "Cancel my reservation",
        "Do you have vegan options?",
    ]
    print("\nQuick predictions:")
    for t in test_inputs:
        r = clf.predict(t)
        print(f"  '{t}' → {r['intent']} ({r['confidence']*100:.1f}%)")
