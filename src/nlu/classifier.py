import json
import numpy as np
from pathlib import Path

from .preprocess import preprocess

class IntentClassifier:
    MAX_LEN = 50

    def __init__(
        self,
        model_dir: str | Path = "models",
        threshold: float = 0.84,
        backend: str = "keras",
    ):
        self.model_dir     = Path(model_dir)
        self.threshold     = threshold
        self.backend       = backend
        self.model         = None
        self.tokenizer     = None
        self.nb_pipeline   = None
        self.label_names   = None

        # Tu dong doc nguong tu threshold_config.json
        thresh_path = self.model_dir / "threshold_config.json"
        if thresh_path.exists():
            try:
                with open(thresh_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.threshold = float(cfg.get("confidence_threshold", self.threshold))
            except Exception:
                pass

    # Load artifacts

    def load(self) -> "IntentClassifier":
        """Load model, tokenizer và labels từ model_dir."""
        # Labels
        with open(self.model_dir / "labels.json", encoding="utf-8") as f:
            raw_labels = json.load(f)
            if isinstance(raw_labels, dict):
                self.label_names = [
                    raw_labels[str(i)] if str(i) in raw_labels else raw_labels[i]
                    for i in range(len(raw_labels))
                ]
            else:
                self.label_names = list(raw_labels)

        if self.backend in ["keras", "conv1d", "lstm"]:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import tokenizer_from_json

            # Uu tien conv1d_best.keras
            if self.backend == "lstm" or not (self.model_dir / "conv1d_best.keras").exists():
                model_file = "lstm_best.keras"
            else:
                model_file = "conv1d_best.keras"

            self.model = tf.keras.models.load_model(self.model_dir / model_file)
            with open(self.model_dir / "tokenizer.json", encoding="utf-8") as f:
                self.tokenizer = tokenizer_from_json(f.read())

        elif self.backend == "nb":
            import pickle

            with open(self.model_dir / "nb_pipeline.pkl", "rb") as f:
                self.nb_pipeline = pickle.load(f)

        else:
            raise ValueError(f"Unknown backend: {self.backend!r}. Use 'keras' or 'nb'.")

        return self

    # Predict
    def predict(self, text: str) -> dict:

        clean = preprocess(text)

        if self.backend == "keras":
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            seq    = self.tokenizer.texts_to_sequences([clean])
            padded = pad_sequences(
                seq, maxlen=self.MAX_LEN, padding="post", truncating="post"
            )
            proba = self.model.predict(padded, verbose=0)[0]

        elif self.backend == "nb":
            proba = self.nb_pipeline.predict_proba([clean])[0]

        max_idx    = int(np.argmax(proba))
        confidence = float(proba[max_idx])

        if confidence < self.threshold:
            return {
                "intent":      "ngoai_pham_vi",
                "confidence":  confidence,
                "transferred": True,
            }

        return {
            "intent":      self.label_names[max_idx],
            "confidence":  confidence,
            "transferred": False,
            "all_proba": {
                self.label_names[i]: float(proba[i])
                for i in range(len(self.label_names))
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Phân loại ý định cho nhiều câu cùng lúc."""
        return [self.predict(t) for t in texts]

    # Repr
    def __repr__(self) -> str:
        loaded = self.model is not None or self.nb_pipeline is not None
        return (
            f"IntentClassifier(backend={self.backend!r}, "
            f"threshold={self.threshold}, "
            f"loaded={loaded})"
        )