"""Inference for the locally trained supervised restaurant intent classifier.

Only the packaged local artifact is loaded. Joblib files must come from this
project's trusted training process; they are not a user-upload format.
"""
from pathlib import Path

import joblib
import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "intent_classifier.joblib"


class IntentClassifier:
    """Predict an intent, then apply the development-selected reject threshold."""

    def __init__(self, model_path=None):
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Intent model not found: {path}. Run python training/train.py.")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
            raise ValueError("Unsupported intent model artifact schema.")
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["reject_threshold"])
        self.model_name = artifact["model_name"]
        self.score_kind = artifact["score_kind"]
        self.metadata = artifact.get("metadata", {})
        self.method = f"trained {self.model_name} with word and character TF-IDF ({self.score_kind})"

    def predict(self, text):
        """Return an uncalibrated model score, not a guarantee of correctness.

        Rejection routes low-score or explicitly out-of-domain predictions to
        fallback. The underlying label remains available as ``raw_intent`` for
        transparent diagnostics, without changing the public reply contract.
        """
        if not isinstance(text, str) or not text.strip():
            return {"intent": "fallback", "score": 0.0, "method": self.method,
                    "accepted": False, "raw_intent": "fallback"}
        if hasattr(self.pipeline, "predict_proba"):
            scores = np.asarray(self.pipeline.predict_proba([text]), dtype=float)[0]
        else:
            # LinearSVC margins are not probabilities. Softmax gives a bounded
            # ranking score so a development-only reject rule can be applied.
            margins = np.asarray(self.pipeline.decision_function([text]), dtype=float)[0]
            weights = np.exp(margins - np.max(margins))
            scores = weights / weights.sum()
        index = int(np.argmax(scores))
        raw_intent = str(self.pipeline.classes_[index])
        score = float(scores[index])
        accepted = raw_intent != "fallback" and score >= self.threshold
        return {"intent": raw_intent if accepted else "fallback", "score": score,
                "method": self.method, "accepted": bool(accepted), "raw_intent": raw_intent}
