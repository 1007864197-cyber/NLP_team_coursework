"""Check packaged model provenance and inference without opening test queries."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from restaurant_bot.classifier import IntentClassifier


def main():
    model = PROJECT / "models" / "intent_classifier.joblib"
    metrics = json.loads(model.with_name("training_metrics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(model.read_bytes()).hexdigest() == metrics["sha256"]["model_artifact"]
    source = PROJECT / "training" / "data" / "intents.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metrics["sha256"]["source_data"]
    manifest = json.loads((PROJECT / "training" / "data" / "split_manifest.json").read_text(encoding="utf-8"))
    families = manifest["families"]
    assert len({row["family_id"] for row in families}) == 380
    assert sum(row["split"] == "train" for row in families) == 304
    assert sum(row["split"] == "development" for row in families) == 76
    classifier = IntentClassifier()
    results = {}
    for label in ["hours", "payment", "allergens", "fallback"]:
        example = next(row["base_text"] for row in families if row["intent"] == label and row["split"] == "train")
        prediction = classifier.predict(example)
        assert set(["intent", "score", "method", "accepted"]) <= set(prediction)
        assert math.isfinite(prediction["score"]) and 0 <= prediction["score"] <= 1
        assert isinstance(prediction["accepted"], bool)
        results[label] = prediction
    assert classifier.predict("")["accepted"] is False
    try:
        IntentClassifier(PROJECT / "models" / "__missing_model_for_check__.joblib")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Absent model did not raise FileNotFoundError")
    print(json.dumps({"artifact_sha_verified": True, "data_sha_verified": True,
                      "family_split_verified": True, "inference_on_training_examples": results}, indent=2))


if __name__ == "__main__":
    main()
