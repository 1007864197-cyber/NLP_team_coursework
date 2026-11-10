"""Reproduce supervised intent training without reading the final evaluation set.

Usage from any directory: python path/to/olive-table/training/train.py
All base-family splitting happens before train-only prefix augmentation. Fitted
features, classifiers and the rejection rule use training/development data only.
"""
import os

# Predictable single-threaded training also avoids slow native-library startup
# on some Windows installations. Set these before importing NumPy/scikit-learn.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "training" / "data" / "intents.json"
MODELS = PROJECT / "models"
SEED = 3242026
DEV_FRACTION = 0.20
PREFIXES = ("", "Hello, ", "Please help me with this: ")
EXPECTED_LABELS = {
    "hours", "location", "reservations", "takeaway", "delivery", "payment",
    "accessibility", "parking", "children", "pets", "service_charge",
    "cancellation", "allergens", "menu", "recommendation", "greeting",
    "thanks", "goodbye", "fallback",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(true, predicted, labels):
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class_report": classification_report(true, predicted, labels=labels, output_dict=True, zero_division=0),
    }


def model_scores(estimator, matrix):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(matrix), "uncalibrated class probability"
    margins = np.asarray(estimator.decision_function(matrix), dtype=float)
    weights = np.exp(margins - margins.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True), "softmax of uncalibrated SVC margins"


def reject(predictions, confidence, threshold):
    return [str(label) if label != "fallback" and score >= threshold else "fallback"
            for label, score in zip(predictions, confidence)]


def main():
    start = time.perf_counter()
    corpus = json.loads(SOURCE.read_text(encoding="utf-8"))
    intents = corpus["intents"]
    if set(intents) != EXPECTED_LABELS:
        raise ValueError("Training corpus must contain the agreed 19 restaurant labels.")
    labels = sorted(intents)
    rng = random.Random(SEED)
    train_text, train_y, dev_text, dev_y = [], [], [], []
    manifest = []
    base_train_counts, base_dev_counts = Counter(), Counter()
    all_bases = set()

    for label in labels:
        utterances = intents[label]
        if len(utterances) != 20 or any(not isinstance(text, str) or not text.strip() for text in utterances):
            raise ValueError(f"Expected 20 nonempty base utterances for {label}.")
        indices = list(range(len(utterances)))
        rng.shuffle(indices)
        dev_ids = set(indices[:round(len(utterances) * DEV_FRACTION)])
        for index, text in enumerate(utterances):
            normalized = " ".join(text.lower().split())
            if normalized in all_bases:
                raise ValueError(f"Duplicate base utterance: {text}")
            all_bases.add(normalized)
            split = "development" if index in dev_ids else "train"
            manifest.append({"family_id": f"{label}_{index + 1:03d}", "intent": label,
                             "split": split, "base_text": text,
                             "base_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
            if split == "development":
                dev_text.append(text)
                dev_y.append(label)
                base_dev_counts[label] += 1
            else:
                # Families are split first. Every variant of this family remains
                # in training; no dev text is augmented or fitted by a vectorizer.
                for prefix in PREFIXES:
                    train_text.append(prefix + text)
                    train_y.append(label)
                base_train_counts[label] += 1

    if set(text.lower().strip() for text in train_text) & set(text.lower().strip() for text in dev_text):
        raise ValueError("Train/development text overlap detected.")
    split_path = PROJECT / "training" / "data" / "split_manifest.json"
    split_path.write_text(json.dumps({"seed": SEED, "split_before_augmentation": True,
                                     "source_sha256": sha256(SOURCE), "families": manifest},
                                    indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Fitting features on {len(train_text)} training variants from {sum(base_train_counts.values())} families; {len(dev_text)} untouched development families.", flush=True)
    feature_cache = {}
    for character_weight in (0.65, 0.35, 0.8):
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=6000)),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=8000)),
        ], transformer_weights={"word": 1.0, "character": character_weight})
        train_matrix = features.fit_transform(train_text)
        dev_matrix = features.transform(dev_text)
        feature_cache[character_weight] = (features, train_matrix, dev_matrix)
    candidates = [
        ("LogisticRegression", "LogisticRegression", LogisticRegression(C=12.0, max_iter=2500, random_state=SEED), {"C": 12.0, "max_iter": 2500}, 0.65),
        ("MultinomialNB", "MultinomialNB", MultinomialNB(alpha=0.3), {"alpha": 0.3}, 0.65),
        ("LinearSVC", "LinearSVC", LinearSVC(C=1.5, random_state=SEED, dual="auto"), {"C": 1.5, "dual": "auto"}, 0.65),
    ]
    # One bounded follow-up grid was specified after the initial three-model
    # development comparison. The family split and corpus remain unchanged.
    # Record this repeated development use explicitly; it increases selection
    # bias and requires a separate untouched final evaluation.
    for character_weight in (0.35, 0.8):
        for regularisation in (2.0, 8.0, 32.0):
            name = f"LogisticRegression_C{regularisation:g}_char{character_weight:g}"
            candidates.append((name, "LogisticRegression", LogisticRegression(C=regularisation, max_iter=2500, random_state=SEED),
                               {"C": regularisation, "max_iter": 2500}, character_weight))
        for regularisation in (0.5, 2.0, 8.0):
            name = f"LinearSVC_C{regularisation:g}_char{character_weight:g}"
            candidates.append((name, "LinearSVC", LinearSVC(C=regularisation, random_state=SEED, dual="auto"),
                               {"C": regularisation, "dual": "auto"}, character_weight))
    results = []
    fitted = {}
    # Thresholds are declared before fitting and selected only on development.
    thresholds = [0.0] + [round(float(x), 2) for x in np.arange(0.05, 0.81, 0.01)]
    for name, estimator_name, estimator, parameters, character_weight in candidates:
        features, train_matrix, dev_matrix = feature_cache[character_weight]
        candidate_start = time.perf_counter()
        estimator.fit(train_matrix, train_y)
        scores, score_kind = model_scores(estimator, dev_matrix)
        predictions = estimator.classes_[np.argmax(scores, axis=1)]
        confidence = scores.max(axis=1)
        raw = summary(dev_y, predictions, labels)
        threshold_results = []
        for threshold in thresholds:
            adjusted = reject(predictions, confidence, threshold)
            threshold_results.append({
                "threshold": threshold,
                "macro_f1": float(f1_score(dev_y, adjusted, labels=labels, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(dev_y, adjusted)),
                "accepted_in_domain_count": int(sum(label != "fallback" for label in adjusted)),
            })
        chosen = max(threshold_results, key=lambda result: (result["macro_f1"], result["accuracy"], result["threshold"]))
        final_predicted = reject(predictions, confidence, chosen["threshold"])
        final = summary(dev_y, final_predicted, labels)
        candidate = {"name": name, "estimator_name": estimator_name, "parameters": parameters,
                     "character_weight": character_weight, "score_kind": score_kind,
                     "fit_and_development_seconds": round(time.perf_counter() - candidate_start, 4),
                     "raw_development": raw, "reject_threshold": chosen["threshold"],
                     "development_after_rejection": final, "threshold_grid": threshold_results}
        results.append(candidate)
        fitted[name] = {"estimator": estimator, "features": features, "dimensions": train_matrix.shape[1]}
        print(f"{name}: raw macro F1 {raw['macro_f1']:.4f}; development-selected threshold {chosen['threshold']:.2f}; rejected macro F1 {final['macro_f1']:.4f}.", flush=True)

    # Stable final tie-breaker follows declared candidate order; no final test
    # query, evaluation file or application result is read by this script.
    selected = max(results, key=lambda result: (result["development_after_rejection"]["macro_f1"],
                                                result["development_after_rejection"]["accuracy"],
                                                result["raw_development"]["macro_f1"]))
    selected_fit = fitted[selected["name"]]
    features = selected_fit["features"]
    pipeline = Pipeline([("features", features), ("classifier", selected_fit["estimator"])])
    metadata = {"seed": SEED, "source_sha256": sha256(SOURCE), "split_manifest_sha256": sha256(split_path),
                "training_examples": len(train_text), "development_examples": len(dev_text),
                "selected_on": "development macro F1 after rejection, then development accuracy, then raw development macro F1"}
    artifact = {"schema_version": 1, "pipeline": pipeline, "model_name": selected["estimator_name"],
                "candidate_name": selected["name"],
                "score_kind": selected["score_kind"], "reject_threshold": selected["reject_threshold"],
                "labels": labels, "metadata": metadata}
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "intent_classifier.joblib"
    joblib.dump(artifact, model_path, compress=3)
    metrics = {
        "schema_version": 1, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": corpus["provenance"], "seed": SEED,
        "data_protocol": {
            "base_utterances": len(manifest), "classes": len(labels),
            "base_training_families": sum(base_train_counts.values()), "development_families": len(dev_text),
            "augmented_training_examples": len(train_text), "development_examples": len(dev_text),
            "development_fraction": DEV_FRACTION, "prefix_variants": list(PREFIXES),
            "split_before_augmentation": True, "features_fit_on": "training variants only",
            "base_training_counts": dict(base_train_counts), "base_development_counts": dict(base_dev_counts),
            "augmented_training_counts": dict(Counter(train_y)),
            "test_set_access": "None. This script reads only training/data/intents.json; it does not read evaluation files.",
        },
        "features": {"word_ngram_range": [1, 2], "character_ngram_range": [3, 5],
                     "character_analyzer": "char_wb", "sublinear_tf": True,
                     "weights": {"word": 1.0, "character": selected["character_weight"]},
                     "word_vocabulary_size": len(features.transformer_list[0][1].vocabulary_),
                     "character_vocabulary_size": len(features.transformer_list[1][1].vocabulary_),
                     "total_dimensions": int(selected_fit["dimensions"])},
        "selection_protocol": {
            "model": metadata["selected_on"],
            "threshold": "For each model maximise development macro F1, then accuracy, then choose the higher tied threshold for conservative rejection.",
            "model_tie_break": "Declared candidate order after the recorded selection metrics tie.",
            "bounded_follow_up": "After the original three-model comparison, evaluate LR C=[2,8,32] and SVC C=[0.5,2,8] at character weights [0.35,0.8], keeping the same fixed corpus and family split. All 15 candidate results are retained. No final test results are consulted.",
            "refit": "No refit on development data. The saved model is the fitted training-only model.",
            "score_warning": "Scores are uncalibrated model outputs, not accuracy estimates or probabilities that answers are factually correct.",
        },
        "candidates": results,
        "selected_model": selected["estimator_name"], "selected_candidate": selected["name"], "reject_threshold": selected["reject_threshold"],
        "selected_development": selected["development_after_rejection"],
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "scikit_learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                     "joblib": joblib.__version__},
        "sha256": {"source_data": sha256(SOURCE), "split_manifest": sha256(split_path),
                   "model_artifact": sha256(model_path)},
        "total_training_seconds": round(time.perf_counter() - start, 4),
        "limitations": [
            "Small self-authored synthetic English corpus; not representative of real restaurant customers.",
            "The development set is used for model and threshold selection, so its scores are optimistic selection results, not final test estimates.",
            "One bounded follow-up hyperparameter grid was chosen after reviewing the initial development scores. This repeated development use introduces further selection bias.",
            "Polite-prefix augmentation does not add new semantic content; grouped splitting prevents its leakage across train and development.",
            "Unknown topics beyond the fallback training examples can still be misclassified.",
        ],
    }
    metrics_path = MODELS / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selected {selected['name']}; artifact {model_path}; SHA256 {metrics['sha256']['model_artifact']}", flush=True)
    print(json.dumps({"selected_model": selected["estimator_name"], "selected_candidate": selected["name"], "development_macro_f1": metrics["selected_development"]["macro_f1"],
                      "development_accuracy": metrics["selected_development"]["accuracy"], "reject_threshold": selected["reject_threshold"],
                      "training_examples": len(train_text), "development_examples": len(dev_text)}), flush=True)


if __name__ == "__main__":
    main()
