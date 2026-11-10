"""Run the fixed synthetic evaluation; never hand-enter measured results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation, matching the documented sample-percentile measure."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def validate_corpus(corpus: dict) -> None:
    required = {"id", "message", "expected_intent", "kind"}
    items = corpus["items"]
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation IDs must be unique")
    if len(items) < 48:
        raise ValueError("The authored evaluation needs at least 48 items")
    for item in items:
        if not required <= item.keys() or not item["message"].strip():
            raise ValueError("Incomplete evaluation item: " + str(item))
    if corpus["metadata"]["provenance"] != "synthetic_manually_authored":
        raise ValueError("Keep the synthetic provenance explicit")


def exact_overlap_check(items: list[dict], reference: Path) -> dict:
    """Report literal overlaps only; this cannot prove semantic independence."""
    if not reference.is_file():
        return {"available": False, "overlapping_query_ids": []}
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    def normalize(value):
        return " ".join(re.findall(r"\w+", value.casefold()))
    reference_strings = {normalize(x) for x in strings(json.loads(reference.read_text(encoding="utf-8")))}
    return {
        "available": True,
        "normalization": "casefold; word tokens; whitespace/punctuation ignored; entire string equality only",
        "overlapping_query_ids": [item["id"] for item in items if normalize(item["message"]) in reference_strings],
    }


def score_rows(rows: list[dict], labels: list[str], rejection: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    gold = [row["expected_intent"] for row in rows]
    predicted = [row["predicted_intent"] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=labels, zero_division=0
    )
    accepted = [r for r in rows if r["accepted"]]
    in_domain = [r for r in rows if r["expected_intent"] != rejection]
    out_of_domain = [r for r in rows if r["expected_intent"] == rejection]
    correct_accepted = [r for r in accepted if r["intent_correct"] and r["expected_intent"] != rejection]

    def ratio(numerator: int, denominator: int):
        return numerator / denominator if denominator else None

    latencies = [r["latency_ms"] for r in rows]
    return {
        "n": len(rows),
        "intent_accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1_including_rejection": float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1_in_domain_labels": float(f1_score(gold, predicted, labels=[x for x in labels if x != rejection], average="macro", zero_division=0)),
        "accepted_count": len(accepted),
        "correct_accepted_count": len(correct_accepted),
        "accepted_retrieval_precision": ratio(len(correct_accepted), len(accepted)),
        "acceptance_coverage_all_queries": ratio(len(accepted), len(rows)),
        "in_domain_acceptance_rate": ratio(sum(r["accepted"] for r in in_domain), len(in_domain)),
        "in_domain_correct_retrieval_rate": ratio(len(correct_accepted), len(in_domain)),
        "out_of_domain_rejection_rate": ratio(sum(not r["accepted"] for r in out_of_domain), len(out_of_domain)),
        "out_of_domain_false_acceptance_count": sum(r["accepted"] for r in out_of_domain),
        "latency_ms": {"median": statistics.median(latencies), "p95": percentile(latencies, 0.95)},
        "by_intent": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
            for label, p, r, f, n in zip(labels, precision, recall, f1, support)
        },
        "by_query_kind": {
            kind: {
                "n": sum(row["kind"] == kind for row in rows),
                "intent_accuracy": sum(row["intent_correct"] for row in rows if row["kind"] == kind) / sum(row["kind"] == kind for row in rows),
            }
            for kind in sorted({row["kind"] for row in rows})
        },
        "error_ids": [r["id"] for r in rows if not r["intent_correct"]],
    }


def format_value(value, percent=True) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%" if percent else f"{value:.3f}"


def write_markdown(result: dict, target: Path) -> None:
    modes = result["modes"]
    lines = [
        "# Observed evaluation results",
        "",
        "**Small, synthetic, manually authored examples; not real users or an independent benchmark.**",
        "The figures below were generated by executing the application. They do not establish production quality, calibrated confidence or factual correctness of every answer.",
        "",
        f"Run (UTC): {result['run_utc']}",
        f"Corpus SHA-256: `{result['corpus_sha256']}`",
        f"Source/config manifest SHA-256: `{result['source_manifest_sha256']}`",
        f"Python: {result['python_version']}; measured queries per pipeline/mode pair: {result['query_count_per_mode']}.",
        "",
        "## Protocol and split",
        "",
        result["protocol"]["split"],
        result["protocol"]["latency"],
        result["protocol"]["acceptance"],
        "",
        "## Full dialogue system",
        "",
        "This comparison calls reply(), including the classifier and dialogue routing enabled in each mode.",
        "",
        "| Metric | Baseline | Enhanced |",
        "|---|---:|---:|",
    ]
    metrics = [
        ("Intent accuracy", "intent_accuracy"),
        ("Macro F1, including rejection", "macro_f1_including_rejection"),
        ("Macro F1, in-domain labels", "macro_f1_in_domain_labels"),
        ("Accepted retrieval precision", "accepted_retrieval_precision"),
        ("Acceptance coverage, all queries", "acceptance_coverage_all_queries"),
        ("In-domain correct retrieval rate", "in_domain_correct_retrieval_rate"),
        ("Out-of-domain rejection rate", "out_of_domain_rejection_rate"),
    ]
    for title, key in metrics:
        lines.append(f"| {title} | {format_value(modes['baseline'][key])} | {format_value(modes['enhanced'][key])} |")
    for title, key in [("Median local latency (ms)", "median"), ("95th percentile local latency (ms)", "p95")]:
        lines.append(f"| {title} | {format_value(modes['baseline']['latency_ms'][key], False)} | {format_value(modes['enhanced']['latency_ms'][key], False)} |")
    raw_modes = result["raw_retrieval_modes"]
    lines += ["", "## Raw lexical retrieval only", "", "These calls bypass dialogue routing and the trained classifier by calling retrieve() directly. They are a separate experiment.", "", "| Metric | Baseline | Enhanced |", "|---|---:|---:|"]
    for title, key in metrics:
        lines.append(f"| {title} | {format_value(raw_modes['baseline'][key])} | {format_value(raw_modes['enhanced'][key])} |")
    for title, key in [("Median local latency (ms)", "median"), ("95th percentile local latency (ms)", "p95")]:
        lines.append(f"| {title} | {format_value(raw_modes['baseline']['latency_ms'][key], False)} | {format_value(raw_modes['enhanced']['latency_ms'][key], False)} |")
    lines += ["", "## Per-category scores", "", "| Category | N | Baseline F1 | Enhanced F1 |", "|---|---:|---:|---:|"]
    for label in result["labels"]:
        b, e = modes["baseline"]["by_intent"][label], modes["enhanced"]["by_intent"][label]
        lines.append(f"| {label} | {b['support']} | {format_value(b['f1'])} | {format_value(e['f1'])} |")
    lines += ["", "## Errors and limitations", ""]
    for mode in ("baseline", "enhanced"):
        ids = ", ".join(modes[mode]["error_ids"]) or "none on this sample"
        lines.append(f"- Full system, {mode}: incorrect category/rejection IDs: {ids}.")
        raw_ids = ", ".join(raw_modes[mode]["error_ids"]) or "none on this sample"
        lines.append(f"- Raw retrieval, {mode}: incorrect category/rejection IDs: {raw_ids}.")
    lines += [
        "",
        "See predictions.csv for every message, label, predicted category, acceptance decision, response and latency. Category errors are retained and are not silently excluded.",
        "The set is small and authored with knowledge of the application's scope. Its vocabulary and class balance may be easier or harder than real traffic. It covers 14 FAQ categories plus rejection, rather than every classifier training class. No statistical significance or population-level improvement is claimed. Accuracy and F1 measure category recognition, not response fluency or complete factuality. Conversation behavior is covered separately by the unit-test scenarios.",
        "",
        "## Exact-text overlap check", "", "This checks whole normalized strings only and does not establish semantic independence.", "", "```json", json.dumps(result["exact_text_overlap_check"], indent=2), "```", "",
        "## Reproducibility", "", "```json", json.dumps(result["source_manifest"], indent=2), "```", "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "queries.json")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    validate_corpus(corpus)

    from restaurant_bot import RestaurantBot

    items = corpus["items"]
    rejection = corpus["metadata"]["rejection_intent"]
    labels = sorted({item["expected_intent"] for item in items})
    rows = []
    for pipeline in ("full_system", "raw_retrieval"):
        for mode in ("baseline", "enhanced"):
            bot = RestaurantBot()
            if pipeline == "full_system":
                bot.reply("hello", state=None, mode=mode)  # One excluded warm-up call.
            else:
                bot.retrieve("hello", mode=mode)
            for item in items:
                started = time.perf_counter_ns()
                if pipeline == "full_system":
                    reply = bot.reply(item["message"], state=None, mode=mode)
                else:
                    reply = bot.retrieve(item["message"], mode=mode)
                elapsed = (time.perf_counter_ns() - started) / 1_000_000
                predicted = reply["intent"]
                if not isinstance(predicted, str):
                    raise TypeError("reply['intent'] must be a string")
                sources = reply.get("sources", []) if pipeline == "full_system" else [reply["source_id"]] if reply.get("source_id") else []
                accepted = predicted != rejection and bool(sources)
                rows.append({
                    "pipeline": pipeline,
                    "mode": mode,
                    **item,
                    "predicted_intent": predicted,
                    "accepted": accepted,
                    "intent_correct": predicted == item["expected_intent"],
                    "latency_ms": elapsed,
                    "response": reply.get("reply", ""),
                    "source_ids": json.dumps(sources, ensure_ascii=False),
                    "score": reply.get("confidence", reply.get("score")),
                })

    source_files = [PROJECT / "restaurant_bot.py", PROJECT / "app.py", PROJECT / "requirements.txt", Path(__file__).resolve()]
    source_files += sorted((PROJECT / "restaurant_bot").rglob("*.py"))
    source_files += sorted((PROJECT / "data").glob("*.json"))
    source_files += sorted((PROJECT / "training").rglob("*.py"))
    source_files += sorted((PROJECT / "training" / "data").glob("*.json"))
    source_files += sorted((PROJECT / "models").glob("*.json"))
    source_files += sorted((PROJECT / "models").glob("*.joblib"))
    manifest = {str(p.relative_to(PROJECT)).replace("\\", "/"): digest(p) for p in source_files if p.is_file()}
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "corpus_sha256": digest(args.corpus),
        "source_manifest": manifest,
        "source_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "provenance": corpus["metadata"],
        "exact_text_overlap_check": {
            "classifier_training_corpus": exact_overlap_check(items, PROJECT / "training" / "data" / "intents.json"),
            "faq_reference": exact_overlap_check(items, PROJECT / "data" / "faq.json"),
        },
        "labels": labels,
        "query_count_per_mode": len(items),
        "query_kind_counts": dict(Counter(x["kind"] for x in items)),
        "protocol": {
            "split": corpus["metadata"]["split"],
            "latency": "Each single-turn query has fresh state. Each pipeline/mode pair has one excluded warm-up call. Latency covers a local reply or retrieve call, excluding application startup and HTTP/browser/network work; p95 uses linear interpolation of this sample.",
            "acceptance": "Full-system accepted retrieval is a non-fallback answer with at least one returned reference source. Raw retrieval requires a non-fallback category and non-empty source_id. A correct accepted retrieval also matches the gold category. No confidence calibration, hallucination rate or answer-level factuality metric is claimed.",
            "mode_definitions": corpus["metadata"].get("mode_definitions", {}),
        },
        "modes": {mode: score_rows([r for r in rows if r["pipeline"] == "full_system" and r["mode"] == mode], labels, rejection) for mode in ("baseline", "enhanced")},
        "raw_retrieval_modes": {mode: score_rows([r for r in rows if r["pipeline"] == "raw_retrieval" and r["mode"] == mode], labels, rejection) for mode in ("baseline", "enhanced")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = ["pipeline", "mode", "id", "kind", "message", "expected_intent", "predicted_intent", "accepted", "intent_correct", "score", "latency_ms", "response", "source_ids"]
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(result, args.output / "results.md")
    for mode, values in result["modes"].items():
        print(f"{mode}: accuracy={values['intent_accuracy']:.3f}, macro_F1={values['macro_f1_including_rejection']:.3f}, median={values['latency_ms']['median']:.3f} ms")
    print("Results saved to", args.output.resolve())


if __name__ == "__main__":
    main()
