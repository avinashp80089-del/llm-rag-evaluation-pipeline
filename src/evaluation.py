"""Automated RAGAS evaluation pipeline — runs weekly to catch LLM regressions."""
from typing import List, Dict, Any
from datetime import datetime
import json
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


REGRESSION_THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def build_eval_dataset(results: List[Dict[str, Any]], ground_truths: List[str]) -> Dataset:
    """Convert RAG results + ground truths into RAGAS Dataset format."""
    return Dataset.from_dict(
        {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": ground_truths,
        }
    )


def run_evaluation(results: List[Dict[str, Any]], ground_truths: List[str]) -> Dict[str, float]:
    """Run full RAGAS evaluation suite and return metric scores."""
    dataset = build_eval_dataset(results, ground_truths)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return scores


def check_regressions(scores: Dict[str, float], run_id: str = None) -> Dict[str, Any]:
    """
    Compare scores against thresholds.
    Catches regressions before production deployment — same pattern that
    caught 3 regression events before they reached prod at Erasmus.AI.
    """
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    failures = {}
    for metric, threshold in REGRESSION_THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        if score < threshold:
            failures[metric] = {"score": round(score, 4), "threshold": threshold, "delta": round(score - threshold, 4)}

    report = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "regressions": failures,
        "passed": len(failures) == 0,
    }

    print(json.dumps(report, indent=2))
    if failures:
        print(f"\n⚠️  REGRESSION DETECTED in {list(failures.keys())} — block deployment")
    else:
        print("\n✅ All metrics above threshold — safe to deploy")

    return report


def save_report(report: Dict[str, Any], output_path: str = "eval_report.json"):
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_path}")
