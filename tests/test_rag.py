"""Unit tests for RAG pipeline components."""
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from src.ingestion import semantic_chunk
from src.evaluation import check_regressions, build_eval_dataset


def test_semantic_chunk_splits_correctly():
    docs = [Document(page_content="Para one.\n\nPara two.\n\nPara three.")]
    chunks = semantic_chunk(docs, chunk_size=30, chunk_overlap=0)
    assert len(chunks) >= 2


def test_semantic_chunk_preserves_content():
    text = "Hello world.\n\nThis is a test."
    docs = [Document(page_content=text)]
    chunks = semantic_chunk(docs, chunk_size=500)
    combined = " ".join(c.page_content for c in chunks)
    assert "Hello world" in combined


def test_check_regressions_passes():
    scores = {"faithfulness": 0.90, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.82}
    report = check_regressions(scores, run_id="test_001")
    assert report["passed"] is True
    assert report["regressions"] == {}


def test_check_regressions_detects_failure():
    scores = {"faithfulness": 0.60, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.82}
    report = check_regressions(scores, run_id="test_002")
    assert report["passed"] is False
    assert "faithfulness" in report["regressions"]


def test_build_eval_dataset():
    results = [{"question": "What is X?", "answer": "X is Y.", "contexts": ["X is Y because Z."]}]
    ground_truths = ["X is Y."]
    dataset = build_eval_dataset(results, ground_truths)
    assert len(dataset) == 1
    assert dataset["question"][0] == "What is X?"
