import json

import numpy as np
import pytest

pytest.importorskip("torch")

from formatter import train as train_module
from formatter.dataset import IGNORE, LABEL2ID
from formatter.train import compute_metrics, load_tagged

K, C, P, Q, D = (LABEL2ID[name] for name in ["KEEP_NONE", "KEEP_COMMA", "KEEP_PERIOD", "KEEP_QUESTION", "DELETE"])


def logits_for(predicted, pieces):
    """Logits where each piece's highest score is the given label."""
    logits = np.zeros((1, pieces, 5))
    for piece, label in enumerate(predicted):
        logits[0, piece, label] = 1
    return logits


def test_perfect_predictions():
    labels = np.array([[IGNORE, K, C, D, P, Q, IGNORE]])
    metrics = compute_metrics((logits_for([0, K, C, D, P, Q, 0], 7), labels))
    assert metrics == {"f1_macro": 1.0, "real_deleted": 0.0, "delete_recall": 1.0}


def test_ignored_pieces_do_not_count():
    labels = np.array([[IGNORE, K, IGNORE, D, IGNORE]])
    # Wrong guesses only on ignored pieces.
    metrics = compute_metrics((logits_for([D, K, D, D, D], 5), labels))
    assert metrics["real_deleted"] == 0.0
    assert metrics["delete_recall"] == 1.0


def test_real_deleted_and_delete_recall():
    labels = np.array([[K, K, K, K, D, D]])
    # 1 of 4 real words deleted, 1 of 2 fillers deleted.
    metrics = compute_metrics((logits_for([D, K, K, K, D, K], 6), labels))
    assert metrics["real_deleted"] == 0.25
    assert metrics["delete_recall"] == 0.5


def test_metrics_are_plain_floats():
    labels = np.array([[K, D]])
    metrics = compute_metrics((logits_for([K, D], 2), labels))
    assert all(type(value) is float for value in metrics.values())


def test_load_tagged_with_and_without_limit(tmp_path, monkeypatch):
    rows = [{"tokens": ["oi", str(i)], "labels": [K, P], "language": "pt"} for i in range(10)]
    (tmp_path / "tagged_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(train_module, "PROCESSED_DIR", tmp_path)
    assert len(load_tagged("val", None)) == 10
    assert load_tagged("val", 3)["tokens"] == [r["tokens"] for r in rows[:3]]
    assert len(load_tagged("val", 100)) == 10
