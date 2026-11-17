"""Testes de formatter/train.py: as métricas de avaliação e a leitura dos dados."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from formatter import train as train_module
from formatter.dataset import IGNORE, LABEL2ID
from formatter.train import checkpoint_to_resume, compute_metrics, load_tagged

K, C, P, Q, D = (LABEL2ID[name] for name in ["KEEP_NONE", "KEEP_COMMA", "KEEP_PERIOD", "KEEP_QUESTION", "DELETE"])


def logits_for(predicted, pieces):
    """Cria logits em que cada peça tem o rótulo dado como o de maior nota."""
    logits = np.zeros((1, pieces, 5))
    for piece, label in enumerate(predicted):
        logits[0, piece, label] = 1
    return logits


def test_perfect_predictions():
    """Previsões perfeitas: F1 1, nenhuma palavra real apagada, todas as hesitações apagadas."""
    labels = np.array([[IGNORE, K, C, D, P, Q, IGNORE]])
    metrics = compute_metrics((logits_for([0, K, C, D, P, Q, 0], 7), labels))
    assert metrics == {"f1_macro": 1.0, "real_deleted": 0.0, "delete_recall": 1.0}


def test_ignored_pieces_do_not_count():
    """Erros em peças IGNORE não entram nas métricas."""
    labels = np.array([[IGNORE, K, IGNORE, D, IGNORE]])
    # Erros só nas peças ignoradas.
    metrics = compute_metrics((logits_for([D, K, D, D, D], 5), labels))
    assert metrics["real_deleted"] == 0.0
    assert metrics["delete_recall"] == 1.0


def test_real_deleted_and_delete_recall():
    """Confere real_deleted e delete_recall com contas pequenas."""
    labels = np.array([[K, K, K, K, D, D]])
    # 1 of 4 real words deleted, 1 of 2 fillers deleted.
    metrics = compute_metrics((logits_for([D, K, K, K, D, K], 6), labels))
    assert metrics["real_deleted"] == 0.25
    assert metrics["delete_recall"] == 0.5


def test_metrics_are_plain_floats():
    """As métricas são floats do Python (o Trainer precisa disso para salvar em JSON)."""
    labels = np.array([[K, D]])
    metrics = compute_metrics((logits_for([K, D], 2), labels))
    assert all(type(value) is float for value in metrics.values())


def test_load_tagged_with_and_without_limit(tmp_path, monkeypatch):
    """load_tagged lê tudo, ou só os primeiros N exemplos."""
    rows = [{"tokens": ["oi", str(i)], "labels": [K, P], "language": "pt"} for i in range(10)]
    (tmp_path / "tagged_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(train_module, "PROCESSED_DIR", tmp_path)
    assert len(load_tagged("val", None)) == 10
    assert load_tagged("val", 3)["tokens"] == [r["tokens"] for r in rows[:3]]
    assert len(load_tagged("val", 100)) == 10


def test_checkpoint_to_resume_without_flag_is_none(tmp_path):
    """Sem --resume, o treino começa do início mesmo que existam checkpoints."""
    (tmp_path / "checkpoint-500").mkdir()
    assert checkpoint_to_resume(tmp_path, False) is None


def test_checkpoint_to_resume_picks_the_last_one(tmp_path):
    """Com --resume, usa o checkpoint de maior número."""
    for step in [500, 1500, 1000]:
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert checkpoint_to_resume(tmp_path, True) == str(tmp_path / "checkpoint-1500")


def test_checkpoint_to_resume_with_nothing_saved_starts_over(tmp_path):
    """Com --resume e nenhum checkpoint (ou nem a pasta), começa do início em vez de dar erro."""
    assert checkpoint_to_resume(tmp_path, True) is None
    assert checkpoint_to_resume(tmp_path / "nao-existe", True) is None
