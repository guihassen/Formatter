"""Testes de formatter/pick_corrections.py: leitura das correções e escolha dos trechos."""

import pytest

from formatter import pick_corrections
from formatter.pick_corrections import correction_labels, pick, segments, word_and_mark


@pytest.mark.parametrize(
    "word, expected",
    [("feature.", ("feature", "PERIOD")), ("Okay?", ("Okay", "QUESTION")), ("happen:", ("happen", "COLON")),
     ("isso...", ("isso", "ELLIPSIS")), ("uau!", ("uau", "EXCLAMATION")), ("so", ("so", "NONE")), (".", (".", "NONE"))],
)
def test_word_and_mark(word, expected):
    """Separa a palavra do sinal final, incluindo sinais que o modelo não tem."""
    assert word_and_mark(word) == expected


def test_correction_labels_marks_missing_words_as_delete():
    """Palavras que sumiram na correção viram DELETE."""
    tokens = ["the", "team", "was", "going", "we", "need", "to", "leave", "now"]
    labels = correction_labels(tokens, "We need to leave now.")
    assert labels == ["DELETE", "DELETE", "DELETE", "DELETE", "KEEP_NONE", "KEEP_NONE", "KEEP_NONE", "KEEP_NONE",
                      "KEEP_PERIOD"]


def test_correction_labels_ignores_case_and_keeps_new_marks():
    """Maiúsculas não importam e sinais novos (: !) são registrados pelo nome."""
    assert correction_labels(["perfeito", "agora", "sim"], "Perfeito. Agora: sim!") == \
        ["KEEP_PERIOD", "KEEP_COLON", "KEEP_EXCLAMATION"]


def test_correction_labels_rejects_changed_words():
    """Trocar uma palavra na correção é um erro."""
    with pytest.raises(ValueError):
        correction_labels(["vou", "comprar"], "Vou vender.")


def test_segments_follow_predicted_sentences(monkeypatch):
    """Os trechos crescem frase a frase até ter o tamanho certo."""
    monkeypatch.setattr(pick_corrections, "MIN_WORDS", 3)
    monkeypatch.setattr(pick_corrections, "MAX_WORDS", 5)
    labels = ["KEEP_NONE", "KEEP_PERIOD", "KEEP_NONE", "KEEP_NONE", "KEEP_QUESTION", "KEEP_NONE"]
    margins = [5, 4, 3, 2, 1, 0.5]
    # As frases terminam nas palavras 1 e 4; cada trecho cresce frase a frase até ter de 3 a 5 palavras.
    assert segments(labels, margins) == [(0, 5, 1), (2, 5, 1)]


def test_pick_skips_overlaps_and_limits_per_dictation(monkeypatch):
    """A escolha pula trechos sobrepostos e respeita o limite por ditado."""
    monkeypatch.setattr(pick_corrections, "HOW_MANY", 10)
    candidates = [
        {"at": "a", "start": 0, "end": 10, "score": 0.1},
        {"at": "a", "start": 5, "end": 15, "score": 0.2},    # sobrepõe o primeiro
        {"at": "a", "start": 20, "end": 30, "score": 0.3},
        {"at": "a", "start": 40, "end": 50, "score": 0.4},   # terceiro trecho do ditado "a"
        {"at": "b", "start": 0, "end": 10, "score": 0.5},
    ]
    assert [(c["at"], c["start"]) for c in pick(candidates)] == [("a", 0), ("a", 20), ("b", 0)]
