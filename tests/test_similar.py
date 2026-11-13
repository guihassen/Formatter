import json

import numpy as np
import pytest

from formatter import similar as sim
from formatter.augment import FILLERS, MARKERS
from formatter.similar import blocked_words, build_similar, load_similar, read_sentences, save_similar, train


class FakeVectors:
    """Stands in for model.wv with fixed neighbors, so tests don't depend on training."""

    def __init__(self, neighbors):
        self.neighbors = neighbors
        self.index_to_key = list(neighbors)

    def most_similar(self, word, topn=10):
        return self.neighbors[word][:topn]


class FakeModel:
    def __init__(self, neighbors):
        self.wv = FakeVectors(neighbors)


def test_read_sentences_lowercases_and_strips_punctuation(tmp_path):
    path = tmp_path / "train_pt.jsonl"
    rows = [{"text": "Oi, tudo bem? Vou sair.", "language": "pt"}, {"text": "Versão 3.5 saiu.", "language": "pt"}]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert read_sentences(path) == [["oi", "tudo", "bem", "vou", "sair"], ["versão", "3.5", "saiu"]]


@pytest.mark.parametrize("language", ["pt", "en"])
def test_blocked_words_has_fillers_and_every_marker_word(language):
    blocked = blocked_words(language)
    assert set(FILLERS[language]) <= blocked
    for marker in MARKERS[language]:
        assert set(marker.split()) <= blocked
    # Multiword markers are split, never stored whole.
    assert all(" " not in word for word in blocked)


def test_build_similar_applies_every_filter():
    model = FakeModel({
        "quarta": [("sexta", 0.9), ("quarta-feira", 0.9), ("não", 0.8), ("terça", 0.7), ("mesa", 0.4)],
        "de": [("da", 0.9)],          # too short
        "tipo": [("jeito", 0.9)],     # blocked word
        "carro": [("mesa", 0.3)],     # nothing passes the threshold
    })
    assert build_similar(model, "pt") == {"quarta": ["sexta", "terça"]}


def test_build_similar_keeps_the_best_top_n():
    neighbors = [(f"palavra{i}", 0.9 - i / 100) for i in range(10)]
    model = FakeModel({"quarta": neighbors})
    assert build_similar(model, "pt")["quarta"] == [w for w, _ in neighbors[:sim.TOP_N]]


def test_build_similar_keeps_the_threshold_itself():
    model = FakeModel({"quarta": [("sexta", sim.MIN_SIMILARITY)]})
    assert build_similar(model, "pt") == {"quarta": ["sexta"]}


def test_train_learns_the_vocabulary_and_is_reproducible(monkeypatch):
    monkeypatch.setattr(sim, "MIN_COUNT", 1)
    sentences = [["vou", "na", "quarta"], ["vou", "na", "sexta"], ["vou", "na", "terça"]] * 20
    first, second = train(sentences), train(sentences)
    assert set(first.wv.index_to_key) == {"vou", "na", "quarta", "sexta", "terça"}
    assert first.wv.vector_size == sim.VECTOR_SIZE
    assert np.array_equal(first.wv["quarta"], second.wv["quarta"])


def test_save_then_load_gives_the_same_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "PROCESSED_DIR", tmp_path)
    similar = {"manhã": ["tarde", "noite"], "quarta": ["sexta"]}
    save_similar(similar, tmp_path / "similar_pt.json")
    assert load_similar("pt") == similar
    # Accents are written as-is, not as \u escapes.
    assert "manhã" in (tmp_path / "similar_pt.json").read_text(encoding="utf-8")
