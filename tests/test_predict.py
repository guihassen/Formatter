from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from formatter import predict
from formatter.dataset import LABEL2ID, LABELS
from formatter.predict import format_text, predict_labels, to_tags

MODEL_DIR = Path(predict.MODEL_DIR)


@pytest.mark.parametrize(
    "labels, actions, puncts",
    [
        (["KEEP_NONE", "DELETE", "KEEP_COMMA", "KEEP_PERIOD", "KEEP_QUESTION"],
         ["KEEP", "DELETE", "KEEP", "KEEP", "KEEP"],
         ["NONE", "NONE", "COMMA", "PERIOD", "QUESTION"]),
        ([], [], []),
    ],
)
def test_to_tags(labels, actions, puncts):
    assert to_tags(labels) == (actions, puncts)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    except OSError:
        pytest.skip("xlm-roberta-base tokenizer not available offline")


class FakeModel:
    """Predicts a fixed label id for each piece, in order, ignoring the input."""

    def __init__(self, piece_label_ids):
        self.piece_label_ids = piece_label_ids

    def __call__(self, input_ids, **kwargs):
        pieces = input_ids.shape[1]
        logits = torch.zeros(1, pieces, len(LABELS))
        for piece in range(pieces):
            logits[0, piece, self.piece_label_ids[piece % len(self.piece_label_ids)]] = 1
        return SimpleNamespace(logits=logits)


def test_predict_labels_uses_the_first_piece_of_each_word(tokenizer):
    # Pieces: <s> ▁vou ▁ah n ▁comprar ▁p ão </s>
    K, P, D, Q = (LABEL2ID[n] for n in ["KEEP_NONE", "KEEP_PERIOD", "DELETE", "KEEP_QUESTION"])
    model = FakeModel([Q, K, D, Q, K, P, Q, Q])
    labels = predict_labels(["vou", "ahn", "comprar", "pão"], tokenizer, model)
    assert labels == ["KEEP_NONE", "DELETE", "KEEP_NONE", "KEEP_PERIOD"]


def test_predict_labels_fills_words_cut_by_truncation(tokenizer):
    model = FakeModel([LABEL2ID["DELETE"]])
    labels = predict_labels(["palavra"] * 300, tokenizer, model)
    assert len(labels) == 300
    assert set(labels[:100]) == {"DELETE"}
    assert set(labels[-100:]) == {"KEEP_NONE"}


def test_format_text_of_empty_text_does_not_call_the_model():
    def explode(*args, **kwargs):
        raise AssertionError("model should not be called")

    assert format_text("", explode, explode) == ""
    assert format_text("   ", explode, explode) == ""


def test_format_text_applies_the_predicted_tags(tokenizer, monkeypatch):
    monkeypatch.setattr(predict, "predict_labels",
                        lambda tokens, tok, model: ["KEEP_NONE", "DELETE", "KEEP_NONE", "KEEP_PERIOD", "KEEP_QUESTION"])
    assert format_text("eu ahn vou embora agora", tokenizer, None) == "Eu vou embora. Agora?"


@pytest.fixture(scope="module")
def trained():
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("trained model not available")
    return predict.load_model()


def test_trained_model_formats_a_dictation(trained):
    tokenizer, model = trained
    text = format_text("ahn eu vou comprar pão amanhã você vem", tokenizer, model)
    assert "ahn" not in text.lower()
    assert text.startswith("Eu vou comprar pão")
    assert text.endswith("?")


def test_trained_model_gives_one_label_per_word(trained):
    tokenizer, model = trained
    words = "então a reunião é na terça às três horas".split()
    labels = predict_labels(words, tokenizer, model)
    assert len(labels) == len(words)
    assert set(labels) <= set(LABELS)
