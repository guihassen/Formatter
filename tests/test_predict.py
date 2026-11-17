"""Testes de formatter/predict.py, com um modelo falso (rótulo escolhido por peça) e, se existir,
com o modelo treinado.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from formatter import predict
from formatter.dataset import LABEL2ID, LABELS
from formatter.predict import (format_text, normalize_whisper, predict_labels, predict_with_confidence,
                               to_tags, windows)

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
    """Rótulos voltam a ser ação e pontuação."""
    assert to_tags(labels) == (actions, puncts)


@pytest.fixture(scope="module")
def tokenizer():
    """Carrega o tokenizador xlm-roberta-base (o teste é pulado se não estiver disponível)."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    except OSError:
        pytest.skip("xlm-roberta-base tokenizer not available offline")


class FakeModel:
    """Modelo falso: devolve, para cada peça, o rótulo de uma lista fixa, sem olhar a entrada."""

    def __init__(self, piece_label_ids):
        self.piece_label_ids = piece_label_ids

    def __call__(self, input_ids, **kwargs):
        pieces = input_ids.shape[1]
        logits = torch.zeros(1, pieces, len(LABELS))
        for piece in range(pieces):
            logits[0, piece, self.piece_label_ids[piece % len(self.piece_label_ids)]] = 1
        return SimpleNamespace(logits=logits)


def test_predict_labels_uses_the_first_piece_of_each_word(tokenizer):
    """Só a primeira peça de cada palavra decide o rótulo."""
    # Peças: <s> ▁vou ▁ah n ▁comprar ▁p ão </s>
    K, P, D, Q = (LABEL2ID[n] for n in ["KEEP_NONE", "KEEP_PERIOD", "DELETE", "KEEP_QUESTION"])
    model = FakeModel([Q, K, D, Q, K, P, Q, Q])
    labels = predict_labels(["vou", "ahn", "comprar", "pão"], tokenizer, model)
    assert labels == ["KEEP_NONE", "DELETE", "KEEP_NONE", "KEEP_PERIOD"]


def test_predict_labels_covers_long_texts_with_windows(tokenizer):
    """Com janelas, todas as palavras de um texto longo recebem rótulo do modelo."""
    model = FakeModel([LABEL2ID["DELETE"]])
    labels = predict_labels(["palavra"] * 300, tokenizer, model)
    assert labels == ["DELETE"] * 300            # nenhuma palavra sobra para o KEEP_NONE de reserva


@pytest.mark.parametrize("pieces", [[1] * 300, [1] * 50, [3] * 200, [1, 200, 1, 1], [2] * 127, [130] * 3])
def test_windows_label_every_word_once_and_fit(pieces):
    """Cada palavra é rotulada uma vez só e cada janela cabe no limite de peças."""
    result = windows(pieces)
    kept = [i for _, _, keep_from, keep_to in result for i in range(keep_from, keep_to)]
    assert kept == list(range(len(pieces)))
    for start, end, keep_from, keep_to in result:
        assert start <= keep_from < keep_to <= end
        assert sum(pieces[start:end]) <= predict.WINDOW_PIECES or end - start == 1 or keep_to - keep_from == 1


def test_windows_keep_context_around_kept_words():
    """Nas janelas do meio, as palavras usadas têm MARGIN_WORDS de contexto dos dois lados."""
    for start, end, keep_from, keep_to in windows([1] * 300)[1:-1]:
        assert keep_from - start >= predict.MARGIN_WORDS
        assert end - keep_to >= predict.MARGIN_WORDS


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Que legal!", "Que legal."),
        ("a gente vai explorar...", "a gente vai explorar."),
        ("Fim… e depois", "Fim. e depois"),
        ("são três: a; b", "são três, a, b"),
        ("é isso?! ok", "é isso? ok"),
        ('ele disse "oi" (baixinho) - tchau', "ele disse oi baixinho tchau"),
        ("alt- path, R$3.50 ok", "alt- path, R$3.50 ok"),
        ("... !", ""),
    ],
)
def test_normalize_whisper(text, expected):
    """A pontuação do Whisper é reescrita com os sinais que o modelo conhece."""
    assert normalize_whisper(text) == expected


def test_predict_with_confidence_gives_one_margin_per_word(tokenizer):
    """Há uma nota de confiança por palavra."""
    model = FakeModel([LABEL2ID["KEEP_NONE"]])
    labels, margins = predict_with_confidence(["vou", "comprar", "pão"], tokenizer, model)
    assert len(labels) == len(margins) == 3
    assert margins == [1.0, 1.0, 1.0]


def logit_row(**scores):
    """Uma linha de logits com as notas dadas por rótulo (os outros ficam em 0)."""
    row = torch.zeros(len(LABELS))
    for label, score in scores.items():
        row[LABEL2ID[label]] = score
    return row


@pytest.mark.parametrize(
    "scores, expected",
    [
        ({"DELETE": 3.0, "KEEP_NONE": 1.0}, "DELETE"),          # folga 2 >= 1: apaga
        ({"DELETE": 1.5, "KEEP_COMMA": 1.0}, "KEEP_COMMA"),     # folga 0,5: fica o melhor KEEP_
        ({"DELETE": 2.0, "KEEP_PERIOD": 1.0}, "DELETE"),        # folga igual ao limite: apaga
        ({"KEEP_QUESTION": 2.0, "DELETE": 1.9}, "KEEP_QUESTION"),
    ],
)
def test_cautious_argmax_only_deletes_when_sure(scores, expected):
    """DELETE só vence se passar o melhor KEEP_ por pelo menos DELETE_MARGIN."""
    winner = predict.cautious_argmax(logit_row(**scores).unsqueeze(0), delete_margin=1.0)
    assert winner.tolist() == [LABEL2ID[expected]]


def test_cautious_argmax_with_zero_margin_is_plain_argmax():
    """Com limite 0 a regra some e o resultado é o argmax comum."""
    logits = torch.randn(50, len(LABELS), generator=torch.Generator().manual_seed(0))
    assert torch.equal(predict.cautious_argmax(logits, delete_margin=0.0), logits.argmax(dim=-1))


def test_format_text_of_empty_text_does_not_call_the_model():
    """Texto vazio (ou só pontuação) devolve "" sem chamar o modelo."""
    def explode(*args, **kwargs):
        raise AssertionError("model should not be called")

    assert format_text("", explode, explode) == ""
    assert format_text("   ", explode, explode) == ""
    assert format_text("... !", explode, explode) == ""


def test_format_text_applies_the_predicted_tags(tokenizer, monkeypatch):
    """Os rótulos previstos viram o texto final certo."""
    monkeypatch.setattr(predict, "predict_labels",
                        lambda tokens, tok, model: ["KEEP_NONE", "DELETE", "KEEP_NONE", "KEEP_PERIOD", "KEEP_QUESTION"])
    assert format_text("eu ahn vou embora agora", tokenizer, None) == "Eu vou embora. Agora?"


@pytest.fixture(scope="module")
def trained():
    """Carrega o modelo treinado (o teste é pulado se ele não existir)."""
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("trained model not available")
    return predict.load_model()


def test_trained_model_formats_a_dictation(trained):
    """Com o modelo real: tira a hesitação, pontua e reconhece a pergunta."""
    tokenizer, model = trained
    text = format_text("ahn eu vou comprar pão amanhã você vem", tokenizer, model)
    assert "ahn" not in text.lower()
    assert text.startswith("Eu vou comprar pão")
    assert text.endswith("?")


def test_trained_model_gives_one_label_per_word(trained):
    """Com o modelo real: um rótulo válido por palavra."""
    tokenizer, model = trained
    words = "então a reunião é na terça às três horas".split()
    labels = predict_labels(words, tokenizer, model)
    assert len(labels) == len(words)
    assert set(labels) <= set(LABELS)
