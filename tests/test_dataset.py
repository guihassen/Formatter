"""Testes de formatter/dataset.py: rótulos, criação dos exemplos e alinhamento com o tokenizador."""

import json
import random

import pandas as pd
import pytest

from formatter import dataset
from formatter.dataset import (
    ID2LABEL, IGNORE, LABEL2ID, LABELS, build_split, make_example, save_examples, to_labels, tokenize_and_align,
)
from formatter.tags import apply_tags

SIMILAR = {"quarta": ["terça", "sexta"], "comprar": ["alugar"], "monday": ["tuesday"]}


def test_label_maps_are_inverse():
    """LABEL2ID e ID2LABEL são inversos um do outro."""
    assert len(LABELS) == 5
    assert all(ID2LABEL[LABEL2ID[label]] == label for label in LABELS)


@pytest.mark.parametrize(
    "actions, puncts, expected",
    [
        (["KEEP", "DELETE", "KEEP", "KEEP"], ["NONE", "NONE", "COMMA", "QUESTION"],
         ["KEEP_NONE", "DELETE", "KEEP_COMMA", "KEEP_QUESTION"]),
        (["KEEP"], ["PERIOD"], ["KEEP_PERIOD"]),
        (["DELETE"], ["COMMA"], ["DELETE"]),  # a pontuação de uma palavra apagada some
        ([], [], []),
    ],
)
def test_to_labels(actions, puncts, expected):
    """Ação e pontuação viram o rótulo certo (a pontuação de uma palavra apagada some)."""
    assert to_labels(actions, puncts) == expected


def labels_back_to_text(tokens, label_ids):
    """Converte rótulos numéricos de volta em texto, para comparar com o original."""
    names = [ID2LABEL[i] for i in label_ids]
    actions = ["DELETE" if n == "DELETE" else "KEEP" for n in names]
    puncts = ["NONE" if n == "DELETE" else n.removeprefix("KEEP_") for n in names]
    return apply_tags(tokens, actions, puncts)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("text, language", [("Vou comprar na quarta. Você vem?", "pt"), ("See you Monday, ok?", "en")])
def test_make_example_labels_give_back_the_text(text, language, seed, monkeypatch):
    """Sem enrich, os rótulos de um exemplo reconstroem o texto original."""
    monkeypatch.setattr(dataset, "enrich", lambda text, *args: text)   # só as palavras apagadas, nada acrescentado
    example = make_example(text, language, random.Random(seed), SIMILAR)
    assert set(example) == {"tokens", "labels", "language"}
    assert example["language"] == language
    assert len(example["tokens"]) == len(example["labels"])
    assert all(isinstance(i, int) for i in example["labels"])
    assert labels_back_to_text(example["tokens"], example["labels"]) == text


def test_make_example_labels_give_back_the_enriched_text():
    """Com enrich, os rótulos reconstroem o texto enriquecido (que é o novo texto certo)."""
    from formatter.augment import enrich

    text = "Vou comprar na quarta de manhã. Você vem?"
    for seed in range(20):
        enriched = enrich(text, random.Random(seed), "pt", SIMILAR)
        example = make_example(text, "pt", random.Random(seed), SIMILAR)
        assert labels_back_to_text(example["tokens"], example["labels"]) == enriched


def test_make_example_same_seed_gives_same_result():
    """Mesma semente, mesmo exemplo."""
    text = "Vou comprar na quarta de manhã."
    assert make_example(text, "pt", random.Random(4), SIMILAR) == make_example(text, "pt", random.Random(4), SIMILAR)


@pytest.fixture
def fake_processed(tmp_path, monkeypatch):
    """Cria arquivos de corpus falsos numa pasta temporária e troca as parecidas por uma lista fixa."""
    texts = {
        "pt": [f"Eu comprei {i} maçãs hoje." for i in range(30)],
        "en": [f"I bought {i} apples today." for i in range(20)],
    }
    for language, rows in texts.items():
        pd.DataFrame({"text": rows, "language": language}).to_json(
            tmp_path / f"train_{language}.jsonl", orient="records", lines=True, force_ascii=False)
    monkeypatch.setattr(dataset, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(dataset, "load_similar", lambda language: SIMILAR)
    return texts


def test_build_split_has_every_text_of_every_language(fake_processed, monkeypatch):
    """build_split usa todos os textos dos dois idiomas."""
    monkeypatch.setattr(dataset, "enrich", lambda text, *args: text)
    examples = build_split("train", 1)
    assert isinstance(examples, list)
    assert len(examples) == 50
    rebuilt = {labels_back_to_text(e["tokens"], e["labels"]) for e in examples}
    assert rebuilt == set(fake_processed["pt"]) | set(fake_processed["en"])


def test_build_split_mixes_the_languages(fake_processed):
    """Os exemplos saem misturados, não um idioma depois do outro."""
    languages = [e["language"] for e in build_split("train", 1)]
    # Não pode ser todo o pt primeiro e depois todo o en.
    assert languages != sorted(languages, key=["pt", "en"].index)


def test_build_split_same_seed_gives_same_result(fake_processed):
    """Mesma semente, mesma divisão."""
    assert build_split("train", 1) == build_split("train", 1)


def test_save_examples_writes_one_json_per_line(tmp_path):
    """save_examples grava um JSON por linha, com acentos."""
    examples = [{"tokens": ["manhã"], "labels": [2], "language": "pt"},
                {"tokens": ["ok"], "labels": [3], "language": "en"}]
    path = tmp_path / "tagged.jsonl"
    save_examples(examples, path)
    text = path.read_text(encoding="utf-8")
    assert [json.loads(line) for line in text.splitlines()] == examples
    assert "manhã" in text


@pytest.fixture(scope="module")
def tokenizer():
    """Carrega o tokenizador xlm-roberta-base (o teste é pulado se não estiver disponível)."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    except OSError:
        pytest.skip("xlm-roberta-base tokenizer not available offline")


def test_tokenize_and_align_labels_only_the_first_piece(tokenizer):
    """Só a primeira peça de cada palavra recebe rótulo; o resto é IGNORE."""
    batch = {"tokens": [["vou", "ahn", "comprar", "na", "quarta"], ["abri", "o", "DronsioFlow"]],
             "labels": [[0, 4, 0, 0, 2], [0, 0, 2]]}
    encoded = tokenize_and_align(batch, tokenizer)
    assert encoded["labels"] == [
        [IGNORE, 0, 4, IGNORE, 0, 0, 2, IGNORE],
        [IGNORE, 0, 0, 2, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE],
    ]


@pytest.mark.parametrize("seed", range(10))
def test_tokenize_and_align_keeps_one_label_per_word(tokenizer, seed):
    """Depois do alinhamento, continua havendo exatamente um rótulo por palavra."""
    example = make_example("Vou comprar o DronsioFlow na quarta. Você vem?", "pt", random.Random(seed), SIMILAR)
    encoded = tokenize_and_align({"tokens": [example["tokens"]], "labels": [example["labels"]]}, tokenizer)
    labels = encoded["labels"][0]
    assert len(labels) == len(encoded["input_ids"][0])
    assert [label for label in labels if label != IGNORE] == example["labels"]


def test_tokenize_and_align_cuts_long_examples(tokenizer):
    """Textos longos são cortados em MAX_LENGTH peças."""
    words = ["palavra"] * 1000
    encoded = tokenize_and_align({"tokens": [words], "labels": [[0] * 1000]}, tokenizer)
    assert len(encoded["input_ids"][0]) == len(encoded["labels"][0]) == dataset.MAX_LENGTH
