import json
import random

import pytest

from formatter import corpus
from formatter.corpus import MAX_SENTENCES, build, clean_line, make_examples


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Eu vou sair amanhã.", "Eu vou sair amanhã."),
        ("Oi, tudo bem com você?", "Oi, tudo bem com você?"),
        ("Eu   vou\tsair amanhã.", "Eu vou sair amanhã."),
        ("Isso é ótimo!", "Isso é ótimo."),
        ("Você está louco?!", "Você está louco?"),
        ("Você está louco!?", "Você está louco?"),
        ("Eu não sei...", "Eu não sei."),
        ("Versão 3.5 saiu hoje.", "Versão 3.5 saiu hoje."),
    ],
)
def test_clean_line_keeps_and_normalizes(line, expected):
    assert clean_line(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Não sei.",  # too short
        " ".join(["palavra"] * 41) + ".",  # too long
        "Ela disse: vamos embora.",
        "Ele falou \"oi\" para mim.",
        "Eu (acho) que sim.",
        "Eu vou — talvez — amanhã.",
        "Primeiro isso; depois aquilo.",
        "Eu vou , sim.",  # lone punctuation
        "Eu vou sair.,",  # two punctuation marks at the end
        "eu vou sair.",  # would come back capitalized
    ],
)
def test_clean_line_rejects(line):
    assert clean_line(line) is None


@pytest.mark.parametrize("line", ["Eu vou sair amanhã.", "Isso é ótimo!", "Oi, tudo bem com você?"])
def test_clean_lines_survive_the_round_trip(line):
    from formatter.tags import apply_tags, extract_tags

    cleaned = clean_line(line)
    assert apply_tags(*extract_tags(cleaned)) == cleaned


SENTENCES = [f"Frase número {i}." for i in range(50)]


@pytest.mark.parametrize("seed", range(20))
def test_make_examples_keeps_every_sentence_in_order(seed):
    examples = make_examples(SENTENCES, random.Random(seed))
    assert " ".join(examples) == " ".join(SENTENCES)


@pytest.mark.parametrize("seed", range(20))
def test_make_examples_groups_one_to_max_sentences(seed):
    examples = make_examples(SENTENCES, random.Random(seed))
    sizes = [example.count("Frase") for example in examples]
    assert all(1 <= size <= MAX_SENTENCES for size in sizes)
    assert sum(sizes) == len(SENTENCES)


def test_make_examples_uses_different_sizes():
    examples = make_examples(SENTENCES, random.Random(0))
    assert len({example.count("Frase") for example in examples}) > 1


def test_make_examples_same_seed_gives_same_result():
    assert make_examples(SENTENCES, random.Random(3)) == make_examples(SENTENCES, random.Random(3))


def test_make_examples_empty():
    assert make_examples([], random.Random(0)) == []


def test_build_writes_train_and_val(tmp_path, monkeypatch):
    raw = tmp_path / "por_sentences.tsv"
    good = [f"Eu comprei {i} maçãs hoje." for i in range(100)]
    bad = ["Não sei.", "Ela disse: vamos embora."]
    raw.write_text("".join(f"{i}\tpor\t{s}\n" for i, s in enumerate(good + bad)), encoding="utf-8")
    monkeypatch.setattr(corpus, "PROCESSED_DIR", tmp_path)

    build(raw, "pt", random.Random(42))

    rows = {}
    for name in ("train", "val"):
        lines = (tmp_path / f"{name}_pt.jsonl").read_text(encoding="utf-8").splitlines()
        rows[name] = [json.loads(line) for line in lines]
        assert rows[name], f"{name} is empty"
        assert all(row["language"] == "pt" for row in rows[name])

    # Every good sentence ends up in exactly one split, and no bad one gets in.
    text = " ".join(row["text"] for split in rows.values() for row in split)
    assert all(text.count(s) == 1 for s in good)
    assert "Não sei." not in text and "disse:" not in text
    # Accents are written as-is, not as \u escapes.
    assert "maçãs" in (tmp_path / "train_pt.jsonl").read_text(encoding="utf-8")
