import random
from functools import partial

import pytest

from formatter.augment import FILLERS, MARKERS, add_corrections, add_fillers, add_restart, corrupt
from formatter.tags import apply_tags, extract_tags

TEXTS = [
    "Eu vou sair amanhã.",
    "Oi, tudo bem? Vou sair.",
    "Abri o DronsioFlow",
    "Versão 3.5 saiu.",
    "",
]
AUGMENTERS = [partial(add_fillers, language="pt"), partial(add_fillers, language="en"), add_restart]


def delete_runs(tokens, actions):
    """Yields (start, end) of each run of consecutive DELETE tokens."""
    i = 0
    while i < len(tokens):
        if actions[i] == "DELETE":
            j = i
            while j < len(tokens) and actions[j] == "DELETE":
                j += 1
            yield i, j
            i = j
        else:
            i += 1


@pytest.mark.parametrize("augment", AUGMENTERS)
@pytest.mark.parametrize("text", TEXTS)
def test_prob_zero_changes_nothing(augment, text):
    tags = extract_tags(text)
    assert augment(*tags, random.Random(0), 0) == tags


@pytest.mark.parametrize("augment", AUGMENTERS)
@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("seed", range(20))
def test_deleting_what_was_added_gives_back_the_text(augment, text, seed):
    tokens, actions, puncts = augment(*extract_tags(text), random.Random(seed), 0.5)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text


@pytest.mark.parametrize("augment", AUGMENTERS)
def test_same_seed_gives_same_result(augment):
    tags = extract_tags("Oi, tudo bem? Vou sair.")
    assert augment(*tags, random.Random(7), 0.5) == augment(*tags, random.Random(7), 0.5)


@pytest.mark.parametrize("augment", AUGMENTERS)
def test_does_not_change_the_input_lists(augment):
    tokens, actions, puncts = extract_tags("Oi, tudo bem? Vou sair.")
    augment(tokens, actions, puncts, random.Random(0), 1)
    assert (tokens, actions, puncts) == extract_tags("Oi, tudo bem? Vou sair.")


@pytest.mark.parametrize("augment", AUGMENTERS)
def test_added_tokens_have_no_punctuation(augment):
    tokens, actions, puncts = augment(*extract_tags("Oi, tudo bem? Vou sair."), random.Random(0), 1)
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


def test_fillers_go_before_every_token_with_prob_one():
    tokens, actions, puncts = extract_tags("Oi, tudo bem? Vou sair.")
    new_tokens, new_actions, new_puncts = add_fillers(tokens, actions, puncts, random.Random(0), 1, "pt")
    assert len(new_tokens) == 2 * len(tokens)
    assert new_tokens[1::2] == tokens
    assert new_actions[1::2] == actions
    assert new_puncts[1::2] == puncts
    assert new_actions[0::2] == ["DELETE"] * len(tokens)
    assert all(filler in FILLERS["pt"] for filler in new_tokens[0::2])


@pytest.mark.parametrize("seed", range(20))
def test_restart_copies_one_to_three_following_tokens(seed):
    original = extract_tags("Eu vou sair amanhã cedo.")[0]
    tokens, actions, _ = add_restart(*extract_tags("Eu vou sair amanhã cedo."), random.Random(seed), 1)
    runs = list(delete_runs(tokens, actions))
    # With prob=1 every original token gets its own copy run, right before it.
    assert len(runs) == len(original)
    for index, (start, end) in enumerate(runs):
        copy = tokens[start:end]
        assert 1 <= len(copy) <= 3
        assert copy == original[index:index + len(copy)]
        assert tokens[end] == original[index]


SIMILAR = {"quarta": ["terça", "sexta"], "comprar": ["alugar", "vender"], "manhã": ["tarde", "noite"], "monday": ["tuesday"]}


def correct(text, seed, prob, language="pt"):
    return add_corrections(*extract_tags(text), random.Random(seed), prob, SIMILAR, language)


@pytest.mark.parametrize("text", TEXTS + ["Vou comprar na quarta de manhã."])
def test_corrections_with_prob_zero_change_nothing(text):
    assert correct(text, 0, 0) == extract_tags(text)


@pytest.mark.parametrize("text", TEXTS + ["Vou comprar na quarta de manhã.", "Quarta, comprar. Manhã?"])
@pytest.mark.parametrize("seed", range(20))
def test_corrections_delete_back_to_the_text(text, seed):
    tokens, actions, puncts = correct(text, seed, 0.5)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


@pytest.mark.parametrize("seed", range(20))
def test_correction_is_a_similar_word_then_a_marker(seed):
    text = "Quarta vou comprar na quarta de manhã."
    original = extract_tags(text)[0]
    tokens, actions, _ = correct(text, seed, 1)
    runs = list(delete_runs(tokens, actions))
    # Only words in SIMILAR get a correction, and with prob=1 all of them do.
    assert len(runs) == sum(t.lower() in SIMILAR for t in original)
    for start, end in runs:
        right = tokens[end]
        wrong, *marker = tokens[start:end]
        assert wrong in SIMILAR[right.lower()]
        assert " ".join(marker) in MARKERS["pt"]


def test_corrections_use_the_markers_of_the_language():
    tokens, actions, _ = correct("See you Monday morning.", 0, 1, language="en")
    start, end = next(delete_runs(tokens, actions))
    assert tokens[start] == "tuesday"
    assert " ".join(tokens[start + 1:end]) in MARKERS["en"]


def test_corrections_do_not_change_the_input_lists():
    tags = extract_tags("Vou comprar na quarta.")
    add_corrections(*tags, random.Random(0), 1, SIMILAR, "pt")
    assert tags == extract_tags("Vou comprar na quarta.")


def test_corrections_same_seed_gives_same_result():
    assert correct("Vou comprar na quarta de manhã.", 3, 0.5) == correct("Vou comprar na quarta de manhã.", 3, 0.5)


CORRUPT_TEXTS = [("Vou comprar na quarta de manhã.", "pt"), ("Oi, tudo bem? Vou sair.", "pt"), ("See you Monday morning.", "en"), ("", "pt")]


@pytest.mark.parametrize("text, language", CORRUPT_TEXTS)
@pytest.mark.parametrize("seed", range(50))
def test_corrupt_deletes_back_to_the_text(text, language, seed):
    tokens, actions, puncts = corrupt(text, random.Random(seed), language, SIMILAR)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


def test_corrupt_gives_both_clean_and_dirty_examples():
    text = "Vou comprar na quarta de manhã e enviar o arquivo amanhã."
    results = [corrupt(text, random.Random(seed), "pt", SIMILAR) for seed in range(200)]
    clean = sum(r == extract_tags(text) for r in results)
    assert 0 < clean < len(results)


def test_corrupt_same_seed_gives_same_result():
    text = "Vou comprar na quarta de manhã."
    assert corrupt(text, random.Random(5), "pt", SIMILAR) == corrupt(text, random.Random(5), "pt", SIMILAR)
