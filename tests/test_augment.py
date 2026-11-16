"""Testes de formatter/augment.py.

A garantia principal: tudo o que corrupt e as funções add_* inserem é DELETE, então apagar essas
palavras devolve o texto original. enrich, ao contrário, só acrescenta palavras que ficam.
"""

import random
from functools import partial

import pytest

from formatter import augment
from formatter.augment import (CLAUSE_MARKERS, DISCOURSE, FILLERS, MARKERS, add_clause_correction, add_corrections,
                               add_discourse, add_false_start, add_fillers, add_parallel, add_restart, clause_starts,
                               corrupt, enrich, sentence_starts)
from formatter.tags import apply_tags, extract_tags

TEXTS = [
    "Eu vou sair amanhã.",
    "Oi, tudo bem? Vou sair.",
    "Abri o DronsioFlow",
    "Versão 3.5 saiu.",
    "",
]
AUGMENTERS = [partial(add_fillers, language="pt"), partial(add_fillers, language="en"), add_restart]
DELETE_AUGMENTERS = AUGMENTERS + [
    partial(add_clause_correction, similar={"quarta": ["terça"], "sair": ["voltar"]}, language="pt"),
    partial(add_false_start, similar={"quarta": ["terça"], "sair": ["voltar"]}),
]


def delete_runs(tokens, actions):
    """Devolve (início, fim) de cada sequência seguida de palavras DELETE."""
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
    """Com probabilidade zero, nada muda."""
    tags = extract_tags(text)
    assert augment(*tags, random.Random(0), 0) == tags


@pytest.mark.parametrize("augment", AUGMENTERS)
@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("seed", range(20))
def test_deleting_what_was_added_gives_back_the_text(augment, text, seed):
    """Apagar as palavras inseridas devolve o texto original."""
    tokens, actions, puncts = augment(*extract_tags(text), random.Random(seed), 0.5)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text


@pytest.mark.parametrize("augment", AUGMENTERS)
def test_same_seed_gives_same_result(augment):
    """A mesma semente gera sempre o mesmo resultado."""
    tags = extract_tags("Oi, tudo bem? Vou sair.")
    assert augment(*tags, random.Random(7), 0.5) == augment(*tags, random.Random(7), 0.5)


@pytest.mark.parametrize("augment", AUGMENTERS)
def test_does_not_change_the_input_lists(augment):
    """As funções não alteram as listas que recebem."""
    tokens, actions, puncts = extract_tags("Oi, tudo bem? Vou sair.")
    augment(tokens, actions, puncts, random.Random(0), 1)
    assert (tokens, actions, puncts) == extract_tags("Oi, tudo bem? Vou sair.")


@pytest.mark.parametrize("augment", AUGMENTERS)
def test_added_tokens_have_no_punctuation(augment):
    """Palavras inseridas para apagar nunca têm pontuação."""
    tokens, actions, puncts = augment(*extract_tags("Oi, tudo bem? Vou sair."), random.Random(0), 1)
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


def test_fillers_go_before_every_token_with_prob_one():
    """Com probabilidade 1, há uma hesitação antes de cada palavra."""
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
    """Um recomeço é a cópia das 1 a 3 palavras seguintes, logo antes delas."""
    original = extract_tags("Eu vou sair amanhã cedo.")[0]
    tokens, actions, _ = add_restart(*extract_tags("Eu vou sair amanhã cedo."), random.Random(seed), 1)
    runs = list(delete_runs(tokens, actions))
    # Com prob=1, cada palavra original ganha a sua própria cópia, logo antes dela.
    assert len(runs) == len(original)
    for index, (start, end) in enumerate(runs):
        copy = tokens[start:end]
        assert 1 <= len(copy) <= 3
        assert copy == original[index:index + len(copy)]
        assert tokens[end] == original[index]


SIMILAR = {"quarta": ["terça", "sexta"], "comprar": ["alugar", "vender"], "manhã": ["tarde", "noite"], "monday": ["tuesday"]}


def correct(text, seed, prob, language="pt"):
    """Atalho: aplica add_corrections a um texto com as parecidas de teste."""
    return add_corrections(*extract_tags(text), random.Random(seed), prob, SIMILAR, language)


@pytest.mark.parametrize("text", TEXTS + ["Vou comprar na quarta de manhã."])
def test_corrections_with_prob_zero_change_nothing(text):
    """Correções com probabilidade zero não mudam nada."""
    assert correct(text, 0, 0) == extract_tags(text)


@pytest.mark.parametrize("text", TEXTS + ["Vou comprar na quarta de manhã.", "Quarta, comprar. Manhã?"])
@pytest.mark.parametrize("seed", range(20))
def test_corrections_delete_back_to_the_text(text, seed):
    """Apagar as correções devolve o texto original."""
    tokens, actions, puncts = correct(text, seed, 0.5)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


@pytest.mark.parametrize("seed", range(20))
def test_correction_is_a_similar_word_then_a_marker(seed, monkeypatch):
    """Uma correção é: palavras anteriores (opcional), palavra parecida errada e um gatilho."""
    monkeypatch.setattr(augment, "GROUP_SIZES", [0])   # uma correção por sequência, para poder contar
    text = "Quarta vou comprar na quarta de manhã."
    original = extract_tags(text)[0]
    tokens, actions, _ = correct(text, seed, 1)
    runs = list(delete_runs(tokens, actions))
    # Só palavras que estão em SIMILAR recebem correção, e com prob=1 todas recebem.
    assert len(runs) == sum(t.lower() in SIMILAR for t in original)
    for start, end in runs:
        run = tokens[start:end]
        # A sequência pode começar com 0 a 2 palavras copiadas de logo antes da palavra corrigida ("na terça não na").
        lead = next(size for size in range(3) if run[size] in SIMILAR.get(tokens[end + size].lower(), []))
        assert run[:lead] == tokens[end:end + lead]
        assert run[lead] in SIMILAR[tokens[end + lead].lower()]
        assert " ".join(run[lead + 1:]) in MARKERS["pt"]


def test_corrections_sometimes_take_the_words_before():
    """As correções às vezes levam 0, 1 ou 2 palavras anteriores ("na terça não na quarta")."""
    text = "Eu vou comprar na quarta de manhã."
    leads = set()
    for seed in range(100):
        tokens, actions, _ = correct(text, seed, 1)
        for start, end in delete_runs(tokens, actions):
            run = tokens[start:end]
            leads.add(next(size for size in range(3) if run[size] in SIMILAR.get(tokens[end + size].lower(), [])))
    assert leads == {0, 1, 2}


def test_corrections_use_the_markers_of_the_language(monkeypatch):
    """Correções em inglês usam os gatilhos em inglês."""
    monkeypatch.setattr(augment, "GROUP_SIZES", [0])
    tokens, actions, _ = correct("See you Monday morning.", 0, 1, language="en")
    start, end = next(delete_runs(tokens, actions))
    assert tokens[start] == "tuesday"
    assert " ".join(tokens[start + 1:end]) in MARKERS["en"]


def test_corrections_do_not_change_the_input_lists():
    """add_corrections não altera as listas que recebe."""
    tags = extract_tags("Vou comprar na quarta.")
    add_corrections(*tags, random.Random(0), 1, SIMILAR, "pt")
    assert tags == extract_tags("Vou comprar na quarta.")


def test_corrections_same_seed_gives_same_result():
    """Mesma semente, mesmas correções."""
    assert correct("Vou comprar na quarta de manhã.", 3, 0.5) == correct("Vou comprar na quarta de manhã.", 3, 0.5)


CORRUPT_TEXTS = [("Vou comprar na quarta de manhã.", "pt"), ("Oi, tudo bem? Vou sair.", "pt"), ("See you Monday morning.", "en"), ("", "pt")]


@pytest.mark.parametrize("text, language", CORRUPT_TEXTS)
@pytest.mark.parametrize("seed", range(50))
def test_corrupt_deletes_back_to_the_text(text, language, seed):
    """corrupt completo: apagar o que foi inserido devolve o texto."""
    tokens, actions, puncts = corrupt(text, random.Random(seed), language, SIMILAR)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


def test_corrupt_gives_both_clean_and_dirty_examples():
    """corrupt gera tanto exemplos limpos quanto com erros."""
    text = "Vou comprar na quarta de manhã e enviar o arquivo amanhã."
    results = [corrupt(text, random.Random(seed), "pt", SIMILAR) for seed in range(200)]
    clean = sum(r == extract_tags(text) for r in results)
    assert 0 < clean < len(results)


def test_corrupt_same_seed_gives_same_result():
    """Mesma semente, mesmo resultado de corrupt."""
    text = "Vou comprar na quarta de manhã."
    assert corrupt(text, random.Random(5), "pt", SIMILAR) == corrupt(text, random.Random(5), "pt", SIMILAR)


LONG_TEXT = "Eu vou sair amanhã cedo, depois volto para casa. Você vem comigo na quarta de manhã?"


@pytest.mark.parametrize("augment", DELETE_AUGMENTERS)
@pytest.mark.parametrize("text", TEXTS + [LONG_TEXT])
@pytest.mark.parametrize("seed", range(20))
def test_new_augmenters_delete_back_to_the_text(augment, text, seed):
    """Correção de oração e falso começo também voltam ao texto original quando apagados."""
    tokens, actions, puncts = augment(*extract_tags(text), random.Random(seed), 0.7)
    assert len(tokens) == len(actions) == len(puncts)
    assert apply_tags(tokens, actions, puncts) == text
    assert all(p == "NONE" for a, p in zip(actions, puncts) if a == "DELETE")


@pytest.mark.parametrize("augment", DELETE_AUGMENTERS)
def test_new_augmenters_with_prob_zero_change_nothing(augment):
    """Correção de oração e falso começo com probabilidade zero não mudam nada."""
    tags = extract_tags(LONG_TEXT)
    assert augment(*tags, random.Random(0), 0) == tags


def test_sentence_and_clause_starts():
    """Encontra onde começam as frases e as orações."""
    _, _, puncts = extract_tags(LONG_TEXT)
    assert sentence_starts(puncts) == [0, 9]
    assert clause_starts(puncts) == [0, 5, 9]


@pytest.mark.parametrize("seed", range(30))
def test_clause_correction_is_a_wrong_idea_then_a_clause_marker(seed):
    """Correção de oração: ideia errada de 3 a 15 palavras e um gatilho, logo antes de uma oração real."""
    similar = {"amanhã": ["hoje"], "cedo": ["tarde"], "quarta": ["terça"]}
    original = extract_tags(LONG_TEXT)
    tokens, actions, _ = add_clause_correction(*original, random.Random(seed), 1, similar, "pt")
    for start, end in delete_runs(tokens, actions):
        run = tokens[start:end]
        marker = next(m for m in sorted(CLAUSE_MARKERS["pt"], key=len, reverse=True) if " ".join(run).endswith(m))
        assert 3 <= len(run) - len(marker.split()) <= 15
        # Logo depois da sequência começa uma oração real.
        kept_before = sum(a == "KEEP" for a in actions[:start])
        assert kept_before in clause_starts(original[2])


@pytest.mark.parametrize("seed", range(20))
def test_clause_correction_changes_some_words_of_the_clause(seed):
    """A ideia errada é o começo da oração com pelo menos uma palavra trocada por parecida."""
    similar = {"amanhã": ["hoje"], "cedo": ["tarde"]}
    original = extract_tags("Amanhã cedo eu vou sair.")[0]
    tokens, actions, _ = add_clause_correction(*extract_tags("Amanhã cedo eu vou sair."), random.Random(seed), 1,
                                               similar, "pt")
    run = [t for t, a in zip(tokens, actions) if a == "DELETE"]
    marker = next(m for m in sorted(CLAUSE_MARKERS["pt"], key=len, reverse=True) if " ".join(run).endswith(m))
    wrong = run[:len(run) - len(marker.split())]
    assert {"hoje", "tarde"} & set(wrong)               # pelo menos uma palavra foi trocada
    assert all(w == o or w in similar.get(o, []) for w, o in zip(wrong, original))


@pytest.mark.parametrize("seed", range(30))
def test_false_start_comes_right_before_a_sentence(seed):
    """O falso começo tem de 1 a 5 palavras e vem logo antes de uma frase."""
    original = extract_tags(LONG_TEXT)
    tokens, actions, _ = add_false_start(*original, random.Random(seed), 1, {})
    for start, end in delete_runs(tokens, actions):
        assert 1 <= end - start <= 5
        assert sum(a == "KEEP" for a in actions[:start]) in sentence_starts(original[2])


@pytest.mark.parametrize("seed", range(30))
def test_parallel_repeats_a_phrase_with_a_new_last_word(seed):
    """A enumeração repete um trecho com a última palavra trocada, e tudo fica (KEEP)."""
    similar = {"amanhã": ["hoje"], "casa": ["escola"], "quarta": ["terça"]}
    tokens, actions, puncts = add_parallel(*extract_tags(LONG_TEXT), random.Random(seed), 1, similar)
    assert set(actions) == {"KEEP"}
    text = apply_tags(tokens, actions, puncts)
    assert any(f"{word}, " in text for word in ["amanhã", "casa", "quarta"])
    assert len(tokens) > len(extract_tags(LONG_TEXT)[0])


def test_parallel_example():
    """Exemplo concreto: "o arroz" ganha uma repetição terminada em "feijão"."""
    tags = extract_tags("Compre o arroz.")
    tokens, actions, puncts = add_parallel(*tags, random.Random(0), 1, {"arroz": ["feijão"]})
    assert apply_tags(tokens, actions, puncts).startswith("Compre ")
    assert apply_tags(tokens, actions, puncts).endswith("o arroz, " + " ".join(tokens[tokens.index("arroz") + 1:]) + ".")
    assert tokens[-1] == "feijão"


def test_discourse_marker_is_kept_with_a_comma():
    """"Na verdade," no começo da frase fica no texto, com vírgula."""
    tokens, actions, puncts = add_discourse(*extract_tags("Eu vou sair. Você vem?"), random.Random(0), 1, "pt")
    assert apply_tags(tokens, actions, puncts) == "Na verdade, eu vou sair. Na verdade, você vem?"
    assert set(actions) == {"KEEP"}


@pytest.mark.parametrize("seed", range(20))
def test_enrich_only_adds_kept_words_and_is_clean_text(seed):
    """enrich devolve texto limpo válido e nunca menor que o original."""
    text = "Vou comprar na quarta de manhã. Você vem?"
    enriched = enrich(text, random.Random(seed), "pt", SIMILAR)
    assert apply_tags(*extract_tags(enriched)) == enriched
    assert len(enriched.split()) >= len(text.split())


def test_enrich_with_zero_probs_changes_nothing(monkeypatch):
    """Com as probabilidades zeradas, enrich não muda o texto."""
    monkeypatch.setattr(augment, "PARALLEL_PROB", 0)
    monkeypatch.setattr(augment, "DISCOURSE_PROB", 0)
    assert enrich("Vou comprar na quarta. Você vem?", random.Random(0), "pt", SIMILAR) == "Vou comprar na quarta. Você vem?"
