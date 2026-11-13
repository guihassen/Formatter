import pytest

from formatter.tags import apply_tags, extract_tags

K, D = "KEEP", "DELETE"
N, C, P, Q = "NONE", "COMMA", "PERIOD", "QUESTION"


@pytest.mark.parametrize(
    "text, actions, puncts, expected",
    [
        ("marca na terça não na quarta", [K, D, D, D, K, K], [N, N, N, N, N, P], "Marca na quarta."),
        ("você vem amanhã", [K, K, K], [N, N, Q], "Você vem amanhã?"),
        ("vou sair depois eu ligo", [K, K, K, K, K], [N, P, N, N, P], "Vou sair. Depois eu ligo."),
        ("abri o DronsioFlow", [K, K, K], [N, N, P], "Abri o DronsioFlow."),
        ("água gelada", [K, K], [N, N], "Água gelada"),
        ("oi tudo bem", [K, K, K], [C, N, Q], "Oi, tudo bem?"),
        # A deleted token drops its punctuation.
        ("vou tipo sair", [K, D, K], [N, C, N], "Vou sair"),
        # Capitalization skips deleted tokens.
        ("tipo vou sair", [D, K, K], [N, N, P], "Vou sair."),
        ("vou sair tipo depois", [K, K, D, K], [N, P, N, P], "Vou sair. Depois."),
        ("tipo né", [D, D], [N, N], ""),
        ("", [], [], ""),
    ],
)
def test_apply_tags(text, actions, puncts, expected):
    assert apply_tags(text.split(), actions, puncts) == expected


@pytest.mark.parametrize(
    "tokens, actions, puncts",
    [
        (["oi"], [K, K], [N]),
        (["oi"], [K], [N, N]),
        (["oi", "tudo"], [K], [N]),
    ],
)
def test_rejects_different_lengths(tokens, actions, puncts):
    with pytest.raises(ValueError):
        apply_tags(tokens, actions, puncts)


def test_does_not_change_the_input_lists():
    tokens, actions, puncts = ["oi", "tipo"], [K, D], [N, N]
    apply_tags(tokens, actions, puncts)
    assert (tokens, actions, puncts) == (["oi", "tipo"], [K, D], [N, N])


@pytest.mark.parametrize(
    "text, tokens, puncts",
    [
        ("Você vem amanhã?", ["você", "vem", "amanhã"], [N, N, Q]),
        ("Oi, tudo bem?", ["oi", "tudo", "bem"], [C, N, Q]),
        ("Vou sair. Depois eu ligo.", ["vou", "sair", "depois", "eu", "ligo"], [N, P, N, N, P]),
        ("Abri o DronsioFlow", ["abri", "o", "DronsioFlow"], [N, N, N]),
        # Punctuation inside a token stays.
        ("Versão 3.5 saiu.", ["versão", "3.5", "saiu"], [N, N, P]),
        ("Água gelada", ["água", "gelada"], [N, N]),
        # Only the first letter is lowercased.
        ("DronsioFlow abriu.", ["dronsioFlow", "abriu"], [N, P]),
        ("Oi,  tudo bem?", ["oi", "tudo", "bem"], [C, N, Q]),
        ("", [], []),
    ],
)
def test_extract_tags(text, tokens, puncts):
    assert extract_tags(text) == (tokens, [K] * len(tokens), puncts)


@pytest.mark.parametrize(
    "text",
    [
        "Oi, tudo bem? Vou sair.",
        "Vou sair. Depois eu ligo.",
        "Abri o DronsioFlow",
        "Versão 3.5 saiu.",
        "Você vem amanhã? Eu vou, sim.",
        "",
    ],
)
def test_extract_then_apply_returns_the_same_text(text):
    assert apply_tags(*extract_tags(text)) == text
