"""Aumento de dados: transforma um texto limpo em um "ditado" com os erros típicos da fala.

Há dois tipos de mudança:
- `enrich` muda o próprio texto certo, acrescentando coisas que DEVEM ficar no resultado
  (enumerações com estrutura repetida e "na verdade," no começo da frase). Serve para o modelo
  aprender que nem toda repetição ou todo gatilho é um erro.
- `corrupt` insere palavras que DEVEM ser apagadas (hesitações, recomeços, correções). Toda palavra
  inserida por ele recebe a ação DELETE, então apagar essas palavras devolve exatamente o texto
  de entrada.

Todas as funções `add_*` recebem e devolvem as três listas paralelas de `tags.extract_tags`
(tokens, actions, puncts), não alteram as listas de entrada e usam só o `rng` recebido, para que
a mesma semente gere sempre o mesmo resultado.
"""

import random

from formatter.tags import apply_tags, extract_tags

CLEAN_PROB = 0.25               # chance de um exemplo ficar sem nenhum erro inserido
RESTART_PROB = 0.02             # por palavra: repetir as próximas 1 a 3 palavras ("eu vou eu vou")
CORRECTION_PROB = 0.06          # por palavra com parecidas: "terça não quarta"
MAX_FILLER_PROB = 0.1           # teto da chance de hesitação por palavra (sorteada por exemplo)
CLAUSE_CORRECTION_PROB = 0.04   # por oração: uma ideia errada, um gatilho e a ideia certa
FALSE_START_PROB = 0.05         # por frase: um começo que desvia antes da frase verdadeira
PARALLEL_PROB = 0.05            # por frase: enumeração com estrutura repetida, que fica no texto
DISCOURSE_PROB = 0.04           # por frase: "na verdade," no começo, que fica no texto

GROUP_SIZES = [0, 0, 1, 2]      # quantas palavras antes da errada entram na correção ("na terça")
CLAUSE_SIZE = (3, 15)           # tamanho, em palavras, da ideia errada numa correção de oração
FALSE_START_PREFIX = (0, 2)     # palavras iguais à frase verdadeira no começo do falso começo
FALSE_START_DIVERGE = (1, 3)    # palavras diferentes que vêm depois

FILLERS = {
    "pt": ["ahn", "hum", "hmm", "eh", "né", "tipo"],
    "en": ["uh", "um", "uhm", "hmm", "er", "like"],
}
MARKERS = {
    "pt": ["não", "quer dizer", "digo", "ou melhor", "na verdade"],
    "en": ["no", "i mean", "sorry", "actually"],
}
CLAUSE_MARKERS = {
    "pt": ["na verdade não", "não espera", "quer dizer", "ou melhor", "na verdade", "não não", "não calma"],
    "en": ["actually no", "no wait", "i mean", "sorry", "actually", "no no"],
}
DISCOURSE = {
    "pt": ["na verdade"],
    "en": ["actually"],
}

SENTENCE_END = ("PERIOD", "QUESTION")


def sentence_starts(puncts: list[str]) -> list[int]:
    """Posições em que começa uma frase: a primeira palavra e as que vêm depois de "." ou "?"."""
    return [i for i in range(len(puncts)) if i == 0 or puncts[i - 1] in SENTENCE_END]


def clause_starts(puncts: list[str]) -> list[int]:
    """Posições em que começa uma oração: o começo de cada frase e as palavras depois de uma vírgula."""
    return [i for i in range(len(puncts)) if i == 0 or puncts[i - 1] != "NONE"]


def clause_end(puncts: list[str], start: int) -> int:
    """Posição logo depois da última palavra da oração que começa em `start`."""
    end = start
    while end < len(puncts) and puncts[end] == "NONE":
        end += 1
    return min(end + 1, len(puncts))


def add_fillers(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float,
                language: str) -> tuple[list[str], list[str], list[str]]:
    """Coloca hesitações ("ahn", "uh"...) antes de cada palavra, com chance `prob`. Elas são DELETE."""
    new_tokens = []
    new_actions = []
    new_puncts = []

    for token, action, punct in zip(tokens, actions, puncts):
        if rng.random() < prob:
            new_tokens.append(rng.choice(FILLERS[language]))
            new_actions.append("DELETE")
            new_puncts.append("NONE")

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)

    return new_tokens, new_actions, new_puncts


def add_restart(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random,
                prob: float) -> tuple[list[str], list[str], list[str]]:
    """Repete as próximas 1 a 3 palavras antes de cada palavra, com chance `prob` ("eu vou eu vou").

    A cópia vem primeiro e é DELETE; as palavras originais continuam como estavam.
    """
    new_tokens = []
    new_actions = []
    new_puncts = []

    for index, (token, action, punct) in enumerate(zip(tokens, actions, puncts)):

        if rng.random() < prob :

            size = rng.randint(1, 3)

            selected_tokens = tokens[index:index + size]

            new_tokens.extend(selected_tokens)

            for selected in selected_tokens :
                new_actions.append("DELETE")
                new_puncts.append("NONE")

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)

    return new_tokens, new_actions, new_puncts


def add_corrections(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float,
                    similar: dict[str, list[str]], language: str) -> tuple[list[str], list[str], list[str]]:
    """Antes de uma palavra que tem parecidas, fala primeiro a palavra errada e um gatilho.

    "quarta" vira "terça não quarta". Às vezes a correção leva junto 1 ou 2 palavras anteriores da
    mesma oração: "na quarta" vira "na terça não na quarta". Tudo o que foi inserido é DELETE.
    """
    new_tokens = []
    new_actions = []
    new_puncts = []

    for index, (token, action, punct) in enumerate(zip(tokens, actions, puncts)):

        if token.lower() in similar and rng.random() < prob :
            wrong_word = rng.choice(similar[token.lower()])
            marker = rng.choice(MARKERS[language])

            # As palavras logo antes só entram na correção se ainda forem originais intactas
            # da mesma oração (sem pontuação, não apagadas e não movidas antes).
            lead = rng.choice(GROUP_SIZES)
            if not (lead and index >= lead and new_tokens[-lead:] == tokens[index - lead:index]
                    and all(p == "NONE" for p in new_puncts[-lead:])
                    and all(a == "KEEP" for a in new_actions[-lead:])):
                lead = 0

            kept = list(zip(new_tokens[-lead:], new_actions[-lead:], new_puncts[-lead:])) if lead else []
            if lead:
                del new_tokens[-lead:], new_actions[-lead:], new_puncts[-lead:]

            for word in [t for t, _, _ in kept] + [wrong_word] + marker.split():
                new_tokens.append(word)
                new_actions.append("DELETE")
                new_puncts.append("NONE")

            for kept_token, kept_action, kept_punct in kept:
                new_tokens.append(kept_token)
                new_actions.append(kept_action)
                new_puncts.append(kept_punct)

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)


    return new_tokens, new_actions, new_puncts


def wrong_idea(tokens: list[str], puncts: list[str], start: int, rng: random.Random,
               similar: dict[str, list[str]]) -> list[str]:
    """Uma "ideia errada" para a oração que começa em `start`, ou [] se não der para criar.

    É o começo da oração (3 a 15 palavras) com 1 a 3 palavras trocadas por parecidas, como quem
    começa a dizer uma coisa e muda de ideia. Se nenhuma palavra tiver parecidas, usa o começo de
    outra oração do mesmo texto.
    """
    end = clause_end(puncts, start)
    if end - start < CLAUSE_SIZE[0]:
        return []
    size = rng.randint(CLAUSE_SIZE[0], min(CLAUSE_SIZE[1], end - start))
    piece = [token.lower() for token in tokens[start:start + size]]

    swappable = [i for i, word in enumerate(piece) if word in similar]
    if swappable:
        for i in rng.sample(swappable, min(len(swappable), rng.randint(1, 3))):
            piece[i] = rng.choice(similar[piece[i]])
        return piece

    others = [s for s in clause_starts(puncts) if s != start and clause_end(puncts, s) - s >= CLAUSE_SIZE[0]]
    if not others:
        return []
    other = rng.choice(others)
    return [token.lower() for token in tokens[other:other + size]]


def add_clause_correction(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random,
                          prob: float, similar: dict[str, list[str]],
                          language: str) -> tuple[list[str], list[str], list[str]]:
    """Antes de uma oração, com chance `prob`, fala uma ideia errada e um gatilho de correção.

    "e espere o ônibus" vira "e pegue um táxi na verdade não e espere o ônibus".
    A ideia errada e o gatilho são DELETE; a oração certa continua como estava.
    """
    chosen = {start for start in clause_starts(puncts) if rng.random() < prob}
    new_tokens, new_actions, new_puncts = [], [], []

    for index, (token, action, punct) in enumerate(zip(tokens, actions, puncts)):
        if index in chosen:
            wrong = wrong_idea(tokens, puncts, index, rng, similar)
            if wrong:
                for word in wrong + rng.choice(CLAUSE_MARKERS[language]).split():
                    new_tokens.append(word)
                    new_actions.append("DELETE")
                    new_puncts.append("NONE")

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)

    return new_tokens, new_actions, new_puncts


def add_false_start(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float,
                    similar: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    """Antes de uma frase, com chance `prob`, coloca um começo que desvia e é abandonado.

    "we need to leave now" vira "the team was going we need to leave now":
    o falso começo tem 0 a 2 palavras iguais às da frase e depois 1 a 3 palavras diferentes
    (parecidas ou tiradas de outra parte do texto). Ele todo é DELETE.
    """
    vocabulary = [token.lower() for token in tokens]
    chosen = {start for start in sentence_starts(puncts) if rng.random() < prob}
    new_tokens, new_actions, new_puncts = [], [], []

    for index, (token, action, punct) in enumerate(zip(tokens, actions, puncts)):
        if index in chosen and clause_end(puncts, index) - index >= 4:
            prefix = [t.lower() for t in tokens[index:index + rng.randint(*FALSE_START_PREFIX)]]
            diverge = []
            for offset in range(rng.randint(*FALSE_START_DIVERGE)):
                original = tokens[index + len(prefix) + offset].lower() if index + len(prefix) + offset < len(tokens) else ""
                options = similar.get(original) or [w for w in vocabulary if w != original] or ["the"]
                diverge.append(rng.choice(options))
            for word in prefix + diverge:
                new_tokens.append(word)
                new_actions.append("DELETE")
                new_puncts.append("NONE")

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)

    return new_tokens, new_actions, new_puncts


def add_parallel(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float,
                 similar: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    """Em uma frase, com chance `prob`, repete um trecho trocando só a última palavra. Tudo fica (KEEP).

    "compre o arroz." vira "compre o arroz, compre o feijão."
    Ensina que estrutura repetida com conteúdo diferente é uma enumeração, não um recomeço.
    """
    starts = sentence_starts(puncts) + [len(tokens)]
    new_tokens, new_actions, new_puncts = list(tokens), list(actions), list(puncts)

    # Da última frase para a primeira, para as posições anteriores não mudarem.
    for start, end in reversed(list(zip(starts, starts[1:]))):
        if rng.random() >= prob:
            continue
        candidates = [i for i in range(start + 1, end)
                      if tokens[i].lower() in similar and puncts[i - 1] == "NONE"]
        if not candidates:
            continue
        last = rng.choice(candidates)
        first = last - 1
        while first > start and puncts[first - 1] == "NONE" and last - first < 3:
            first -= 1
        first = rng.randint(first, last - 1)

        copy = [t.lower() for t in tokens[first:last]] + [rng.choice(similar[tokens[last].lower()])]
        new_tokens[last + 1:last + 1] = copy
        new_actions[last + 1:last + 1] = ["KEEP"] * len(copy)
        new_puncts[last + 1:last + 1] = ["NONE"] * (len(copy) - 1) + [puncts[last]]
        new_puncts[last] = "COMMA"

    return new_tokens, new_actions, new_puncts


def add_discourse(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float,
                  language: str) -> tuple[list[str], list[str], list[str]]:
    """No começo de uma frase, com chance `prob`, coloca "na verdade," ou "actually,". Fica (KEEP).

    Ensina que o gatilho sozinho, sem uma ideia errada antes, não é uma correção.
    """
    chosen = {start for start in sentence_starts(puncts) if rng.random() < prob}
    new_tokens, new_actions, new_puncts = [], [], []

    for index, (token, action, punct) in enumerate(zip(tokens, actions, puncts)):
        if index in chosen:
            words = rng.choice(DISCOURSE[language]).split()
            new_tokens.extend(words)
            new_actions.extend(["KEEP"] * len(words))
            new_puncts.extend(["NONE"] * (len(words) - 1) + ["COMMA"])

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)

    return new_tokens, new_actions, new_puncts


def enrich(text: str, rng: random.Random, language: str, similar: dict[str, list[str]]) -> str:
    """Devolve o texto certo com enumerações e "na verdade," acrescentados, que devem ficar."""
    tags = extract_tags(text)
    tags = add_parallel(*tags, rng, PARALLEL_PROB, similar)
    tags = add_discourse(*tags, rng, DISCOURSE_PROB, language)
    return apply_tags(*tags)


def corrupt(text: str, rng: random.Random, language: str,
            similar: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    """Transforma o texto em um ditado com erros; apagar as palavras DELETE devolve o texto.

    Em CLEAN_PROB dos casos o texto volta sem erros, para o modelo também ver ditados limpos.
    """
    tokens, actions, puncts = extract_tags(text)

    if rng.random() < CLEAN_PROB :
       return tokens, actions, puncts

    hesitation_prob = rng.uniform(0, MAX_FILLER_PROB)

    tokens, actions, puncts = add_clause_correction(tokens, actions, puncts, rng, CLAUSE_CORRECTION_PROB, similar, language)
    tokens, actions, puncts = add_false_start(tokens, actions, puncts, rng, FALSE_START_PROB, similar)
    tokens, actions, puncts = add_restart(tokens, actions, puncts, rng, RESTART_PROB)
    tokens, actions, puncts = add_corrections(tokens, actions, puncts, rng, CORRECTION_PROB, similar, language)
    tokens, actions, puncts = add_fillers(tokens, actions, puncts, rng, hesitation_prob, language)

    return tokens, actions, puncts
