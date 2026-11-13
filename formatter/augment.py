import random
from formatter.tags import extract_tags

CLEAN_PROB = 0.25
RESTART_PROB = 0.02
CORRECTION_PROB = 0.1
MAX_FILLER_PROB = 0.1

FILLERS = {
    "pt": ["ahn", "hum", "hmm", "eh", "né", "tipo"],
    "en": ["uh", "um", "uhm", "hmm", "er", "like"],
}
MARKERS = {
    "pt": ["não", "quer dizer", "digo", "ou melhor", "na verdade"],
    "en": ["no", "i mean", "sorry", "actually"],
}


def add_fillers(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float, language: str) -> tuple[list[str], list[str], list[str]]:
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


def add_restart(tokens: list[str], actions: list[str], puncts: list[str], rng: random.Random, prob: float) -> list[str] :
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


def add_corrections(tokens: list[str], actions: list[str], puncts: list[str], rng:random.Random, prob:float, similar: dict[str, list[str]], language: str) -> tuple[list[str], list[str], list[str]]:
    new_tokens = []
    new_actions = []
    new_puncts = []

    for token, action, punct in zip(tokens, actions, puncts):

        if token.lower() in similar and rng.random() < prob :
            wrong_word = rng.choice(similar[token.lower()])

            new_tokens.append(wrong_word)
            new_actions.append("DELETE")
            new_puncts.append("NONE")

                    
            selected = rng.choice(MARKERS[language])

            for word in selected.split() :
                new_tokens.append(word)
                new_actions.append("DELETE")
                new_puncts.append("NONE")

        new_tokens.append(token)
        new_actions.append(action)
        new_puncts.append(punct)


    return new_tokens, new_actions, new_puncts


def corrupt(text: str, rng: random.Random, language: str, similar: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]: 

    tokens, actions, puncts = extract_tags(text)

    if rng.random() < CLEAN_PROB :
       return tokens, actions, puncts

    hesitation_prob = rng.uniform(0, MAX_FILLER_PROB)

    tokens, actions, puncts = add_restart(tokens, actions, puncts, rng, RESTART_PROB)
    tokens, actions,puncts = add_corrections(tokens, actions, puncts, rng, CORRECTION_PROB, similar, language)
    tokens, actions, puncts = add_fillers(tokens, actions, puncts, rng, hesitation_prob, language)


    return tokens, actions, puncts 




                 

       



        


