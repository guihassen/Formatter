"""Escolhe trechos curtos de ditados reais em que o modelo está menos seguro, para correção à mão,
e depois mede o modelo contra essas correções.

Cada trecho é uma sequência de frases do próprio modelo com MIN_WORDS a MAX_WORDS palavras, e sua nota
é a da palavra mais insegura. Gera data/eval/corrections.txt (edite a linha "correto:") e
data/eval/corrections_source.jsonl (palavras e rótulos para ler as correções de volta).
Rode com: python -m formatter.pick_corrections (gerar) ou --score (medir).
"""

import json
from pathlib import Path

from formatter.ab_eval import HISTORY_PATH, RESULTS_PATH, load_dictations, load_results
from formatter.predict import load_model, normalize_whisper, predict_with_confidence, to_tags
from formatter.tags import apply_tags, extract_tags

EVAL_DIR = Path("data/eval")
TEXT_PATH = EVAL_DIR / "corrections.txt"
SOURCE_PATH = EVAL_DIR / "corrections_source.jsonl"
HOW_MANY = 20
PER_DICTATION = 2
MIN_WORDS = 8
MAX_WORDS = 20

HEADER = """# Corrija a linha "correto:" de cada trecho e salve o arquivo.
# Pode: mudar a pontuação (, . ? ! : ; ...), maiúsculas e minúsculas, e apagar palavras
#       (hesitações, repetições, correções do tipo "terça não quarta").
# Não pode: trocar, juntar ou acrescentar palavras. Se o Whisper errou uma palavra, deixe como está.
# Se o trecho já estiver certo, não mexa.

"""


def segments(labels: list[str], margins: list[float]) -> list[tuple[int, int, float]]:
    """(start, end, nota) de sequências de frases previstas que cabem entre MIN_WORDS e MAX_WORDS palavras."""
    ends = [i + 1 for i, label in enumerate(labels) if label in ("KEEP_PERIOD", "KEEP_QUESTION")]
    if not ends or ends[-1] != len(labels):
        ends.append(len(labels))
    starts = [0] + ends[:-1]

    result = []
    for first, start in enumerate(starts):
        for end in ends[first:]:
            size = end - start
            if size > MAX_WORDS:
                break
            if size >= MIN_WORDS:
                result.append((start, end, min(margins[start:end])))
                break
    return result


def pick(candidates: list[dict]) -> list[dict]:
    """Os trechos mais inseguros, no máximo PER_DICTATION por ditado e sem sobreposição."""
    chosen, per_dictation = [], {}
    for candidate in sorted(candidates, key=lambda c: c["score"]):
        if per_dictation.get(candidate["at"], 0) >= PER_DICTATION:
            continue
        if any(c["at"] == candidate["at"] and c["start"] < candidate["end"] and candidate["start"] < c["end"]
               for c in chosen):
            continue
        chosen.append(candidate)
        per_dictation[candidate["at"]] = per_dictation.get(candidate["at"], 0) + 1
        if len(chosen) == HOW_MANY:
            break
    return chosen


def write(chosen: list[dict]) -> None:
    """Escreve o arquivo para correção e o arquivo com as palavras e rótulos de cada trecho."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for number, piece in enumerate(chosen, start=1):
        lines.append(f"## {number}\nmodelo:  {piece['model']}\ncorreto: {piece['model']}\n\n")
    TEXT_PATH.write_text("".join(lines), encoding="utf-8")
    with open(SOURCE_PATH, "w", encoding="utf-8") as f:
        for number, piece in enumerate(chosen, start=1):
            f.write(json.dumps({"number": number, **piece}, ensure_ascii=False) + "\n")


def main() -> None:
    """Roda o modelo nos ditados ainda não julgados no A/B e escreve os trechos escolhidos."""
    judged = {r["at"] for r in load_results(RESULTS_PATH)}
    dictations = [d for d in load_dictations(HISTORY_PATH) if d["at"] not in judged]
    tokenizer, model = load_model()

    candidates = []
    for dictation in dictations:
        tokens, _, _ = extract_tags(normalize_whisper(dictation["raw"]))
        if not tokens:
            continue
        labels, margins = predict_with_confidence(tokens, tokenizer, model)
        for start, end, score in segments(labels, margins):
            actions, puncts = to_tags(labels[start:end])
            candidates.append({
                "at": dictation["at"],
                "start": start,
                "end": end,
                "tokens": tokens[start:end],
                "labels": labels[start:end],
                "model": apply_tags(tokens[start:end], actions, puncts),
                "score": score,
            })

    chosen = pick(candidates)
    write(chosen)
    print(f"{len(chosen)} trechos de {len(dictations)} ditados em {TEXT_PATH}")




MARKS = {",": "COMMA", ".": "PERIOD", "?": "QUESTION", "!": "EXCLAMATION", ":": "COLON", ";": "SEMICOLON",
         "...": "ELLIPSIS", "…": "ELLIPSIS"}


def word_and_mark(word: str) -> tuple[str, str]:
    """Separa a palavra do sinal final: "feature." -> ("feature", "PERIOD"). Aceita também ! : ; ..."""
    for mark in sorted(MARKS, key=len, reverse=True):
        if word.endswith(mark) and len(word) > len(mark):
            return word[:-len(mark)], MARKS[mark]
    return word, "NONE"


def correction_labels(tokens: list[str], corrected: str) -> list[str]:
    """Um rótulo por palavra original a partir da correção; palavras que sumiram viram DELETE.

    Levanta ValueError se a correção tiver uma palavra que não está no original, na mesma ordem.
    """
    fixed = [word_and_mark(word) for word in corrected.split()]
    labels, position = [], 0
    for token in tokens:
        if position < len(fixed) and fixed[position][0].lower() == token.lower():
            labels.append("KEEP_" + fixed[position][1])
            position += 1
        else:
            labels.append("DELETE")
    if position != len(fixed):
        raise ValueError(f"word not found in the original: {fixed[position][0]!r}")
    return labels


def read_corrections() -> dict[int, str]:
    """Lê o arquivo corrigido e devolve {número do trecho: texto da linha "correto:"}."""
    corrected, number = {}, None
    for line in TEXT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            number = int(line[3:])
        elif line.startswith("correto:") and number is not None:
            corrected[number] = line.removeprefix("correto:").strip()
    return corrected


def score() -> None:
    """Compara os rótulos do modelo com os das correções e mostra o acerto e os erros mais comuns."""
    from collections import Counter

    sources = [json.loads(line) for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines()]
    corrected = read_corrections()
    pairs, untouched = Counter(), 0
    for source in sources:
        gold = correction_labels(source["tokens"], corrected[source["number"]])
        untouched += gold == source["labels"]
        for predicted, true in zip(source["labels"], gold):
            pairs[(true, predicted)] += 1

    words = sum(pairs.values())
    right = sum(n for (true, predicted), n in pairs.items() if true == predicted)
    print(f"Trechos sem nenhuma correção: {untouched}/{len(sources)}")
    print(f"Palavras com o rótulo certo: {right}/{words} ({right / words:.1%})")
    print("Erros (certo <- modelo):")
    for (true, predicted), n in sorted(pairs.items(), key=lambda item: -item[1]):
        if true != predicted:
            print(f"  {n:3d}  {true:<16} <- {predicted}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--score"]:
        score()
    else:
        main()
