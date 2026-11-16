"""Comparação cega (A/B) entre a saída atual do DronsioFlow e a do formatador.

Lê o texto cru do Whisper de cada ditado do histórico do DronsioFlow (só leitura), formata com o
modelo e pergunta qual versão é melhor, sem dizer qual é qual. Cada resposta é salva na hora, então
dá para parar e continuar depois. Os resultados ficam em data/eval/ (fora do git, porque são ditados
pessoais). Rode com: python -m formatter.ab_eval
"""

import argparse
import difflib
import json
import random
from collections import Counter
from pathlib import Path

HISTORY_PATH = Path.home() / ".config" / "DronsioFlow" / "history.json"
RESULTS_PATH = Path("data/eval/ab_results.jsonl")
SEED = 7

BOLD_YELLOW = "\033[1;33m"
DIM = "\033[2m"
RESET = "\033[0m"

ANSWERS = {"1": 0, "2": 1}


def load_dictations(path: Path) -> list[dict]:
    """Ditados do histórico que têm o texto cru do Whisper (comandos, agentes e polimentos ficam de fora)."""
    history = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in history if e.get("kind", "dictation") == "dictation" and e.get("raw", "").strip()]


def load_results(path: Path) -> list[dict]:
    """Carrega os julgamentos já feitos (lista vazia se ainda não há nenhum)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_result(result: dict, path: Path) -> None:
    """Acrescenta um julgamento ao arquivo de resultados, criando a pasta se precisar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def pending(dictations: list[dict], results: list[dict]) -> list[dict]:
    """Ditados que ainda não foram julgados."""
    done = {r["at"] for r in results}
    return [d for d in dictations if d["at"] not in done]


def highlight(text: str, other: str) -> str:
    """Devolve o texto com as palavras diferentes da outra versão destacadas em amarelo."""
    words, other_words = text.split(), other.split()
    out = []
    matcher = difflib.SequenceMatcher(a=words, b=other_words, autojunk=False)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        chunk = " ".join(words[i1:i2])
        if chunk:
            out.append(chunk if tag == "equal" else f"{BOLD_YELLOW}{chunk}{RESET}")
    return " ".join(out)


def summarize(results: list[dict]) -> Counter:
    """Conta quantas vezes cada lado venceu (model, current, tie, same)."""
    return Counter(r["winner"] for r in results)


def print_summary(results: list[dict]) -> None:
    """Mostra o placar no terminal."""
    counts = summarize(results)
    judged = counts["model"] + counts["current"] + counts["tie"]
    print(f"\nResultado ({len(results)} ditados):")
    print(f"  modelo melhor:     {counts['model']}")
    print(f"  DronsioFlow melhor: {counts['current']}")
    print(f"  empate:            {counts['tie']}")
    print(f"  textos idênticos:  {counts['same']} (não perguntados)")
    if judged:
        print(f"  modelo vence ou empata em {(counts['model'] + counts['tie']) / judged:.0%} dos julgados")


def main(history_path: Path, results_path: Path) -> None:
    """Laço principal: mostra cada ditado pendente em ordem sorteada e grava a resposta."""
    from formatter.predict import format_text, load_model

    results = load_results(results_path)
    todo = pending(load_dictations(history_path), results)
    if not todo:
        print_summary(results)
        return

    print(f"{len(todo)} ditados para julgar. Carregando o modelo...")
    tokenizer, model = load_model()
    rng = random.Random(SEED)

    print("Responda 1 ou 2 (a melhor versão), = (empate), p (pular) ou s (sair).")
    for number, dictation in enumerate(todo, start=1):
        candidates = {"model": format_text(dictation["raw"], tokenizer, model), "current": dictation["text"].strip()}
        result = {"at": dictation["at"], **candidates}

        if candidates["model"] == candidates["current"]:
            save_result({**result, "winner": "same"}, results_path)
            results.append({**result, "winner": "same"})
            continue

        order = ["model", "current"]
        rng.shuffle(order)
        first, second = (candidates[name] for name in order)

        print(f"\n{DIM}[{number}/{len(todo)}] Whisper: {dictation['raw'].strip()}{RESET}")
        print(f"1) {highlight(first, second)}")
        print(f"2) {highlight(second, first)}")

        answer = ""
        while answer not in ("1", "2", "=", "p", "s"):
            answer = input("> ").strip().lower()
        if answer == "s":
            break
        if answer == "p":
            continue

        winner = "tie" if answer == "=" else order[ANSWERS[answer]]
        save_result({**result, "winner": winner}, results_path)
        results.append({**result, "winner": winner})

    print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparação cega entre o DronsioFlow atual e o modelo.")
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    options = parser.parse_args()

    main(options.history, options.results)
