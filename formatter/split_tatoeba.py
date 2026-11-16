"""Divide o arquivo de pares inglês-português do Tatoeba em um arquivo de frases por idioma.

O Tatoeba entrega os pares em um arquivo só (id_en, frase_en, id_pt, frase_pt). Uma frase pode
aparecer em vários pares, então cada idioma fica só com a primeira ocorrência de cada id. A saída tem o
formato que corpus.read_tatoeba lê: id, idioma e texto separados por tab.
Rode com: python -m formatter.split_tatoeba "data/raw/<arquivo de pares>.tsv"
"""

import argparse
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUTS = {"eng": RAW_DIR / "eng_sentences.tsv", "por": RAW_DIR / "por_sentences.tsv"}


def read_pairs(path: Path) -> list[tuple[str, str, str, str]]:
    """Lê os pares (id_en, frase_en, id_pt, frase_pt), ignorando o BOM e as quebras de linha do Windows."""
    pairs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) == 4:
                pairs.append(tuple(fields))
    return pairs


def split_pairs(pairs: list[tuple[str, str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Separa as frases por idioma, sem repetir ids e mantendo a ordem em que aparecem."""
    sentences = {"eng": {}, "por": {}}
    for en_id, en_text, pt_id, pt_text in pairs:
        sentences["eng"].setdefault(en_id, en_text)
        sentences["por"].setdefault(pt_id, pt_text)
    return {language: list(by_id.items()) for language, by_id in sentences.items()}


def save_sentences(sentences: list[tuple[str, str]], language: str, path: Path) -> None:
    """Escreve uma frase por linha no formato id<tab>idioma<tab>texto."""
    with open(path, "w", encoding="utf-8") as f:
        for sentence_id, text in sentences:
            f.write(f"{sentence_id}\t{language}\t{text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide os pares do Tatoeba em um arquivo por idioma.")
    parser.add_argument("pairs", type=Path, help="arquivo de pares baixado do Tatoeba")
    options = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for language, sentences in split_pairs(read_pairs(options.pairs)).items():
        save_sentences(sentences, language, OUTPUTS[language])
        print(f"{language}: {len(sentences)} frases em {OUTPUTS[language]}")
