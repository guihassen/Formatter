"""Monta o corpus de textos limpos a partir das frases do Tatoeba.

Lê os arquivos de frases em português e inglês, descarta as que o formatador não consegue
representar, separa treino e validação e junta as frases em textos de 1 a MAX_SENTENCES frases,
parecidos com um ditado. Rode com: python -m formatter.corpus
"""

import csv
import json
import random
import re
from pathlib import Path
from datasets import Dataset

import pandas as pd

from formatter.tags import apply_tags, extract_tags

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VAL_FRACTION = 0.05
MIN_WORDS = 3
MAX_WORDS = 40
MAX_SENTENCES = 6


def read_tatoeba(path: Path) -> list[str] :
    """Lê um arquivo do Tatoeba (id, idioma, texto separados por tab) e devolve só os textos."""
    df = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    text = df[2].to_list()

    return text

def clean_line(line: str) -> str | None :
    """Normaliza uma frase e devolve None se ela não servir para o treino.

    Troca "!" e "..." por ".", descarta frases com símbolos que o modelo não tem (";", ":", aspas,
    parênteses, travessões), com palavras só de pontuação, curtas ou longas demais, ou que não sobrevivem
    à ida e volta extract_tags/apply_tags.
    """

    c_line = " ".join(line.split())

    c_line = c_line.replace("?!", "?").replace("!?", "?").replace("...", ".").replace("!", ".")
    
    SYMBOLS = [";", ":",'"',"(",")","—", "–",'“','”']

    PUNCTUATION = ",.?"

    words = c_line.split()

    for word in words :
        if word.strip(PUNCTUATION) == "" :
            return None
        if len(word) >= 2 and word[-1] in PUNCTUATION and word[-2] in PUNCTUATION :
            return None

    for symbol in SYMBOLS :
        if symbol in c_line :
            return None 

    if len(words) < MIN_WORDS or len(words) > MAX_WORDS :
        return None

    if apply_tags(*extract_tags(c_line)) != c_line :
        return None


    return c_line
    

def make_examples(sentences: list[str], rng: random.Random) -> list[str] :
    """Junta frases em sequência em grupos de 1 a MAX_SENTENCES, mantendo a ordem."""

    examples = []
    start = 0
    
    while start < len(sentences) :
        size = rng.randint(1,MAX_SENTENCES)

        group = sentences[start:start + size]

        output = " ".join(group)

        examples.append(output)

        start += size

    return examples




def build(raw_file: Path, language: str, rng: random.Random) -> None :
    """Gera os arquivos {train|val}_{idioma}.jsonl de um idioma.

    Limpa as frases, separa VAL_FRACTION para validação ANTES de agrupar (assim nenhuma frase aparece
    nos dois lados) e salva os textos agrupados.
    """

    clean = []

    raw = read_tatoeba(raw_file)

    for sentence in raw :
        clean_sentence = clean_line(sentence)

        if clean_sentence != None :
            clean.append(clean_sentence)

    df = Dataset.from_dict({"text": clean})

    splited_df = df.train_test_split(test_size=VAL_FRACTION, seed=42)

    for name, part in splited_df.items() :

        file_name = "val" if name == "test" else name

        examples = make_examples(part["text"], rng)

        corpus = Dataset.from_dict({"text": examples,
                                    "language": [language] * len(examples) })

        # Um texto por linha (JSON Lines), com acentos escritos como estão.
        corpus.to_json(
            PROCESSED_DIR / f"{file_name}_{language}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False
                )

        print(corpus)

    print(f"lidas:{len(raw)}")
    print(f"fora:{len(raw) - len(clean)}")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    build(RAW_DIR / "por_sentences.tsv", "pt", rng)
    build(RAW_DIR / "eng_sentences.tsv", "en", rng)    