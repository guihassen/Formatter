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
MAX_SENTENCES = 4


def read_tatoeba(path: Path) -> list[str] :
    df = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    text = df[2].to_list()

    return text

def clean_line(line: str) -> str | None :

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

        # O comando exato que você deve usar:
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