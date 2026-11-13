import json
import random
from pathlib import Path
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer

from formatter.augment import corrupt
from formatter.similar import load_similar

PROCESSED_DIR = Path("data/processed")
LANGUAGES = ["pt", "en"]
LABELS = ["KEEP_NONE", "KEEP_COMMA", "KEEP_PERIOD", "KEEP_QUESTION", "DELETE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
MAX_LENGTH = 256   
IGNORE = -100      


def to_labels(actions: list[str], puncts: list[str]) -> list[str] :

    labels = []
    
    for action, punct in zip(actions, puncts):
        if action == "DELETE" :
            labels.append("DELETE")

        else :
                labels.append("KEEP_" + punct)
    
    return labels

def make_example(text: str, language: str, rng: random.Random, similar: dict[str, list[str]]) -> dict :
        
        token, actions, puncts = corrupt(text, rng, language, similar)

        labels = to_labels(actions, puncts)

        num_labels = [LABEL2ID[label] for label in labels]

        return {"tokens": token, "labels": num_labels, "language": language,}


def build_split(split: str, seed: int) -> list[dict] :
    rng = random.Random(seed)

    examples = []

    for language in LANGUAGES :
        similar = load_similar(language)

        df = pd.read_json(PROCESSED_DIR / f"{split}_{language}.jsonl", lines=True)
        
        for text in df["text"] :    
            examples.append(make_example(text,language, rng, similar))

    rng.shuffle(examples)

    return examples


def save_examples(examples: list[dict], path: Path) -> None :
     with open(path, "w", encoding="utf-8") as f:
          for example in examples :
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
def tokenize_and_align(batch: dict, tokenizer) -> dict :
          
    encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)

    all_labels = []

    for i, labels in enumerate(batch["labels"]) :
        aligned = []
        previous = None

        for word_id in encoded.word_ids(batch_index=i) :
            if word_id == None :
                
                aligned.append(IGNORE)

            elif word_id == previous :
                aligned.append(IGNORE)

            else :
            
                new_word = labels[word_id]

                aligned.append(new_word)

                previous = word_id 
            
        all_labels.append(aligned)

    encoded["labels"] = all_labels

    return encoded


if __name__ == "__main__":
    for split, seed in [("train", 1), ("val", 2)]:
        examples = build_split(split, seed)
        save_examples(examples, PROCESSED_DIR / f"tagged_{split}.jsonl")

        counts = Counter(label for example in examples for label in example["labels"])
        total = sum(counts.values())

        print(f"{split}: {len(examples)} exemplos, {total} palavras")
        for label_id, label in ID2LABEL.items():
            print(f"  {label:<14} {counts[label_id]:>8} ({counts[label_id] / total:.1%})")

        for example in examples[:3]:
            print("  " + " ".join(f"{token}/{ID2LABEL[label_id]}"
                                  for token, label_id in zip(example["tokens"], example["labels"])))