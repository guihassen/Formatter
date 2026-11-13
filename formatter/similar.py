import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from formatter.augment import FILLERS, MARKERS
import json

PROCESSED_DIR = Path("data/processed")
VECTOR_SIZE = 100
MIN_COUNT = 20
TOP_N = 5
MIN_SIMILARITY = 0.5

def read_sentences(path) -> list[list[str]] :
    df = pd.read_json(path, lines=True)

    sentences = []

    for sentence in df["text"] :
        words = []

        for word in sentence.split() :

            word = word.lower().strip(",.?")
        
            words.append(word)

        sentences.append(words)

    return sentences


def train(sentences: list[list[str]]) -> Word2Vec :
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=MIN_COUNT,
        workers=1,
        seed=42,
        epochs=10
    )

    return model


def blocked_words(language: str) -> set[str] :
    
    blocked = set()

    for filler in FILLERS[language] :
        blocked.add(filler)

    for marker in MARKERS[language] :
        for word in marker.split() :
            blocked.add(word)

    return blocked


def build_similar(model: Word2Vec, language: str) -> dict[str, list[str]] :
    similar = {}
    blocked = blocked_words(language)

    for word in model.wv.index_to_key :
        if len(word) < 3 or word in blocked :
            continue

        options = []
        
        for other, score in model.wv.most_similar(word, topn=10) :
            if score < MIN_SIMILARITY or other in blocked or other[:4] == word[:4] :
                continue

            
            options.append(other)

        options = options[:TOP_N]

        if options :
            similar[word] = options

    return similar

def save_similar(similar: dict[str, list[str]], path: Path) -> None :
    with open(path, "w", encoding="utf-8") as f :
        json.dump(similar, f, ensure_ascii=False, indent=1)


def load_similar(language: str) -> dict[str, list[str]] :
    path =  PROCESSED_DIR / f"similar_{language}.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    examples = {
        "pt": ["quarta", "comprar", "manhã", "ontem", "janeiro"],
        "en": ["wednesday", "buy", "morning", "yesterday", "january"],
    }

    for language, words in examples.items():
        sentences = read_sentences(PROCESSED_DIR / f"train_{language}.jsonl")
        model = train(sentences)
        similar = build_similar(model, language)
        save_similar(similar, PROCESSED_DIR / f"similar_{language}.json")

        print(f"{language}: {len(similar)} palavras com parecidas")
        for word in words:
            print(f"  {word}: {similar.get(word)}")