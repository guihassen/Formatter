"""Usa o modelo treinado para formatar um ditado de verdade (inferência).

Caminho: normaliza a pontuação do Whisper -> separa as palavras (extract_tags) -> o modelo dá um
rótulo por palavra (em janelas, para textos longos) -> rótulos viram etiquetas (to_tags) -> texto
final (apply_tags). Rode com: python -m formatter.predict, e digite ditados no terminal.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from formatter.dataset import ID2LABEL, MAX_LENGTH
from formatter.tags import apply_tags, extract_tags

MODEL_DIR = "models/formatter/best"
WINDOW_PIECES = MAX_LENGTH - 2   # espaço reservado para <s> e </s>
MARGIN_WORDS = 15                # palavras de contexto em cada borda da janela, cujos rótulos não são usados

SENTENCE_END = ".!?…"
PAUSE = ",;:"
QUOTES = "\"'“”‘’()[]{}«»"

def load_model(model_dir: str = MODEL_DIR) :
    """Carrega o tokenizador e o modelo salvos em `model_dir` e coloca o modelo em modo de avaliação."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    return tokenizer, model

def normalize_whisper(text: str) -> str:
    """Reescreve a pontuação do Whisper só com os sinais que extract_tags conhece.

    O treino trocou "!" e "..." por "." (veja corpus.clean_line); ";" e ":" viram ",". Aspas e
    parênteses em volta das palavras saem, e palavras que são só pontuação são descartadas.
    """
    words = []
    for word in text.split():
        core = word.strip(QUOTES)
        stripped = core.rstrip(SENTENCE_END + PAUSE)
        trailing = core[len(stripped):]
        stripped = stripped.strip(QUOTES)
        if not stripped.strip(SENTENCE_END + PAUSE + "-–—"):
            continue
        if "?" in trailing:
            stripped += "?"
        elif any(mark in trailing for mark in SENTENCE_END):
            stripped += "."
        elif trailing:
            stripped += ","
        words.append(stripped)
    return " ".join(words)


def windows(pieces_per_word: list[int]) -> list[tuple[int, int, int, int]]:
    """Divide as palavras em janelas sobrepostas de no máximo WINDOW_PIECES peças.

    Devolve (start, end, keep_from, keep_to): a janela cobre words[start:end], mas seus rótulos só são
    usados para words[keep_from:keep_to]. Assim cada palavra é rotulada tendo contexto dos dois lados.
    """
    total = len(pieces_per_word)
    result = []
    start, kept = 0, 0
    while kept < total:
        end, used = start, 0
        while end < total and used + pieces_per_word[end] <= WINDOW_PIECES:
            used += pieces_per_word[end]
            end += 1
        end = max(end, kept + 1)   # uma palavra maior que a janela não trava o laço
        keep_to = end if end == total else max(end - MARGIN_WORDS, kept + 1)
        result.append((start, end, kept, keep_to))
        kept = keep_to
        start = max(kept - MARGIN_WORDS, start + 1) if kept < total else start
    return result


def predict_labels(tokens: list[str], tokenizer, model) -> list[str] :
    """Um rótulo por palavra; textos longos são rotulados janela por janela."""
    return predict_with_confidence(tokens, tokenizer, model)[0]


def predict_with_confidence(tokens: list[str], tokenizer, model) -> tuple[list[str], list[float]]:
    """Rótulos e, para cada palavra, a distância entre a nota do rótulo vencedor e a do segundo.

    Distância pequena = modelo inseguro. É usada pelo pick_corrections para escolher trechos a corrigir.
    """
    pieces_per_word = [max(len(tokenizer.tokenize(token)), 1) for token in tokens]
    labels, margins = [], []
    for start, end, keep_from, keep_to in windows(pieces_per_word):
        window_labels, window_margins = predict_window(tokens[start:end], tokenizer, model)
        labels.extend(window_labels[keep_from - start:keep_to - start])
        margins.extend(window_margins[keep_from - start:keep_to - start])
    return labels, margins


def predict_window(tokens: list[str], tokenizer, model) -> list[str] :
    """Roda o modelo em um trecho que cabe em MAX_LENGTH peças e lê o rótulo da primeira peça de cada palavra."""
    encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        piece = torch.argmax(logits, dim=2)
        example = piece.tolist()[0]
        top_two = torch.topk(logits[0], 2, dim=1).values
        gaps = (top_two[:, 0] - top_two[:, 1]).tolist()
        word_ids = encoding.word_ids()

        labels = []
        margins = []
        previous = None

        for pos, word_id in enumerate(word_ids) :
            if word_id is None :
                continue 

            if word_id == previous :
                continue 

            labels.append(ID2LABEL[example[pos]])
            margins.append(gaps[pos])
            previous = word_id

        while len(labels) < len(tokens) :
            labels.append("KEEP_NONE")
            margins.append(0.0)

    return labels, margins


def to_tags(labels: list[str]) -> tuple[list[str], list[str]] :
    """Faz o caminho inverso de dataset.to_labels: rótulo -> (ação, pontuação)."""
    actions, puncts = [], []

    for label in labels :
        if "DELETE" in label :
            actions.append("DELETE")
            puncts.append("NONE")

        elif "KEEP_" in label :
            actions.append("KEEP")
            punct = label.removeprefix("KEEP_")
            puncts.append(punct)

    return actions, puncts

def format_text(text: str, tokenizer, model) -> str :
    """Formata um ditado cru do Whisper e devolve o texto pronto (ou "" para texto vazio)."""
    tokens, _, _ = extract_tags(normalize_whisper(text))

    if not tokens :
         return ""

    labels = predict_labels(tokens, tokenizer, model) 
    actions, puncts = to_tags(labels)
    return apply_tags(tokens, actions, puncts )



if __name__ == "__main__":
    tokenizer, model = load_model()
    print("Digite um ditado (linha vazia para sair).")

    while True:
        line = input("> ")
        if not line.strip():
            break
        print(format_text(line, tokenizer, model))
