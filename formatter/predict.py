import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from formatter.dataset import ID2LABEL, MAX_LENGTH
from formatter.tags import apply_tags, extract_tags

MODEL_DIR = "models/formatter/best"

def load_model(model_dir: str = MODEL_DIR) :
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    return tokenizer, model

def predict_labels(tokens: list[str], tokenizer, model) -> list[str] :
    encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        piece = torch.argmax(logits, dim=2)
        example = piece.tolist()[0]
        word_ids = encoding.word_ids()

        labels = []
        previous = None

        for pos, word_id in enumerate(word_ids) :
            if word_id is None :
                continue 

            if word_id == previous :
                continue 

            labels.append(ID2LABEL[example[pos]])
            previous = word_id

        while len(labels) < len(tokens) :
            labels.append("KEEP_NONE")

    return labels


def to_tags(labels: list[str]) -> tuple[list[str], list[str]] :
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
    tokens, _, _ = extract_tags(text)

    if not tokens :
         return ""

    labels = predict_labels(tokens, tokenizer, model) 
    actions, puncts = to_tags(labels)
    return apply_tags(tokens, actions, puncts )



if __name__ == "__main__":
    tokenizer, model = load_model()
    print("Type a dictation (empty line to quit).")

    while True:
        line = input("> ")
        if not line.strip():
            break
        print(format_text(line, tokenizer, model))
