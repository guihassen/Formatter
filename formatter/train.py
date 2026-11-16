import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from formatter.dataset import ID2LABEL, IGNORE, LABEL2ID, PROCESSED_DIR, tokenize_and_align


MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER_NAME = "xlm-roberta-base"
OUTPUT_DIR = Path("models/formatter")
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
PATIENCE = 2
VAL_LIMIT = 2000      # val examples used during training (keeps each evaluation quick)
EVAL_STEPS = 500      # evaluate and save every N training steps


def load_tagged(split: str, limit: int | None) :
    data = load_dataset("json", data_files=str(PROCESSED_DIR / f"tagged_{split}.jsonl"))["train"]

    if limit is not None:
        data = data.select(range(min(limit, len(data))))

    return data

def compute_metrics(eval_pred) -> dict :
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=2)

    mask = labels != IGNORE 

    true = labels[mask]

    pred = predictions[mask]

    delete = LABEL2ID["DELETE"]

    f1_macro = f1_score(true, pred, average="macro", labels=list(ID2LABEL), zero_division=0)

    is_real = true != delete                
    
    real_deleted = np.mean(pred[is_real] == delete)  

    is_delete = true == delete

    delete_recall = np.mean(pred[is_delete] == delete)

    return {
        "f1_macro": float(f1_macro),
        "real_deleted": float(real_deleted),
        "delete_recall": float(delete_recall),
    }

def main(limit: int | None, epochs: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train = load_tagged("train", limit)
    val = load_tagged("val", VAL_LIMIT)

    train = train.map(tokenize_and_align, batched=True, fn_kwargs={"tokenizer": tokenizer},
                      remove_columns=train.column_names)
    val = val.map(tokenize_and_align, batched=True, fn_kwargs={"tokenizer": tokenizer},
                  remove_columns=val.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)

    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    best = OUTPUT_DIR / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training examples")
    parser.add_argument("--epochs", type=int, default=10)
    options = parser.parse_args()

    main(options.limit, options.epochs)


