"""Treina (fine-tuning) o modelo de classificação de tokens com o Trainer do Hugging Face.

Parte do Multilingual-MiniLM (ou de um modelo já treinado, com --init), avalia a cada EVAL_STEPS
passos, para cedo se o F1 macro não melhorar por PATIENCE avaliações e salva o melhor modelo em
<saída>/best. Rode com: python -m formatter.train [--limit N] [--epochs N] [--init pasta]
[--output pasta] [--lr taxa] [--resume]

Com --resume, um treino interrompido continua do último checkpoint salvo na pasta de saída (perde no
máximo EVAL_STEPS passos). O comando precisa ser o mesmo da primeira vez, mais o --resume.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from formatter.dataset import ID2LABEL, IGNORE, LABEL2ID, PROCESSED_DIR, tokenize_and_align


MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER_NAME = "xlm-roberta-base"
OUTPUT_DIR = Path("models/formatter")
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
PATIENCE = 4
VAL_LIMIT = 3000      # exemplos de validação usados durante o treino (cada avaliação fica rápida)
EVAL_STEPS = 500      # avalia e salva a cada N passos de treino


def load_tagged(split: str, limit: int | None) :
    """Carrega tagged_{split}.jsonl como Dataset, opcionalmente só com os primeiros `limit` exemplos."""
    data = load_dataset("json", data_files=str(PROCESSED_DIR / f"tagged_{split}.jsonl"))["train"]

    if limit is not None:
        data = data.select(range(min(limit, len(data))))

    return data

def compute_metrics(eval_pred) -> dict :
    """Calcula as métricas de uma avaliação a partir dos logits e dos rótulos certos.

    - f1_macro: média do F1 dos 5 rótulos (cada rótulo pesa igual, mesmo os raros).
    - real_deleted: fração das palavras reais que o modelo apagou (o erro mais grave; quanto menor, melhor).
    - delete_recall: fração das palavras a apagar que o modelo apagou.
    Peças com rótulo IGNORE não entram na conta.
    """
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

def checkpoint_to_resume(output_dir: Path, resume: bool) -> str | None:
    """O último checkpoint da pasta de saída, se --resume foi pedido e existe algum; senão None."""
    if not resume:
        return None
    last = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if last is None:
        print(f"Nenhum checkpoint em {output_dir}; começando do início.")
    else:
        print(f"Continuando de {last}.")
    return last


def main(limit: int | None, epochs: int, init: str = MODEL_NAME, output_dir: Path = OUTPUT_DIR,
         learning_rate: float = LEARNING_RATE, resume: bool = False) -> None:
    """Executa o treino completo: tokeniza os dados, monta o modelo e o Trainer, treina e salva o melhor."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME if init == MODEL_NAME else init)

    train = load_tagged("train", limit)
    val = load_tagged("val", VAL_LIMIT)

    train = train.map(tokenize_and_align, batched=True, fn_kwargs={"tokenizer": tokenizer},
                      remove_columns=train.column_names)
    val = val.map(tokenize_and_align, batched=True, fn_kwargs={"tokenizer": tokenizer},
                  remove_columns=val.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        init, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)

    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
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

    trainer.train(resume_from_checkpoint=checkpoint_to_resume(output_dir, resume))

    best = output_dir / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o formatador de ditado.")
    parser.add_argument("--limit", type=int, default=None, help="usa só os primeiros N exemplos de treino")
    parser.add_argument("--epochs", type=int, default=10, help="número de épocas")
    parser.add_argument("--init", default=MODEL_NAME, help="modelo de partida (uma pasta continua um modelo treinado)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="pasta dos checkpoints e do best/")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="taxa de aprendizado")
    parser.add_argument("--resume", action="store_true", help="continua do último checkpoint da pasta de saída")
    options = parser.parse_args()

    main(options.limit, options.epochs, options.init, options.output, options.lr, options.resume)


