---
language:
  - pt
  - en
license: mit
library_name: transformers
pipeline_tag: token-classification
base_model: microsoft/Multilingual-MiniLM-L12-H384
tags:
  - dictation
  - punctuation
  - disfluency-removal
  - speech-to-text
  - whisper
---

# Formatter: dictation formatter (pt/en)

> **Em português:** um modelo pequeno, que roda em CPU, para limpar o texto que o Whisper transcreve
> de um ditado. Ele apaga hesitações, recomeços e correções, e coloca vírgulas, pontos e interrogações.
> Código, dados e explicação completa: [github.com/guihassen/Formatter](https://github.com/guihassen/Formatter).

A small CPU-friendly model that cleans up raw speech-to-text output (e.g. Whisper) from dictation in
**Portuguese and English**. It removes fillers, restarts and self-corrections, and adds commas,
periods and question marks. It was built for [DronsioFlow](https://github.com/guihassen/DronsioFlow),
a 100% local voice dictation app for Linux.

```text
Whisper:    ahn eu vou eu vou comprar leite digo vou comprar pão na quarta você vem comigo
Formatter:  Eu vou comprar pão na quarta. Você vem comigo?
```

## How it works

Instead of rewriting the text like an LLM, the model **tags every word** with one of 5 labels, so it
cannot invent or change words:

| Label | Meaning |
|---|---|
| `KEEP_NONE` | keep the word |
| `KEEP_COMMA` | keep it and add a comma after |
| `KEEP_PERIOD` | keep it and add a period after |
| `KEEP_QUESTION` | keep it and add a question mark after |
| `DELETE` | remove the word |

Capitalization comes from the punctuation: the word after a period or question mark is capitalized.

- **Architecture:** Multilingual-MiniLM-L12-H384 (12 layers, hidden size 384, ~117M parameters, 82% of
  them in the embedding table) with a 5-way token classification head. Uses the XLM-RoBERTa tokenizer.
- **Input:** up to 128 pieces. Longer texts are labeled in overlapping windows (see the repository's
  `predict.py`).
- **Speed:** about 30–50 ms for a 60-word dictation on a laptop CPU.

## Usage

The simplest way is the repository's `formatter/predict.py`, which handles Whisper punctuation, long
texts and the delete safety rule:

```python
from formatter.predict import load_model, format_text

tokenizer, model = load_model("path/to/this/model")
format_text("so um I think the tests are passing can you check the logs", tokenizer, model)
# 'So I think the tests are passing. Can you check the logs?'
```

With plain `transformers` (short texts, lowercase words without punctuation):

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

repo = "<user>/formatter"  # this model
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForTokenClassification.from_pretrained(repo).eval()

words = "compre o bolo na verdade não compre a torta".split()
encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():
    predictions = model(**encoding).logits.argmax(-1)[0].tolist()

labels, previous = [], None
for position, word_id in enumerate(encoding.word_ids()):
    if word_id is not None and word_id != previous:     # first piece of each word
        labels.append(model.config.id2label[predictions[position]])
    previous = word_id
print(list(zip(words, labels)))
```

**Recommended safety rule:** only accept `DELETE` when its logit beats the best `KEEP_*` logit by at
least 1.0; otherwise keep the word with the best `KEEP_*` label. Deleting a real word is the worst
error, and this rule cuts it by about 40% for a 1-point drop in delete recall (see below).

## Training data

There is no large public corpus of dictations labeled this way, so the training data is
**synthetic**. Clean sentences from [Tatoeba](https://tatoeba.org) (English–Portuguese pairs, CC BY
2.0 FR) are grouped into texts of up to 6 sentences, lowercased, stripped of punctuation and corrupted
with typical speech errors, whose correct labels are known by construction:

- fillers ("ahn", "tipo", "um", "like");
- repeated words and restarts ("eu vou eu vou");
- word and group corrections ("na terça, não, na quarta"), using Word2Vec neighbors to pick
  plausible wrong words;
- whole-clause corrections ("compre o bolo, na verdade não, compre a torta");
- false starts without a trigger word.

Hard negatives that must be **kept** are also added: parallel enumerations ("o preço do arroz, o
preço do feijão") and discourse "na verdade," / "actually,".

- 135,346 training examples, 7,103 validation examples; 17.3% of the words are `DELETE`.
- v1: trained from the base model. v2 (this model): continued from v1 for 1 epoch on the data above
  (learning rate 3e-5, batch size 32, CPU only, about 6 hours).

## Evaluation

On the synthetic validation set (7,103 examples, plain argmax):

| Label | Precision | Recall | F1 |
|---|---|---|---|
| KEEP_NONE | 0.988 | 0.992 | 0.990 |
| DELETE | 0.986 | 0.978 | 0.982 |
| KEEP_PERIOD | 0.945 | 0.958 | 0.952 |
| KEEP_QUESTION | 0.910 | 0.844 | 0.876 |
| KEEP_COMMA | 0.828 | 0.784 | 0.805 |

- Macro F1: **0.921** (v1 on the same set: 0.840).
- Real words deleted: **0.30%** (v1: 1.46%). Delete recall: **97.8%** (v1: 85.8%).
- Whole example exactly right: 66.1% (v1: 47.2%).

With the safety rule at 1.0 (3,000 validation examples): 0.18% of real words deleted, 96.7% delete
recall.

On 20 short excerpts of real dictations corrected by hand (277 words, chosen where v1 was least
sure), v1 got 88.1% of the labels right and v2 with the safety rule 85.9%. This sample is small and
most of the gap is in commas, so treat it as a sanity check, not a benchmark.

## Limitations

- **Synthetic training data.** Real speech has error patterns that the generator does not cover.
- **Commas** are the weakest label, and many are optional.
- **Questions without a cue in the words** ("você vem amanhã") are ambiguous; keep Whisper's "?" when
  integrating.
- **Short corrections with little context** ("we should deploy on friday no wait on monday") can
  delete the wrong word.
- **Restarts that share no words** ("the team was going we need to leave now") are often kept.
- **Sentences abandoned at the end of a dictation** are rarely in the training data and usually kept.
- **No "!", ":" or ";".**
- **It never fixes words.** If Whisper misheard a word, it stays wrong.
- Only Portuguese and English were trained.

## Credits

- Sentences: [Tatoeba](https://tatoeba.org), CC BY 2.0 FR.
- Base model: [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384), MIT.
- Tokenizer: [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base), MIT.
