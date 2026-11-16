"""Testes de formatter/ab_eval.py, com um histórico falso e respostas simuladas."""

import json

import pytest

from formatter import ab_eval
from formatter.ab_eval import highlight, load_dictations, load_results, pending, save_result, summarize


@pytest.fixture
def history(tmp_path):
    """Cria um histórico falso com ditados, comandos, agentes e entradas vazias."""
    entries = [
        {"at": "t1", "kind": "dictation", "raw": "ahn vou sair", "text": "Vou sair."},
        {"at": "t2", "raw": "oi tudo bem", "text": "Oi, tudo bem?"},          # entradas antigas não têm o campo kind
        {"at": "t3", "kind": "command", "raw": "traduz", "text": "translate"},
        {"at": "t4", "kind": "dictation", "raw": "   ", "text": ""},
        {"at": "t5", "kind": "agent", "raw": "abre o site", "text": "ok"},
        {"at": "t6", "kind": "dictation", "raw": "é isso", "text": "É isso."},
    ]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_load_dictations_keeps_only_dictations_with_raw_text(history):
    """Só entram ditados com texto cru (entradas antigas sem tipo contam como ditado)."""
    assert [d["at"] for d in load_dictations(history)] == ["t1", "t2", "t6"]


def test_results_round_trip_and_resume(tmp_path, history):
    """Os julgamentos são salvos e lidos de volta, e os já julgados não voltam."""
    path = tmp_path / "eval" / "results.jsonl"
    assert load_results(path) == []
    save_result({"at": "t1", "winner": "model"}, path)
    save_result({"at": "t6", "winner": "tie"}, path)
    results = load_results(path)
    assert results == [{"at": "t1", "winner": "model"}, {"at": "t6", "winner": "tie"}]
    assert [d["at"] for d in pending(load_dictations(history), results)] == ["t2"]


def test_highlight_marks_only_the_different_words():
    """Só as palavras diferentes ficam destacadas."""
    out = highlight("Vou sair amanhã.", "Vou sair hoje.")
    assert out.startswith("Vou sair ")
    assert f"{ab_eval.BOLD_YELLOW}amanhã.{ab_eval.RESET}" in out
    assert highlight("igual", "igual") == "igual"


def test_summarize_counts_winners():
    """O placar conta cada vencedor."""
    results = [{"winner": w} for w in ["model", "model", "current", "tie", "same"]]
    assert summarize(results) == {"model": 2, "current": 1, "tie": 1, "same": 1}


def test_main_asks_blind_and_saves_each_answer(tmp_path, history, monkeypatch):
    """O laço não mostra qual é qual, aceita só respostas válidas e traduz "1"/"2" para o lado certo."""
    pytest.importorskip("torch")
    from formatter import predict

    outputs = {"ahn vou sair": "Vou sair.", "oi tudo bem": "Oi tudo bem.", "é isso": "É isso!"}
    monkeypatch.setattr(predict, "load_model", lambda: (None, None))
    monkeypatch.setattr(predict, "format_text", lambda text, tok, model: outputs[text])

    shown = []
    answers = iter(["x", "1", "="])                       # "x" is invalid and asked again
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: shown.append(" ".join(map(str, args))))

    results_path = tmp_path / "results.jsonl"
    ab_eval.main(history, results_path)
    results = {r["at"]: r for r in load_results(results_path)}

    assert results["t1"]["winner"] == "same"               # textos idênticos não são perguntados
    assert results["t6"]["winner"] == "tie"
    first_shown = next(line for line in shown if line.startswith("1) "))
    expected = "model" if "bem." in first_shown else "current"   # o modelo escreveu "bem.", o DronsioFlow "bem?"
    assert results["t2"]["winner"] == expected             # a resposta "1" é traduzida de volta, desfazendo o sorteio
    assert not any("model" in line or "current" in line for line in shown if line[:3] in ("1) ", "2) "))
