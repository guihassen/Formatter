"""Testes de formatter/split_tatoeba.py: leitura dos pares, separação por idioma e gravação."""

from formatter.split_tatoeba import read_pairs, save_sentences, split_pairs


def test_read_pairs_ignores_bom_and_windows_line_breaks(tmp_path):
    """O BOM do começo e o \\r\\n do Windows não entram nos textos."""
    path = tmp_path / "pairs.tsv"
    path.write_bytes("﻿1\tHi.\t10\tOi.\r\n2\tBye.\t20\tTchau.\r\n".encode("utf-8"))
    assert read_pairs(path) == [("1", "Hi.", "10", "Oi."), ("2", "Bye.", "20", "Tchau.")]


def test_read_pairs_skips_broken_lines(tmp_path):
    """Linhas sem os 4 campos são ignoradas."""
    path = tmp_path / "pairs.tsv"
    path.write_text("1\tHi.\t10\tOi.\nquebrada\n", encoding="utf-8")
    assert read_pairs(path) == [("1", "Hi.", "10", "Oi.")]


def test_split_pairs_keeps_each_id_once_in_order():
    """Uma frase que aparece em vários pares entra uma vez só, na ordem em que apareceu primeiro."""
    pairs = [("1", "Hi.", "10", "Oi."), ("1", "Hi.", "11", "Olá."), ("2", "Bye.", "10", "Oi.")]
    assert split_pairs(pairs) == {
        "eng": [("1", "Hi."), ("2", "Bye.")],
        "por": [("10", "Oi."), ("11", "Olá.")],
    }


def test_save_sentences_writes_the_corpus_format(tmp_path):
    """A saída tem o formato id<tab>idioma<tab>texto, com acentos."""
    path = tmp_path / "por.tsv"
    save_sentences([("10", "Olá, você vem?")], "por", path)
    assert path.read_text(encoding="utf-8") == "10\tpor\tOlá, você vem?\n"
