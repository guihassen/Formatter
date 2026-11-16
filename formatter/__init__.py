"""Formatador de ditado do DronsioFlow.

Transforma o texto cru do Whisper em texto pronto: apaga hesitações, recomeços e correções, e coloca
vírgulas, pontos, interrogações e maiúsculas. É um modelo de classificação de tokens: cada palavra
recebe um de 5 rótulos (KEEP_NONE, KEEP_COMMA, KEEP_PERIOD, KEEP_QUESTION, DELETE).

Ordem do pipeline: corpus -> similar -> dataset -> train -> predict. Avaliação: ab_eval e
pick_corrections.
"""

