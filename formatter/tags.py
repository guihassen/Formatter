"""Converte entre texto pontuado e etiquetas por palavra, nos dois sentidos.

Cada palavra tem uma ação (KEEP ou DELETE) e uma pontuação depois dela (NONE, COMMA, PERIOD ou
QUESTION). `extract_tags` tira essas etiquetas de um texto pronto e `apply_tags` monta o texto de volta.
Propriedade central do projeto: apply_tags(*extract_tags(texto)) == texto.
"""

ACTIONS = ["KEEP", "DELETE"]
PUNCTS = {"NONE":"", "COMMA":',', "PERIOD": ".", "QUESTION":"?"}

def apply_tags(tokens: list[str], actions: list[str], puncts: list[str]) -> str :
    """Monta o texto final a partir das palavras e das etiquetas.

    Pula as palavras DELETE, cola a pontuação de cada palavra e põe maiúscula na primeira letra depois
    de "." ou "?" (e no começo do texto).
    """
    
    if not (len(tokens) == len(actions) == len(puncts)):
        raise ValueError("tokens, actions e puncts precisam ter o mesmo tamanho")
    
    output = []
    capitalize_next = True

    for token, action, punct in zip(tokens, actions, puncts) :
        if action == "DELETE" :
           continue

        if capitalize_next :
            token = token[0].upper() + token[1:]

        token = token + PUNCTS[punct]

        output.append(token)

        if punct == "PERIOD" or punct == "QUESTION" :
            capitalize_next = True

        else :
            capitalize_next = False
                
    return " ".join(output) 



SYMBOLS = {",": "COMMA", ".": "PERIOD", "?": "QUESTION"}

def extract_tags(text):
    """Faz o caminho inverso de apply_tags: separa um texto em palavras e etiquetas.

    Tira a pontuação final de cada palavra (só ",", "." e "?"), deixa minúscula a primeira letra de cada
    frase e marca todas as palavras como KEEP. Devolve (tokens, actions, puncts).
    """
    tokens, actions, puncts = [], [], []
    lowercase_next = True

    for word in text.split():
        punct = "NONE"
        action = "KEEP"


        if word[-1] in SYMBOLS:         
            punct = SYMBOLS[word[-1]]   
            word = word[:-1]
            

        if lowercase_next :
            word = word[0].lower() + word[1:]
        
        puncts.append(punct)
        actions.append(action)
        tokens.append(word)

        lowercase_next = punct in ("PERIOD", "QUESTION")
        
        

    return tokens, actions, puncts


            

        