

ACTIONS = ["KEEP", "DELETE"]
PUNCTS = {"NONE":"", "COMMA":',', "PERIOD": ".", "QUESTION":"?"}

def apply_tags(tokens: list[str], actions: list[str], puncts: list[str]) -> str :
    
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


            

        