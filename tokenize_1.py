from transformers import LlamaTokenizerFast

# Initialiser le tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained("openlm-research/open_llama_3b")

# Définir les caractères de fin de phrase courants
end_of_sentence_chars = ['.', '!', '?', '...']

# Fonction pour obtenir le token d'un caractère
def get_token_id(char):
    return tokenizer.encode(char, add_special_tokens=False)[0]

# Récupérer les tokens pour chaque caractère de fin de phrase
eos_tokens = {char: get_token_id(char) for char in end_of_sentence_chars}

# Afficher les résultats
for char, token_id in eos_tokens.items():
    print(f"Caractère: '{char}', Token ID: {token_id}")

# Vérifier les différences de tokenization en contexte
test_sentences = [
    "This is a test.",
    "This is a test!",
    "This is a test?",
    "This is a test...",
    ". ! ? ...",
]

for sentence in test_sentences:
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    print(f"\nPhrase: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Dernier token: {tokens[-1]}")