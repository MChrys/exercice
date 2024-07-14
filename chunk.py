def chunk_speech(self, text):
    tokens = self.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= self.max_tokens * 0.9:
        return [text]

    end_sentence_tokens = set(self.tokenizer.encode('. ! ?', add_special_tokens=False))
    reversed_tokens = list(reversed(tokens))
    chunks = []
    current_chunk = []
    current_chunk_length = 0

    for token in reversed_tokens:
        if current_chunk_length + 1 > self.max_tokens * 0.9:
            if any(t in end_sentence_tokens for t in current_chunk):
                # Trouver le dernier token de fin de phrase dans le chunk actuel
                for i, t in enumerate(current_chunk):
                    if t in end_sentence_tokens:
                        break
                chunks.append(self.tokenizer.decode(list(reversed(current_chunk[:i+1]))))
                current_chunk = current_chunk[i+1:] + [token]
                current_chunk_length = len(current_chunk)
            else:
                chunks.append(self.tokenizer.decode(list(reversed(current_chunk))))
                current_chunk = [token]
                current_chunk_length = 1
        else:
            current_chunk.append(token)
            current_chunk_length += 1

    if current_chunk:
        chunks.append(self.tokenizer.decode(list(reversed(current_chunk))))

    return list(reversed(chunks))