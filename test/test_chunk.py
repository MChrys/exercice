def get_chunks(tokens, max_tokens_per_line, end_of_sentence_tokens):
    chunks = [[]] if len(tokens) else []
    prev = 0

    for i in range(len(tokens)):
        cur_sentence_len = i - prev + 1
        if tokens[i] in end_of_sentence_tokens or cur_sentence_len == max_tokens_per_line:
            if len(chunks[-1]) and len(chunks[-1]) + cur_sentence_len > max_tokens_per_line:
                chunks.append([])
            chunks[-1].extend(tokens[prev:i+1])
            prev = i + 1

    if prev < len(tokens):
        if len(chunks[-1]) and len(chunks[-1]) + i - prev + 1 > max_tokens_per_line:
            chunks.append([])
        chunks[-1].extend(tokens[prev:i+1])

    return chunks


print(
    get_chunks(
        [1, 16644, 31843, 433, 322, 260,260,260,260,260,260, 1397, 31843, 16644, 31844, 433, 322, 260, 1397,250,250,250,250,250,250,360],
        # [1, 16644, 31843, 433, 31843, 260, 1397],
        # [1, 2, 3, 4, 5, 6, 31843],
        # [1],
        # [],
        3,
        [31843]
    )
)