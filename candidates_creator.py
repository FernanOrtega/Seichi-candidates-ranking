def max_hits(n):
    # return (n + n**2)/2
    return sum([compute_score(i + 1) for i in range(n)])


def compute_score(x):
    # return 0.5+ 1 / np.math.sqrt(2 * x)
    # return 1 / np.math.sqrt(x)
    # return max(1-0.05*x, 0)
    return 0 if x <= 0 else 1 / x


def flat_sent(sub_sequence):
    return [token for block in sub_sequence for token in block]


def candidates_of_sentence(blocks):
    candidates = []
    for i, item in enumerate(blocks):
        for j in range(i + 1, len(blocks) + 1):
            condition = flat_sent(blocks[i:j])
            if len(condition) > 1:
                candidates.append([flat_sent(blocks[:i])] + [condition] + [flat_sent(blocks[j:])])

    return candidates


def compute_candidates(row, w2v_model):
    l_candidates = []
    blocks = row[0]
    conditions = row[1]
    tokens = flat_sent(blocks)

    for sequence in candidates_of_sentence(blocks):
        if len(sequence) > 1:
            sequence.sort(key=lambda x: x[0])

            l_index_sequence = [w_index for (w_index, dep_idx) in sequence]
            set_index_sequence = set(l_index_sequence)
            tokens_sequence = [(w2v_model.word2idx(tokens[i - 1]), i_dep) for (i, i_dep) in sequence]

            if len(conditions) > 0:
                array_hits = [sum([value * compute_score(index + 1) for index, value in
                                   enumerate([int(i in set_index_sequence) for i in cond])])
                             for cond in conditions]
                score = max([2.0 * array_hits[index] / max_hits(len(condition)) + max_hits(len(sequence))
                             for index, condition in enumerate(conditions)])
            else:
                score = 0.0

            l_candidates.append([l_index_sequence, tokens_sequence, score])

    return l_candidates
