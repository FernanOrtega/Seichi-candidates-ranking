def max_hits(n):
    # return (n + n**2)/2
    return sum([score_funct(i + 1) for i in range(n)])


def score_funct(x):
    # return 0.5+ 1 / np.math.sqrt(2 * x)
    # return 1 / np.math.sqrt(x)
    # return max(1-0.05*x, 0)
    return 0 if x <= 0 else 1 / x


# todo: we now consider all of the possible candidates built from
def compute_candidates(row, w2v_model):
    l_cand_of_deptree = []
    tokens = row[0]
    deptree = row[1]
    conditions = row[2]
    '''
    for sequence in candidates_of_node(deptree)[1]:
        if len(sequence) > 1:
            sequence.sort(key=lambda x: x[0])

            if not [True for token in sequence if token[0] - 1 >= len(tokens)]:
                l_index_sequence = [w_index for (w_index, dep_idx) in sequence]
                set_index_sequence = set(l_index_sequence)
                tokens_sequence = [(w2v_model.word2idx(tokens[i - 1]), i_dep) for (i, i_dep) in sequence]

                if len(conditions) > 0:
                    array_hits = [sum([value * score_funct(index + 1) for index, value in
                                       enumerate([int(i in set_index_sequence) for i in cond])])
                                  for cond in conditions]
                    score = max([2.0 * array_hits[index] / max_hits(len(condition)) + max_hits(len(sequence))
                                 for index, condition in enumerate(conditions)])
                else:
                    score = 0.0

                l_cand_of_deptree.append([l_index_sequence, tokens_sequence, score])
            else:
                print('Problem with sequence: ', sequence)
    '''
    return l_cand_of_deptree
