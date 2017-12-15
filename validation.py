from collections import Counter


def measure_macro(l_conf_matrices, function_micro):
    measure_per_matrix = [function_micro(conf_matrix) for conf_matrix in l_conf_matrices]
    return sum(measure_per_matrix) / len(measure_per_matrix)


def f1_micro(conf_matrix):
    dividend = 2 * conf_matrix['tp']
    divisor = 2 * conf_matrix['tp'] + conf_matrix['fp'] + conf_matrix['fn']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def precision_micro(conf_matrix):
    dividend = conf_matrix['tp']
    divisor = conf_matrix['tp'] + conf_matrix['fp']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def recall_micro(conf_matrix):
    dividend = conf_matrix['tp']
    divisor = conf_matrix['tp'] + conf_matrix['fn']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def tnr_micro(conf_matrix):
    dividend = conf_matrix['tn']
    divisor = conf_matrix['tn'] + conf_matrix['fp']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def validate(l_predicted, l_expected, l_sizes):
    # confusion matrix
    l_conf_matrices = []
    l_conf_matrices_t = []
    for predicted, expected, size_sentence in zip(l_predicted, l_expected, l_sizes):
        flat_predicted = [index for elem in predicted for index in elem]
        flat_expected = [index for elem in expected for index in elem]
        # Chunk based computation
        tp = sum([len([index for index in elem if index in flat_expected]) / len(elem) for elem in predicted])
        fp = sum([len([index for index in elem if index not in flat_expected]) / len(elem) for elem in predicted])
        fn = sum([len([index for index in elem if index not in flat_predicted]) / len(elem) for elem in expected])
        tn = sum([len([index for index in elem if index in flat_predicted]) / len(elem) for elem in expected])
        l_conf_matrices.append({'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})

        # Token based computation
        tp = len(set(flat_predicted) & set(flat_expected))
        fp = len(flat_predicted) - tp
        fn = len(flat_expected) - tp
        tn = size_sentence - tp - fp - fn
        l_conf_matrices_t.append({'tp': tp / size_sentence, 'tn': tn / size_sentence, 'fp': fp / size_sentence,
                                  'fn': fn / size_sentence})

    confusion_matrix = Counter()
    for d in l_conf_matrices:
        for k, v in d.items():
            confusion_matrix[k] += v
    confusion_matrix_t = Counter()
    for d in l_conf_matrices:
        for k, v in d.items():
            confusion_matrix_t[k] += v

    return [measure_macro(l_conf_matrices, precision_micro), measure_macro(l_conf_matrices, recall_micro),
            measure_macro(l_conf_matrices, f1_micro), measure_macro(l_conf_matrices, tnr_micro),
            precision_micro(confusion_matrix), recall_micro(confusion_matrix), f1_micro(confusion_matrix),
            tnr_micro(confusion_matrix),
            measure_macro(l_conf_matrices_t, precision_micro), measure_macro(l_conf_matrices_t, recall_micro),
            measure_macro(l_conf_matrices_t, f1_micro), measure_macro(l_conf_matrices_t, tnr_micro),
            precision_micro(confusion_matrix_t), recall_micro(confusion_matrix_t), f1_micro(confusion_matrix_t),
            tnr_micro(confusion_matrix_t)]


def overlapped(candidate, non_overlapped):
    if len(non_overlapped) == 0:
        return False
    c_min = candidate[0]
    c_max = candidate[-1]
    for other_candidate in non_overlapped:
        o_min = other_candidate[0]
        o_max = other_candidate[-1]

        if max(c_min, o_min) <= min(c_max, o_max):
            return True

    return False


def get_best_candidates(l_candidates):
    result = []
    candidates_sorted = sorted(l_candidates, key=lambda x: x[1], reverse=True)
    for candidate in candidates_sorted:
        if candidate[1] >= 0.5 and not overlapped(candidate[0], result):
            result.append(candidate[0])

    return result


def evaluate(test, model):

    # line -> [tokens, deptree, conditions, candidates]
    # candidate -> [token_indexes, tokens_and_deptag, score]
    l_expected = [line[2] for line in test]
    l_predicted = []
    l_sizes = []

    for line in test:
        l_sizes.append(len(line[0]))
        x = [candidate[1] for candidate in line[3]]
        sequences = [candidate[0] for candidate in line[3]]
        l_scores = model.predict(x)
        l_candidates = [[sequence, score[0]] for sequence, candidate, score in zip(sequences, x, l_scores)]
        l_predicted.append(get_best_candidates(l_candidates))

    return validate(l_predicted, l_expected, l_sizes)