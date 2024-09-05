def sequences(labels):
    seqs = []
    start_index = None
    for i, x in enumerate(labels):
        if start_index is None and x == 1:
            start_index = i
        if start_index is not None and x == 0:
            seqs.append((start_index, i))
            start_index = None
    return seqs


def evaluate_sequences(y_true, y_pred):
    seqs_true = sequences(y_true)
    seqs_pred = sequences(y_pred)

    tp_for_recall = 0
    tp_for_precision = 0
    fp = 0
    used_true_seqs = []

    for seq in seqs_pred:
        start_index, end_index = seq
        for i, true_seq in enumerate(seqs_true):
            true_start_index, true_end_index = true_seq
            if true_start_index <= start_index < true_end_index or \
                    true_start_index <= end_index < true_end_index or \
                    start_index < true_start_index and end_index > true_end_index:
                tp_for_precision += 1
                if i not in used_true_seqs:
                    tp_for_recall += 1
                    used_true_seqs.append(i)
            else:
                fp += 1


    all_for_precision = tp_for_precision + fp
    precision = tp_for_precision / all_for_precision if all_for_precision > 0 else 0
    recall = tp_for_recall / len(seqs_true) if len(seqs_true) > 0 else 1
    if precision + recall <= 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall
