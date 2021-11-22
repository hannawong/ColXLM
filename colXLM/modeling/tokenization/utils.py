import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches, positive_batches, negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((Q, D))

    return batches


def tensorize_triples_qlm(query_tokenizer, doc_tokenizer, queries, positives,bsize):
    assert len(queries) == len(positives) 
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    positive_ids, positive_mask = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask)in zip(query_batches, positive_batches):
        Q = (q_ids, q_mask)
        D = (p_ids, p_mask)
        batches.append((Q, D))

    return batches


def tensorize_triples_prop(query_tokenizer, doc_tokenizer, docs, query_positives, query_negatives, bsize):
    assert len(docs) == len(query_positives) == len(query_negatives)
    assert bsize is None or len(docs) % bsize == 0

    N = len(docs)
    Q_ids, Q_mask = query_tokenizer.tensorize(query_positives + query_negatives)
    Q_ids = Q_ids.view(2, N, -1)
    Q_mask = Q_mask.view(2, N, -1) ## [2,12,32]
    D_ids, D_mask = doc_tokenizer.tensorize(docs) #[12,180]
    D_ids = D_ids.unsqueeze(0)
    D_mask = D_mask.unsqueeze(0)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[:,indices], Q_mask[:,indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = Q_ids, Q_mask

    doc_batches = _split_into_batches(D_ids.squeeze(), D_mask.squeeze(), bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)
    batches = []
    for (d_ids, d_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(doc_batches, positive_batches, negative_batches):
        D = (torch.cat((d_ids, d_ids)), torch.cat((d_mask, d_mask)))
        Q = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((D, Q))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
