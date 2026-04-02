import json

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# A custom JSON dumper that pretty-prints dictionaries with indentation, while keeping lists in a compact format.
def dump_pretty(obj, fp, indent=4, ensure_ascii=False):
    def _serialize(o, level):
        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            indent_str = " " * (indent * level)
            child_indent_str = " " * (indent * (level + 1))

            items.append("{\n")
            kvs = list(o.items())
            for i, (k, v) in enumerate(kvs):
                key = json.dumps(k, ensure_ascii=ensure_ascii)
                value = _serialize(v, level + 1)
                items.append(f"{child_indent_str}{key}: {value}")
                if i < len(kvs) - 1:
                    items.append(",")
                items.append("\n")
            items.append(indent_str + "}")
            return "".join(items)

        elif isinstance(o, list):
            inner = ", ".join(_serialize(v, level) for v in o)
            return "[" + inner + "]"

        else:
            return json.dumps(o, ensure_ascii=ensure_ascii)

    fp.write(_serialize(obj, 0))


def _normalize_rows(features, eps=1e-12):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, eps, None)


def _estimate_mu_cov(features, shrink=0.2, diag=False, eps=1e-6):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        raise ValueError("Cannot estimate covariance from an empty feature matrix.")

    mu = features.mean(axis=0, keepdims=True)
    centered = features - mu
    denom = max(features.shape[0] - 1, 1)
    cov = centered.T @ centered / float(denom)
    cov = cov.astype(np.float32, copy=False)
    if diag:
        cov = np.diag(np.diag(cov))
    dim = cov.shape[0]
    trace_mean = float(np.trace(cov)) / max(dim, 1)
    cov = (1.0 - float(shrink)) * cov + float(shrink) * trace_mean * np.eye(dim, dtype=np.float32)
    cov = cov + eps * np.eye(dim, dtype=np.float32)
    return mu.astype(np.float32, copy=False), cov


def _inv_sqrt_cov(cov, eps=1e-6):
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    inv_sqrt = evecs @ np.diag(np.power(evals, -0.5, dtype=np.float32)) @ evecs.T
    return inv_sqrt.astype(np.float32, copy=False)


def subject_adaptive_whiten(features, subject_ids, shrink=0.2, diag=False):
    features = np.asarray(features, dtype=np.float32)
    subject_ids = np.asarray(subject_ids)
    if features.shape[0] != subject_ids.shape[0]:
        raise ValueError("features and subject_ids must have the same first dimension.")

    processed = np.empty_like(features, dtype=np.float32)
    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        subject_features = features[mask]
        if subject_features.shape[0] == 0:
            continue
        mu, cov = _estimate_mu_cov(subject_features, shrink=shrink, diag=diag)
        inv_sqrt = _inv_sqrt_cov(cov)
        whitened = (subject_features - mu) @ inv_sqrt.T
        processed[mask] = _normalize_rows(whitened.astype(np.float32, copy=False))
    return processed


def candidate_whiten(features, shrink=0.05, diag=False):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        return features
    mu, cov = _estimate_mu_cov(features, shrink=shrink, diag=diag)
    inv_sqrt = _inv_sqrt_cov(cov)
    whitened = (features - mu) @ inv_sqrt.T
    return _normalize_rows(whitened.astype(np.float32, copy=False))


def csls_scores(similarities, k=10):
    similarities = np.asarray(similarities, dtype=np.float32)
    if similarities.ndim != 2:
        raise ValueError("CSLS expects a 2-D similarity matrix.")
    n_q, n_c = similarities.shape
    if n_q == 0 or n_c == 0:
        return similarities

    k_eff = max(1, min(int(k), n_q, n_c))
    row_topk = np.partition(similarities, kth=n_c - k_eff, axis=1)[:, -k_eff:]
    col_topk = np.partition(similarities, kth=n_q - k_eff, axis=0)[-k_eff:, :]
    rx = row_topk.mean(axis=1, keepdims=True)
    ry = col_topk.mean(axis=0, keepdims=True)
    return (2.0 * similarities - rx - ry).astype(np.float32, copy=False)


def topk(matrix, k, target_indices=None):
    sorted_indices = np.argsort(-matrix, axis=1)
    rankings = np.argsort(sorted_indices, axis=1)
    if target_indices is None:
        target_ranks = np.diag(rankings) + 1
    else:
        target_indices = np.asarray(target_indices, dtype=np.int64)
        target_ranks = rankings[np.arange(sorted_indices.shape[0]), target_indices] + 1

    count_k = 0
    count_1  = 0
    for i in range(sorted_indices.shape[0]):
        if target_ranks[i] <= k:
            count_k += 1
        if target_ranks[i] == 1:
            count_1 += 1
    return count_k, count_1


def _build_candidate_bank(image_features, object_indices, image_indices):
    image_features = np.asarray(image_features, dtype=np.float32)
    object_indices = np.asarray(object_indices)
    image_indices = np.asarray(image_indices)

    candidate_map = {}
    candidate_features = []
    target_indices = []
    for feature, object_idx, image_idx in zip(image_features, object_indices, image_indices):
        key = (int(object_idx), int(image_idx))
        if key not in candidate_map:
            candidate_map[key] = len(candidate_features)
            candidate_features.append(feature)
        target_indices.append(candidate_map[key])

    return np.stack(candidate_features, axis=0), np.asarray(target_indices, dtype=np.int64)


def compute_retrieval_scores(
    eeg_features,
    image_features,
    subject_ids,
    object_indices,
    image_indices,
    eval_mode='plain_cosine',
    sattc_params=None,
):
    sattc_params = sattc_params or {}
    eeg_features = np.asarray(eeg_features, dtype=np.float32)
    subject_ids = np.asarray(subject_ids)
    object_indices = np.asarray(object_indices)
    image_indices = np.asarray(image_indices)

    candidate_features, target_indices = _build_candidate_bank(image_features, object_indices, image_indices)
    if sattc_params.get('cw_enabled', False):
        candidate_features = candidate_whiten(
            candidate_features,
            shrink=sattc_params.get('cw_shrink', 0.05),
            diag=sattc_params.get('cw_diag', False),
        )

    if eval_mode == 'plain_cosine':
        similarity_matrix = cosine_similarity(eeg_features, candidate_features).astype(np.float32, copy=False)
        return similarity_matrix, target_indices

    processed_queries = eeg_features
    if eval_mode in {'saw', 'saw_csls'}:
        processed_queries = subject_adaptive_whiten(
            eeg_features,
            subject_ids,
            shrink=sattc_params.get('saw_shrink', 0.2),
            diag=sattc_params.get('saw_diag', False),
        )
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    pre_csls = cosine_similarity(processed_queries, candidate_features).astype(np.float32, copy=False)
    if eval_mode == 'saw':
        return pre_csls, target_indices
    scores = csls_scores(pre_csls, k=sattc_params.get('csls_k', 12))
    return scores, target_indices


def retrieve_all(
    eeg_features,
    image_features,
    average: bool,
    subject_ids=None,
    object_indices=None,
    image_indices=None,
    eval_mode='plain_cosine',
    sattc_params=None,
):
    if object_indices is None or image_indices is None:
        similarity_matrix = cosine_similarity(eeg_features, image_features)
        count_5, count_1 = topk(similarity_matrix, 5)
        return count_5, count_1, eeg_features.shape[0]

    similarity_matrix, target_indices = compute_retrieval_scores(
        eeg_features,
        image_features,
        subject_ids,
        object_indices,
        image_indices,
        eval_mode=eval_mode,
        sattc_params=sattc_params,
    )
    count_5, count_1 = topk(similarity_matrix, 5, target_indices=target_indices)
    return count_5, count_1, eeg_features.shape[0]