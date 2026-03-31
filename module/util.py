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


def csls_adaptive(
    similarities,
    k0=12,
    kmin=5,
    kmax=20,
    alpha=1.0,
    m=10,
    col_alpha=None,
    col_m=None,
    col_kmin=None,
    col_kmax=None,
):
    similarities = np.asarray(similarities, dtype=np.float32)
    if similarities.ndim != 2:
        raise ValueError("Adaptive CSLS expects a 2-D similarity matrix.")
    n_q, n_c = similarities.shape
    if n_q == 0 or n_c == 0:
        return similarities

    kmin = max(1, min(int(kmin), n_c))
    kmax = max(kmin, min(int(kmax), n_c))
    k0 = max(kmin, min(int(k0), kmax))
    m_eff = max(1, min(int(m), n_c))

    row_topm = np.partition(similarities, kth=n_c - m_eff, axis=1)[:, -m_eff:]
    rho_row = row_topm.mean(axis=1)
    med_row = np.median(rho_row) + 1e-9
    scale_row = np.power(np.clip(rho_row / med_row, 1e-6, None), float(alpha))
    k_row = np.clip(np.rint(k0 * scale_row), kmin, kmax).astype(np.int32)

    r_q = np.empty(n_q, dtype=np.float32)
    for k_val in np.unique(k_row):
        mask = k_row == k_val
        k_int = max(1, int(k_val))
        row_vals = np.partition(similarities[mask], kth=n_c - k_int, axis=1)[:, -k_int:]
        r_q[mask] = row_vals.mean(axis=1)

    col_alpha_eff = float(alpha if col_alpha is None else col_alpha)
    col_m_eff = max(1, min(int(col_m if col_m is not None else m_eff), n_c))
    col_kmin_eff = max(1, min(int(col_kmin if col_kmin is not None else kmin), n_q))
    col_kmax_eff = max(col_kmin_eff, min(int(col_kmax if col_kmax is not None else kmax), n_q))
    base_k_col = max(col_kmin_eff, min(int(k0), col_kmax_eff))

    row_top_idx = np.argpartition(similarities, kth=n_c - col_m_eff, axis=1)[:, -col_m_eff:]
    hits = np.zeros(n_c, dtype=np.float32)
    np.add.at(hits, row_top_idx.reshape(-1), 1.0)
    rho_col = hits / max(1, n_q)
    med_col = np.median(rho_col) + 1e-9
    scale_col = np.power(np.clip(rho_col / med_col, 1e-6, None), col_alpha_eff)
    k_col = np.clip(np.rint(base_k_col * scale_col), col_kmin_eff, col_kmax_eff).astype(np.int32)

    r_c = np.empty(n_c, dtype=np.float32)
    for k_val in np.unique(k_col):
        mask = k_col == k_val
        k_int = max(1, int(k_val))
        col_vals = np.partition(similarities[:, mask], kth=n_q - k_int, axis=0)[-k_int:, :]
        r_c[mask] = col_vals.mean(axis=0)

    return (2.0 * similarities - r_q[:, None] - r_c[None, :]).astype(np.float32, copy=False)


def build_structural_expert(
    pre_csls_similarities,
    geom_scores,
    top_l=5,
    popularity_k=5,
    hub_high_quantile=0.95,
    hub_mid_quantile=0.80,
    lambda_pen=None,
    lambda_bonus=None,
):
    pre_csls_similarities = np.asarray(pre_csls_similarities, dtype=np.float32)
    geom_scores = np.asarray(geom_scores, dtype=np.float32)
    q, c = pre_csls_similarities.shape
    if q == 0 or c == 0:
        return np.zeros_like(pre_csls_similarities, dtype=np.float32)

    top_l = max(1, min(int(top_l), q, c))
    popularity_k = max(1, min(int(popularity_k), c))

    row_sorted = np.argsort(-pre_csls_similarities, axis=1)
    row_rank = np.argsort(row_sorted, axis=1) + 1
    col_sorted = np.argsort(-pre_csls_similarities, axis=0)
    col_rank = np.argsort(col_sorted, axis=0) + 1

    mnn1_mask = (row_rank == 1) & (col_rank == 1)
    bidirectional_topl_mask = (row_rank <= top_l) & (col_rank <= top_l)
    popularity = (row_rank <= popularity_k).sum(axis=0).astype(np.float32)

    if popularity.max() > popularity.min():
        hub_score = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    else:
        hub_score = np.zeros_like(popularity, dtype=np.float32)

    high_threshold = float(np.quantile(hub_score, hub_high_quantile)) if hub_score.size else 1.0
    mid_threshold = float(np.quantile(hub_score, hub_mid_quantile)) if hub_score.size else 1.0

    high_hub = hub_score >= high_threshold
    mid_hub = (hub_score >= mid_threshold) & (~high_hub)
    hub_like = (row_rank > popularity_k) & (col_rank <= top_l)
    protect_mask = bidirectional_topl_mask

    geom_std = float(np.std(geom_scores)) if geom_scores.size else 1.0
    geom_std = max(geom_std, 1e-6)
    lambda_pen = float(lambda_pen) if lambda_pen is not None else 0.5 * geom_std
    lambda_bonus = float(lambda_bonus) if lambda_bonus is not None else 0.25 * geom_std

    s_struct = np.zeros_like(pre_csls_similarities, dtype=np.float32)
    s_struct[bidirectional_topl_mask] += 0.5 * lambda_bonus
    s_struct[mnn1_mask] += 0.5 * lambda_bonus

    high_penalty = hub_like & high_hub[None, :] & (~protect_mask)
    mid_penalty = hub_like & mid_hub[None, :] & (~protect_mask)
    hub_matrix = np.broadcast_to(hub_score[None, :], pre_csls_similarities.shape)
    s_struct[high_penalty] -= lambda_pen * hub_matrix[high_penalty]
    s_struct[mid_penalty] -= 0.5 * lambda_pen * hub_matrix[mid_penalty]
    return s_struct


def fuse_poe_scores(geom_scores, struct_scores, beta=1.9):
    return np.asarray(geom_scores, dtype=np.float32) + float(beta) * np.asarray(struct_scores, dtype=np.float32)


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

    if eval_mode == 'plain_cosine':
        similarity_matrix = cosine_similarity(eeg_features, candidate_features).astype(np.float32, copy=False)
        return similarity_matrix, target_indices

    processed_queries = eeg_features
    if eval_mode in {'saw', 'saw_csls', 'saw_adacsls', 'saw_adacsls_poe'}:
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
    if eval_mode == 'saw_csls':
        scores = csls_scores(pre_csls, k=sattc_params.get('csls_k', 12))
        return scores, target_indices

    geom_scores = csls_adaptive(
        pre_csls,
        k0=sattc_params.get('csls_k', 12),
        kmin=sattc_params.get('csls_kmin', 5),
        kmax=sattc_params.get('csls_kmax', 20),
        alpha=sattc_params.get('csls_alpha', 1.0),
        m=sattc_params.get('csls_m', 10),
        col_alpha=sattc_params.get('csls_col_alpha'),
        col_m=sattc_params.get('csls_col_m'),
        col_kmin=sattc_params.get('csls_col_kmin'),
        col_kmax=sattc_params.get('csls_col_kmax'),
    )
    if eval_mode == 'saw_adacsls':
        return geom_scores, target_indices

    struct_scores = build_structural_expert(
        pre_csls,
        geom_scores,
        top_l=sattc_params.get('struct_top_l', 5),
        popularity_k=sattc_params.get('struct_popularity_k', 5),
        hub_high_quantile=sattc_params.get('struct_hub_high_quantile', 0.95),
        hub_mid_quantile=sattc_params.get('struct_hub_mid_quantile', 0.80),
        lambda_pen=sattc_params.get('poe_lambda_pen'),
        lambda_bonus=sattc_params.get('poe_lambda_bonus'),
    )
    final_scores = fuse_poe_scores(geom_scores, struct_scores, beta=sattc_params.get('poe_beta', 1.9))
    return final_scores, target_indices


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