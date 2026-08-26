"""Subject-level augmentations for cross-subject EEG domain generalization.

The key distinction from ordinary per-example MixStyle is coherence: one sampled
style map is shared by every example from a source subject in the current batch.
With grouped same-stimulus batches this estimates subject statistics from matched
content and produces a small collection of internally consistent virtual domains.
"""

import torch


def cross_subject_spectral_style_mix(
    features,
    object_indices,
    image_indices,
    subject_ids,
    alpha=0.5,
    eps=1e-6,
):
    """Sample same-stimulus subject styles in channel-frequency magnitude space.

    A trial's Fourier phase retains its latency/temporal structure, while its log
    magnitude is replaced by a Dirichlet barycenter of the matched source
    subjects. Geometric (log-space) averaging prevents high-power people or bins
    from dominating. Every output therefore combines a measured phase with a
    data-supported synthetic spectrum for the exact same stimulus.
    """
    if features.ndim != 3:
        raise ValueError(
            f'features must have shape (batch, channels, time), got {features.shape}'
        )
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if eps <= 0:
        raise ValueError('eps must be positive')

    groups = {}
    for index, key in enumerate(zip(object_indices.tolist(), image_indices.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)
    out = features.clone()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        # Duplicate observations from one person cannot define cross-person style.
        if len({int(subject_ids[index]) for index in indices}) < 2:
            continue
        group_idx = torch.tensor(indices, device=features.device, dtype=torch.long)
        group = features[group_idx]
        spectrum = torch.fft.rfft(group.float(), dim=-1)
        log_magnitude = torch.log(torch.abs(spectrum).clamp_min(float(eps)))
        concentration = torch.full(
            (len(indices),), max(float(alpha), 1e-3),
            device=features.device, dtype=torch.float32,
        )
        weights = torch.distributions.Dirichlet(concentration).sample((len(indices),))
        mixed_log_magnitude = torch.einsum('os,sct->oct', weights, log_magnitude)
        unit_phase = spectrum / torch.abs(spectrum).clamp_min(float(eps))
        mixed_spectrum = torch.exp(mixed_log_magnitude) * unit_phase
        mixed = torch.fft.irfft(mixed_spectrum, n=features.shape[-1], dim=-1)
        out[group_idx] = mixed.to(features.dtype)
    return out


def translate_auxiliary_subjects(
    features,
    subject_ids,
    partition_boundary,
    channel_map,
    source_mean,
    target_mean,
    blend=1.0,
):
    """Map an auxiliary EEG cohort into the primary acquisition coordinates.

    ``channel_map`` is fitted *only on training stimuli* from the paired cohort
    means, i.e. ``mean(EEG1 meta subjects) -> mean(EEG2 source subjects)``.  The
    same affine map is applied to each auxiliary observation, which retains its
    deviation from the EEG1 cohort while correcting the dataset-common spatial
    acquisition shift. Primary rows are returned exactly unchanged.

    The expected shapes are ``features=(B,C,T)``, ``channel_map=(C,C)``, and
    ``source_mean=target_mean=(C,T)``. ``blend=0`` is an exact no-op and
    ``blend=1`` applies the complete fitted counterfactual map.
    """
    if features.ndim != 3:
        raise ValueError(f'features must have shape (batch, channels, time), got {features.shape}')
    if subject_ids.ndim != 1 or subject_ids.shape[0] != features.shape[0]:
        raise ValueError('subject_ids must be a vector matching the feature batch')
    if partition_boundary <= 0:
        raise ValueError('partition_boundary must be positive')
    if not 0.0 <= blend <= 1.0:
        raise ValueError('blend must be in [0, 1]')
    channels, time = features.shape[-2:]
    if tuple(channel_map.shape) != (channels, channels):
        raise ValueError(
            f'channel_map must have shape {(channels, channels)}, got {tuple(channel_map.shape)}'
        )
    if tuple(source_mean.shape) != (channels, time):
        raise ValueError(
            f'source_mean must have shape {(channels, time)}, got {tuple(source_mean.shape)}'
        )
    if tuple(target_mean.shape) != (channels, time):
        raise ValueError(
            f'target_mean must have shape {(channels, time)}, got {tuple(target_mean.shape)}'
        )

    auxiliary_mask = subject_ids >= partition_boundary
    if not auxiliary_mask.any() or blend == 0:
        return features
    auxiliary = features[auxiliary_mask]
    translated = target_mean.unsqueeze(0) + torch.einsum(
        'oi,bit->bot', channel_map, auxiliary - source_mean.unsqueeze(0)
    )
    out = features.clone()
    out[auxiliary_mask] = auxiliary.lerp(translated, float(blend))
    return out


def _deranged_subject_partners(num_subjects, device):
    """Return a random no-fixed-point mapping over subject positions."""
    order = torch.randperm(num_subjects, device=device)
    offset = int(torch.randint(1, num_subjects, (1,), device=device).item())
    partners = torch.empty_like(order)
    partners[order] = torch.roll(order, shifts=offset)
    return partners


def sample_coherent_subject_weights(subject_ids, alpha=0.5, self_anchor=0.0, device=None):
    """Sample one convex source mixture for each virtual subject in a cohort.

    The returned table is intended to be reused across all stimuli in an epoch or run.
    ``self_anchor`` optionally keeps each virtual subject near its corresponding real
    subject while the remaining mass follows a Dirichlet draw.
    """
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if not 0.0 <= self_anchor < 1.0:
        raise ValueError('self_anchor must be in [0, 1)')
    subjects = torch.as_tensor(subject_ids, dtype=torch.long, device=device)
    subjects = torch.unique(subjects, sorted=True)
    if subjects.numel() < 2:
        raise ValueError('coherent subject mixing requires at least two subjects')
    concentration = torch.full(
        (subjects.numel(),), max(float(alpha), 1e-3),
        device=subjects.device, dtype=torch.float32,
    )
    weights = torch.distributions.Dirichlet(concentration).sample((subjects.numel(),))
    if self_anchor:
        weights = (1.0 - self_anchor) * weights
        weights = weights + self_anchor * torch.eye(subjects.numel(), device=weights.device)
    return subjects, weights


def coherent_cross_subject_stimulus_mix(
    features,
    object_indices,
    image_indices,
    subject_ids,
    cohort_subjects,
    cohort_weights,
    residual_scale=1.0,
    jitter=0.0,
    jitter_alpha=0.5,
):
    """Render fixed barycentric virtual subjects for every complete stimulus group.

    Unlike ordinary group SubjectMix, ``cohort_weights`` is not resampled for each
    stimulus. Row *j* therefore describes the same virtual subject throughout the
    episode. Incomplete or duplicate-subject groups are left untouched rather than
    silently applying a different generative rule.
    """
    if residual_scale < 0:
        raise ValueError('residual_scale must be non-negative')
    if not 0.0 <= jitter <= 1.0:
        raise ValueError('jitter must be in [0, 1]')
    if jitter_alpha <= 0:
        raise ValueError('jitter_alpha must be positive')
    cohort_subjects = torch.as_tensor(
        cohort_subjects, dtype=subject_ids.dtype, device=subject_ids.device
    )
    cohort_weights = torch.as_tensor(
        cohort_weights, dtype=features.dtype, device=features.device
    )
    num_subjects = cohort_subjects.numel()
    if cohort_weights.shape != (num_subjects, num_subjects):
        raise ValueError(
            f'cohort_weights must have shape {(num_subjects, num_subjects)}, '
            f'got {tuple(cohort_weights.shape)}'
        )
    if not torch.allclose(
        cohort_weights.sum(dim=1),
        torch.ones(num_subjects, device=features.device, dtype=features.dtype),
        atol=1e-5,
    ) or torch.any(cohort_weights < 0):
        raise ValueError('cohort_weights must be non-negative and sum to one by row')

    groups = {}
    for index, key in enumerate(zip(object_indices.tolist(), image_indices.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)
    subject_to_position = {
        int(subject): position for position, subject in enumerate(cohort_subjects.tolist())
    }
    out = features.clone()
    for indices in groups.values():
        if len(indices) != num_subjects:
            continue
        by_subject = {int(subject_ids[index]): index for index in indices}
        if len(by_subject) != num_subjects or set(by_subject) != set(subject_to_position):
            continue
        ordered_indices = torch.tensor(
            [by_subject[int(subject)] for subject in cohort_subjects.tolist()],
            device=features.device, dtype=torch.long,
        )
        ordered_features = features[ordered_indices]
        effective_weights = cohort_weights
        if jitter:
            concentration = torch.full(
                (num_subjects,), max(float(jitter_alpha), 1e-3),
                device=features.device, dtype=torch.float32,
            )
            random_weights = torch.distributions.Dirichlet(concentration).sample(
                (num_subjects,)
            ).to(features.dtype)
            effective_weights = (
                (1.0 - float(jitter)) * cohort_weights
                + float(jitter) * random_weights
            )
        mixed = torch.einsum('vs,s...->v...', effective_weights, ordered_features)
        if residual_scale != 1.0:
            centroid = ordered_features.mean(dim=0, keepdim=True)
            mixed = centroid + float(residual_scale) * (mixed - centroid)
        # Store virtual-subject row v at the example originally owned by subject v.
        out[ordered_indices] = mixed
    return out


def cross_dataset_residual_subject_mix(
    features,
    object_indices,
    image_indices,
    subject_ids,
    partition_boundary,
    alpha=0.5,
    residual_scale=0.5,
):
    """Inject centered auxiliary-cohort variability into primary-domain mixtures.

    For every complete matched-stimulus group, subjects below ``partition_boundary``
    define the primary acquisition domain (EEG2) and subjects at/above it define an
    auxiliary natural-variability cohort (EEG1 meta-subjects).  We first draw one
    Dirichlet primary-domain base for every output row.  Auxiliary rows then receive
    their own centered auxiliary residual:

        x* = sum_s w_s x_primary,s + gamma (x_aux,j - mean_j x_aux,j).

    Centering exactly removes the auxiliary cohort's protocol-common response.  The
    primary rows remain ordinary stochastic group mixtures; the auxiliary rows become
    extra virtual primary-protocol subjects carrying measured natural residuals.
    """
    if partition_boundary <= 0:
        raise ValueError('partition_boundary must be positive')
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if residual_scale < 0:
        raise ValueError('residual_scale must be non-negative')

    groups = {}
    for index, key in enumerate(zip(object_indices.tolist(), image_indices.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)
    out = features.clone()
    for indices in groups.values():
        primary_indices = [i for i in indices if int(subject_ids[i]) < partition_boundary]
        auxiliary_indices = [i for i in indices if int(subject_ids[i]) >= partition_boundary]
        if len(primary_indices) < 2 or not auxiliary_indices:
            continue
        primary_idx = torch.tensor(primary_indices, device=features.device, dtype=torch.long)
        auxiliary_idx = torch.tensor(auxiliary_indices, device=features.device, dtype=torch.long)
        output_idx = torch.tensor(indices, device=features.device, dtype=torch.long)
        primary = features[primary_idx]
        concentration = torch.full(
            (len(primary_indices),), max(float(alpha), 1e-3),
            device=features.device, dtype=torch.float32,
        )
        weights = torch.distributions.Dirichlet(concentration).sample((len(indices),))
        bases = torch.einsum('os,s...->o...', weights.to(features.dtype), primary)
        out[output_idx] = bases

        auxiliary = features[auxiliary_idx]
        residuals = auxiliary - auxiliary.mean(dim=0, keepdim=True)
        # output_idx follows the original group order, so locate each auxiliary row's
        # output position explicitly before adding its own natural residual.
        output_position = {int(original): pos for pos, original in enumerate(indices)}
        aux_positions = torch.tensor(
            [output_position[i] for i in auxiliary_indices],
            device=features.device, dtype=torch.long,
        )
        out[auxiliary_idx] = bases[aux_positions] + float(residual_scale) * residuals
    return out


def subject_coherent_mixstyle(
    features,
    subject_ids,
    alpha=0.5,
    granularity="filter_channel",
    eps=1e-5,
    return_metadata=False,
):
    """Mix temporal-feature statistics once per subject, not once per trial.

    Args:
        features: TSConv temporal activations shaped ``(batch, filters, channels, time)``.
        subject_ids: Integer subject labels shaped ``(batch,)``.
        alpha: Symmetric Beta concentration for interpolation weights.
        granularity: ``filter_channel`` retains a style statistic for each temporal
            filter and sensor; ``filter`` pools sensors for a cheaper, coarser style.
        eps: Minimum standard deviation used by the affine map.
        return_metadata: Return sampled style tensors and partner indices for tests and
            diagnostics.

    Statistics are detached, as in MixStyle, so the augmentation cannot be defeated by
    changing the statistics estimator. Gradients still pass through the transformed
    activations. All examples carrying a given subject ID receive the same affine map.
    """
    if features.ndim != 4:
        raise ValueError(
            "subject_coherent_mixstyle expects (batch, filters, channels, time), "
            f"got {tuple(features.shape)}"
        )
    if subject_ids.ndim != 1 or subject_ids.shape[0] != features.shape[0]:
        raise ValueError("subject_ids must be a vector matching the feature batch")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if granularity not in {"filter", "filter_channel"}:
        raise ValueError("granularity must be 'filter' or 'filter_channel'")

    unique_subjects, inverse = torch.unique(subject_ids, sorted=True, return_inverse=True)
    num_subjects = unique_subjects.numel()
    if num_subjects < 2:
        metadata = {
            "subjects": unique_subjects,
            "partners": torch.arange(num_subjects, device=features.device),
            "lambda": torch.ones(num_subjects, device=features.device, dtype=features.dtype),
            "source_mean": None,
            "source_std": None,
            "target_mean": None,
            "target_std": None,
        }
        return (features, metadata) if return_metadata else features

    # Pool over examples and time. Keeping channels exposes diagonal sensor gain/style;
    # pooling them gives conventional filter-wise MixStyle.
    reduce_dims = (0, 3) if granularity == "filter_channel" else (0, 2, 3)
    means = []
    stds = []
    for subject_pos in range(num_subjects):
        subject_features = features[inverse == subject_pos]
        means.append(subject_features.mean(dim=reduce_dims, keepdim=True))
        stds.append(
            subject_features.var(dim=reduce_dims, unbiased=False, keepdim=True)
            .add(eps)
            .sqrt()
        )
    means = torch.cat(means, dim=0).detach()
    stds = torch.cat(stds, dim=0).detach()

    partners = _deranged_subject_partners(num_subjects, features.device)
    concentration = torch.full(
        (num_subjects,), max(float(alpha), 1e-3),
        device=features.device, dtype=torch.float32,
    )
    lam = torch.distributions.Beta(concentration, concentration).sample().to(features.dtype)
    view_shape = (num_subjects,) + (1,) * (features.ndim - 1)
    lam_view = lam.view(view_shape)
    target_means = lam_view * means + (1.0 - lam_view) * means[partners]
    target_stds = lam_view * stds + (1.0 - lam_view) * stds[partners]

    out = torch.empty_like(features)
    for subject_pos in range(num_subjects):
        mask = inverse == subject_pos
        out[mask] = (
            (features[mask] - means[subject_pos])
            * (target_stds[subject_pos] / stds[subject_pos].clamp_min(eps))
            + target_means[subject_pos]
        )

    if not return_metadata:
        return out
    return out, {
        "subjects": unique_subjects,
        "partners": partners,
        "lambda": lam,
        "source_mean": means,
        "source_std": stds,
        "target_mean": target_means,
        "target_std": target_stds,
    }


def subject_style_manifold(
    features,
    subject_ids,
    rank=2,
    latent_scale=1.0,
    max_radius=2.0,
    eps=1e-5,
    return_metadata=False,
):
    """Render new coherent subject styles from a low-rank source manifold.

    A style point concatenates the temporal activation mean and log standard
    deviation for every (filter, sensor) pair.  The batch must contain balanced
    source domains, as produced by ``GroupedImageBatchSampler``.  Coordinate-wise
    standardization prevents either the mean or log-scale block from dominating
    merely because of units, after which PCA is fit across subjects.  New bounded
    Gaussian codes are sampled in only ``rank`` source directions and decoded to
    target statistics.  One decoded point is used for every example belonging to
    the corresponding virtual subject in the batch.

    This is deliberately a small-sample linear model: with S source subjects the
    rank can never exceed S-1.  Statistics and the fitted basis are detached, while
    gradients continue through the normalized source activations.
    """
    if features.ndim != 4:
        raise ValueError(
            "subject_style_manifold expects (batch, filters, channels, time), "
            f"got {tuple(features.shape)}"
        )
    if subject_ids.ndim != 1 or subject_ids.shape[0] != features.shape[0]:
        raise ValueError("subject_ids must be a vector matching the feature batch")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if latent_scale < 0:
        raise ValueError("latent_scale must be non-negative")
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")

    unique_subjects, inverse = torch.unique(subject_ids, sorted=True, return_inverse=True)
    num_subjects = unique_subjects.numel()
    if num_subjects < 2:
        metadata = {
            "subjects": unique_subjects,
            "rank": 0,
            "basis": None,
            "sampled_codes": None,
            "target_mean": None,
            "target_std": None,
        }
        return (features, metadata) if return_metadata else features

    means = []
    stds = []
    for subject_pos in range(num_subjects):
        subject_features = features[inverse == subject_pos]
        means.append(subject_features.mean(dim=(0, 3), keepdim=True))
        stds.append(
            subject_features.var(dim=(0, 3), unbiased=False, keepdim=True)
            .add(eps)
            .sqrt()
        )
    means = torch.cat(means, dim=0).detach()
    stds = torch.cat(stds, dim=0).detach()
    stat_shape = means.shape[1:]

    styles = torch.cat(
        [means.flatten(1), stds.clamp_min(eps).log().flatten(1)], dim=1
    )
    style_center = styles.mean(dim=0, keepdim=True)
    # This scale is a change of units, not an estimate used in the forward gradient.
    coordinate_scale = styles.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    standardized = (styles - style_center) / coordinate_scale
    effective_rank = min(int(rank), num_subjects - 1)
    _, singular_values, basis = torch.linalg.svd(
        standardized, full_matrices=False
    )
    basis = basis[:effective_rank]
    latent_std = singular_values[:effective_rank] / max((num_subjects - 1) ** 0.5, 1.0)

    sampled_codes = torch.randn(
        num_subjects, effective_rank, device=features.device, dtype=features.dtype
    )
    sampled_codes = sampled_codes.clamp(-float(max_radius), float(max_radius))
    sampled_codes = sampled_codes * latent_std.unsqueeze(0) * float(latent_scale)
    target_standardized = sampled_codes @ basis
    target_styles = style_center + target_standardized * coordinate_scale
    split = means[0].numel()
    target_means = target_styles[:, :split].reshape((num_subjects,) + stat_shape)
    target_stds = target_styles[:, split:].reshape((num_subjects,) + stat_shape).exp()

    out = torch.empty_like(features)
    for subject_pos in range(num_subjects):
        mask = inverse == subject_pos
        out[mask] = (
            (features[mask] - means[subject_pos])
            * (target_stds[subject_pos] / stds[subject_pos].clamp_min(eps))
            + target_means[subject_pos]
        )

    if not return_metadata:
        return out
    return out, {
        "subjects": unique_subjects,
        "rank": effective_rank,
        "basis": basis,
        "latent_std": latent_std,
        "sampled_codes": sampled_codes,
        "target_mean": target_means,
        "target_std": target_stds,
    }


def coherent_subject_time_shift(
    features,
    subject_ids,
    shift_subject_ids,
    shifts,
):
    """Apply one zero-padded latency shift to every row of a virtual subject.

    ``shift_subject_ids`` and ``shifts`` define an epoch-level subject-style table.
    Keeping the table outside this function makes the intervention coherent across
    stimuli and minibatches, unlike ordinary per-trial time jitter.  Positive shifts
    delay a response.  Zero padding is used instead of circular wraparound so activity
    at the end of the analysis window cannot reappear before stimulus onset.
    """
    if features.ndim < 2:
        raise ValueError("features must have a batch and time dimension")
    if subject_ids.ndim != 1 or subject_ids.shape[0] != features.shape[0]:
        raise ValueError("subject_ids must be a vector matching the feature batch")
    if shift_subject_ids.ndim != 1 or shifts.ndim != 1:
        raise ValueError("shift_subject_ids and shifts must be vectors")
    if shift_subject_ids.numel() != shifts.numel():
        raise ValueError("shift_subject_ids and shifts must have equal length")

    row_shifts = torch.zeros_like(subject_ids, dtype=torch.long)
    matched = torch.zeros_like(subject_ids, dtype=torch.bool)
    for table_subject, table_shift in zip(shift_subject_ids, shifts):
        mask = subject_ids == table_subject
        row_shifts[mask] = table_shift.to(dtype=torch.long)
        matched |= mask
    if not torch.all(matched):
        missing = torch.unique(subject_ids[~matched]).detach().cpu().tolist()
        raise ValueError(f"latency table has no entry for subject IDs {missing}")

    time_points = features.shape[-1]
    source_time = (
        torch.arange(time_points, device=features.device).unsqueeze(0)
        - row_shifts.unsqueeze(1)
    )
    valid = (source_time >= 0) & (source_time < time_points)
    gather_index = source_time.clamp(0, time_points - 1)
    gather_shape = (features.shape[0],) + (1,) * (features.ndim - 2) + (time_points,)
    gather_index = gather_index.view(gather_shape).expand_as(features)
    shifted = torch.gather(features, dim=-1, index=gather_index)
    valid = valid.view(gather_shape).expand_as(features)
    return shifted * valid.to(features.dtype)


def sample_coherent_subject_time_shifts(
    subject_ids,
    max_shift,
    std,
    *,
    device,
):
    """Sample a bounded, integer latency style for each virtual subject."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if std < 0:
        raise ValueError("std must be non-negative")
    subjects = torch.as_tensor(subject_ids, dtype=torch.long, device=device)
    if max_shift == 0 or std == 0:
        return subjects, torch.zeros_like(subjects)
    sampled = torch.randn(subjects.numel(), device=device) * float(std)
    sampled = sampled.round().clamp(-int(max_shift), int(max_shift)).to(torch.long)
    return subjects, sampled
