import math

import torch


def _inferred_step_capacity(input_ids, attention_mask, labels, marker_ids):
    marker_length = marker_ids.numel()
    if input_ids.shape[1] < marker_length:
        return 1
    valid_lengths = attention_mask.sum(-1)
    supervised = labels != -100
    response_starts = torch.where(
        supervised.any(-1), supervised.long().argmax(-1) + 1, valid_lengths)
    marker_starts = torch.arange(
        input_ids.shape[1] - marker_length + 1, device=input_ids.device)
    matches = (input_ids.unfold(1, marker_length, 1) == marker_ids.to(input_ids.device)).all(-1)
    matches &= marker_starts[None, :] >= response_starts[:, None]
    matches &= marker_starts[None, :] + marker_length <= valid_lengths[:, None]
    return int(matches.sum(-1).max().item()) + 1


def find_step_spans(input_ids, attention_mask, labels, marker_ids, max_steps=None):
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected input_ids and attention_mask with shape [B, L]")
    if labels.shape != input_ids.shape:
        raise ValueError("Labels must have the same shape as input_ids")
    if marker_ids.ndim != 1 or marker_ids.numel() == 0:
        raise ValueError("marker_ids must be a nonempty 1D tensor")
    max_steps = (_inferred_step_capacity(input_ids, attention_mask, labels, marker_ids)
                 if max_steps is None else int(max_steps))
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    spans = input_ids.new_full((input_ids.shape[0], max_steps, 2), -1)
    if max_steps == 0 or input_ids.shape[1] == 0:
        return spans
    marker_ids = marker_ids.to(input_ids.device)
    marker_length = marker_ids.numel()
    valid_lengths = attention_mask.sum(dim=-1)
    supervised = labels != -100
    response_starts = supervised.long().argmax(dim=-1) + 1
    response_starts = torch.where(
        supervised.any(dim=-1), response_starts, valid_lengths
    )
    if input_ids.shape[1] < marker_length:
        valid = response_starts < valid_lengths
        spans[:, 0, 0] = torch.where(valid, response_starts, -1)
        spans[:, 0, 1] = torch.where(valid, valid_lengths, -1)
        return spans
    marker_starts = torch.arange(
        input_ids.shape[1] - marker_length + 1, device=input_ids.device
    )
    matches = (input_ids.unfold(1, marker_length, 1) == marker_ids).all(dim=-1)
    matches &= marker_starts[None, :] >= response_starts[:, None]
    matches &= marker_starts[None, :] + marker_length <= valid_lengths[:, None]

    # Select markers left-to-right. Keeping the cursor on device avoids a CPU/GPU
    # synchronization for every trajectory and also rejects overlapping matches.
    cursor = response_starts
    step_counts = torch.zeros_like(response_starts)
    sentinel = marker_starts.numel()
    for _ in range(max_steps):
        eligible = matches & (marker_starts[None, :] >= cursor[:, None])
        next_marker = torch.where(eligible, marker_starts, sentinel).amin(dim=-1)
        valid = next_marker < sentinel
        if not valid.any():
            break
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        valid_indices = batch_indices[valid]
        valid_steps = step_counts[valid]
        marker_end = next_marker + marker_length
        spans[valid_indices, valid_steps, 0] = cursor[valid]
        spans[valid_indices, valid_steps, 1] = marker_end[valid]
        step_counts = step_counts + valid.long()
        cursor = torch.where(valid, marker_end, cursor)

    tail_valid = (cursor < valid_lengths) & (step_counts < max_steps)
    if tail_valid.any():
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        tail_indices = batch_indices[tail_valid]
        tail_steps = step_counts[tail_valid]
        spans[tail_indices, tail_steps, 0] = cursor[tail_valid]
        spans[tail_indices, tail_steps, 1] = valid_lengths[tail_valid]
    return spans


def pool_steps(hidden_states, step_spans, pooling="mean"):
    if pooling not in ("mean", "last_token"):
        raise ValueError(f"Unknown step pooling: {pooling}")
    if hidden_states.ndim != 3 or step_spans.ndim != 3 or step_spans.shape[-1] != 2:
        raise ValueError("Expected hidden states [B, L, D] and step spans [B, T, 2]")
    if hidden_states.shape[0] != step_spans.shape[0]:
        raise ValueError("Hidden states and spans must have the same batch size")
    start, end = step_spans.unbind(-1)
    valid = (start >= 0) & (end > start) & (end <= hidden_states.shape[1])
    start = start.masked_fill(~valid, 0)
    end = end.masked_fill(~valid, 0)
    hidden = hidden_states.float()
    batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    if pooling == "last_token":
        pooled = hidden[batch, (end - 1).clamp_min(0)]
    else:
        # Keep a long context-bearing reference prompt out of the
        # prefix sum. Otherwise its values cancel only approximately in float32.
        if step_spans.shape[1]:
            first_start = start.masked_fill(~valid, hidden.shape[1]).amin(dim=-1)
            prompt_positions = torch.arange(hidden.shape[1], device=hidden.device)
            hidden = hidden.masked_fill(prompt_positions[None, :, None] < first_start[:, None, None], 0)
        prefix = torch.cat(
            (
                hidden.new_zeros(hidden.shape[0], 1, hidden.shape[2]),
                hidden.cumsum(dim=1),
            ),
            dim=1,
        )
        pooled = (prefix[batch, end] - prefix[batch, start]) / (end - start).clamp_min(
            1
        )[..., None]
    return pooled.masked_fill(~valid[..., None], 0), valid


def _relative_magnitudes(magnitudes, mask, normalization, eps):
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (magnitudes * mask).sum(dim=-1, keepdim=True) / count
    if normalization == "mean":
        result = magnitudes / (mean + eps)
    elif normalization == "zscore":
        centered = (magnitudes - mean).masked_fill(~mask, 0)
        # Population std; vector_norm has a finite gradient at an all-zero input.
        std = torch.linalg.vector_norm(centered, dim=-1, keepdim=True) / count.sqrt()
        result = centered / (std + eps)
    else:
        raise ValueError(f"Unknown magnitude normalization: {normalization}")
    return result.masked_fill(~mask, 0)


def _trajectory_average(values, mask):
    counts = mask.sum(dim=-1)
    per_sample = values.masked_fill(~mask, 0).sum(dim=-1) / counts.clamp_min(1)
    eligible = counts > 0
    return per_sample.sum() / eligible.sum().clamp_min(1)


def reasoning_velocity_loss(
    student_hidden,
    teacher_hidden,
    student_spans,
    teacher_spans=None,
    pooling="mean",
    normalization="zscore",
    eps=1e-6,
):
    """Compare velocities between consecutive response steps."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if teacher_spans is None:
        teacher_spans = student_spans
    if student_spans.shape != teacher_spans.shape:
        raise ValueError("Teacher/student spans must align on batch and step indices")
    student, s_mask = pool_steps(student_hidden, student_spans, pooling)
    teacher, t_mask = pool_steps(teacher_hidden.detach(), teacher_spans, pooling)
    return velocity_loss_from_steps(
        student, teacher, s_mask, t_mask, normalization, eps
    )


def cka_loss_from_steps(student, teacher, s_mask, t_mask, eps=1e-6):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (
        student.shape[:2] != teacher.shape[:2]
        or s_mask.shape != student.shape[:2]
        or t_mask.shape != teacher.shape[:2]
    ):
        raise ValueError(
            "Representations and masks must align on batch and step indices"
        )

    student = student.float()
    teacher = teacher.detach().float()
    step_mask = s_mask & t_mask
    pair_mask = step_mask[:, :, None] & step_mask[:, None, :]
    pair_weight = pair_mask.to(student.dtype)
    counts = step_mask.sum(-1)
    safe_counts = counts.clamp_min(1).to(student.dtype)

    student_kernel = student @ student.transpose(-1, -2)
    teacher_kernel = teacher @ teacher.transpose(-1, -2)

    def center(kernel):
        masked = kernel * pair_weight
        row_mean = masked.sum(-1) / safe_counts[:, None]
        grand_mean = masked.sum((-1, -2)) / safe_counts.square()
        centered = (
            kernel
            - row_mean[:, :, None]
            - row_mean[:, None, :]
            + grand_mean[:, None, None]
        )
        return centered * pair_weight

    student_kernel = center(student_kernel)
    teacher_kernel = center(teacher_kernel)
    hsic = (student_kernel * teacher_kernel).sum((-1, -2))
    student_norm = student_kernel.square().sum((-1, -2))
    teacher_norm = teacher_kernel.square().sum((-1, -2))
    denominator = (student_norm * teacher_norm).clamp_min(eps ** 2).sqrt()
    # Cauchy-Schwarz bounds CKA by one; clamping only guards round-off.
    losses = 1.0 - (hsic / denominator).clamp(min=0.0, max=1.0)
    eligible = counts >= 2
    return (losses * eligible).sum() / eligible.sum().clamp_min(1)


def reasoning_cka_loss(
    student_hidden,
    teacher_hidden,
    student_spans,
    teacher_spans=None,
    pooling="mean",
    eps=1e-6,
):
    """Pool each reasoning step and align student/teacher trajectories with CKA."""
    if teacher_spans is None:
        teacher_spans = student_spans
    if student_spans.shape != teacher_spans.shape:
        raise ValueError("Teacher/student spans must align on batch and step indices")
    student, s_mask = pool_steps(student_hidden, student_spans, pooling)
    teacher, t_mask = pool_steps(teacher_hidden.detach(), teacher_spans, pooling)
    return cka_loss_from_steps(student, teacher, s_mask, t_mask, eps)


def _pack_supervised_states(hidden, labels, width):
    """Align hidden states by response token index despite different prompt lengths."""
    supervised = labels != -100
    positions = (supervised.long().cumsum(-1) - 1).clamp_min(0)
    states = hidden.float().new_zeros(hidden.shape[0], width, hidden.shape[-1])
    states.scatter_add_(
        1, positions[..., None].expand_as(hidden),
        hidden.float() * supervised[..., None])
    return states


def _response_token_gram_loss(student_hidden, teacher_hidden,
                              student_labels, teacher_labels, rows, eps):
    student_counts = (student_labels != -100).sum(-1)
    teacher_counts = (teacher_labels != -100).sum(-1)
    if not torch.equal(student_counts, teacher_counts):
        raise ValueError("Teacher/student response lengths must match for geometry")
    width = max(1, int(student_counts.max().item()))
    student = _pack_supervised_states(student_hidden, student_labels, width)
    teacher = _pack_supervised_states(teacher_hidden.detach(), teacher_labels, width)
    student = student / torch.linalg.vector_norm(student, dim=-1, keepdim=True).clamp_min(eps)
    teacher = teacher / torch.linalg.vector_norm(teacher, dim=-1, keepdim=True).clamp_min(eps)
    student_gram = student @ student.transpose(-1, -2)
    teacher_gram = teacher @ teacher.transpose(-1, -2)
    token_mask = torch.arange(width, device=student.device)[None, :] < student_counts[:, None]
    pair_mask = (token_mask[:, :, None] & token_mask[:, None, :]).triu(diagonal=1)
    pair_mask &= rows[:, None, None]
    pair_counts = pair_mask.sum((-1, -2))
    per_sample = ((student_gram - teacher_gram).square() * pair_mask).sum((-1, -2)) / pair_counts.clamp_min(1)
    return per_sample.sum() / rows.sum().clamp_min(1)


def reasoning_geometry_loss(student_hidden, teacher_hidden, student_spans,
                            teacher_spans, student_labels, teacher_labels,
                            pooling="mean", normalization="zscore", eps=1e-6):
    """Align step velocities, or pooled-step scale and token relations when steps are scarce.

    RMS and within-response Gram comparisons work when hidden sizes differ.
    Two velocities require at least three detected steps; shorter trajectories
    use the fallback so a single pooled step still receives geometry gradients.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if student_spans.shape != teacher_spans.shape:
        raise ValueError("Teacher/student spans must align on batch and step indices")
    if student_labels.shape != student_hidden.shape[:2] or teacher_labels.shape != teacher_hidden.shape[:2]:
        raise ValueError("Geometry labels must match hidden-state sequence dimensions")
    student, s_mask = pool_steps(student_hidden, student_spans, pooling)
    teacher, t_mask = pool_steps(teacher_hidden.detach(), teacher_spans, pooling)
    step_mask = s_mask & t_mask
    velocities = step_mask[:, 1:] & step_mask[:, :-1]
    multi_rows = velocities.sum(-1) >= 2
    short_rows = step_mask.any(-1) & ~multi_rows

    velocity_mag, velocity_gram = velocity_loss_from_steps(
        student, teacher, s_mask & multi_rows[:, None],
        t_mask & multi_rows[:, None], normalization, eps)
    if not short_rows.any():
        return velocity_mag, velocity_gram

    student_rms = student.square().mean(-1).add(eps ** 2).sqrt()
    teacher_rms = teacher.square().mean(-1).add(eps ** 2).sqrt()
    step_loss = (student_rms.log() - teacher_rms.log()).square()
    per_sample_mag = step_loss.masked_fill(~step_mask, 0).sum(-1) / step_mask.sum(-1).clamp_min(1)
    short_mag = per_sample_mag.masked_fill(~short_rows, 0).sum() / short_rows.sum().clamp_min(1)
    short_gram = _response_token_gram_loss(
        student_hidden, teacher_hidden, student_labels, teacher_labels, short_rows, eps)
    multi_count, short_count = multi_rows.sum(), short_rows.sum()
    total = (multi_count + short_count).clamp_min(1)
    return ((velocity_mag * multi_count + short_mag * short_count) / total,
            (velocity_gram * multi_count + short_gram * short_count) / total)


def velocity_loss_from_steps(
    student, teacher, s_mask, t_mask, normalization="zscore", eps=1e-6
):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (
        student.shape[:2] != teacher.shape[:2]
        or s_mask.shape != student.shape[:2]
        or t_mask.shape != teacher.shape[:2]
    ):
        raise ValueError(
            "Representations and masks must align on batch and step indices"
        )
    student = student.float()
    teacher = teacher.detach().float()
    step_mask = s_mask & t_mask
    velocity_mask = step_mask[:, 1:] & step_mask[:, :-1]
    student_delta = student[:, 1:] - student[:, :-1]
    teacher_delta = teacher[:, 1:] - teacher[:, :-1]
    student_mag = torch.linalg.vector_norm(student_delta, dim=-1)
    teacher_mag = torch.linalg.vector_norm(teacher_delta, dim=-1)
    student_z = _relative_magnitudes(student_mag, velocity_mask, normalization, eps)
    teacher_z = _relative_magnitudes(teacher_mag, velocity_mask, normalization, eps)
    mag_loss = _trajectory_average((student_z - teacher_z).square(), velocity_mask)

    student_dir = student_delta / (student_mag[..., None] + eps)
    teacher_dir = teacher_delta / (teacher_mag[..., None] + eps)
    student_gram = student_dir @ student_dir.transpose(-1, -2)
    teacher_gram = teacher_dir @ teacher_dir.transpose(-1, -2)
    pair_mask = (velocity_mask[:, :, None] & velocity_mask[:, None, :]).triu(diagonal=1)
    gram_loss = _trajectory_average(
        (student_gram - teacher_gram).square().flatten(1), pair_mask.flatten(1)
    )
    return mag_loss, gram_loss
