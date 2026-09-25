"""Menger-curvature distillation over marker-delimited reasoning steps."""

import math
import os
from functools import lru_cache

import torch


def _response_tokens(labels, eos_token_id):
    positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
    tokens = labels[positions]
    if tokens.numel() and eos_token_id is not None and tokens[-1].item() == eos_token_id:
        positions = positions[:-1]
        tokens = tokens[:-1]
    return positions, tokens


def _hidden_positions(input_ids, attention_mask, label_positions, response_tokens):
    """Map target-token indices to states of those tokens in the causal input.

    Canonical batches store labels one position to the left, while rollout
    batches store them at the response-token positions. Select the alignment
    that actually matches the response token IDs.
    """
    count = response_tokens.numel()
    if count == 0:
        return {}
    valid_length = int(attention_mask.sum().item())
    candidates = (label_positions, label_positions + 1)
    best = None
    best_matches = -1
    for candidate in candidates:
        in_bounds = candidate < valid_length
        matched = in_bounds.clone()
        if in_bounds.any():
            matched[in_bounds] &= input_ids[candidate[in_bounds]].eq(
                response_tokens[in_bounds]
            )
        matches = int(matched.sum().item())
        if matches > best_matches:
            best = (candidate, matched)
            best_matches = matches

    # A causally shifted sequence can omit only its final target token when
    # training data was truncated before EOS. Anything worse is not a safe
    # response/hidden-state alignment.
    if best_matches < count - 1:
        raise ValueError("Cannot align response tokens with model hidden states")
    candidate, matched = best
    return {
        index: int(position.item())
        for index, (position, is_match) in enumerate(zip(candidate, matched))
        if bool(is_match)
    }


def _step_token_ranges(tokenizer, token_ids, separator):
    """Convert literal text-step boundaries to ranges of response token IDs."""
    if not separator:
        raise ValueError("step separator must be a nonempty string")
    ids = token_ids.detach().cpu().tolist()
    if not ids:
        return []
    text = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Menger loss requires a fast tokenizer with offset mappings"
        )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    round_trips = list(encoded["input_ids"]) == ids
    offsets = encoded["offset_mapping"] if round_trips else None
    if not round_trips:
        # Re-encoding a response without its prompt can change BPE merges.
        # Decode prefixes of the original IDs to keep boundaries aligned with
        # the hidden states used by this loss.
        @lru_cache(maxsize=None)
        def decoded_prefix(count):
            return tokenizer.decode(
                ids[:count], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        def first_prefix_at(position, *, past):
            low, high = 1, len(ids)
            while low < high:
                middle = (low + high) // 2
                prefix = decoded_prefix(middle)
                if past:
                    # A partial UTF-8 character can decode as U+FFFD. Count
                    # its first token when locating the start of a step.
                    reached = len(prefix) > position
                else:
                    # Wait until the text through the end of a step is stable.
                    stable_length = len(os.path.commonprefix((text, prefix)))
                    reached = stable_length >= position
                if reached:
                    high = middle
                else:
                    low = middle + 1
            return low

    ranges = []
    cursor = 0
    parts = text.split(separator)
    for part in parts:
        part_end = cursor + len(part)
        stripped = part.strip()
        if stripped:
            content_start = cursor + len(part) - len(part.lstrip())
            content_end = part_end - (len(part) - len(part.rstrip()))
            if round_trips:
                covered = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if end > content_start and start < content_end
                ]
                if covered:
                    ranges.append((covered[0], covered[-1] + 1))
            else:
                start = first_prefix_at(content_start, past=True) - 1
                end = first_prefix_at(content_end, past=False)
                if start < end:
                    ranges.append((start, end))
        cursor = part_end + len(separator)
    return ranges


def _pool_response_steps(hidden, token_to_hidden, step_ranges):
    pooled = []
    valid = []
    hidden = hidden.float()
    for start, end in step_ranges:
        positions = [
            token_to_hidden[index]
            for index in range(start, end)
            if index in token_to_hidden
        ]
        if positions:
            pooled.append(hidden[positions].mean(dim=0))
            valid.append(True)
        else:
            pooled.append(hidden.new_zeros(hidden.shape[-1]))
            valid.append(False)
    if not pooled:
        return hidden.new_zeros((0, hidden.shape[-1])), torch.zeros(
            0, dtype=torch.bool, device=hidden.device
        )
    return torch.stack(pooled), torch.tensor(valid, device=hidden.device)


def _curvatures(steps, step_valid, eps):
    if steps.shape[0] < 3:
        return steps.new_zeros(0), torch.zeros(
            0, dtype=torch.bool, device=steps.device
        )
    u = steps[1:-1] - steps[:-2]
    v = steps[2:] - steps[1:-1]
    chord = steps[2:] - steps[:-2]
    u_norm = torch.linalg.vector_norm(u, dim=-1)
    v_norm = torch.linalg.vector_norm(v, dim=-1)
    chord_norm = torch.linalg.vector_norm(chord, dim=-1)
    geometric_valid = (u_norm > eps) & (v_norm > eps) & (chord_norm > eps)
    triple_valid = (
        step_valid[:-2]
        & step_valid[1:-1]
        & step_valid[2:]
        & geometric_valid
    )
    cosine = (u * v).sum(dim=-1) / (u_norm * v_norm).clamp_min(eps)
    cosine = cosine.clamp(min=-1.0, max=1.0)
    sine_squared = (1.0 - cosine.square()).clamp_min(0.0)
    # sqrt has an infinite derivative at zero. Preserve exact zero curvature
    # for collinear steps while keeping backward finite in that case.
    sine = sine_squared.clamp_min(eps * eps).sqrt()
    sine = sine.masked_fill(sine_squared <= eps * eps, 0.0)
    curvature = 2.0 * sine / chord_norm.clamp_min(eps)
    return curvature, triple_valid


def menger_loss_per_response(
    student_hidden,
    teacher_hidden,
    student_batch,
    student_labels,
    teacher_batch,
    teacher_labels,
    tokenizer,
    separator="\n\n",
    eps=1e-6,
):
    """Return one plain curvature-MSE value for every response in the batch."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Menger epsilon must be finite and positive")
    if student_hidden.ndim != 3 or teacher_hidden.ndim != 3:
        raise ValueError("Expected hidden states with shape [B, L, D]")
    batch_size = student_hidden.shape[0]
    if teacher_hidden.shape[0] != batch_size:
        raise ValueError("Teacher/student hidden-state batch sizes differ")
    if student_labels.shape[0] != batch_size or teacher_labels.shape[0] != batch_size:
        raise ValueError("Labels and hidden states must have the same batch size")

    losses = []
    eos_token_id = tokenizer.eos_token_id
    for index in range(batch_size):
        s_label_positions, s_tokens = _response_tokens(
            student_labels[index], eos_token_id
        )
        t_label_positions, t_tokens = _response_tokens(
            teacher_labels[index], eos_token_id
        )
        if not torch.equal(s_tokens, t_tokens):
            raise ValueError("Teacher/student Menger targets are not identical")
        zero = student_hidden[index].sum() * 0.0
        step_ranges = _step_token_ranges(tokenizer, s_tokens, separator)
        if len(step_ranges) < 3:
            losses.append(zero)
            continue

        s_map = _hidden_positions(
            student_batch["input_ids"][index],
            student_batch["attention_mask"][index],
            s_label_positions,
            s_tokens,
        )
        t_map = _hidden_positions(
            teacher_batch["input_ids"][index],
            teacher_batch["attention_mask"][index],
            t_label_positions,
            t_tokens,
        )
        # Pool exactly the same response-token indices on both sides.
        common = set(s_map).intersection(t_map)
        s_map = {key: value for key, value in s_map.items() if key in common}
        t_map = {key: value for key, value in t_map.items() if key in common}
        student_steps, student_valid = _pool_response_steps(
            student_hidden[index], s_map, step_ranges
        )
        teacher_steps, teacher_valid = _pool_response_steps(
            teacher_hidden[index].detach(), t_map, step_ranges
        )
        student_curvature, student_triples = _curvatures(
            student_steps, student_valid, eps
        )
        teacher_curvature, teacher_triples = _curvatures(
            teacher_steps, teacher_valid, eps
        )
        valid = student_triples & teacher_triples
        if valid.any():
            losses.append(
                (student_curvature[valid] - teacher_curvature[valid])
                .square()
                .mean()
            )
        else:
            losses.append(zero)
    return torch.stack(losses)
