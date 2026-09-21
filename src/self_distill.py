"""Build contextual trajectories for a frozen earlier student."""

import copy

import torch

from .batching import pack_trajectories
from data_utils.records import get_raw_prompt


def make_reference_model(student_module):
    reference = copy.deepcopy(student_module)
    reference.requires_grad_(False)
    reference.eval()
    return reference


def refresh_reference_model(reference, student_module):
    """Copy weights into the reference, which was frozen when created."""
    reference.load_state_dict(student_module.state_dict())


def complete_visible_steps(sample, tokenizer, separator):
    """Return complete text steps visible in the canonical response window."""
    # Labels include the final response token even when the causal input ends
    # one token earlier; use that exact supervised region.
    visible = sample["label"][len(sample["prompt_ids"]) - 1:]
    if visible and visible[-1] == tokenizer.eos_token_id:
        visible = visible[:-1]
    if not visible:
        return []
    response = sample["response"]
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"][:len(visible)] != visible:
        raise ValueError("Canonical response text does not align with visible training tokens")
    visible_end = encoded["offset_mapping"][len(visible) - 1][1]
    steps = []
    start = 0
    for part in response.split(separator):
        end = start + len(part)
        if end > visible_end:
            break
        if part.strip():
            steps.append(part)
        start = end + len(separator)
    return steps


def sample_context(steps, separator, max_drop_ratio, rng):
    if len(steps) < 2:
        return ""
    drop_ratio = rng.uniform(0.0, max_drop_ratio)
    kept = [step for step in steps if rng.random() >= drop_ratio]
    if not kept:
        kept = [steps[rng.randrange(len(steps))]]
    elif len(kept) == len(steps):
        kept.pop(rng.randrange(len(kept)))
    return separator.join(kept)


def render_context_prompt(record, tokenizer, context, template):
    if not context and record.get("prompt"):
        return record["prompt"]
    addition = template.format(context=context) if context else ""
    if record.get("prompt"):
        prompt = record["prompt"]
        special = any(marker and marker in prompt for marker in
                      getattr(tokenizer, "all_special_tokens", []))
        if not special:
            return prompt + addition
        marker = "__SELF_DISTILL_CONTEXT_MARKER__"
        exemplar = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        suffix = exemplar.split(marker, 1)[1]
        if not suffix or not prompt.endswith(suffix):
            raise ValueError("Cannot insert self-distillation context into rendered prompt")
        return prompt[:-len(suffix)] + addition + suffix
    raw = get_raw_prompt(record, tokenizer)
    messages = []
    if record.get("system_prompt"):
        messages.append({"role": "system", "content": record["system_prompt"]})
    messages.append({"role": "user", "content": raw + addition})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return "\n\n".join(message["content"] for message in messages)


def _fit_reference_prompt(record, tokenizer, context, template, separator, budget,
                          context_budget):
    def encode(candidate):
        context_tokens = len(tokenizer.encode(candidate, add_special_tokens=False)) if candidate else 0
        if context_tokens > context_budget:
            return None, context_tokens
        text = render_context_prompt(record, tokenizer, candidate, template)
        return tokenizer.encode(text, add_special_tokens=False), context_tokens

    ref_ids, context_tokens = encode(context)
    if ref_ids is not None and len(ref_ids) <= budget:
        return ref_ids, context, context_tokens
    if not context:
        return ref_ids[-budget:], "", 0

    # Keep whole steps while respecting both the context cap and the prompt cap.
    # Preserve the question before dropping its prefix to make room for context.
    base_ids, _ = encode("")
    if len(base_ids) >= budget:
        return base_ids[-budget:], "", 0

    steps = context.split(separator)
    best_ids, best_count, best_tokens = base_ids, 0, 0
    low, high = 1, len(steps) - 1
    while low <= high:
        count = (low + high) // 2
        candidate = separator.join(steps[:count])
        candidate_ids, candidate_tokens = encode(candidate)
        if candidate_ids is not None and len(candidate_ids) <= budget:
            best_ids, best_count, best_tokens = candidate_ids, count, candidate_tokens
            low = count + 1
        else:
            high = count - 1
    return best_ids, separator.join(steps[:best_count]), best_tokens


def _rebase_step_spans(spans, original_labels, packed_batch, packed_labels):
    """Move canonical response spans to the packed prompt and clip truncated steps."""
    original_start = (original_labels != -100).long().argmax(-1) + 1
    packed_start = (packed_labels != -100).long().argmax(-1) + 1
    shifted = spans - original_start[:, None, None] + packed_start[:, None, None]
    response_end = packed_batch["attention_mask"].sum(-1)[:, None]
    start = shifted[..., 0]
    end = shifted[..., 1].minimum(response_end)
    valid = (spans[..., 0] >= original_start[:, None]) & (start < end)
    return torch.stack((start, end), dim=-1).masked_fill(~valid[..., None], -1)


def prepare_self_distill_batches(args, tokenizer, student_batch, metadata, rng):
    """Use the canonical response with plain and contextual student prompts."""
    batch_size = student_batch["input_ids"].shape[0]
    if any(len(metadata[key]) != batch_size for key in
           ("self_distill_steps", "self_distill_records")):
        raise ValueError("Self-distillation metadata must match the batch size")
    labels = metadata["label"]
    # Dataset labels have one contiguous supervised response per row. Copy the
    # start and length for the whole batch in a single device transfer.
    supervised = labels != -100
    response_ranges = torch.stack(
        (supervised.long().argmax(-1), supervised.sum(-1)), dim=-1
    ).tolist()
    prompts, ref_prompts, responses, counts = [], [], [], []
    for index, (ids, row_labels, steps, record, (start, length)) in enumerate(zip(
            student_batch["input_ids"], labels, metadata["self_distill_steps"],
            metadata["self_distill_records"], response_ranges)):
        prompt = ids[:start + 1][-args.max_prompt_length:]
        response = row_labels[start:start + length]
        response = response[:min(args.max_length - len(prompt), args.t_max_length - 1)]
        row_rng = rng[index] if isinstance(rng, list) else rng
        context = sample_context(steps, args.step_separator,
                                 args.self_distill_context_drop_ratio, row_rng)
        budget = min(args.t_max_prompt_length, args.t_max_length - len(response))
        ref_ids, context, context_tokens = _fit_reference_prompt(
            record, tokenizer, context, args.self_distill_context_template,
            args.step_separator, budget, args.self_distill_context_max_tokens)
        prompts.append(prompt)
        ref_prompts.append(ref_ids)
        responses.append(response)
        counts.append(context_tokens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = student_batch["input_ids"].device
    student, student_meta = pack_trajectories(
        prompts, responses, pad_id, args.model_type, args.max_length, device)
    reference, ref_meta = pack_trajectories(
        ref_prompts, responses, pad_id, args.model_type, args.t_max_length, device)
    if "step_spans" in metadata:
        student_meta["step_spans"] = _rebase_step_spans(
            metadata["step_spans"], labels, student, student_meta["label"])
        ref_meta["step_spans"] = _rebase_step_spans(
            metadata["step_spans"], labels, reference, ref_meta["label"])
    student_meta["self_distill_context_tokens"] = student_meta["label"].new_tensor(counts)
    return student, student_meta, reference, ref_meta
