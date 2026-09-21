"""On-policy self-distillation with a fixed, solution-conditioned reference."""

import torch
import torch.nn.functional as F

from .batching import pack_trajectories
from .self_distill import render_context_prompt


def prepare_opsd_reference_batch(args, tokenizer, student_batch, labels, records, solutions):
    """Score a student rollout under a teacher that sees the full gold solution."""
    batch_size = labels.shape[0]
    if len(records) != batch_size or len(solutions) != batch_size:
        raise ValueError("OPSD reference metadata must match the generated batch")
    supervised = labels != -100
    ranges = torch.stack((supervised.long().argmax(-1), supervised.sum(-1)), -1).tolist()
    prompts, responses = [], []
    for record, solution, row_labels, (start, length) in zip(records, solutions, labels, ranges):
        if not length:
            raise ValueError("OPSD requires a nonempty student rollout")
        response = row_labels[start:start + length]
        prompt_text = render_context_prompt(
            record, tokenizer, solution, args.opsd_context_template)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > args.t_max_prompt_length or len(prompt_ids) + len(response) > args.t_max_length:
            raise ValueError(
                "OPSD teacher prompt or rollout exceeds reference limits; increase "
                "T_MAX_PROMPT_LENGTH/T_MAX_LENGTH so the full gold solution is retained")
        prompts.append(prompt_ids)
        responses.append(response)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return pack_trajectories(
        prompts, responses, pad_id, args.model_type, None, student_batch["input_ids"].device)


def fixed_base_teacher_forward(model, teacher_batch):
    """Mirror OPSD fixed_teacher: same model, LoRA disabled for teacher only."""
    adapter_context = model.module.disable_adapter()
    with torch.no_grad(), adapter_context:
        return model(**teacher_batch, use_cache=False,
                     output_hidden_states=False, return_dict=True)


def opsd_forward_kl(student_logits, teacher_logits, pointwise_clip, response_lengths):
    """Full-vocabulary FKL with OPSD's positive pointwise-contribution cap."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("OPSD needs aligned student/teacher response logits [N,V]")
    if student_logits.shape[0] == 0:
        return student_logits.sum()
    if (response_lengths.ndim != 1 or (response_lengths <= 0).any()
            or response_lengths.sum() != student_logits.shape[0]):
        raise ValueError("OPSD response lengths must cover all aligned logits")
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    contributions = teacher_logp.exp() * (teacher_logp - student_logp)
    if pointwise_clip > 0:
        contributions = contributions.clamp(max=pointwise_clip)
    per_token = contributions.sum(-1)
    row_ids = torch.repeat_interleave(
        torch.arange(response_lengths.numel(), device=per_token.device),
        response_lengths, output_size=per_token.numel())
    row_totals = per_token.new_zeros(response_lengths.numel()).scatter_add_(0, row_ids, per_token)
    return (row_totals / response_lengths).mean()
