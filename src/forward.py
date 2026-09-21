from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .trajectory import find_step_spans, pool_steps


@dataclass
class ModelFeatures:
    step_representations: torch.Tensor  # [B, T, D]
    step_mask: torch.Tensor  # [B, T]
    step_spans: torch.Tensor  # [B, T, 2]
    topk_logits: torch.Tensor  # [B, L, K]
    topk_indices: torch.Tensor  # [B, L, K], -1 outside supervised tokens
    token_mask: torch.Tensor  # [B, L]
    ce_loss: torch.Tensor


def _forward_last_hidden(model, batch):
    causal_lm = model
    while hasattr(causal_lm, "module"):
        causal_lm = causal_lm.module
    if callable(getattr(causal_lm, "get_base_model", None)):
        causal_lm = causal_lm.get_base_model()
    backbone = getattr(causal_lm, getattr(causal_lm, "base_model_prefix", ""), None)
    if not isinstance(backbone, torch.nn.Module) or backbone is causal_lm:
        raise ValueError("Last-hidden extraction requires a causal LM with a backbone "
                         "identified by base_model_prefix")

    captured = []

    def capture(_module, _inputs, output):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, tuple):
            hidden = output[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("Expected backbone last_hidden_state with shape [B, L, D]")
        captured.append(hidden)

    handle = backbone.register_forward_hook(capture)
    try:
        outputs = model(
            **batch, output_hidden_states=False, return_dict=True, use_cache=False
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one backbone forward for hidden extraction")
    return outputs, captured[0]


def forward_features(
    model,
    batch,
    metadata,
    pooling="mean",
    top_k=32,
    support=None,
    compute_ce=False,
    sample_mask=None,
):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    outputs, last_hidden = _forward_last_hidden(model, batch)
    step_spans = metadata.get("step_spans")
    if step_spans is None:
        step_spans = find_step_spans(
            batch["input_ids"],
            batch["attention_mask"],
            metadata["label"],
            metadata["step_marker_ids"],
            metadata["max_step_markers"],
        )
    representations, step_mask = pool_steps(
        last_hidden, step_spans, pooling
    )
    logits = outputs.logits
    labels = metadata["label"]
    token_mask = (labels != -100) & batch["attention_mask"].bool()
    if sample_mask is not None:
        step_mask = step_mask & sample_mask[:, None]
        token_mask = token_mask & sample_mask[:, None]
    if support is None:
        values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
    else:
        if support.shape[:2] != logits.shape[:2]:
            raise ValueError(
                "Teacher/student prediction positions must match for top-k"
            )
        if ((support >= logits.size(-1)) | (support < -1)).any():
            raise ValueError(
                "Teacher top-k token IDs are outside the student vocabulary"
            )
        indices = support
        values = logits.gather(-1, indices.clamp_min(0))
    values = values.float().masked_fill(~token_mask[..., None], 0)
    indices = indices.masked_fill(~token_mask[..., None], -1)
    # A differentiable zero also supports batches with no eligible trajectories.
    ce_loss = representations.sum() * 0
    if compute_ce:
        targets = labels.masked_fill(~token_mask, -100)
        ce_sum = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        ce_loss = ce_sum / token_mask.sum().clamp_min(1)
    return ModelFeatures(
        representations,
        step_mask,
        step_spans,
        values,
        indices,
        token_mask,
        ce_loss,
    )
