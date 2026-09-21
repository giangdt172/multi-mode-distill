"""Pack prompt + response IDs with causal labels, preserving EOS == PAD."""

import torch


def pack_trajectories(prompts, responses, pad_id, model_type, max_length, device):
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("Expected equally sized nonempty prompt/response batches")
    if any(not len(p) or not len(r) for p, r in zip(prompts, responses)):
        raise ValueError("Every trajectory requires a prompt and response")
    if max_length is not None:
        if max_length < 2:
            raise ValueError("max_length must leave space for at least one prompt and response token")
        # Keep the prompt suffix and response prefix, as in the dataset packer.
        # Reserve at least one response label even if the prompt alone is long.
        prompts = [p[-(max_length - 1):] for p in prompts]
        responses = [r[:max_length - len(p)] for p, r in zip(prompts, responses)]
    lengths = [len(p) + len(r) - 1 for p, r in zip(prompts, responses)]
    width = max(lengths)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        sequence = torch.cat((torch.as_tensor(prompt, device=device),
                              torch.as_tensor(response, device=device))).long()
        length = lengths[i]
        ids[i, :length] = sequence[:-1]
        mask[i, :length] = 1
        labels[i, len(prompt) - 1:length] = sequence[len(prompt):]
    batch = {"input_ids": ids, "attention_mask": mask}
    if model_type == "gpt2":
        batch["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0) * mask
    return batch, {"label": labels, "loss_mask": (labels != -100).float()}
