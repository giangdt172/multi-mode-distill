import torch
import torch.distributed as dist
from transformers import GenerationConfig

from .batching import pack_trajectories


class SampleGenerator:
    def __init__(self, args, tokenizer, do_sample=None):
        self.args = args
        self.tokenizer = tokenizer
        self.max_new_token = args.max_length - args.max_prompt_length
        self.pad_id = tokenizer.pad_token_id
        self.generation_config = GenerationConfig(
            do_sample=args.do_sample if do_sample is None else do_sample,
            top_p=(getattr(args, "gen_top_p", None) or args.top_p) if do_sample is not False else 1.0,
            top_k=args.top_k if do_sample is not False else 50,
            temperature=args.temperature if do_sample is not False else 1.0,
            repetition_penalty=args.repetition_penalty or 1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=self.pad_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    @torch.no_grad()
    def run_sample(self, model, gen_data):
        generation_model = model
        while not callable(getattr(generation_model, "generate", None)):
            generation_model = generation_model.module
        was_training = model.training
        model.eval()
        try:
            output = generation_model.generate(
                input_ids=gen_data["input_ids"],
                attention_mask=gen_data["attention_mask"],
                generation_config=self.generation_config,
                max_new_tokens=self.max_new_token,
                # Synchronize EOS termination for sharded model collectives.
                synced_gpus=dist.is_initialized() and dist.get_world_size() > 1,
            )
        finally:
            model.train(was_training)
        width = gen_data["input_ids"].shape[1]
        responses = []
        for row in output.sequences[:, width:]:
            eos = (row == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            # Include first EOS as a label, even when EOS == PAD.
            responses.append(row[:int(eos[0]) + 1] if eos.numel() else row)
        prompts = [row[mask.bool()] for row, mask in
                   zip(gen_data["input_ids"], gen_data["attention_mask"])]
        batch, metadata = pack_trajectories(
            prompts, responses, self.pad_id, self.args.model_type,
            self.args.max_length, gen_data["input_ids"].device,
        )
        batch["no_model_batch"] = metadata["label"]
        return batch
