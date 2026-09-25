import time
import os
import copy
import random
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

import json
from tqdm import tqdm
import math

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import get_tokenizer, get_model

from src.sampler import SampleGenerator
from src.losses import forward_kl, reverse_kl, js_distance, tv_distance
from src.losses import skewed_forward_kl, skewed_reverse_kl
from src.menger import menger_loss_per_response
from src.trajectory import find_step_spans, reasoning_geometry_loss, reasoning_cka_loss
from src.modes import (align_response_logits,
                           validate_mode_args, require_shared_vocabulary,
                           geometry_enabled_for_mode, menger_enabled_for_mode)
from src.self_distill import (make_reference_model, prepare_self_distill_batches,
                              refresh_reference_model)
from src.opsd import (prepare_opsd_reference_batch, opsd_forward_kl,
                      fixed_base_teacher_forward)
from src.adaptive import (AdaptiveConfig, AdaptiveScheduler,
                               OptimizerStepModeRouter)

torch.set_num_threads(4)


@contextmanager
def capture_last_hidden(model, enabled):
    """Capture only the final hidden state passed into the LM head."""
    captured = []
    if not enabled:
        yield captured
        return
    module = model.module if hasattr(model, "module") else model
    output_embeddings = module.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model does not expose output embeddings for Menger loss")

    def capture(_module, inputs):
        if not inputs:
            raise RuntimeError("LM head received no hidden-state input")
        captured.append(inputs[0])

    handle = output_embeddings.register_forward_pre_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


def captured_hidden(captured):
    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one LM-head invocation, captured {len(captured)}"
        )
    return captured[0]


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    config.is_model_parallel = False
    try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
    except:
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
        model = model.half()

    if args.teacher_peft_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.teacher_peft_path)
        model = model.merge_and_unload()
        print("merge_and_unload")

    if dist.get_rank() == 0:
        print(' > number of parameters: {}'.format(
            sum([p.nelement() for p in model.parameters()])), flush=True)

    model.requires_grad_(False)
    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    import deepspeed
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer, teacher_tokenizer=None):
    data = {}
    if args.do_train:
        data["train"] = LMTrainDataset(
            args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio,
            teacher_tokenizer=teacher_tokenizer,
            with_teacher=args.teacher_model_path is not None,
            distill_mode=args.distill_mode,
            geometry=args.geometry or args.off_policy_geometry or args.cka,
        )
        print_rank("train num", len(data["train"]))
        if args.eval_interval:
            adaptive = getattr(args, "adaptive_on_policy", False)
            dev_args = copy.copy(args)
            data["dev"] = LMTrainDataset(dev_args, tokenizer, args.data_dir, "valid",
                                        args.dev_num, args.dev_ratio,
                                        teacher_tokenizer=teacher_tokenizer,
                                        with_teacher=adaptive,
                                        distill_mode="off_policy" if adaptive else None)
    if args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test",
                                     args.dev_num, args.dev_ratio, with_teacher=False)
    return data


def _teacher_topk_logits(args, teacher_logits, student_logits):
    """Gather each model's logits at the teacher's top-k token IDs per position."""
    vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
    top_k = getattr(args, "distill_top_k", 32)
    if top_k < 2 or vocab_size < 2:
        raise ValueError("Top-k distillation requires at least two shared vocabulary tokens")
    teacher_logits, token_ids = teacher_logits[..., :vocab_size].topk(
        min(top_k, vocab_size), dim=-1)
    student_logits = student_logits[..., :vocab_size].gather(-1, token_ids)
    return teacher_logits, student_logits


def _distil_loss_on_topk(args, teacher_logits, no_model_batch, student_logits):
    temperature = getattr(args, "distill_temperature", 1.0)
    logits = student_logits.float() / temperature
    teacher_logits = teacher_logits.float() / temperature

    if args.kd_loss == "sfkl":
        distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif args.kd_loss == "srkl":
        distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif args.kd_loss == "jsd":
        distil_loss = js_distance(logits, teacher_logits, no_model_batch)
    elif args.kd_loss == "tvd":
        distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
    elif args.kd_loss == "fkl":
        distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
    elif args.kd_loss == "rkl":
        distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
    else:
        raise NotImplementedError(args.kd_loss)
    # TV is a probability distance; temperature-squared scaling is for KL/JSD.
    return distil_loss if args.kd_loss == "tvd" else distil_loss * temperature ** 2


def get_distil_loss(args, teacher_logits, no_model_batch, logits):
    """Distill on teacher-selected top-k tokens at supervised response positions."""
    if not (no_model_batch["label"] != -100).any():
        return logits.reshape(-1)[:0].sum()
    teacher_topk, student_topk = _teacher_topk_logits(args, teacher_logits, logits)
    return _distil_loss_on_topk(args, teacher_topk, no_model_batch, student_topk)


def get_adaptive_discrepancy(args, teacher_logits, labels, student_logits):
    """Measure teacher/student divergence on already selected response positions."""
    teacher_topk, student_topk = _teacher_topk_logits(args, teacher_logits, student_logits)
    kd = _distil_loss_on_topk(args, teacher_topk, {"label": labels}, student_topk)
    if args.kd_loss not in ("fkl", "sfkl"):
        return kd
    # Both forward losses optimize cross-entropy. On fresh ON trajectories,
    # teacher entropy varies even when student and teacher match exactly.
    temperature = getattr(args, "distill_temperature", 1.0)
    teacher_logprobs = F.log_softmax(
        teacher_topk.float() / temperature, dim=-1)
    teacher_entropy = -(teacher_logprobs.exp() * teacher_logprobs).sum(-1).mean()
    return (kd - teacher_entropy * temperature ** 2).clamp_min(0)


def teacher_batch_for_response(args, student_batch):
    teacher_batch = {key: value for key, value in student_batch.items() if key != "position_ids"}
    if (args.teacher_model_type or args.model_type) == "gpt2":
        mask = teacher_batch["attention_mask"]
        teacher_batch["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0) * mask
    return teacher_batch


def evaluate_adaptive_losses(args, tokenizer, model, teacher_model, reference_model,
                             dataset, device, monitored_modes=AdaptiveScheduler.MODES):
    """Return token-weighted dev discrepancies for monitored adaptive modes."""
    if not dataset.with_teacher or not dataset.needs_self_distill_context:
        raise ValueError("Adaptive dev data must include canonical self-distillation steps")
    evaluation_modes = tuple(
        mode for mode in ("self_distill", "on_policy") if mode in monitored_modes)
    if not evaluation_modes:
        raise ValueError("Adaptive routing requires SELF or ON evaluation")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False,
                                 rank=rank, num_replicas=world_size)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                            num_workers=args.num_workers, collate_fn=dataset.collate)
    generator = (SampleGenerator(args, tokenizer, do_sample=False)
                 if "on_policy" in evaluation_modes else None)
    # One token-loss sum and token count per evaluated mode.
    stats = torch.zeros(2 * len(evaluation_modes), dtype=torch.float64, device=device)
    offset = 0
    was_training = model.training
    model.eval()
    teacher_model.eval()
    reference_model.eval()
    try:
        with torch.no_grad():
            for model_batch, metadata, gen_data, _, _ in dataloader:
                dataset.move_to_device(model_batch, metadata, gen_data, device)
                rows = torch.arange(model_batch["input_ids"].shape[0], device=device) + offset
                valid_rows = rows * world_size + rank < len(dataset)
                offset += rows.numel()
                for mode_index, mode in enumerate(evaluation_modes):
                    start = 2 * mode_index
                    if mode == "self_distill":
                        sample_ids = (rows * world_size + rank).tolist()
                        rngs = [random.Random(args.self_distill_eval_seed + sample_id)
                                for sample_id in sample_ids]
                        student_batch, student_meta, teacher_batch, teacher_meta = prepare_self_distill_batches(
                            args, tokenizer, model_batch, metadata, rngs)
                        source_model = reference_model
                    else:
                        student_batch = generator.run_sample(model, gen_data)
                        labels = student_batch.pop("no_model_batch")
                        student_meta = {"label": labels}
                        teacher_batch = teacher_batch_for_response(args, student_batch)
                        teacher_meta = student_meta
                        source_model = teacher_model
                    student_logits = model(**student_batch, use_cache=False, return_dict=True).logits
                    teacher_logits = source_model(**teacher_batch, use_cache=False, return_dict=True).logits
                    aligned_student, aligned_teacher, response_labels = align_response_logits(
                        student_logits, student_meta["label"], teacher_logits, teacher_meta["label"])
                    keep = valid_rows[:, None].expand_as(student_meta["label"])
                    keep = keep[student_meta["label"] != -100]
                    count = keep.sum()
                    if count:
                        kd = get_adaptive_discrepancy(
                            args, aligned_teacher[keep], response_labels[keep], aligned_student[keep])
                        stats[start] += kd.double() * count
                        stats[start + 1] += count
    finally:
        model.train(was_training)
    dist.all_reduce(stats, dist.ReduceOp.SUM)
    losses = {}
    for mode_index, mode in enumerate(evaluation_modes):
        loss_sum, token_count = stats[2 * mode_index:2 * mode_index + 2]
        if (token_count == 0).item():
            raise ValueError(f"Adaptive {mode} dev evaluation has no response tokens")
        loss = loss_sum / token_count
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite adaptive {mode} dev KD loss")
        losses[mode] = loss.item()
    return losses


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")
    if args.load:
        raise ValueError("--load is unavailable without training-state checkpoints; start a new run with --model-path or --peft-path")

    if args.model_parallel:
        raise NotImplementedError
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset["train"], sampler=sampler, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    adaptive = getattr(args, "adaptive_on_policy", False)
    scheduler = (AdaptiveScheduler(
        AdaptiveConfig.from_args(args), seed=getattr(args, "seed", 42),
        mode_set=getattr(args, "adaptive_mode_set", "all"))
        if adaptive else None)
    if adaptive and (teacher_model is None or "dev" not in dataset or args.eval_interval < 1):
        raise ValueError("Adaptive training requires a teacher, dev data, and positive eval_interval")
    student_gen = ((adaptive and "on_policy" in scheduler.enabled_modes)
                   or args.distill_mode in ("on_policy", "opsd"))
    student_generator = SampleGenerator(args, tokenizer) if student_gen else None
    if teacher_model is not None:
        teacher_model.requires_grad_(False)
        teacher_model.eval()
    reference_model = None
    reference_eval_step = 0
    if adaptive or args.distill_mode == "self_distill":
        reference_model = make_reference_model(model.module)
    if args.distill_mode == "opsd" and not callable(getattr(model.module, "disable_adapter", None)):
        raise ValueError("OPSD fixed teacher requires a PEFT model with disable_adapter()")
    context_rng = random.Random(args.seed + dp_rank)

    step, global_step = 1, 1
    mode_router = OptimizerStepModeRouter(scheduler, args.distill_mode)
    total_time, log_steps = 0.0, 0
    # loss, distil, lm, kl, magnitude, direction, CKA, Menger, context tokens,
    # generated samples
    total_losses = torch.zeros(10, dtype=torch.float64, device=device)
    log_modes = (*AdaptiveScheduler.MODES, "opsd")
    mode_kl_totals = dict.fromkeys(log_modes, 0.)
    mode_log_counts = dict.fromkeys(log_modes, 0)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for model_batch, no_model_batch, gen_data, t_model_batch, t_no_model_batch in train_dataloader:
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)

            if model_batch["input_ids"].is_cuda:
                torch.cuda.synchronize()
            st_time = time.time()

            selected_mode = mode_router.for_microbatch(device)
            student_gen = selected_mode in ("on_policy", "opsd")
            if selected_mode == "opsd":
                source_model = model
            elif selected_mode == "self_distill":
                source_model = reference_model
            else:
                source_model = teacher_model
            use_geometry = source_model is not None and geometry_enabled_for_mode(args, selected_mode)
            use_menger = source_model is not None and menger_enabled_for_mode(args, selected_mode)
            data_source = "fresh_on_policy" if student_gen else "canonical"

            if selected_mode == "self_distill":
                data_source = "canonical_context_subset"
            elif selected_mode == "opsd":
                data_source = "student_rollout_gold_conditioned_reference"

            # Generate a fresh student trajectory only for modes that require one.
            if student_gen:
                if selected_mode == "on_policy":
                    step_marker_ids = no_model_batch["step_marker_ids"]
                if selected_mode == "opsd":
                    opsd_records = no_model_batch["opsd_records"]
                    opsd_solutions = no_model_batch["opsd_solutions"]
                model_batch = student_generator.run_sample(model, gen_data)
                labels = model_batch.pop("no_model_batch")
                no_model_batch = {"label": labels, "loss_mask": (labels != -100).float()}
                if selected_mode == "on_policy":
                    no_model_batch["step_marker_ids"] = step_marker_ids

            # Prepare the reference batch on exactly the selected response tokens.
            if selected_mode == "self_distill":
                model_batch, no_model_batch, t_model_batch, t_no_model_batch = prepare_self_distill_batches(
                    args, tokenizer, model_batch, no_model_batch, context_rng)
            elif selected_mode == "on_policy":
                if use_geometry:
                    no_model_batch["step_spans"] = find_step_spans(
                        model_batch["input_ids"], model_batch["attention_mask"],
                        no_model_batch["label"], no_model_batch["step_marker_ids"])
                t_model_batch = teacher_batch_for_response(args, model_batch)
                t_no_model_batch = no_model_batch
            elif selected_mode == "opsd":
                t_model_batch, t_no_model_batch = prepare_opsd_reference_batch(
                    args, tokenizer, model_batch, no_model_batch["label"],
                    opsd_records, opsd_solutions)
            else:
                t_model_batch = teacher_batch_for_response(args, model_batch)
                t_no_model_batch = no_model_batch

            with capture_last_hidden(model, use_menger) as student_capture:
                outputs = model(
                    **model_batch, use_cache=False,
                    output_hidden_states=use_geometry, return_dict=True)

            logits = outputs.logits
            h_stu = (captured_hidden(student_capture) if use_menger else
                     outputs.hidden_states[-1] if use_geometry else None)
            # Fresh ON-policy trajectories are optimized only through their
            # teacher/reference objectives, without a next-token CE term.
            use_lm_loss = not args.disable_lm_loss and selected_mode != "on_policy"
            lm_loss = logits.reshape(-1)[:0].sum()
            if use_lm_loss:
                lm_loss = loss_func(
                    logits.float().reshape(-1, logits.shape[-1]), no_model_batch["label"].reshape(-1))
                lm_loss = lm_loss / (no_model_batch["label"] != -100).sum().clamp_min(1)

            kl_loss = magnitude_loss = gram_loss = cka_loss = menger_loss = distil_loss = logits.reshape(-1)[:0].sum()
            if source_model is not None:
                if selected_mode == "opsd":
                    teacher_outputs = fixed_base_teacher_forward(model, t_model_batch)
                else:
                    with torch.no_grad():
                        with capture_last_hidden(source_model, use_menger) as teacher_capture:
                            teacher_outputs = source_model(
                                **t_model_batch, use_cache=False,
                                output_hidden_states=use_geometry, return_dict=True)
                h_tea = (captured_hidden(teacher_capture) if use_menger else
                         teacher_outputs.hidden_states[-1] if use_geometry else None)

                response_lengths = (no_model_batch["label"] != -100).sum(-1)
                logits, teacher_logits, response_labels = align_response_logits(
                    logits, no_model_batch["label"], teacher_outputs.logits, t_no_model_batch["label"])
                new_no_model_batch = {"label": response_labels}

                kl_loss = (opsd_forward_kl(logits, teacher_logits, args.opsd_token_clip,
                                           response_lengths)
                           if selected_mode == "opsd" else
                           get_distil_loss(args, teacher_logits, new_no_model_batch, logits))
                distil_loss = kl_loss
                if use_geometry:
                    if args.cka:
                        cka_loss = reasoning_cka_loss(
                            h_stu, h_tea, no_model_batch["step_spans"],
                            t_no_model_batch["step_spans"],
                            pooling=args.step_pooling, eps=args.eps)
                        distil_loss = distil_loss + args.cka_weight * cka_loss
                    else:
                        magnitude_loss, gram_loss = reasoning_geometry_loss(
                            h_stu, h_tea, no_model_batch["step_spans"], t_no_model_batch["step_spans"],
                            no_model_batch["label"], t_no_model_batch["label"],
                            pooling=args.step_pooling, normalization=args.magnitude_normalization, eps=args.eps)
                        distil_loss = distil_loss + args.mag_weight * magnitude_loss + args.gram_weight * gram_loss

                if use_menger:
                    menger_loss = menger_loss_per_response(
                        h_stu, h_tea,
                        model_batch, no_model_batch["label"],
                        t_model_batch, t_no_model_batch["label"],
                        tokenizer, separator=args.step_separator,
                        eps=args.menger_eps).mean()

                if use_lm_loss:
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
                else:
                    loss = distil_loss
                loss = loss + args.menger_weight * menger_loss
            else:
                loss = lm_loss

            finite = torch.isfinite(loss).long()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=dp_group)
            if not bool(finite):
                raise FloatingPointError("Non-finite loss on at least one rank")

            model.backward(loss)
            # Query before step(): DeepSpeed owns optimizer/accumulation boundaries.
            boundary = model.is_gradient_accumulation_boundary()
            model.step()
            if boundary:
                mode_router.on_optimizer_step()

            context_tokens = no_model_batch.get("self_distill_context_tokens") if selected_mode == "self_distill" else None
            context_count = context_tokens.sum() if context_tokens is not None else loss.new_zeros(())
            fresh_count = loss.new_tensor(model_batch["input_ids"].shape[0] if student_gen else 0)
            global_losses = torch.stack((
                loss, distil_loss, lm_loss, kl_loss, magnitude_loss, gram_loss, cka_loss,
                menger_loss, context_count, fresh_count)).detach().double()
            dist.all_reduce(global_losses, dist.ReduceOp.SUM, group=dp_group)
            global_losses[:8] /= dp_world_size
            mode_kl_totals[selected_mode] += global_losses[3].item()
            mode_log_counts[selected_mode] += 1
            total_losses += global_losses
            log_steps += 1

            if model_batch["input_ids"].is_cuda:
                torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_time += elapsed_time

            # Logging
            def get_log(log_losses, log_time, aggregate=False):
                (log_loss, log_distil_loss, log_lm, log_kl, log_mag, log_dir,
                 log_cka, log_menger, context, fresh) = log_losses.tolist()
                log_str = (
                    "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
                    "loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:.4f} | "
                    "micro time: {:.3f} | step time: {:.3f}"
                ).format(
                    epoch, step, args.total_iters * args.gradient_accumulation_steps,
                    global_step, args.total_iters, log_loss, log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time, log_time)
                log_str += (
                    f" | mode: {'adaptive' if adaptive and aggregate else selected_mode} | loss/lm: {log_lm:.6f} | loss/total: {log_loss:.6f}"
                    f" | loss/mag: {log_mag:.6f} | loss/dir: {log_dir:.6f} | loss/cka: {log_cka:.6f}"
                    f" | loss/menger: {log_menger:.6f}"
                    f" | data/source: {'mixed' if adaptive and aggregate else data_source} | data/on_policy_fresh_count: {fresh:.0f}"
                    f" | data/self_distill_context_tokens: {context:.0f}")
                for mode in log_modes:
                    mode_kl = (mode_kl_totals[mode] / max(1, mode_log_counts[mode]) if aggregate
                               else log_kl if mode == selected_mode else 0.)
                    log_str += f" | loss/{mode}_kl: {mode_kl:.6f}"
                return log_str

            if args.mid_log_num > 0:
                mid_log_step = max(1, args.gradient_accumulation_steps // args.mid_log_num)
                if step % mid_log_step == 0:
                    print_rank(get_log(global_losses, 0))

            if boundary and (global_step % args.log_interval == 0 or global_step == args.total_iters):
                log_losses = total_losses.clone()
                log_losses[:8] /= log_steps
                log_str = get_log(log_losses, total_time, aggregate=True)
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_losses.zero_()
                total_time, log_steps = 0.0, 0
                mode_kl_totals = dict.fromkeys(log_modes, 0.)
                mode_log_counts = dict.fromkeys(log_modes, 0)

            # Evaluation
            if boundary and args.eval_interval and global_step % args.eval_interval == 0:
                evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device,
                         global_step=global_step)
                if scheduler:
                    adaptive_losses = evaluate_adaptive_losses(
                        args, tokenizer, model, teacher_model, reference_model,
                        dataset["dev"], device, scheduler.monitored_modes)
                    scheduler_log = scheduler.on_evaluation(
                        self_loss=adaptive_losses.get("self_distill"),
                        on_loss=adaptive_losses.get("on_policy"))
                    scheduler_log["global_step"] = global_step
                    log_str = "scheduler | " + json.dumps(scheduler_log, sort_keys=True, allow_nan=False)
                    print_rank(log_str)
                    save_rank(log_str, os.path.join(args.save, "log.txt"))
                if args.do_eval:
                    evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device,
                             global_step=global_step)
                model.train()
                if reference_model is not None:
                    refresh_reference_model(reference_model, model.module)
                    reference_eval_step = global_step

            # Save model weights for evaluation or inference.
            if boundary and args.save and (
                (args.save_interval and global_step % args.save_interval == 0) or global_step == args.total_iters
            ):
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    print_rank(f"Model save to {save_dir_path}")
                    tokenizer.save_pretrained(save_dir_path)
                    model.module.save_pretrained(save_dir_path, safe_serialization=False)
                    if reference_model is not None:
                        reference_dir = os.path.join(save_dir_path, "self_distill_ref")
                        reference_model.save_pretrained(reference_dir, safe_serialization=False)
                        with open(os.path.join(save_dir_path, "self_distill_reference.json"), "w") as handle:
                            json.dump({"reference_eval_step": reference_eval_step,
                                       "reference_dir": "self_distill_ref"}, handle)
                    checkpoint_file = os.environ.get("FINAL_CHECKPOINT_FILE")
                    if global_step == args.total_iters and checkpoint_file:
                        with open(checkpoint_file, "w") as handle:
                            handle.write(os.path.abspath(save_dir_path) + "\n")
                dist.barrier()

            step += 1
            if boundary:
                if global_step >= args.total_iters:
                    return model
                global_step += 1

    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device,
             global_step=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    # Sum NLL and valid tokens over unique samples, then reduce once across ranks.
    loss_stats = torch.zeros(2, dtype=torch.float64, device=device)
    local_offset = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for model_batch, no_model_batch, gen_data, _, _ in tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0)):
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            labels = no_model_batch["label"]
            # DistributedSampler may repeat rows to balance ranks; exclude repeats.
            positions = (torch.arange(labels.shape[0], device=labels.device) + local_offset) * dp_world_size + dp_rank
            labels = labels.masked_fill((positions >= len(dataset))[:, None], -100)
            token_losses = loss_func(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss_stats[0] += token_losses.double().sum()
            loss_stats[1] += (labels != -100).sum()
            local_offset += labels.shape[0]
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
    dist.all_reduce(loss_stats, dist.ReduceOp.SUM, group=dp_group)
    if loss_stats[1].item() == 0:
        raise ValueError("Evaluation dataset has no valid response tokens")
    avg_loss = (loss_stats[0] / loss_stats[1]).item()
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))[:len(dataset)]
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    res = {}
    if get_rank() == 0:
        if args.eval_gen:
            from rouge_metric import compute_metrics
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)

        
            event = str(global_step) if global_step is not None else "standalone"
            eval_dir = os.path.join(args.save, "eval", event, split)
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        log_str = f"{split} | avg_loss: {avg_loss} | {res} | epoch: {epoch} | global_step: {global_step}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return avg_loss


def main():
    from arguments import get_args
    torch.backends.cudnn.enabled = False
    
    args = get_args(default_type="kd")
    if args.load:
        raise ValueError("--load is unavailable without training-state checkpoints; start a new run with --model-path or --peft-path")
    validate_mode_args(args)
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.bf16 = ds_config.get("bf16", {}).get("enabled", False)
    args.fp32 = not ds_config.get("fp16", {}).get("enabled", False) and not args.bf16
    if ds_config.get("zero_optimization", {}).get("stage", 0) > 2:
        raise ValueError("finetune_v2 checkpoint saving supports ZeRO stages 0, 1, 2")
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    teacher_tokenizer = None
    if args.do_train and args.teacher_model_path:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
        require_shared_vocabulary(tokenizer, teacher_tokenizer)

    dataset = prepare_dataset(
        args,
        tokenizer,
        teacher_tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        micro_batches = len(dataset["train"]) // (args.batch_size * dp_world_size)
        args.train_iters_per_epoch = micro_batches // args.gradient_accumulation_steps
        if args.train_iters_per_epoch < 1:
            raise ValueError("Training data must contain at least one full optimizer step")
        if args.total_iters is None:
            if args.epochs is None or args.epochs < 1:
                raise ValueError("Provide positive --epochs or --total-iters")
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.total_iters < 1:
            raise ValueError("--total-iters must be positive")
        required_epochs = math.ceil(args.total_iters * args.gradient_accumulation_steps / micro_batches)
        args.epochs = max(args.epochs or 0, required_epochs)
        print_rank("total_iters", args.total_iters)

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)

    if not args.do_train:
        if args.do_eval:
            evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        return
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
        
    
if __name__ == "__main__":
    main()
