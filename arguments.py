# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
import numpy as np


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default="gpt2")
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-ckpt-name", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--no-value", action="store_true")
    group.add_argument("--dropout-path-rate", type=float, default=None)
    group.add_argument("--fp32", action="store_true")
    group.add_argument("--bf16", action="store_true")
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--type", type=str, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")
    group.add_argument("--save-rollout", action="store_true")
    group.add_argument("--eb-sample-times", type=int, default=3)
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--gen-num", type=int, default=-1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--t-max-prompt-length", type=int, default=1536,
                       help="Reference prompt limit, including self-distillation context")
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    
    group.add_argument("--prompt-data-dir", type=str)
    group.add_argument("--lm-data-dir", type=str)
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-rw", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    
    group.add_argument("--only-prompt", action="store_true")
    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=-1,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--t-max-length', type=int, default=2560,
                       help='Reference prompt + response limit in self-distillation')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-ppo", type=int, default=42)
    group.add_argument("--seed-lm", type=int, default=7)
    group.add_argument('--epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument('--training-epochs', type=int, default=10000)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument("--kd-ratio", type=float, default=None)

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--warmup-ratio', type=float, default=0.0),
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam', 'wrmup_cosine'],
                       help='learning rate decay function')
    group.add_argument("--scheduler-name", type=str, default="constant_trm")

    group.add_argument("--w-span-loss", type=float, default=1.0)

    return parser


def add_ppo_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('ppo', 'ppo configurations')
    
    group.add_argument("--reward-scaling", type=float, default=None)
    group.add_argument("--cliprange-reward", type=float, default=1)
    group.add_argument("--ppo-epochs", type=int, default=None)
    group.add_argument("--num-rollouts", type=int, default=256)
    group.add_argument("--num-rollouts-per-device", type=int, default=None)
    group.add_argument("--cliprange", type=float, default=0.2)
    group.add_argument("--chunk-size", type=int, default=None)
    group.add_argument("--gamma", type=float, default=0.95)
    
    return parser


def add_minillm_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('minillm', 'minillm configurations')
    
    group.add_argument("--length-norm", action="store_true")
    group.add_argument("--single-step-reg", action="store_true")
    group.add_argument("--teacher-mixed-alpha", type=float, default=None)
    group.add_argument("--lm-coef", type=float, default=1)
    
    return parser


def add_distillation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('rvd', 'reasoning velocity and logit distillation')

    # Reasoning velocity distillation on marker-delimited JSONL trajectories.
    group.add_argument("--step-pooling", choices=["mean", "last_token"], default="mean")
    group.add_argument("--step-separator", type=str, default="\n\n",
                       help="Literal reasoning-step separator; defaults to a blank line (LF LF)")
    group.add_argument("--magnitude-normalization", choices=["zscore", "mean"], default="zscore")
    group.add_argument("--eps", type=float, default=1e-6)
    group.add_argument("--ce-weight", type=float, default=1.0)
    group.add_argument("--mag-weight", type=float, default=1.0)
    group.add_argument("--gram-weight", type=float, default=1.0)
    group.add_argument("--logit-weight", type=float, default=1.0)
    group.add_argument("--distill-top-k", type=int, default=32,
                       help="Teacher top-k vocabulary candidates per response position for distillation")
    group.add_argument("--distill-temperature", type=float, default=1.0,
                       help="Distillation temperature for teacher-selected top-k logits")

    # Explicit modes belong to finetune_v2; the original RVD entrypoint is unchanged.
    group.add_argument("--distill-mode", choices=["off_policy", "on_policy", "self_distill", "opsd"], default=None)
    group.add_argument("--kd-loss", choices=["fkl", "rkl", "sfkl", "srkl", "jsd", "tvd"], default=None)
    group.add_argument("--skew-alpha", type=float, default=0.1)
    group.add_argument("--off-policy-geometry", action="store_true",
                       help="Enable magnitude/Gram velocity losses for off_policy only (legacy flag)")
    group.add_argument("--geometry", action="store_true",
                       help="Enable magnitude/Gram velocity losses for off_policy, self_distill, and on_policy batches")
    group.add_argument("--cka", action="store_true",
                       help="Enable step-level linear CKA for off_policy, self_distill, and on_policy batches")
    group.add_argument("--cka-weight", type=float, default=1.0,
                       help="Weight of CKA loss when --cka is enabled")
    group.add_argument("--menger-weight", type=float, default=0.0,
                       help="Weight of Menger-curvature MSE in OFF/SELF/ON modes; 0 disables it")
    group.add_argument("--menger-eps", type=float, default=1.0e-6,
                       help="Numerical threshold for degenerate Menger-curvature triples")
    group.add_argument("--disable-lm-loss", action="store_true",
                       help="Optimize distillation alone, without kd-ratio scaling")
    group.add_argument("--self-distill-context-drop-ratio", type=float, default=0.5,
                       help="Upper bound for a per-sample drop probability drawn uniformly from [0, bound]")
    group.add_argument("--self-distill-context-fixed-drop-ratio", type=float, default=None,
                       help="Fixed fraction of randomly selected context steps to remove; "
                            "overrides stochastic context dropping when set")
    group.add_argument("--self-distill-context-max-tokens", type=int, default=512,
                       help="Maximum token count of sampled self-distillation context, before template text")
    group.add_argument("--self-distill-eval-seed", type=int, default=1234,
                       help="Fixed seed for comparable self-distillation dev contexts")
    group.add_argument("--self-distill-context-template",
                       default="\n\nAdditional context:\n{context}\n\n")
    group.add_argument("--opsd-context-template",
                       default="\n\nVerified solution:\n{context}\n\nSolve the problem again using this solution as guidance.\n\n")
    group.add_argument("--opsd-token-clip", type=float, default=0.05,
                       help="Cap each positive vocabulary-level FKL contribution; 0 disables clipping")
    # Legacy alias for on_policy in finetune_v2.
    group.add_argument("--student-gen", action="store_true", help=argparse.SUPPRESS)

    from src.adaptive import AdaptiveConfig, DEPRECATED_ADAPTIVE_ARGUMENTS
    defaults = AdaptiveConfig()
    group.add_argument("--dual-adaptive-exposure", "--adaptive-on-policy",
                       dest="adaptive_on_policy", action="store_true",
                       help="Sample one enabled distillation mode per optimizer step using dev discrepancies")
    group.add_argument("--adaptive-mode-set", choices=["all", "on_self", "off_self"],
                       default="all",
                       help="Modes available to adaptive routing; pairwise choices are ablations")
    for name in ("rho_self_init", "rho_on_init", "rho_self_max", "rho_on_max",
                 "rho_self_increment", "rho_on_increment"):
        group.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    group.add_argument("--adaptive-deterioration-threshold", "--adaptive-threshold",
                       dest="adaptive_deterioration_threshold", type=float,
                       default=defaults.deterioration_threshold)
    group.add_argument("--adaptive-eps", type=float, default=defaults.eps)
    for name in DEPRECATED_ADAPTIVE_ARGUMENTS:
        group.add_argument("--" + name.replace("_", "-"), default=argparse.SUPPRESS,
                           help=argparse.SUPPRESS)

    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    
    return parser


def add_peft_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--peft", type=str, default=None)
    group.add_argument("--peft-lora-r", type=int, default=16)
    group.add_argument("--peft-lora-alpha", type=int, default=64)
    group.add_argument("--peft-lora-dropout", type=float, default=0.1)
    group.add_argument("--peft-name", type=str, default=None)
    group.add_argument("--peft-path", type=str, default=None)
    group.add_argument("--teacher-peft-name", type=str, default=None)
    group.add_argument("--teacher-peft-path", type=str, default=None)
    return parser


def get_args(default_type=None):
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_ppo_args(parser)
    parser = add_minillm_args(parser)
    parser = add_distillation_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    if default_type is not None:
        parser.set_defaults(type=default_type)
    args, unknown = parser.parse_known_args()

    if args.type == "sft_cot":
        if args.teacher_model_path is not None or args.teacher_peft_path is not None:
            parser.error("--type sft_cot trains with CE only and must not receive a teacher")
        if args.kd_ratio is not None or args.student_gen:
            parser.error("--type sft_cot does not use --kd-ratio or --student-gen")
        if not 0 < args.max_prompt_length < args.max_length:
            parser.error("Require 0 < --max-prompt-length < --max-length")

    if args.type == "rvd":
        import math
        if args.do_train and not args.teacher_model_path:
            parser.error("--type rvd requires --teacher-model-path for training")
        if args.student_gen:
            parser.error("--type rvd uses paired training responses and does not support --student-gen")
        if args.kd_ratio is not None:
            parser.error("--type rvd uses --{ce,mag,gram}-weight instead of --kd-ratio")
        if args.do_train:
            if args.distill_top_k < 1 or (args.logit_weight > 0 and args.distill_top_k < 2):
                parser.error("--distill-top-k must be at least 2 when top-k KL is enabled")
            if not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
                parser.error("--distill-temperature must be finite and positive")
            weights = (args.ce_weight, args.mag_weight, args.gram_weight, args.logit_weight)
            if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) == 0:
                parser.error("RVD weights must be finite, nonnegative, and not all zero")
            if not math.isfinite(args.eps) or args.eps <= 0:
                parser.error("--eps must be finite and positive")
        if not 0 < args.max_prompt_length < args.max_length:
            parser.error("Require 0 < --max-prompt-length < --max-length")
        if args.do_train and not 0 < args.t_max_prompt_length < args.t_max_length:
            parser.error("Require 0 < --t-max-prompt-length < --t-max-length")
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
    args.n_gpu = args.n_gpu * args.n_nodes
        
    if args.type == "eval_main":
        ckpt_name = None
        if args.ckpt_name is not None:
            ckpt_name = args.ckpt_name
        if args.peft_name is not None:
            ckpt_name = args.peft_name

        if ckpt_name is not None:
            tmp = ckpt_name.split("/")
            if tmp[-1].isdigit():
                ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                ckpt_name = "_".join(tmp)

        save_path = os.path.join(
            args.save,
            f"{args.data_names}-{args.max_length}" + (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else ""),
            ckpt_name,
            f"{args.seed}",
        )
        args.save = save_path
    elif args.type == "lm":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "kd":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-kd{args.kd_ratio}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "gen":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}"),
            (f"t{args.temperature}-l{args.max_length}"),
        )
        args.save = save_path
    elif args.type == "minillm":
        ppo_prefix = f"pe{args.ppo_epochs}" + \
                     (f"_rs{args.reward_scaling}" if args.ppo_epochs is not None else "") + \
                     (f"_nr{args.num_rollouts}" if args.num_rollouts is not None else "") + \
                     (f"_ln" if args.length_norm else "") + \
                     (f"_sr" if args.single_step_reg else "") + \
                     (f"_tm{args.teacher_mixed_alpha}" if args.teacher_mixed_alpha is not None else "")
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-lm{args.lm_coef}-len{args.max_length}" + \
                (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")) + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else ""),
            ppo_prefix + args.save_additional_suffix
        )
        args.save = save_path
        args.num_rollouts_per_device = args.num_rollouts // args.n_gpu
        
        if args.warmup_iters > 0:
            assert args.scheduler_name is not None

    return args
